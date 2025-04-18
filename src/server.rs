use std::collections::HashMap;
use std::time::Duration;
use std::time::Instant;

use crate::model::*;
use crate::LWE;
use crate::THREADS;
use rand::seq::SliceRandom;
use revolut::*;
use tfhe::core_crypto::prelude::lwe_ciphertext_plaintext_add_assign;
use tfhe::core_crypto::prelude::slice_algorithms::slice_wrapping_sub_scalar_mul_assign;
use tfhe::core_crypto::prelude::Plaintext;
use tfhe::shortint::ClassicPBSParameters;

const DEBUG: bool = false;

use crate::Query;
use rayon::{prelude::*, ThreadPoolBuilder};
pub struct Server {
    public_key: PublicKey,
    model: Model,
}

const BEST_MODEL_TRIES: usize = 100;

// #[allow(dead_code)]
pub struct KnnClear {
    pub top_k_distances_and_labels: Vec<(u64, u64)>,
}

impl KnnClear {
    pub fn run(k: usize, client_feature_vector: &Vec<u64>, model: &Model, delta: u64) -> Self {
        let mut distances_and_labels = model
            .model_points
            .iter()
            .map(|point| {
                (
                    squared_distance_in_clear(&point.feature_vector, &client_feature_vector),
                    point.label,
                )
            })
            .collect::<Vec<(u64, u64)>>();
        let delta_dist = (1u64 << 63) / model.dist_modulus;
        let ratio = delta / delta_dist;
        distances_and_labels
            .iter_mut()
            .for_each(|(d, _)| *d /= ratio);

        let mut distances_and_labels_sorted = distances_and_labels.clone();
        distances_and_labels_sorted.sort_by_key(|&(distance, _)| distance);

        let top_k_distances_and_labels = distances_and_labels_sorted[..k].to_vec();

        KnnClear {
            top_k_distances_and_labels,
        }
    }
}

impl Server {
    pub fn new(public_key: PublicKey, model: Model) -> Self {
        Server { public_key, model }
    }

    pub fn encode_model(&self, ctx: &Context) -> Vec<ModelPointEncoded> {
        let model_points = &self.model.model_points;
        encode_model_points(model_points, ctx)
    }

    pub fn delta_dist(&self) -> u64 {
        (1u64 << 63) / self.model.dist_modulus
    }

    // Function to calculate the squared distance of a query vector to a model point
    pub fn squared_distance(&self, query: &Query, point: &ModelPointEncoded, ctx: &Context) -> LWE {
        let m = point.m.clone();
        let m_prime = point.m_prime.clone();
        let delta_dist = self.delta_dist();

        // Step 1: Compute the inner product m(X)q(X)
        let inner_product = self
            .public_key
            .glwe_absorption_polynomial_with_fft(&query.ct, &m);

        // Step 2 : Sample extract at gamma - 1 which is <m,c>
        let m_times_c = self
            .public_key
            .glwe_extract(&inner_product, self.model.gamma - 1, ctx);

        // Step 3 : c'' = c'' - 2<m,c>
        let mut dist = query.ct_second.clone();
        slice_wrapping_sub_scalar_mul_assign(dist.as_mut(), m_times_c.as_ref(), 2);
        lwe_ciphertext_plaintext_add_assign(&mut dist, Plaintext(m_prime * delta_dist));

        // Step 4 : Lower the precision if needed
        if delta_dist != ctx.delta() as u64 {
            self.public_key
                .lower_precision(&mut dist, &ctx, self.model.dist_modulus);
        }
        dist
    }

    fn topk_distances_and_labels(
        &self,
        many_lwes: &Vec<Vec<LWE>>,
        k: usize,
        ctx: &Context,
    ) -> Vec<Vec<LWE>> {
        let topk_distances_and_labels = self
            .public_key
            .blind_topk_many_lut_par(many_lwes, k, THREADS, ctx);
        if DEBUG {
            let private_key = key(ctx.parameters());
            println!(
                "[DEBUG] top{k} distances: {:?}",
                private_key.decrypt_lwe_vector(&topk_distances_and_labels[0], ctx)
            );
            println!(
                "[DEBUG] top{k} labels: {:?}",
                private_key.decrypt_lwe_vector(&topk_distances_and_labels[1], ctx)
            );
        }
        topk_distances_and_labels
    }

    pub fn predict(
        &self,
        query: &Query,
        model_points: &Vec<ModelPointEncoded>,
        k: usize,
        ctx: &Context,
    ) -> (Vec<Vec<LWE>>, Duration, Duration) {
        let pool = ThreadPoolBuilder::new()
            .num_threads(THREADS)
            .build()
            .unwrap();

        let start = Instant::now();

        // Step 1Compute the distances
        let distances: Vec<LWE> = pool.install(|| {
            model_points
                .par_iter()
                .map(|point| self.squared_distance(query, point, ctx))
                .collect()
        });
        let end_distances = Instant::now();

        // print the distances
        if DEBUG {
            let private_key = key(ctx.parameters());
            let decrypted_distances = private_key.decrypt_lwe_vector(&distances, ctx);
            println!(
                "[DEBUG] Decrypted distances: {:?}",
                decrypted_distances
                    .iter()
                    .take(self.model.d)
                    .collect::<Vec<_>>()
            );
        }

        let dist_dur = end_distances - start;

        let start_topk = Instant::now();
        // Step 0: Encrypt the labels as LWE ciphertexts trivially
        let labels = model_points
            .iter()
            .map(|p| {
                self.public_key
                    .allocate_and_trivially_encrypt_lwe(p.label, ctx)
            })
            .collect::<Vec<LWE>>();
        // Step 2: Compute the topk labels
        let topk = self.topk_distances_and_labels(&vec![distances, labels], k, ctx);
        let end_topk = Instant::now();
        let topk_dur = end_topk - start_topk;

        (topk, dist_dur, topk_dur)
    }
}

// Repeatedly train and find the set that has the highest accuracy
// the accuracy is computed for all possible test vectors
pub fn find_best_model(
    model_size: usize,
    output_test_size: usize,
    k: usize,
    dataset: &Vec<Vec<u64>>,
    delta: u64,
    dist_modulus: u64,
) -> (Vec<Vec<u64>>, Vec<u64>, Vec<Vec<u64>>, Vec<u64>, f64) {
    let mut final_model_vec: Vec<Vec<u64>> = vec![];
    let mut final_test_vec: Vec<Vec<u64>> = vec![];
    let mut final_model_labels: Vec<u64> = vec![];
    let mut final_test_labels: Vec<u64> = vec![];
    let mut highest_accuracy: usize = 0;

    let mut rng = rand::thread_rng();
    let test_size = dataset.len() - model_size;

    // Try 10 times and take the best model
    for _ in 0..BEST_MODEL_TRIES {
        // shuffle and split model/test vector
        let mut rows = dataset.clone();
        rows.shuffle(&mut rng);
        let (model_vec, model_labels, test_vec, test_labels) =
            split_model_test(model_size, test_size, rows);

        // do knn and check accuracy
        let mut oks: usize = 0;
        for (target, expected) in test_vec.iter().zip(&test_labels) {
            // When dist modulus is set to the full message modulus,
            // the distance is the same as the squared distance in clear
            let model = Model::new(model_vec.clone(), model_labels.clone(), dist_modulus);
            let knn_clear = KnnClear::run(k, &target, &model, delta);
            let out_labels = knn_clear
                .top_k_distances_and_labels
                .iter()
                .map(|(_, l)| *l)
                .collect::<Vec<u64>>();
            let res = majority(&out_labels);
            if res == *expected {
                oks += 1;
            }
        }

        // check if our accuracy is higher
        if oks > highest_accuracy {
            final_model_vec = model_vec;
            final_model_labels = model_labels;
            final_test_vec = test_vec[..output_test_size].to_vec();
            final_test_labels = test_labels[..output_test_size].to_vec();
            highest_accuracy = oks;
        }
    }

    (
        final_model_vec,
        final_model_labels,
        final_test_vec,
        final_test_labels,
        highest_accuracy as f64 / test_size as f64,
    )
}

/// Split the feature vectors into a training and testing set.
/// The feature vectors are specified in `rows` and the last
/// element of every vector is the label.
pub fn split_model_test(
    model_size: usize,
    test_size: usize,
    rows: Vec<Vec<u64>>,
) -> (Vec<Vec<u64>>, Vec<u64>, Vec<Vec<u64>>, Vec<u64>) {
    let mut model_vec: Vec<Vec<u64>> = vec![];
    let mut test_vec: Vec<Vec<u64>> = vec![];
    let mut model_labels: Vec<u64> = vec![];
    let mut test_labels: Vec<u64> = vec![];

    for (i, mut row) in rows.into_iter().enumerate() {
        let last = row.pop().unwrap();
        if i < model_size {
            // Add to training set
            model_vec.push(row);
            model_labels.push(last);
        } else if i >= model_size && i < model_size + test_size {
            // Add to test set up to test_size
            test_vec.push(row);
            test_labels.push(last);
        } else {
            // Stop once we have enough samples
            break;
        }
    }

    (model_vec, model_labels, test_vec, test_labels)
}

pub fn majority(vs: &[u64]) -> u64 {
    assert!(!vs.is_empty());
    let max = vs
        .iter()
        .fold(HashMap::<u64, usize>::new(), |mut m, x| {
            *m.entry(*x).or_default() += 1;
            m
        })
        .into_iter()
        .max_by_key(|(_, v)| *v)
        .map(|(k, _)| k);
    max.unwrap()
}

// Calculate the squared distance between two vectors
fn squared_distance_in_clear(xs: &[u64], ys: &[u64]) -> u64 {
    xs.iter()
        .zip(ys)
        .map(|(x, y)| {
            let diff = if x > y { x - y } else { y - x };
            diff * diff
        })
        .sum()
}

#[allow(dead_code)]
pub fn decode(params: ClassicPBSParameters, x: u64) -> u64 {
    let delta = (1u64 << 63) / (params.message_modulus.0 * params.carry_modulus.0) as u64;

    //The bit before the message
    let rounding_bit = delta >> 1;

    //compute the rounding bit
    let rounding = (x & rounding_bit) << 1;

    // add the rounding bit and divide by delta
    x.wrapping_add(rounding) / delta
}
