use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::time::Duration;
use std::time::Instant;

use crate::model::*;
use crate::DEBUG;
use crate::LWE;
use crate::THREADS;
use rand::seq::SliceRandom;
use revolut::*;
use tfhe::core_crypto::prelude::lwe_ciphertext_plaintext_add_assign;
use tfhe::core_crypto::prelude::slice_algorithms::slice_wrapping_sub_scalar_mul_assign;
use tfhe::core_crypto::prelude::Plaintext;
use tfhe::shortint::ClassicPBSParameters;

use crate::Query;
use rayon::{prelude::*, ThreadPoolBuilder};
pub struct Server {
    public_key: PublicKey,
    model: Model,
}

pub const DISTANCES: [u64; 10] = [5, 5, 5, 4, 2, 6, 4, 6, 6, 6];

#[allow(dead_code)]
pub struct KnnClear {
    pub distances: Vec<u64>,
    pub distances_and_labels_sorted: Vec<(u64, u64)>,
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
        // let ratio = ctx.delta() / delta_dist;
        let ratio = delta / delta_dist;
        distances_and_labels
            .iter_mut()
            .for_each(|(d, _)| *d /= ratio);

        let mut distances_and_labels_sorted = distances_and_labels.clone();
        distances_and_labels_sorted.sort_by_key(|&(distance, _)| distance);

        let distances = distances_and_labels
            .iter()
            .map(|(d, _)| *d)
            .collect::<Vec<u64>>();

        let top_k_distances_and_labels = distances_and_labels_sorted[..k].to_vec();

        KnnClear {
            distances,
            distances_and_labels_sorted,
            top_k_distances_and_labels,
        }
    }
}

#[allow(dead_code)]
pub fn calculate_and_print_noise(dist: LWE, expected: u64, ctx: &Context, delta: u64) {
    let private_key = key(ctx.parameters());
    let noise = private_key.lwe_noise_delta(&dist, expected, delta);
    println!("noise : {:.2}", noise);
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

        let f_size = self.model.gamma;

        // Step 1: Compute the inner product m(X)q(X)
        let inner_product = self
            .public_key
            .glwe_absorption_polynomial_with_fft(&query.ct, &m);

        // Step 2 : Sample extract at feature_vector_size - 1 which is <m,c>
        let m_times_c = self
            .public_key
            .glwe_extract(&inner_product, f_size - 1, ctx);

        // Step 3 : c'' = c'' - 2<m,c>
        let mut dist = query.ct_second.clone();
        slice_wrapping_sub_scalar_mul_assign(dist.as_mut(), m_times_c.as_ref(), 2);

        lwe_ciphertext_plaintext_add_assign(&mut dist, Plaintext(m_prime * delta_dist));

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
                "top{k} distances: {:?}",
                private_key.decrypt_lwe_vector(&topk_distances_and_labels[0], ctx)
            );
            println!(
                "top{k} labels: {:?}",
                private_key.decrypt_lwe_vector(&topk_distances_and_labels[1], ctx)
            );
        }
        topk_distances_and_labels
    }

    #[allow(dead_code)]
    pub fn serialize_lwe_vector_to_file(&self, lwe_vector: &Vec<LWE>, file_path: &str) {
        let json = serde_json::to_string(lwe_vector).expect("Failed to serialize LWE vector");
        let mut file = File::create(file_path).expect("Failed to create file");
        file.write_all(json.as_bytes())
            .expect("Failed to write to file");
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
                "Decrypted distances: {:?}",
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
    println!("dataset: {:?}", dataset.len());
    println!("model_size: {:?}", model_size);
    let test_size = dataset.len() - model_size;
    println!("test_size: {:?}", test_size);

    // Try 10 times and take the best model
    for _ in 0..10 {
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

#[allow(dead_code)]
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
pub fn knn_predict_in_clear(model: &Model, query: &Vec<u64>, ctx: &Context) -> Vec<(u64, u64)> {
    let initial_modulus = model.dist_modulus;
    let final_modulus = ctx.full_message_modulus() as u64;
    let ratio = (initial_modulus / final_modulus) as u64;
    let mut distances_and_labels: Vec<(u64, u64)> = model
        .model_points
        .iter()
        .map(|point| {
            // let distance = squared_distance_in_clear(&point.feature_vector, query, modulo);
            let distance =
                squared_distance_in_clear(point.feature_vector.as_slice(), query.as_slice());
            (distance / ratio, point.label)
        })
        .collect();

    // print the 10 first distances
    println!(
        "Distances in clear: {:?}",
        distances_and_labels
            .iter()
            .take(10)
            .map(|&(distance, _)| distance)
            .collect::<Vec<_>>()
    );

    // Sort by distance
    distances_and_labels.sort_by_key(|&(distance, _)| distance);

    distances_and_labels
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

#[cfg(test)]
mod tests {
    use parameters::PARAM_MESSAGE_4_CARRY_0;
    use prelude::{
        DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
        StandardDev,
    };
    use tfhe::core_crypto::prelude::{decrypt_lwe_ciphertext, Polynomial};
    use tfhe::shortint::parameters;
    use tfhe::shortint::*;

    use super::*;
    use crate::{
        client::Client,
        model::{Model, ModelPoint},
    };

    const TEST_PARAMS: ClassicPBSParameters = ClassicPBSParameters {
        lwe_dimension: LweDimension(742),
        glwe_dimension: GlweDimension(1),
        polynomial_size: PolynomialSize(2048),
        lwe_noise_distribution: parameters::DynamicDistribution::new_gaussian_from_std_dev(
            StandardDev(0.000007069849454709433),
        ),
        glwe_noise_distribution: parameters::DynamicDistribution::new_gaussian_from_std_dev(
            StandardDev(0.00000000000000029403601535432533),
        ),
        pbs_base_log: DecompositionBaseLog(23),
        pbs_level: DecompositionLevelCount(1),
        ks_level: DecompositionLevelCount(5),
        ks_base_log: DecompositionBaseLog(3),
        message_modulus: MessageModulus(16),
        carry_modulus: CarryModulus(1),
        ..PARAM_MESSAGE_4_CARRY_0
    };

    #[test]
    fn test_knn_predict_in_clear() {
        let ctx = Context::from(TEST_PARAMS);
        let model_points = vec![
            ModelPoint {
                feature_vector: vec![1, 2, 3],
                label: 2,
            },
            ModelPoint {
                feature_vector: vec![0, 1, 0],
                label: 1,
            },
            ModelPoint {
                feature_vector: vec![1, 0, 0],
                label: 1,
            },
            ModelPoint {
                feature_vector: vec![0, 2, 0],
                label: 1,
            },
            ModelPoint {
                feature_vector: vec![2, 3, 1],
                label: 2,
            },
        ];

        let model = Model {
            model_points,
            d: 5,
            gamma: 3,
            dist_modulus: 16,
        };

        let query = vec![1, 1, 1];

        let result = knn_predict_in_clear(&model, &query, &ctx);

        assert_eq!(result.len(), 5);
        assert_eq!(result[0].1, 2);
        assert_eq!(result[1].1, 1);
        assert_eq!(result[2].1, 1);
    }

    #[test]
    fn test_lower_precision() {
        let final_params = TEST_PARAMS;
        let mut ctx = Context::from(final_params);

        let initial_modulus = MessageModulus(64);
        let initial_params = ClassicPBSParameters {
            message_modulus: initial_modulus,
            ..final_params
        };

        let client = Client::new(&ctx.parameters(), vec![0; 3]);

        let final_modulus = ctx.message_modulus();
        let ratio = (initial_modulus.0 / final_modulus.0) as u64;

        for m in 0..initial_modulus.0 as u64 {
            let mut ct =
                client
                    .private_key
                    .lwe_encrypt_with_modulus(m, initial_modulus.0 as u64, &mut ctx);

            // check for correct decryption
            let encoded = decrypt_lwe_ciphertext(&client.private_key.get_big_lwe_sk(), &ct);
            let pt = decode(initial_params, encoded.0);
            assert_eq!(pt, m);

            // lower the precision
            client
                .public_key
                .lower_precision(&mut ct, &ctx, initial_modulus.0 as u64);

            let expected = m / ratio;
            let actual = client.private_key.decrypt_lwe(&ct, &ctx);
            println!(
                "m={m}, actual={actual}, expected={expected}, noise={:.2}",
                client.private_key.lwe_noise(&ct, expected, &ctx)
            );
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_compute_distance_lower_precision() {
        let final_params = TEST_PARAMS;
        let mut ctx = Context::from(final_params);
        let initial_modulus = MessageModulus(64);

        let client = Client::new(&ctx.parameters(), vec![0; 3]);
        let final_modulus = ctx.message_modulus();
        let ratio = (initial_modulus.0 / final_modulus.0) as u64;

        for i in 0..8u64 {
            // 8*8 = 64 (initial_modulus)
            for j in 0..4 {
                // we want the maximum to be 7*7 + 3*3 < 64
                let data = vec![vec![i, 0, 0, 0u64]];
                let target = vec![0, j, 0, 0u64];
                let model_points = data
                    .iter()
                    .map(|x| ModelPoint {
                        feature_vector: x.clone(),
                        label: 0,
                    })
                    .collect();
                let model = Model {
                    model_points,
                    d: data.len(),
                    gamma: target.len(),
                    dist_modulus: initial_modulus.0 as u64,
                };

                let server = Server::new(client.public_key.clone(), model.clone());
                let model_point_encoded = encode_model_points(&model.model_points, &ctx);
                let query = client.create_query(&mut ctx, model.dist_modulus); // chiffre avec le delta de la distance

                // compute the distance and lower the precision if needed (inside the function)
                let distances = server.squared_distance(&query, &model_point_encoded[0], &ctx);

                let expected = (j * j + i * i) / ratio;
                // let actual = client.key.decrypt(&distances);
                let actual = client.private_key.decrypt_lwe(&distances, &ctx);
                println!(
                    "i={i}, j={j}, actual={actual}, expected={expected}, noise={:.2}",
                    client.private_key.lwe_noise(&distances, expected, &ctx)
                );
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn test_encrypt_and_bootstrap() {
        let params = ClassicPBSParameters {
            lwe_dimension: LweDimension(742),

            glwe_dimension: GlweDimension(1),
            polynomial_size: PolynomialSize(2048),
            lwe_noise_distribution: parameters::DynamicDistribution::new_gaussian_from_std_dev(
                StandardDev(0.000007069849454709433),
            ),
            glwe_noise_distribution: parameters::DynamicDistribution::new_gaussian_from_std_dev(
                StandardDev(0.00000000000000029403601535432533),
            ),
            pbs_base_log: DecompositionBaseLog(23),
            pbs_level: DecompositionLevelCount(1),
            ks_level: DecompositionLevelCount(5),
            ks_base_log: DecompositionBaseLog(3),
            message_modulus: MessageModulus(16),
            carry_modulus: CarryModulus(1),
            ..PARAM_MESSAGE_4_CARRY_0
        };
        let mut ctx = Context::from(params);
        let client = Client::new(&ctx.parameters(), vec![0; 3]);

        // Encrypt a value with the small LWE key
        let input = 0;
        let lwe = client.private_key.allocate_and_encrypt_lwe(input, &mut ctx);

        // Check initial encryption
        let actual = client.private_key.decrypt_lwe(&lwe, &ctx);
        assert_eq!(actual, input);
        println!(
            "Initial noise={:.2}",
            client.private_key.lwe_noise(&lwe, input, &ctx)
        );

        let mut lwe_sum = lwe.clone();
        let it = 1000;
        for _ in 1..it {
            tfhe::core_crypto::prelude::lwe_ciphertext_add_assign(&mut lwe_sum, &lwe);
        }

        let expected = (it * input) % ctx.full_message_modulus() as u64;
        println!(
            "Noise after {} additions={:.2}",
            it,
            client.private_key.lwe_noise(&lwe_sum, expected, &ctx)
        );

        // Bootstrap the ciphertext
        client.public_key.bootstrap_lwe(&mut lwe_sum, &ctx);
        println!(
            "Noise after bootstrap={:.2}",
            client.private_key.lwe_noise(&lwe_sum, expected, &ctx)
        );

        // Check result after bootstrap
        let actual_after_bootstrap = client
            .private_key
            .decrypt_lwe_without_reduction(&lwe_sum, &ctx);
        // assert_eq!(actual_after_bootstrap, expected);
        println!("expected={}", expected);
        println!("actual={}", actual_after_bootstrap);
    }

    #[test]
    fn test_encrypt() {
        let mut ctx = Context::from(TEST_PARAMS);
        let client = Client::new(&ctx.parameters(), vec![0; 3]);

        let input = 15;
        let lwe = client.private_key.lwe_encrypt_with_modulus(
            input,
            ctx.full_message_modulus() as u64,
            &mut ctx,
        );
        let actual = client.private_key.decrypt_lwe(&lwe, &ctx);
        let expected = input;
        println!(
            "noise={:.2}",
            client.private_key.lwe_noise(&lwe, expected, &ctx)
        );
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_absorption_noise() {
        let mut ctx = Context::from(TEST_PARAMS);
        let client = Client::new(&ctx.parameters(), vec![0; 3]);

        let data = vec![1, 2, 3];
        let target = vec![1, 0, 0];

        let index_to_extract = 1;
        let expected = 2;

        let glwe = client.private_key.allocate_and_encrypt_glwe_with_modulus(
            &target,
            ctx.full_message_modulus() as u64,
            &mut ctx,
        );

        // Encode data into a polynomial
        let mut data_coeffs: Vec<u64> = data.clone();
        data_coeffs.resize(ctx.polynomial_size().0, 0);
        let data_polynomial = Polynomial::from_container(data_coeffs);

        // Perform GLWE absorption between glwe and data_polynomial
        let absorbed_glwe = client
            .public_key
            .glwe_absorption_polynomial_with_fft(&glwe, &data_polynomial);

        // Sample extract (big sk)
        let extracted_lwe = client
            .public_key
            .glwe_extract(&absorbed_glwe, index_to_extract, &ctx);

        let noise = client.private_key.lwe_noise(&extracted_lwe, expected, &ctx);
        println!("noise after sample extract: {:.2}", noise);
    }
}
