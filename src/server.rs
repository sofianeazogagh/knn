use std::fs::File;
use std::io::Write;
use std::time::Instant;

use crate::model::*;
use crate::DEBUG;
use crate::LWE;
use crate::PRINT_NOISE;
use itertools::enumerate;
use revolut::*;
use tfhe::core_crypto::prelude::lwe_ciphertext_add;
use tfhe::core_crypto::prelude::lwe_ciphertext_add_assign;
use tfhe::core_crypto::prelude::lwe_ciphertext_plaintext_add_assign;

use crate::Query;
use rayon::{prelude::*, ThreadPoolBuilder};
pub struct Server {
    public_key: PublicKey,
    model: Model,
}

pub struct KnnClear {
    pub client_feature_vector: Vec<u64>,
    pub distances: Vec<u64>,
    pub distances_and_labels_sorted: Vec<(u64, u64)>,
}

impl KnnClear {
    pub fn new(client_feature_vector: Vec<u64>, model: &Model, ctx: &Context) -> Self {
        let distances_and_labels = model
            .model_points
            .iter()
            .map(|point| {
                (
                    squared_distance_in_clear(&point.feature_vector, &client_feature_vector),
                    point.label,
                )
            })
            .collect::<Vec<(u64, u64)>>();

        let ratio = ctx.delta() / model.delta_dist;
        let distances = distances_and_labels
            .iter()
            .map(|(d, _)| *d / ratio)
            .collect::<Vec<u64>>();

        let mut distances_and_labels_sorted = distances_and_labels.clone();
        distances_and_labels_sorted.sort_by_key(|&(distance, _)| distance);

        KnnClear {
            client_feature_vector,
            distances,
            distances_and_labels_sorted,
        }
    }
}

pub fn calculate_and_print_noise(dist: LWE, expected: u64, ctx: &Context, delta: u64) {
    let private_key = key(ctx.parameters());
    let noise = private_key.lwe_noise_delta(&dist, expected, delta, ctx);
    println!("noise : {:?}", noise);
    private_key.debug_lwe_delta("actual : ", &dist, delta, ctx);
}

impl Server {
    pub fn new(public_key: PublicKey, model: Model) -> Self {
        Server { public_key, model }
    }

    pub fn encode_model(&self, ctx: &Context) -> Vec<ModelPointEncoded> {
        let model_points = &self.model.model_points;
        encode_model_points(model_points, ctx)
    }

    // Function to calculate the squared distance of a query vector to a model point
    // TODO : remove distance_expected once the debug is done
    fn squared_distance(
        &self,
        query: &Query,
        point: &ModelPointEncoded,
        ctx: &Context,
        delta_dist: u64,
        distance_expected: u64,
    ) -> LWE {
        let m = point.m.clone();
        let m_prime = point.m_prime.clone();

        let f_size = self.model.f_size;

        // Step 1: Compute the inner product m(X)q(X)
        let inner_product = self
            .public_key
            .glwe_absorption_polynomial_with_fft(&query.ct, &m);

        // Step 2 : Sample extract at feature_vector_size - 1 which is -2<m,c>
        let mut dist = self
            .public_key
            .sample_extract_in_glwe(&inner_product, f_size - 1, ctx);

        // Do we need to lower precision here?

        if PRINT_NOISE {
            let private_key = key(ctx.parameters());

            private_key.debug_glwe_without_reduction(
                "inner product : ",
                &inner_product,
                ctx,
                f_size - 1,
            );

            println!("------ noise after sample extract ------");
            let second_component_expected =
                private_key.decrypt_lwe_delta(&query.ct_second, delta_dist, ctx);
            let expected = (distance_expected - m_prime - second_component_expected)
                % ctx.full_message_modulus() as u64;
            calculate_and_print_noise(dist.clone(), expected, ctx, delta_dist);
        }

        // Step 5bis : Add the second component of the encoded model point
        dist = self
            .public_key
            .lwe_ciphertext_plaintext_add(&dist, m_prime, ctx);

        if PRINT_NOISE {
            let private_key = key(ctx.parameters());
            let second_component_expected =
                private_key.decrypt_lwe_delta(&query.ct_second, delta_dist, ctx);
            println!("------ noise after m' ------");
            let expected =
                (distance_expected - second_component_expected) % ctx.full_message_modulus() as u64;
            calculate_and_print_noise(dist.clone(), expected, ctx, delta_dist);
        }

        // Step 3: Add the second component of the query
        lwe_ciphertext_add_assign(&mut dist, &query.ct_second);

        if PRINT_NOISE {
            println!("------ noise of the final distance ------");
            let expected = distance_expected;
            calculate_and_print_noise(dist.clone(), expected, ctx, delta_dist);
            println!("----------------------------------------");
        }
        // // Step 6: bootstrap
        // let identity = LUT::from_function_and_delta(|x| x, delta_dist, ctx);
        // dist = self.public_key.run_lut(&dist, &identity, ctx);

        dist
    }

    fn topk_labels(&self, many_lwes: &Vec<Vec<LWE>>, k: usize, ctx: &Context) -> Vec<LWE> {
        let mut topk_many_lut = self.public_key.blind_topk_many_lut_par(many_lwes, k, ctx);

        // The first lut is the distances, the second is the labels
        let k_labels = topk_many_lut.remove(1);
        k_labels
    }

    fn topk_distances_and_labels(
        &self,
        many_lwes: &Vec<Vec<LWE>>,
        k: usize,
        ctx: &Context,
    ) -> Vec<Vec<LWE>> {
        // TODO : take the par version once the debug is done
        let topk_many_lut = self.public_key.blind_topk_many_lut(many_lwes, k, ctx);
        if DEBUG {
            let private_key = key(ctx.parameters());
            println!(
                "topk_many_lut: {:?}",
                private_key.decrypt_lwe_vector_without_mod(&topk_many_lut[0], ctx)
            );
        }
        topk_many_lut
    }

    #[allow(dead_code)]
    pub fn serialize_lwe_vector_to_file(&self, lwe_vector: &Vec<LWE>, file_path: &str) {
        let json = serde_json::to_string(lwe_vector).expect("Failed to serialize LWE vector");
        let mut file = File::create(file_path).expect("Failed to create file");
        file.write_all(json.as_bytes())
            .expect("Failed to write to file");
    }

    pub fn predict_distance_and_labels(
        &self,
        query: &Query,
        model_points: &Vec<ModelPointEncoded>,
        k: usize,
        ctx: &Context,
        distances_expected: &Vec<u64>,
    ) -> Vec<Vec<LWE>> {
        let pool = ThreadPoolBuilder::new().num_threads(4).build().unwrap();

        // Step 0: Encrypt the labels as LWE ciphertexts trivially
        let labels = model_points
            .iter()
            .map(|p| {
                self.public_key
                    .allocate_and_trivially_encrypt_lwe(p.label, ctx)
            })
            .collect::<Vec<LWE>>();

        let start = Instant::now();

        // TODO : uncomment once the debug is done
        // // Step 1Compute the distances
        // let mut distances: Vec<LWE> = pool.install(|| {
        //     model_points
        //         .par_iter()
        //         .map(|point| self.squared_distance(query, point, ctx, self.model.delta_dist))
        //         .collect()
        // });

        let mut distances: Vec<LWE> = model_points
            .iter()
            .enumerate()
            .map(|(i, point)| {
                self.squared_distance(
                    query,
                    point,
                    ctx,
                    self.model.delta_dist,
                    distances_expected[i],
                )
            })
            .collect();

        // Lower reduction precision if needed
        if self.model.delta_dist != ctx.delta() {
            // print the distances before the reduction
            if DEBUG {
                let private_key = key(ctx.parameters());
                println!(
                    "Distances before reduction: {:?}",
                    private_key.decrypt_lwe_vector_without_mod_delta(
                        &distances,
                        self.model.delta_dist,
                        ctx
                    )
                );
            }
            distances.iter_mut().for_each(|x| {
                self.public_key
                    .lower_precision(x, ctx, self.model.delta_dist)
            });
            if DEBUG {
                let private_key = key(ctx.parameters());
                println!(
                    "Distances after reduction: {:?}",
                    private_key.decrypt_lwe_vector_without_mod(&distances, ctx)
                );
            }
        }
        let end_distances = Instant::now();

        // self.serialize_lwe_vector_to_file(&distances, "distances.json");

        // print the distances
        let private_key = key(ctx.parameters());
        let decrypted_distances = private_key.decrypt_lwe_vector_without_mod(&distances, ctx);
        println!(
            "Decrypted distances: {:?}",
            decrypted_distances
                .iter()
                .take(self.model.d)
                .collect::<Vec<_>>()
        );

        println!(
            "Time taken to compute distances: {:?}",
            end_distances - start
        );

        // Step 2: Compute the topk labels
        let topk_many_lut = self.topk_distances_and_labels(&vec![distances, labels], k, ctx);
        let end_topk = Instant::now();
        println!(
            "Time taken to compute topk labels: {:?}",
            end_topk - end_distances
        );

        topk_many_lut
    }
}

#[allow(dead_code)]
// Calculate the squared distance between two vectors, modulo a given parameter
// pub fn squared_distance_in_clear(v1: &Vec<u64>, v2: &Vec<u64>, modulo: u64) -> u64 {
//     let norm_v1_squared: u64 = v1.iter().map(|&x| (x * x) % modulo).sum::<u64>() % modulo;
//     let norm_v2_squared: u64 = v2.iter().map(|&x| (x * x) % modulo).sum::<u64>() % modulo;
//     let scalar_product: u64 = v1
//         .iter()
//         .zip(v2.iter())
//         .map(|(a, b)| (2 * a * b) % modulo)
//         .sum::<u64>()
//         % modulo;

//     (norm_v1_squared + scalar_product + norm_v2_squared) % modulo
// }

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
pub fn knn_predict_in_clear(
    model: &Model,
    query: &Vec<u64>,
    k: usize,
    modulo: u64,
) -> Vec<(u64, u64)> {
    // Calculate distances from the query to each model point, modulo the given parameter
    let mut distances_and_labels: Vec<(u64, u64)> = model
        .model_points
        .iter()
        .map(|point| {
            // let distance = squared_distance_in_clear(&point.feature_vector, query, modulo);
            let distance =
                squared_distance_in_clear(point.feature_vector.as_slice(), query.as_slice());
            (distance, point.label)
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
