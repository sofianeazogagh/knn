use std::fs::File;
use std::io::Write;
use std::time::Instant;

use crate::model::*;
use crate::DEBUG;
use crate::LWE;
use revolut::*;

use crate::Query;
use rayon::{prelude::*, ThreadPoolBuilder};
pub struct Server {
    public_key: PublicKey,
    model: Model,
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
    // TODO : optimization by encoding the model points as one GLWE M and one LWE m_prime as well as the query as one GLWE C and one lwe c_second (much less heavy in memory?)
    fn squared_distance(&self, query: &Query, point: &ModelPointEncoded, ctx: &Context) -> LWE {
        let m = point.m.clone();
        let m_prime = point.m_prime.clone();

        let f_size = self.model.f_size;

        // Step 1: Compute the inner product m(X) * q(X)
        let mut inner_product = self
            .public_key
            .glwe_absorption_polynomial_with_fft(&query.ct, &m);

        // Step 2: Compute -2*inner_product + m_prime
        let dist = self
            .public_key
            .glwe_sum_polynomial(&mut inner_product, &m_prime, ctx);

        // Step 4: Add the second component of the query
        let dist = self.public_key.glwe_sum(&dist, &query.ct_second);

        // Step 5: Sample extract at feature_vector_size - 1
        let mut result = self
            .public_key
            .sample_extract_in_glwe(&dist, f_size - 1, ctx);

        // // Step 6: bootstrap
        // let identity = LUT::from_function(|x| x, ctx);
        // result = self.public_key.run_lut(&result, &identity, ctx);

        result
    }

    fn topk_labels(&self, many_lwes: &Vec<Vec<LWE>>, k: usize, ctx: &Context) -> Vec<LWE> {
        let mut topk_many_lut = self.public_key.blind_topk_many_lut_par(many_lwes, k, ctx);

        // The first lut is the distances, the second is the labels
        let k_labels = topk_many_lut.remove(1);
        k_labels
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
    ) -> Vec<LWE> {
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

        // Step 1Compute the distances
        let distances: Vec<LWE> = pool.install(|| {
            model_points
                .par_iter()
                .map(|point| self.squared_distance(query, point, ctx))
                .collect()
        });
        let end_distances = Instant::now();

        // self.serialize_lwe_vector_to_file(&distances, "distances.json");

        // if DEBUG {
        //     // print the 10 first distances
        //     let private_key = key(ctx.parameters());
        //     let decrypted_distances = private_key.decrypt_lwe_vector(&distances, ctx);
        //     println!(
        //         "Decrypted distances: {:?}",
        //         decrypted_distances.iter().take(10).collect::<Vec<_>>()
        //     );
        // }

        println!(
            "Time taken to compute distances: {:?}",
            end_distances - start
        );

        // Step 2: Compute the topk labels
        let k_labels = self.topk_labels(&vec![distances, labels], k, ctx);
        let end_topk = Instant::now();
        println!(
            "Time taken to compute topk labels: {:?}",
            end_topk - end_distances
        );

        k_labels
    }
}

#[allow(dead_code)]
// Calculate the squared distance between two vectors, modulo a given parameter
pub fn squared_distance_in_clear(v1: &Vec<u64>, v2: &Vec<u64>, modulo: u64) -> u64 {
    let norm_v1_squared: u64 = v1.iter().map(|&x| (x * x) % modulo).sum::<u64>() % modulo;
    let norm_v2_squared: u64 = v2.iter().map(|&x| (x * x) % modulo).sum::<u64>() % modulo;
    let scalar_product: u64 = v1
        .iter()
        .zip(v2.iter())
        .map(|(a, b)| (2 * a * b) % modulo)
        .sum::<u64>()
        % modulo;

    (norm_v1_squared + scalar_product + norm_v2_squared) % modulo
}
#[allow(dead_code)]
pub fn knn_predict_in_clear(model: &Model, query: &Vec<u64>, k: usize, modulo: u64) -> Vec<u64> {
    // Calculate distances from the query to each model point, modulo the given parameter
    let mut distances_and_labels: Vec<(u64, u64)> = model
        .model_points
        .iter()
        .map(|point| {
            let distance = squared_distance_in_clear(&point.feature_vector, query, modulo);
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

    // print the 10 first distances
    println!(
        "Sorted distances in clear: {:?}",
        distances_and_labels
            .iter()
            .take(10)
            .map(|&(distance, _)| distance)
            .collect::<Vec<_>>()
    );

    // Get the labels of the k nearest neighbors
    let k_nearest_labels = distances_and_labels.iter().take(k).map(|&(_, label)| label);

    k_nearest_labels.collect()
}

// #[cfg(test)]
// mod tests {
//     use tfhe::shortint::parameters::PARAM_MESSAGE_4_CARRY_0;

//     use super::*;
//     use crate::model::ModelPointEncoded;
//     use crate::Context;

//     #[test]
//     fn test_squared_distance() {
//         // Setup the context, public key, and model
//         let ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);
//         let private_key = key(ctx.parameters());
//         let public_key = &private_key.public_key;
//         // Create test model points
//         let model_points = vec![
//             ModelPoint {
//                 feature_vector: vec![1, 2, 3],
//                 label: 2,
//             },
//             ModelPoint {
//                 feature_vector: vec![0, 1, 0],
//                 label: 1,
//             },
//             ModelPoint {
//                 feature_vector: vec![1, 0, 0],
//                 label: 1,
//             },
//             ModelPoint {
//                 feature_vector: vec![0, 2, 0],
//                 label: 1,
//             },
//             ModelPoint {
//                 feature_vector: vec![2, 3, 1],
//                 label: 2,
//             },
//         ];
//         let d = model_points.len();
//         let model = Model {
//             d,
//             f_size: model_points[0].feature_vector.len(),
//             model_points,
//         };
//         // Create a server instance
//         let server = Server::new(public_key.clone(), model);

//         // Create a query and a model point
//         let query = Query {
//             ct: vec![1, 0, 0],
//             ct_second: vec![0, 0, 0],
//         };
//         let point = ModelPointEncoded {
//             m: vec![/* ... */],
//             m_prime: vec![/* ... */],
//             label: 0,
//         };

//         // Calculate the squared distance using the encrypted method
//         let encrypted_distance = server.squared_distance(&query, &point, &ctx);

//         // Calculate the squared distance in clear
//         let clear_distance = squared_distance_in_clear(
//             &point.feature_vector,
//             &query.feature_vector,
//             ctx.modulo,
//         );

//         // Decrypt the encrypted distance for comparison
//         let private_key = key(ctx.parameters());
//         let decrypted_distance = private_key.decrypt_lwe(&encrypted_distance, &ctx);

//         // Assert that the decrypted distance matches the clear distance
//         assert_eq!(decrypted_distance, clear_distance);
//     }
// }
