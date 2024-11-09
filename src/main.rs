use std::time::Instant;

use std::fs::File;
use std::io::Write;

// TFHE
use tfhe::core_crypto::prelude::*;
use tfhe::shortint::parameters::*;

// REVOLUT
use revolut::*;

// KNN
mod client;
mod model;
mod server;

use client::Client;
use server::Server;

type GLWE = GlweCiphertext<Vec<u64>>;
type LWE = LweCiphertext<Vec<u64>>;
type Poly = Polynomial<Vec<u64>>;

const PRINT_CSV: bool = false;
const DEBUG: bool = false;
// const PRINT_NOISE: bool = false;

#[allow(dead_code)]
enum QuantizeType {
    None,
    Binary,
    Ternary,
}

pub struct Query {
    pub ct: GLWE,
    pub ct_second: LWE,
}

// const PARAMS: ClassicPBSParameters = ClassicPBSParameters {
//     lwe_dimension: LweDimension(742),
//     glwe_dimension: GlweDimension(1),
//     polynomial_size: PolynomialSize(2048),
//     lwe_noise_distribution: parameters::DynamicDistribution::new_gaussian_from_std_dev(
//         StandardDev(0.000007069849454709433),
//     ),
//     glwe_noise_distribution: parameters::DynamicDistribution::new_gaussian_from_std_dev(
//         StandardDev(0.00000000000000029403601535432533),
//     ),
//     pbs_base_log: DecompositionBaseLog(23),
//     pbs_level: DecompositionLevelCount(1),
//     ks_level: DecompositionLevelCount(5),
//     ks_base_log: DecompositionBaseLog(3),
//     message_modulus: MessageModulus(16),
//     carry_modulus: CarryModulus(1),
//     ..PARAM_MESSAGE_4_CARRY_0
// };

fn main() {
    // let mut ctx = Context::from(PARAMS);
    let mut ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);
    let client = &Client::new(&ctx.parameters());

    // Parameters
    let k = 3;
    let d: usize = 10;

    let t_dist = 32;
    // let t_dist = ctx.message_modulus().0 as u64;

    /* MODEL instantiation */
    // let model = model::parse_csv("data/cancer.csv", QuantizeType::Binary, d, t_dist);
    let model = model::parse_csv("data/mnist-8x8.csv", QuantizeType::Binary, d, t_dist);

    /* QUERY instantiation */
    // Create a query vector from a client
    let client_feature_vector = model.model_points[0].feature_vector.clone();

    let knn_clear = server::KnnClear::new(client_feature_vector.clone(), &model, &ctx);

    let query = client.create_query(client_feature_vector, &mut ctx, model.dist_modulus);

    /* SERVER instantiation */
    let server = &Server::new(client.public_key.clone(), model.clone());

    // Open the CSV file for printing the time taken
    let mut file: Option<File> = None;
    if PRINT_CSV {
        file = Some(File::create(format!("{k}nn_time.csv")).unwrap());
        writeln!(file.as_mut().unwrap(), "d,time").unwrap();
    }

    // Encode the model points
    let encoded_points = server.encode_model(&ctx);

    // Predict the k nearest labels
    let start = Instant::now();
    let predicted_distances_and_labels =
        server.predict_distance_and_labels(&query, &encoded_points, k, &ctx);
    let total_dur = start.elapsed().as_secs_f32();

    if DEBUG {
        println!("Distances in clear: {:?}", knn_clear.distances);
    }

    // Verify the result
    let actual_couples = client
        .private_key
        .decrypt_lwe_vector(&predicted_distances_and_labels[0], &ctx)
        .iter()
        .zip(
            client
                .private_key
                .decrypt_lwe_vector(&predicted_distances_and_labels[1], &ctx)
                .iter(),
        )
        .map(|(d, l)| (*d, *l))
        .collect::<Vec<(u64, u64)>>();

    let expected_couples = knn_clear
        .distances_and_labels_sorted
        .iter()
        .map(|&(d, l)| (d, l))
        .take(k)
        .collect::<Vec<_>>();

    println!("Distances and labels decrypted: {:?}", actual_couples);
    println!("Distances and labels in clear: {:?}", expected_couples);

    assert_eq!(actual_couples, expected_couples);

    println!("Total time taken: {:?}s", total_dur);
    if PRINT_CSV {
        writeln!(file.as_mut().unwrap(), "{d},{total_dur}").unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use model::{Model, ModelPoint};

    #[test]
    fn test_predict() {
        let mut ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);
        let client = Client::new(&ctx.parameters());
        let k = 3;

        // Create test model points
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
        let d = model_points.len();
        let model = Model {
            d,
            gamma: model_points[0].feature_vector.len(),
            model_points,
            dist_modulus: ctx.full_message_modulus() as u64,
        };

        // Create a query vector from a client
        let client_feature_vector = vec![0, 0, 0];
        let query = client.create_query(client_feature_vector, &mut ctx, model.dist_modulus);

        // Instantiate the server
        let server = Server::new(client.public_key.clone(), model);

        // Encode the model points
        let encoded_points = server.encode_model(&ctx);

        // Predict the k nearest labels
        let start = Instant::now();
        let distances_and_labels =
            server.predict_distance_and_labels(&query, &encoded_points, k, &ctx);
        let total_dur = start.elapsed().as_secs_f32();
        println!("Total time taken: {:?}s", total_dur);

        // decrypt the labels
        for label in distances_and_labels[1].iter() {
            let decrypted_label = client.private_key.decrypt_lwe(&label, &ctx);
            println!("{:?}", decrypted_label);
        }
    }

    #[test]
    fn test_params() {
        let params = PARAM_MESSAGE_4_CARRY_0;
        println!("Ciphertext modulus: {:?}", params.ciphertext_modulus);
        println!("Message modulus: {:?}", params.message_modulus);
        println!("Carry modulus: {:?}", params.carry_modulus);

        let q: u64 = 1 << 63;

        println!("{:?}", q);
    }
}
