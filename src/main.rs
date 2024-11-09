use std::time::Instant;

use std::fs::File;
use std::io::Write;

use model::Model;
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

const PRINT_CSV: bool = true;
const DEBUG: bool = false;

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
    let mut ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);
    let client = &Client::new(&ctx.parameters());

    // Parameters
    let dataset_name = "mnist";
    let k_values = vec![3, 5];

    for k in k_values {
        // Open the CSV file for printing the time taken

        let d_values = match dataset_name {
            "mnist" => vec![40, 175, 269, 457, 1000],
            "cancer" => vec![10, 30, 50, 200],
            _ => panic!("Unknown dataset name: {}", dataset_name),
        };
        let mut file: Option<File> = None;
        if PRINT_CSV {
            file = Some(File::create(format!("benchmarks/{k}nn_time_{dataset_name}.csv")).unwrap());
            writeln!(file.as_mut().unwrap(), "d,time,wrong_predictions").unwrap();
        }

        let mut wrong_predictions = false;

        for d in d_values {
            // let dist_modulus = ctx.message_modulus().0 as u64;

            /* MODEL instantiation */
            let model: Model;
            if dataset_name == "cancer" {
                let dist_modulus = ctx.full_message_modulus() as u64;
                model = model::parse_csv("data/cancer.csv", QuantizeType::Binary, d, dist_modulus);
            } else {
                let dist_modulus = 32;
                model = model::parse_csv("data/mnist.csv", QuantizeType::Binary, d, dist_modulus);
            }

            /* QUERY instantiation */
            // Create a query vector from a client
            let target_vector = vec![0; model.gamma];

            let query = client.create_query(&target_vector, &mut ctx, model.dist_modulus);

            /* SERVER instantiation */
            let server = &Server::new(client.public_key.clone(), model.clone());

            // Encode the model points
            let encoded_points = server.encode_model(&ctx);

            // Predict the k nearest labels
            let start = Instant::now();
            let predicted_distances_and_labels =
                server.predict_distance_and_labels(&query, &encoded_points, k, &ctx);
            let total_dur = start.elapsed().as_secs_f32();
            // Verify the result
            let knn_clear = server::KnnClear::new(&target_vector, &model, &ctx);
            if DEBUG {
                println!("Distances in clear: {:?}", knn_clear.distances);
            }
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

            // assert_eq!(actual_couples, expected_couples);
            if actual_couples != expected_couples {
                println!("Wrong result for d = {}", d);
                wrong_predictions = true;
            } else {
                wrong_predictions = false;
            }
            println!("Total time taken: {:?}s", total_dur);
            if PRINT_CSV {
                writeln!(
                    file.as_mut().unwrap(),
                    "{d},{total_dur},{wrong_predictions}"
                )
                .unwrap();
            }
        }
        println!("Wrong predictions: {}", wrong_predictions);
    }
}
