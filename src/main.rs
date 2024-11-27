use std::time::Instant;

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

const DEBUG: bool = true;
const VERBOSE: bool = true;
const THREADS: usize = 4;
const TEST_SIZE: usize = 200;
const REPETITIONS: usize = 1;

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

const PARAMS: ClassicPBSParameters = ClassicPBSParameters {
    lwe_dimension: LweDimension(742),
    glwe_dimension: GlweDimension(1),
    polynomial_size: PolynomialSize(2048),
    lwe_noise_distribution:
        tfhe::boolean::parameters::DynamicDistribution::new_gaussian_from_std_dev(StandardDev(
            0.000007069849454709433,
        )),
    glwe_noise_distribution:
        tfhe::boolean::parameters::DynamicDistribution::new_gaussian_from_std_dev(StandardDev(
            0.00000000000000029403601535432533,
        )),
    pbs_base_log: DecompositionBaseLog(23),
    pbs_level: DecompositionLevelCount(1),
    ks_level: DecompositionLevelCount(5),
    ks_base_log: DecompositionBaseLog(3),
    message_modulus: MessageModulus(16),
    carry_modulus: CarryModulus(1),
    ..PARAM_MESSAGE_4_CARRY_0
};
fn main() {
    // Parameters
    let mut ctx = Context::from(PARAMS);
    let dataset_name = "cancer";
    // let k = 3;
    let k_values = vec![3, 5];
    let d_values = vec![10, 30, 50, 200];

    /* READ CSV FILES */
    let dataset: Vec<Vec<u64>>;
    let dist_modulus: u64;
    if dataset_name == "cancer" {
        dist_modulus = 16 as u64;
        (dataset, _) = model::parse_csv_dataset("data/cancer.csv", QuantizeType::Binary);
    } else {
        dist_modulus = 32;
        (dataset, _) = model::parse_csv_dataset("data/mnist.csv", QuantizeType::Binary);
    }

    let mut actual_errs = 0usize;
    let mut clear_errs = 0usize;
    for (d, k) in d_values.into_iter().zip(k_values.into_iter()) {
        for _ in 0..REPETITIONS {
            // INSTANTIATE MODEL
            let (model_vec, model_labels, test_vec, test_labels, _acc) =
                server::find_best_model(d, TEST_SIZE, k, &dataset, ctx.delta(), dist_modulus);
            let model = Model::new(model_vec, model_labels, dist_modulus);

            /* TEST for all targets */
            for (i, (target, expected)) in test_vec.into_iter().zip(test_labels).enumerate() {
                println!("---------------Target no={i}-----------------");
                let client = &Client::new(&ctx.parameters(), target.clone());
                let query = client.create_query(&mut ctx, dist_modulus);

                /* SERVER instantiation */
                let server = &Server::new(client.public_key.clone(), model.clone());

                // Encode the model points
                let encoded_points = server.encode_model(&ctx);

                // Predict the k nearest labels
                let start = Instant::now();
                let (actual, dist_dur, topk_dur) = server.predict(&query, &encoded_points, k, &ctx);
                let total_dur = start.elapsed().as_secs_f32();

                if VERBOSE {
                    println!("Distance computation time: {:?}ms", dist_dur.as_millis());
                    println!("Topk computation time: {:?}ms", topk_dur.as_millis());
                }

                let actual_labels = client.private_key.decrypt_lwe_vector(&actual[1], &ctx);
                let actual_maj = server::majority(&actual_labels);
                assert_eq!(actual_labels.len(), k);

                // Verify the result
                let knn_clear =
                    server::KnnClear::run(k, &client.target_vector, &model, ctx.delta());
                let clear_labels = knn_clear
                    .top_k_distances_and_labels
                    .iter()
                    .map(|(_, l)| *l)
                    .collect::<Vec<_>>();
                let clear_maj = server::majority(&clear_labels);
                assert_eq!(clear_labels.len(), k);

                if actual_maj != expected {
                    actual_errs += 1;
                }
                if clear_maj != expected {
                    clear_errs += 1;
                }

                let actual_couples = client
                    .private_key
                    .decrypt_lwe_vector(&actual[0], &ctx)
                    .iter()
                    .zip(
                        client
                            .private_key
                            .decrypt_lwe_vector(&actual[1], &ctx)
                            .iter(),
                    )
                    .map(|(d, l)| (*d, *l))
                    .collect::<Vec<(u64, u64)>>();

                let expected_couples = knn_clear
                    .top_k_distances_and_labels
                    .iter()
                    .map(|&(d, l)| (d, l))
                    .take(k)
                    .collect::<Vec<_>>();
                if VERBOSE {
                    println!("Distances and labels decrypted: {:?}", actual_couples);
                    println!("Distances and labels in clear: {:?}", expected_couples);
                }
                println!("Total time taken: {:?}s", total_dur);

                // assert_eq!(actual_couples, expected_couples);
            }
            println!(
                "[SUMMARY]: \
        k={}, \
        model_size={}, \
        test_size={}, \
        actual_errs={actual_errs}, \
        clear_errs={clear_errs}, \
        actual_accuracy={:.2}, \
        clear_accuracy={:.2}",
                k,
                d,
                TEST_SIZE,
                1f64 - ((actual_errs as f64) / (REPETITIONS * TEST_SIZE) as f64),
                1f64 - ((clear_errs as f64) / (REPETITIONS * TEST_SIZE) as f64)
            );
        }
    }
}

// #[allow(dead_code)]
// fn benchmark(dataset_name: &str) {
//     let k_values = vec![3, 5];
//     for k in k_values {
//         // Open a file to store the results
//         let mut file: Option<File> = None;
//         if PRINT_CSV {
//             file = Some(
//                 File::create(format!(
//                     "benchmarks/{k}nn_time_{dataset_name}_one_thread.csv"
//                 ))
//                 .unwrap(),
//             );
//             writeln!(file.as_mut().unwrap(), "d,time").unwrap();
//         }

//         // Set a range of d values depending on the dataset
//         let d_values = match dataset_name {
//             "mnist" => vec![40, 175, 269, 457, 1000],
//             "cancer" => vec![10, 30, 50, 200],
//             _ => panic!("Unknown dataset name: {}", dataset_name),
//         };
//         for d in d_values {
//             let dataset: Vec<Vec<u64>>;
//             let dist_modulus: u64;
//             if dataset_name == "cancer" {
//                 dist_modulus = 16u64;
//                 dataset = model::parse_csv_dataset("data/cancer.csv", QuantizeType::Binary);
//             } else {
//                 dist_modulus = 32u64;
//                 dataset = model::parse_csv_dataset("data/mnist.csv", QuantizeType::Binary);
//             }

//             let model = Model::new(dataset, Vec::new(), dist_modulus);

//             /* CLIENT instantiation */
//             let mut ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);
//             let target_vector = vec![0; model.gamma];
//             let client = &Client::new(&ctx.parameters(), target_vector);
//             let query = client.create_query(&mut ctx, model.dist_modulus);

//             /* SERVER instantiation */
//             let server = &Server::new(client.public_key.clone(), model.clone());

//             // Encode the model points
//             let encoded_points = server.encode_model(&ctx);

//             // Predict the k nearest labels
//             let start = Instant::now();
//             let _ = server.predict(&query, &encoded_points, k, &ctx);
//             let total_dur = start.elapsed().as_secs_f32();

//             if PRINT_CSV {
//                 writeln!(file.as_mut().unwrap(), "{d},{total_dur}").unwrap();
//             }
//         }
//     }
// }
