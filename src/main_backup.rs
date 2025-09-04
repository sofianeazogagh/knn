use std::env;
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

const VERBOSE: bool = true;
const THREADS: usize = 4;

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

fn parse_args() -> (String, Vec<usize>, Vec<usize>, usize, usize) {
    let args: Vec<String> = env::args().collect();
    if args.len() < 6 {
        eprintln!(
            "Usage: {} <dataset_name> <k_values> <d_values> <test_size> <repetitions>",
            args[0]
        );
        std::process::exit(1);
    }

    let dataset_name = args[1].clone();
    let k_values: Vec<usize> = args[2]
        .split(',')
        .map(|s| s.parse().expect("Invalid k value"))
        .collect();
    let d_values: Vec<usize> = args[3]
        .split(',')
        .map(|s| s.parse().expect("Invalid d value"))
        .collect();
    let test_size = args[4].parse().expect("Invalid test size");
    let repetitions = args[5].parse().expect("Invalid repetitions");

    (dataset_name, k_values, d_values, test_size, repetitions)
}

fn main() {
    // Parameters
    let mut ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);
    let (dataset_name, k_values, d_values, test_size, repetitions) = parse_args();

    /* READ DATASET FILES */
    let dataset: Vec<Vec<u64>>;
    let dist_modulus: u64;
    if dataset_name == "cancer" {
        dist_modulus = 16 as u64;
        (dataset, _) = model::parse_csv_dataset("./data/cancer.csv", QuantizeType::Binary);
    } else {
        dist_modulus = 32;
        (dataset, _) = model::parse_csv_dataset("./data/mnist.csv", QuantizeType::Binary);
    }

    for k in &k_values {
        for d in &d_values {
            println!("=============k={k}, d={d}=============");
            let mut actual_errs = 0usize;
            let mut clear_errs = 0usize;
            let mut duration = 0.0;
            for _ in 0..repetitions {
                // INSTANTIATE MODEL
                if VERBOSE {
                    println!("Finding best model...");
                }
                let (model_vec, model_labels, test_vec, test_labels, _acc) =
                    server::find_best_model(*d, test_size, *k, &dataset, ctx.delta(), dist_modulus);
                let model = Model::new(model_vec, model_labels, dist_modulus);

                /* TEST for all targets (i.e each point in the test set) */
                if VERBOSE {
                    println!("Testing for {test_size} targets...");
                }
                for (i, (target, expected_label)) in
                    test_vec.into_iter().zip(test_labels).enumerate()
                {
                    if VERBOSE {
                        println!("----Target no={i}----");
                    }

                    // Once we have the target and the model, we can instantiate the client and server
                    let client = &Client::new(&mut ctx, target.clone());
                    let query = client.create_query(&mut ctx, dist_modulus);
                    let server = &Server::new(client.public_key.clone(), model.clone());

                    // Encode the model points
                    let encoded_points = server.encode_model(&ctx);

                    // Predict the k nearest labels
                    let start = Instant::now();
                    let (actual, dist_dur, topk_dur) =
                        server.predict(&query, &encoded_points, *k, &ctx);
                    let total_dur = start.elapsed().as_secs_f32();

                    if VERBOSE {
                        println!("Distance computation time: {:?}ms", dist_dur.as_millis());
                        println!("Topk computation time: {:?}ms", topk_dur.as_millis());
                    }

                    let actual_labels = client.private_key.decrypt_lwe_vector(&actual[1], &ctx);
                    let actual_maj = server::majority(&actual_labels);
                    assert_eq!(actual_labels.len(), *k);

                    // Verify the result
                    let knn_clear =
                        server::KnnClear::run(*k, &client.target_vector, &model, ctx.delta());
                    let clear_labels = knn_clear
                        .top_k_distances_and_labels
                        .iter()
                        .map(|(_, l)| *l)
                        .collect::<Vec<_>>();
                    let clear_maj = server::majority(&clear_labels);
                    assert_eq!(clear_labels.len(), *k);

                    if actual_maj != expected_label {
                        actual_errs += 1;
                    }
                    if clear_maj != expected_label {
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
                        .take(*k)
                        .collect::<Vec<_>>();

                    duration = duration + total_dur;
                    if VERBOSE {
                        println!("Distances and labels decrypted: {:?}", actual_couples);
                        println!("Distances and labels in clear: {:?}", expected_couples);
                        println!("Total time taken: {:?}s", total_dur);
                    }
                }
            }

            let avg_dur = duration / (repetitions * test_size) as f32;
            println!(
                "[SUMMARY]: \
                dataset={}, \
                k={}, \
                model_size={}, \
                test_size={}, \
                time={:.2}s, \
                fhe_accuracy={:.2}, \
                clear_accuracy={:.2}",
                dataset_name,
                k,
                d,
                test_size,
                avg_dur,
                1f64 - ((actual_errs as f64) / (repetitions * test_size) as f64),
                1f64 - ((clear_errs as f64) / (repetitions * test_size) as f64)
            );
        }
    }
}
