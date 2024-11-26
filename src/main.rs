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
mod clear_knn;
mod client;
mod model;
mod server;

use clear_knn::KnnClear;
use client::Client;
use server::Server;

type GLWE = GlweCiphertext<Vec<u64>>;
type LWE = LweCiphertext<Vec<u64>>;
type Poly = Polynomial<Vec<u64>>;

// TODO : all these should be from argparse
const PRINT_CSV: bool = false;
const DEBUG: bool = true;
const VERBOSE: bool = false;
const BEST_MODEL: bool = true;
const THREADS: usize = 1;
const TEST_SIZE: usize = 200;
const REPETITIONS: usize = 100;
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
    // let dataset_name = "mnist";
    let dataset_name = "cancer";
    let k_values = vec![3, 5];
    let d_values = vec![50, 200, 400];
    let mut ctx = Context::from(PARAMS);

    /* READ DATASET */
    let dataset: Vec<Vec<u64>>;
    let dist_modulus: u64;
    if dataset_name == "cancer" {
        dist_modulus = 16u64;
        (dataset, _) = model::parse_csv("data/cancer.csv", QuantizeType::Binary);
    } else {
        dist_modulus = 32u64;
        (dataset, _) = model::parse_csv("data/mnist.csv", QuantizeType::Binary);
    }

    let mut actual_errs = 0usize;
    let mut clear_errs = 0usize;
    for (k, d) in k_values.into_iter().zip(d_values.into_iter()) {
        for _ in 0..REPETITIONS {
            /* FIND BEST MODEL */
            let (model_vec, model_labels, test_vec, test_labels) = {
                if BEST_MODEL {
                    if VERBOSE {
                        println!("[DEBUG] finding best model");
                    }
                    let (model_vec, model_labels, test_vec, test_labels, acc) =
                        clear_knn::find_best_model(d, TEST_SIZE, k, &dataset, &ctx);
                    if VERBOSE {
                        println!("[DEBUG] expected accuracy: {}", acc);
                    }
                    (model_vec, model_labels, test_vec, test_labels)
                } else {
                    clear_knn::split_model_test(d, TEST_SIZE, dataset.clone())
                }
            };

            /* MODEL instantiation */
            let model = Model::new(model_vec, model_labels, dist_modulus);
            // Run the tests
            for (i, (target, expected)) in test_vec.into_iter().zip(test_labels).enumerate() {
                println!("---------------Target no={i}-----------------");
                if DEBUG {
                    let delta_dist = (1u64 << 63) / model.dist_modulus;
                    let ratio = ctx.delta() / delta_dist;
                    println!("[DEBUG] target_no={i}");
                    println!(
                        "[DEBUG] clear_distances={:?}",
                        model
                            .model_points
                            .iter()
                            .map(|p| {
                                clear_knn::squared_distance_in_clear(&p.feature_vector, &target)
                                    / ratio
                            })
                            .take(model.d)
                            .collect::<Vec<_>>()
                    )
                }

                // PRIVATE KNN
                /* CLIENT instantiation */
                let client = &Client::new(&ctx.parameters(), target.clone());
                let query = client.create_query(&mut ctx, dist_modulus);

                /* SERVER instantiation */
                let server = &Server::new(client.public_key.clone(), model.clone());

                // Encode the model points
                let encoded_points = server.encode_model(&ctx);

                // Predict the k nearest labels
                let (cts_full, dist_dur, topk_dur) =
                    server.predict(&query, &encoded_points, k, &ctx);
                if VERBOSE {
                    println!("Distance computation time: {:?}ms", dist_dur.as_millis());
                    println!("Topk computation time: {:?}ms", topk_dur.as_millis());
                }

                let actual_labels = client.private_key.decrypt_lwe_vector(&cts_full[1], &ctx);
                let actual_maj = clear_knn::majority(&actual_labels);
                assert_eq!(actual_labels.len(), k);

                // CLEAR KNN
                let clear_knn = KnnClear::run(k, &target, &model, &ctx);
                let out_labels = clear_knn.top_k.iter().map(|(_, l)| *l).collect::<Vec<_>>();
                let clear_maj = clear_knn::majority(&out_labels);
                assert_eq!(out_labels.len(), k);

                if actual_maj != expected {
                    actual_errs += 1;
                }
                if clear_maj != expected {
                    clear_errs += 1;
                }

                if actual_maj != clear_maj {
                    println!("[DEBUG] target_no={i}");
                    println!("[DEBUG] distances in clear: {:?}", clear_knn.distances);
                    let actual_couples =
                        split_distances_labels_and_decrypt(&cts_full, &client.private_key, &ctx);
                    let clear_couples = clear_knn
                        .distances_and_labels_sorted
                        .iter()
                        .map(|(d, l)| (d, l))
                        .collect::<Vec<_>>();
                    println!("[DEBUG] actual_couples: {:?}", actual_couples);
                    println!("[DEBUG] clear_couples: {:?}", clear_couples);
                    println!(
                    "[DEBUG] actual_maj={actual_maj}, clear_maj={clear_maj}, expected={expected}"
                );
                }
            }
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

pub fn split_distances_labels_and_decrypt(
    cts: &Vec<Vec<LWE>>,
    private_key: &PrivateKey,
    ctx: &Context,
) -> Vec<(u64, u64)> {
    let actual_couples: Vec<(u64, u64)> = cts[0]
        .iter()
        .zip(cts[1].iter())
        .map(|(d, l)| {
            (
                private_key.decrypt_lwe(d, ctx),
                private_key.decrypt_lwe(l, ctx),
            )
        })
        .collect();
    actual_couples
}
