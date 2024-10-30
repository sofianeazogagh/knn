use std::time::Instant;

use std::fs::File;
use std::io::Write;

use model::ModelPoint;
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

const PRINT_PARAMS: bool = false;
const PRINT_CSV: bool = false;
const DEBUG: bool = true;

pub struct Query {
    pub ct: GLWE,
    pub ct_second: GLWE, // TODO : lwe
}

fn main() {
    let mut ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);
    let client = &Client::new(&ctx.parameters());

    // Parameters
    let k = 3;
    let d = 10;

    /* MODEL instantiation */
    // Read the model points from the csv file
    // let model = model::read_csv("data/cancer.csv", d).expect("Failed to read the model");

    let model = model::model_test(d, 3);

    /* QUERY instantiation */
    // Create a query vector from a client
    let client_feature_vector = vec![0, 0, 0];
    let mut k_clear_labels: Vec<u64> = Vec::new();
    if DEBUG {
        let modulo = ctx.full_message_modulus() as u64;
        k_clear_labels = server::knn_predict_in_clear(&model, &client_feature_vector, k, modulo);
    }
    let query = client.create_query(client_feature_vector, &mut ctx);

    /* SERVER instantiation */
    let server = &Server::new(client.public_key.clone(), model);

    // Open the CSV file for printing the time taken
    let mut file: Option<File> = None;
    if PRINT_CSV {
        file = Some(File::create(format!("{k}nn_time.csv")).unwrap());
        writeln!(file.as_mut().unwrap(), "d,time").unwrap();
    }

    if PRINT_PARAMS {
        println!("PARAM_{}", (ctx.full_message_modulus() as f64).log2());
        println!("d: {:?}", d);
        println!("k: {:?}", k);
    }

    // Encode the model points
    let encoded_points = server.encode_model(&ctx);

    // Predict the k nearest labels
    let start = Instant::now();
    let _k_labels = server.predict(&query, &encoded_points, k, &ctx);
    let total_dur = start.elapsed().as_secs_f32();

    // Decrypt the labels
    if DEBUG {
        let mut decrypted_labels = Vec::new();
        for label in _k_labels {
            let decrypted_label = client.private_key.decrypt_lwe(&label, &ctx);
            decrypted_labels.push(decrypted_label);
        }
        println!("Decrypted labels: {:?}", decrypted_labels);
        println!("Labels in clear: {:?}", k_clear_labels);
    }

    println!("Total time taken: {:?}s", total_dur);
    if PRINT_CSV {
        writeln!(file.as_mut().unwrap(), "{d},{total_dur}").unwrap();
    }
}

#[cfg(test)]
mod tests {
    use model::{Model, ModelPoint};
    use server::knn_predict_in_clear;

    use super::*;
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
            f_size: model_points[0].feature_vector.len(),
            model_points,
        };

        let server = Server::new(client.public_key.clone(), model);

        // Create a query vector from a client
        let client_feature_vector = vec![0, 0, 0];
        let query = client.create_query(client_feature_vector, &mut ctx);

        // Encode the model points
        let encoded_points = server.encode_model(&ctx);

        // Predict the k nearest labels
        let start = Instant::now();
        let k_labels = server.predict(&query, &encoded_points, k, &ctx);
        let total_dur = start.elapsed().as_secs_f32();
        println!("Total time taken: {:?}s", total_dur);

        // decrypt the labels
        for label in k_labels {
            let decrypted_label = client.private_key.decrypt_lwe(&label, &ctx);
            println!("{:?}", decrypted_label);
        }
    }

    #[test]
    fn test_knn_predict_in_clear() {
        // Create test model points
        let model_points = vec![
            ModelPoint {
                feature_vector: vec![1, 2, 3],
                label: 0,
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
                label: 0,
            },
        ];
        let d = model_points.len();
        let model = Model {
            d,
            f_size: model_points[0].feature_vector.len(),
            model_points,
        };

        // Create a query vector from a client
        let clear_query = vec![0, 0, 0];
        let k = 3;
        let modulo = 16;
        let k_labels = knn_predict_in_clear(&model, &clear_query, k, modulo);

        for label in k_labels {
            println!("{:?}", label);
        }
    }
}
