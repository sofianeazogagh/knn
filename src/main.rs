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
const PRINT_NOISE: bool = false;

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
    // message_modulus: MessageModulus(16),
    ..PARAM_MESSAGE_4_CARRY_0
};

fn main() {
    // let mut ctx = Context::from(PARAM_MESSAGE_4_CARRY_1);

    let mut ctx = Context::from(PARAMS);
    let client = &Client::new(&ctx.parameters());

    // Parameters
    let k = 3;
    let d: usize = 40;

    // let t_dist = 64;
    // let delta_dist = (1 << 63) / t_dist;

    let delta_dist = ctx.delta();
    //

    /* MODEL instantiation */
    // Read the model points from the csv file
    let model = model::parse_csv("data/cancer.csv", QuantizeType::Binary, d, delta_dist);

    // let model = model::parse_csv("data/mnist-8x8.csv", QuantizeType::Binary, d, delta_dist);

    model.print_first_point();

    /* QUERY instantiation */
    // Create a query vector from a client
    // let client_feature_vector = vec![0, 0, 0];

    let client_feature_vector = model.model_points[0].feature_vector.clone();

    let knn_clear = server::KnnClear::new(client_feature_vector.clone(), &model, &ctx);
    println!("Distances in clear: {:?}", knn_clear.distances);

    let query = client.create_query(client_feature_vector, &mut ctx, delta_dist);

    /* SERVER instantiation */
    let server = &Server::new(client.public_key.clone(), model.clone());

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
    let predicted_distances_and_labels =
        server.predict_distance_and_labels(&query, &encoded_points, k, &ctx, &knn_clear.distances);
    let total_dur = start.elapsed().as_secs_f32();

    // Verify the result
    let decrypted_distances = client
        .private_key
        .decrypt_lwe_vector(&predicted_distances_and_labels[0], &ctx);
    let decrypted_labels = client
        .private_key
        .decrypt_lwe_vector(&predicted_distances_and_labels[1], &ctx);

    let actual_couples = decrypted_distances
        .iter()
        .zip(decrypted_labels.iter())
        .map(|(d, l)| (d.clone(), l.clone()))
        .collect::<Vec<(u64, u64)>>();

    // FIXME : this is not the expected couples
    // we need to divide the distances by the ratio = ctx.delta() / delta_dist
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

    if DEBUG {
        // print the 10 first distances
        println!(
            "Sorted distances in clear: {:?}",
            knn_clear
                .distances_and_labels_sorted
                .iter()
                .take(10)
                .map(|&(distance, _)| distance)
                .collect::<Vec<_>>()
        );
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
            delta_dist: ctx.delta(),
        };

        // Create a query vector from a client
        let client_feature_vector = vec![0, 0, 0];
        let query = client.create_query(client_feature_vector, &mut ctx, model.delta_dist);

        // Instantiate the server
        let server = Server::new(client.public_key.clone(), model);

        // Encode the model points
        let encoded_points = server.encode_model(&ctx);

        // Predict the k nearest labels
        let start = Instant::now();
        let distances_and_labels =
            server.predict_distance_and_labels(&query, &encoded_points, k, &ctx, &Vec::new());
        let total_dur = start.elapsed().as_secs_f32();
        println!("Total time taken: {:?}s", total_dur);

        // decrypt the labels
        for label in distances_and_labels[1].iter() {
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
            delta_dist: 0,
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

    #[test]
    fn test_params() {
        let params = PARAM_MESSAGE_4_CARRY_0;
        println!("Ciphertext modulus: {:?}", params.ciphertext_modulus);
        println!("Message modulus: {:?}", params.message_modulus);
        println!("Carry modulus: {:?}", params.carry_modulus);

        let q: u64 = 1 << 64;

        println!("{:?}", q);
    }
}
