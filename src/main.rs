use std::time::Instant;

use revolut::*;

// TFHE
use tfhe::core_crypto::prelude::*;
use tfhe::shortint::parameters::*;

mod client;
mod model;
mod server;

use client::Client;
use model::generate_random_model;
use server::Server;

pub struct Query {
    pub ct: GlweCiphertext<Vec<u64>>,
    pub ct_second: GlweCiphertext<Vec<u64>>,
}

fn main() {
    let mut ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);
    let client = &Client::new(&ctx.parameters());

    let f_size = 3;
    let k = 3;
    let d = ctx.full_message_modulus() as usize;
    // let d = 30;

    println!("PARAM_{}", (ctx.full_message_modulus() as f64).log2());
    println!("d: {:?}", d);
    println!("k: {:?}", k);

    let model = generate_random_model(d, f_size, &mut ctx);
    let server = &Server::new(client.public_key.clone(), model);

    // Create a query vector from a client
    let client_feature_vector = vec![0, 0, 0];
    let query = client.create_query(client_feature_vector, &mut ctx);

    // Encode the model points
    let encoded_points = server.encode_model(&ctx);

    // Predict the k nearest labels
    let start = Instant::now();
    let _k_labels = server.predict(&query, &encoded_points, k, &ctx);
    let total_dur = start.elapsed().as_secs_f32();
    println!("Total time taken: {:?}s", total_dur);

    println!("Number of threads: {:?}", rayon::current_num_threads());
}

#[cfg(test)]
mod tests {
    use model::{Model, ModelPoint};

    use super::*;
    #[test]
    fn test_predict() {
        let mut ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);
        let client = Client::new(&ctx.parameters());

        let f_size = 3;
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
            f_size,
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
}
