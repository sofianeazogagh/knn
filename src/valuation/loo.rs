use revolut::*;
use tfhe::core_crypto::prelude::*;

type GLWE = GlweCiphertext<Vec<u64>>;
type LWE = LweCiphertext<Vec<u64>>;
type Poly = Polynomial<Vec<u64>>;
use tfhe::shortint::parameters::*;

use knn::*;

pub fn leave_one_out(
    X: Vec<Vec<u64>>,
    Y: Vec<u64>,
    X_test: Vec<Vec<u64>>,
    Y_test: Vec<u64>,
    k: usize,
    ctx: &mut Context,
    dist_modulus: u64,
) {
    let n = X.len();
    for i in 0..n {
        println!("i = {}", i);
        let mut model_vec_without_i = X.clone();
        model_vec_without_i.remove(i);
        let mut model_labels_without_i = Y.clone();
        model_labels_without_i.remove(i);

        let model = model::Model::new(model_vec_without_i, model_labels_without_i, dist_modulus);

        for (i, (x_test, y_test)) in X_test.iter().zip(Y_test.iter()).enumerate() {
            println!("Testing test point {}", i);

            let client = &client::Client::new(ctx, x_test.clone());
            let query = client.create_query(ctx, dist_modulus);
            
            let server = &server::Server::new(client.public_key.clone(), model.clone());

            let encoded_points = server.encode_model(&ctx);

            let (actual, dist_dur, topk_dur) = server.predict(&query, &encoded_points, k, &ctx);

            let predicted_labels = client.private_key.decrypt_lwe_vector(&actual[1], &ctx);

            // Get the most frequent label among the predicted labels (majority vote)
            let mut counts = std::collections::HashMap::new();
            for label in &predicted_labels {
                *counts.entry(label).or_insert(0) += 1;
            }
            // let predicted_label = *counts
            //     .iter()
            //     .max_by_key(|&(_, count)| count)
            //     .map(|(label, _)| *label)
            //     .unwrap_or(&predicted_labels[0]);

            // if predicted_labels[0] == y_test {
            //     correct += 1;
            // }
        }
    }
}

fn main() {
    let mut ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);

    let dataset_name = "cancer";

    let (dataset, _) = knn::model::parse_csv_dataset(
        &format!("./data/{}.csv", dataset_name),
        knn::QuantizeType::Binary,
    );

    let dist_modulus = 16 as u64;
    let train_size = (dataset.len() as f64 * 0.8) as usize;
    let test_size = dataset.len() - train_size;

    let (X, Y, X_test, Y_test) =
        knn::server::split_model_test(train_size, test_size, dataset.clone());

    leave_one_out(X, Y, X_test, Y_test, 3, &mut ctx, dist_modulus);
}
