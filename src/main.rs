use revolut::*;
use tfhe::core_crypto::prelude::*;

type GLWE = GlweCiphertext<Vec<u64>>;
type LWE = LweCiphertext<Vec<u64>>;
type Poly = Polynomial<Vec<u64>>;
use tfhe::shortint::parameters::*;
use rand::seq::SliceRandom;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;

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

    
  
    let mut accuracies = Vec::new();
    let mut file = File::create("results.txt").unwrap();
    let mut writer = BufWriter::new(&file);


    for i in 0..n {
        let mut correct = 0;
        let mut total = 0;

        let point = X[i].clone();
        let label = Y[i];

        let start = std::time::Instant::now();
        println!(" --------- Leaving out training point {} ---------", i);
        let mut model_vec_without_i = X.clone();
        model_vec_without_i.remove(i);
        let mut model_labels_without_i = Y.clone();
        model_labels_without_i.remove(i);

        let model = model::Model::new(model_vec_without_i, model_labels_without_i, dist_modulus);

        

        for (j, (x_test, y_test)) in X_test.iter().zip(Y_test.iter()).enumerate() {

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
            let predicted_label = *counts
                .iter()
                .max_by_key(|&(_, count)| count)
                .map(|(label, _)| *label)
                .unwrap_or(&predicted_labels[0]);

            if predicted_label == *y_test {
                correct += 1;
            }
            total += 1;


            println!("Accuracy for test point {}: {} / {}", j, correct, total);
            
        }
        
        let end = std::time::Instant::now();
        println!("Time taken: {:?}", end.duration_since(start));

        let accuracy = correct as f64 / total as f64;

        writeln!(writer, "{},{},{}",i, accuracy, end.duration_since(start).as_secs_f64()).unwrap();
        writer.flush().unwrap();
        accuracies.push(accuracy);
     
    }

    println!("Accuracies: {:?}", accuracies);
}

fn main() {
    let mut ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);

    let dataset_name = "cancer";

    let (train_dataset, train_size) = knn::model::parse_csv_dataset(
        &format!("./data/train/{}_train.csv", dataset_name),
        knn::QuantizeType::Binary,
    );

    let (test_dataset, test_size) = knn::model::parse_csv_dataset(
        &format!("./data/test/{}_test.csv", dataset_name),
        knn::QuantizeType::Binary,
    );



    let dist_modulus = 16 as u64;
    let (X, Y) = knn::model::dataset_X_Y(train_dataset);
    let (X_test, Y_test) = knn::model::dataset_X_Y(test_dataset);

    leave_one_out(X, Y, X_test, Y_test, 3, &mut ctx, dist_modulus);
}
