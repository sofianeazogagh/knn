use revolut::*;
use tfhe::core_crypto::prelude::*;

type GLWE = GlweCiphertext<Vec<u64>>;
type LWE = LweCiphertext<Vec<u64>>;
type Poly = Polynomial<Vec<u64>>;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write, BufRead};
use tfhe::shortint::parameters::*;
use std::time::Instant;

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
    let results_file = File::create("results.txt").expect("Unable to create results.txt");
    let mut writer = BufWriter::new(results_file);
    for i in 0..n {
        println!(
            r#"
[VALUATION] Training point {}

      .-""""-.
     / -   -  \
    |  .-. .- |
    |  \o| |o (
    \     ^    \
     '.  )--'  /
       '-...-'`
"#,
            i
        );

        let mut model_vec_without_i = X.clone();
        model_vec_without_i.remove(i);
        let mut model_labels_without_i = Y.clone();
        model_labels_without_i.remove(i);

        let mut correct = 0;
        let mut total = 0;

        let model = model::Model::new(model_vec_without_i, model_labels_without_i, dist_modulus);

        let start = Instant::now();
        for (i, (x_test, y_test)) in X_test.iter().zip(Y_test.iter()).enumerate() {
            println!("[VALUATION] Testing test point {}", i);

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
                .map(|(label, _)| label)
                .unwrap_or(&&predicted_labels[0]);
            
            if *predicted_label == *y_test {
                correct += 1;
            }
            total += 1;
            println!("[VALUATION] Accuracy after test point {}: {} / {}", i, correct, total);
        }
        let duration = start.elapsed().as_secs_f32();
        println!("[VALUATION] Training point valuation :{} - Duration: {}s", correct as f32/total as f32, duration);
        writer.write_fmt(format_args!("{},{},{}\n", i, correct as f32/total as f32, duration)).expect("Unable to write to results.txt");


    }
}

pub fn retrain_without_samples(
    X: Vec<Vec<u64>>,
    Y: Vec<u64>,
    X_eval: Vec<Vec<u64>>,
    Y_eval: Vec<u64>,
    threshold: f64,
    ctx: &mut Context,
    dist_modulus: u64,
    k: usize,
) -> f64 {
    println!(
        "
    ╔════════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║   🎨 Retraining without samples with threshold: {:<8}   ║
    ║                                                          ║
    ╚════════════════════════════════════════════════════════════╝
    ",
        threshold
    );
    let mut results = Vec::new();

    let mut selected_samples = Vec::new();

    // Read results.txt file and print its contents
    let file = File::open("results.txt").expect("Unable to open results.txt");
    let reader = BufReader::new(file);

    let mut correct = 0;
    let mut total = 0;

    for line in reader.lines() {
        let line = line.expect("Unable to read line");
        let parts: Vec<&str> = line.split(',').collect();
        let sample_id = parts[0].parse::<usize>().unwrap();
        let accuracy = parts[1].parse::<f64>().unwrap();
        let time = parts[2].parse::<f64>().unwrap();
        results.push((sample_id, accuracy, time));
        if accuracy <= threshold {
            selected_samples.push(sample_id);
        }
    }

    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let num_selected_samples = threshold * results.len() as f64;

    let model_filtered = results[..num_selected_samples as usize]
        .iter()
        .map(|(sample_id, _, _)| X[*sample_id].clone())
        .collect::<Vec<Vec<u64>>>();
    let model_labels_filtered = results[..num_selected_samples as usize]
        .iter()
        .map(|(sample_id, _, _)| Y[*sample_id])
        .collect::<Vec<u64>>();

    println!(
        "[RETRAIN] Shrink ratio: {}",
        model_filtered.len() as f64 / X.len() as f64
    );

    let model = model::Model::new(model_filtered, model_labels_filtered, dist_modulus);

    for (i, (x_test, y_test)) in X_eval.iter().zip(Y_eval.iter()).enumerate() {
        println!("[RETRAIN] Testing test point {}", i);

        let client = &client::Client::new(ctx, x_test.clone());

        let knn_clear = server::KnnClear::run(k, &client.target_vector, &model, ctx.delta());

        let clear_labels = knn_clear
            .top_k_distances_and_labels
            .iter()
            .map(|(_, l)| *l)
            .collect::<Vec<_>>();
        let clear_maj = server::majority(&clear_labels);

        if clear_maj == *y_test {
            correct += 1;
        }
        total += 1;

        println!("[RETRAIN] Accuracy for test point {}: {} / {}", i, correct, total);
    }

    let accuracy = correct as f64 / total as f64;

    println!("Accuracy on filtered model: {}", accuracy);
    return accuracy;
}

fn main() {
    let mut ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);
    let seed = 42;
    let dataset_name = "cancer";

    let (dataset, _) = knn::model::parse_csv_dataset(
        &format!("./data/{}.csv", dataset_name),
        knn::QuantizeType::Binary,
    );

    let dist_modulus = 16 as u64;
    let train_size = (dataset.len() as f64 * 0.7) as usize;
    let test_size = (dataset.len() as f64 * 0.2) as usize;
    let eval_size = dataset.len() - train_size - test_size;

    let (X, Y, X_test, Y_test, X_eval, Y_eval) =
        knn::server::split_model_test(train_size, test_size, eval_size, dataset.clone(), seed);

    leave_one_out(X, Y, X_test, Y_test, 3, &mut ctx, dist_modulus);

    // let mut accuracies = Vec::new();
    // let threshods = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    // for threshold in threshods {
    //     accuracies.push(retrain_without_samples(
    //         X.clone(),
    //         Y.clone(),
    //         X_eval.clone(),
    //         Y_eval.clone(),
    //         threshold,
    //         &mut ctx,
    //         dist_modulus,
    //         3,
    //     ));
    // }

    // let file = File::create("accuracies.txt").expect("Unable to create accuracies.txt");
    // let mut writer = BufWriter::new(file);
    // for (threshold, accuracy) in threshods.iter().zip(accuracies.iter()) {
    //     writer
    //         .write_fmt(format_args!("{},{}\n", threshold, accuracy))
    //         .expect("Unable to write to accuracies.txt");
    // }
}
