use std::collections::HashMap;

use rand::seq::SliceRandom;
use revolut::Context;

use crate::model::Model;
const BEST_MODEL_TRIES: usize = 10000;

#[allow(dead_code)]
pub struct KnnClear {
    pub distances: Vec<u64>,
    pub distances_and_labels_sorted: Vec<(u64, u64)>,
    pub top_k: Vec<(u64, u64)>,
}

// Calculate the squared distance between two vectors
pub fn squared_distance_in_clear(xs: &Vec<u64>, ys: &Vec<u64>) -> u64 {
    xs.iter()
        .zip(ys)
        .map(|(x, y)| {
            let diff = if x > y { x - y } else { y - x };
            diff * diff
        })
        .sum()
}

impl KnnClear {
    pub fn run(k: usize, client_feature_vector: &Vec<u64>, model: &Model, ctx: &Context) -> Self {
        let mut distances_and_labels = model
            .model_points
            .iter()
            .map(|point| {
                (
                    squared_distance_in_clear(&point.feature_vector, &client_feature_vector),
                    point.label,
                )
            })
            .collect::<Vec<(u64, u64)>>();
        let delta_dist = (1u64 << 63) / model.dist_modulus;
        let ratio = ctx.delta() / delta_dist;
        distances_and_labels
            .iter_mut()
            .for_each(|(d, _)| *d /= ratio);

        let mut distances_and_labels_sorted = distances_and_labels.clone();
        distances_and_labels_sorted.sort_by_key(|&(distance, _)| distance);

        let distances = distances_and_labels
            .iter()
            .map(|(d, _)| *d)
            .collect::<Vec<u64>>();

        let top_k = distances_and_labels_sorted[..k].to_vec();

        KnnClear {
            distances,
            distances_and_labels_sorted,
            top_k,
        }
    }
}

// Repeatedly train and find the set that has the highest accuracy
// the accuracy is computed for all possible test vectors
pub fn find_best_model(
    model_size: usize,
    output_test_size: usize,
    k: usize,
    dataset: &Vec<Vec<u64>>,
    ctx: &Context,
) -> (Vec<Vec<u64>>, Vec<u64>, Vec<Vec<u64>>, Vec<u64>, f64) {
    let mut final_model_vec: Vec<Vec<u64>> = vec![];
    let mut final_test_vec: Vec<Vec<u64>> = vec![];
    let mut final_model_labels: Vec<u64> = vec![];
    let mut final_test_labels: Vec<u64> = vec![];
    let mut highest_accuracy: usize = 0;

    let mut rng = rand::thread_rng();
    let test_size = dataset.len() - model_size;

    for _ in 0..BEST_MODEL_TRIES {
        // shuffle and split model/test vector
        let mut rows = dataset.clone();
        rows.shuffle(&mut rng);
        let (model_vec, model_labels, test_vec, test_labels) =
            split_model_test(model_size, test_size, rows);

        // do knn and check accuracy
        let mut oks: usize = 0;
        for (target, expected) in test_vec.iter().zip(&test_labels) {
            // When dist modulus is set to the full message modulus,
            // the distance is the same as the squared distance in clear
            let model = Model::new(
                model_vec.clone(),
                model_labels.clone(),
                ctx.full_message_modulus() as u64,
            );
            let knn_clear = KnnClear::run(k, &target, &model, &ctx);
            let out_labels = knn_clear
                .top_k
                .iter()
                .map(|(_, l)| *l)
                .collect::<Vec<u64>>();
            let res = majority(&out_labels);
            if res == *expected {
                oks += 1;
            }
        }

        // check if our accuracy is higher
        if oks > highest_accuracy {
            final_model_vec = model_vec;
            final_model_labels = model_labels;
            final_test_vec = test_vec[..output_test_size].to_vec();
            final_test_labels = test_labels[..output_test_size].to_vec();
            highest_accuracy = oks;
        }
    }

    (
        final_model_vec,
        final_model_labels,
        final_test_vec,
        final_test_labels,
        highest_accuracy as f64 / test_size as f64,
    )
}

/// Split the feature vectors into a training and testing set.
/// The feature vectors are specified in `rows` and the last
/// element of every vector is the label.
pub fn split_model_test(
    model_size: usize,
    test_size: usize,
    rows: Vec<Vec<u64>>,
) -> (Vec<Vec<u64>>, Vec<u64>, Vec<Vec<u64>>, Vec<u64>) {
    let mut model_vec: Vec<Vec<u64>> = vec![];
    let mut test_vec: Vec<Vec<u64>> = vec![];
    let mut model_labels: Vec<u64> = vec![];
    let mut test_labels: Vec<u64> = vec![];

    for (i, mut row) in rows.into_iter().enumerate() {
        let last = row.pop().unwrap();
        if i < model_size {
            model_vec.push(row);
            model_labels.push(last);
        } else if i >= model_size && i < model_size + test_size {
            test_vec.push(row);
            test_labels.push(last);
        } else {
            break;
        }
    }

    (model_vec, model_labels, test_vec, test_labels)
}

// /// Split the feature vectors into a training and testing set, which are equivalent to return two models.
// pub fn split_train_test(train_size: usize, test_size: usize, model: Model) -> (Model, Model) {
//     let mut train_vec: Vec<Vec<u64>> = vec![];
//     let mut test_vec: Vec<Vec<u64>> = vec![];
//     let mut train_labels: Vec<u64> = vec![];
//     let mut test_labels: Vec<u64> = vec![];

//     for (i, point) in model.model_points.into_iter().enumerate() {
//         let label = point.label;
//         if i < train_size {
//             train_vec.push(point.feature_vector);
//             train_labels.push(label);
//         } else if i >= train_size && i < train_size + test_size {
//             test_vec.push(point.feature_vector);
//             test_labels.push(label);
//         } else {
//             break;
//         }
//     }

//     let model_train = Model {
//         model_points: train_vec
//             .into_iter()
//             .zip(train_labels)
//             .map(|(v, l)| ModelPoint {
//                 feature_vector: v,
//                 label: l,
//             })
//             .collect(),
//         d: model.d,
//         gamma: model.gamma,
//         dist_modulus: model.dist_modulus,
//     };
//     let model_test = Model {
//         model_points: test_vec
//             .into_iter()
//             .zip(test_labels)
//             .map(|(v, l)| ModelPoint {
//                 feature_vector: v,
//                 label: l,
//             })
//             .collect(),
//         d: model.d,
//         gamma: model.gamma,
//         dist_modulus: model.dist_modulus,
//     };

//     (model_train, model_test)
// }

pub fn majority(vs: &[u64]) -> u64 {
    assert!(!vs.is_empty());
    let max = vs
        .iter()
        .fold(HashMap::<u64, usize>::new(), |mut m, x| {
            *m.entry(*x).or_default() += 1;
            m
        })
        .into_iter()
        .max_by_key(|(_, v)| *v)
        .map(|(k, _)| k);
    max.unwrap()
}
