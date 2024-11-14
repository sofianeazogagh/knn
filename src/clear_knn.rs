use std::collections::HashMap;

use rand::seq::SliceRandom;
use revolut::Context;

use crate::{
    model::{self, Model, ModelPoint},
    DEBUG,
};
const BEST_MODEL_TRIES: usize = 10000;

#[allow(dead_code)]
pub struct KnnClear {
    pub distances: Vec<u64>,
    pub distances_and_labels_sorted: Vec<(u64, u64)>,
}

impl KnnClear {
    // Calculate the squared distance between two vectors
    pub fn squared_distance_in_clear(xs: &[u64], ys: &[u64]) -> u64 {
        xs.iter()
            .zip(ys)
            .map(|(x, y)| {
                let diff = if x > y { x - y } else { y - x };
                diff * diff
            })
            .sum()
    }

    pub fn run(client_feature_vector: &Vec<u64>, model: &Model, ctx: &Context) -> Self {
        let mut distances_and_labels = model
            .model_points
            .iter()
            .map(|point| {
                (
                    Self::squared_distance_in_clear(&point.feature_vector, &client_feature_vector),
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

        if DEBUG {
            println!("Distances in clear: {:?}", distances);
            println!(
                "Distances and labels in clear: {:?}",
                distances_and_labels_sorted
            );
        }

        KnnClear {
            distances,
            distances_and_labels_sorted,
        }
    }
}

// Repeatedly train and find the set that has the highest accuracy
// the accuracy is computed for all possible test vectors
pub fn find_best_model(
    model_size: usize,
    output_test_size: usize,
    k: usize,
    model: Model,
    ctx: &Context,
) -> (Vec<Vec<u64>>, Vec<u64>, Vec<Vec<u64>>, Vec<u64>, f64) {
    let mut final_train_vec: Vec<Vec<u64>> = vec![];
    let mut final_test_vec: Vec<Vec<u64>> = vec![];
    let mut final_train_labels: Vec<u64> = vec![];
    let mut final_test_labels: Vec<u64> = vec![];
    let mut highest_accuracy: usize = 0;

    let mut rng = rand::thread_rng();
    let test_size = model.model_points.len() - model_size;

    for _ in 0..BEST_MODEL_TRIES {
        // shuffle and split model/test vector
        let mut model_tested = model.clone();
        model_tested.model_points.shuffle(&mut rng);

        let (model_train, model_test) = split_train_test(model_size, test_size, model_tested);

        // do knn and check accuracy
        let mut oks: usize = 0;
        for (target, expected) in model_test
            .model_points
            .iter()
            .map(|p| (&p.feature_vector, p.label))
        {
            // let knn_c = run_knn(k, &model_vec, &model_labels, target);

            let knn_clear = KnnClear::run(&target, &model_train, &ctx);

            let out_labels: Vec<_> = knn_clear
                .distances_and_labels_sorted
                .iter()
                .map(|l| l.1)
                .collect();
            let res = majority(&out_labels);
            if res == expected {
                oks += 1;
            }
        }

        // check if our accuracy is higher
        if oks > highest_accuracy {
            final_train_vec = model_train
                .model_points
                .iter()
                .take(output_test_size)
                .map(|p| p.feature_vector.clone())
                .collect();
            final_train_labels = model_train.model_points.iter().map(|p| p.label).collect();
            final_test_vec = model_test
                .model_points
                .iter()
                .take(output_test_size)
                .map(|p| p.feature_vector.clone())
                .collect();
            final_test_labels = model_test.model_points.iter().map(|p| p.label).collect();
            highest_accuracy = oks;
        }
    }

    (
        final_train_vec,
        final_train_labels,
        final_test_vec,
        final_test_labels,
        highest_accuracy as f64 / test_size as f64,
    )
}

/// Split the feature vectors into a training and testing set, which are equivalent to return two models.
pub fn split_train_test(train_size: usize, test_size: usize, model: Model) -> (Model, Model) {
    let mut train_vec: Vec<Vec<u64>> = vec![];
    let mut test_vec: Vec<Vec<u64>> = vec![];
    let mut train_labels: Vec<u64> = vec![];
    let mut test_labels: Vec<u64> = vec![];

    for (i, point) in model.model_points.into_iter().enumerate() {
        let label = point.label;
        if i < train_size {
            train_vec.push(point.feature_vector);
            train_labels.push(label);
        } else if i >= train_size && i < train_size + test_size {
            test_vec.push(point.feature_vector);
            test_labels.push(label);
        } else {
            break;
        }
    }

    let model_train = Model {
        model_points: train_vec
            .into_iter()
            .zip(train_labels)
            .map(|(v, l)| ModelPoint {
                feature_vector: v,
                label: l,
            })
            .collect(),
        d: model.d,
        gamma: model.gamma,
        dist_modulus: model.dist_modulus,
    };
    let model_test = Model {
        model_points: test_vec
            .into_iter()
            .zip(test_labels)
            .map(|(v, l)| ModelPoint {
                feature_vector: v,
                label: l,
            })
            .collect(),
        d: model.d,
        gamma: model.gamma,
        dist_modulus: model.dist_modulus,
    };

    (model_train, model_test)
}

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
