use std::{fs::File, path::Path};

use crate::{Context, Poly, QuantizeType};
use csv::ReaderBuilder;
use rand::Rng;
use tfhe::core_crypto::prelude::*;

const MAX_MODEL: u64 = 16;
// Define the ModelPoint structure
#[derive(Clone)]
pub struct ModelPoint {
    pub feature_vector: Vec<u64>,
    pub label: u64,
}

impl ModelPoint {
    pub fn print(&self) {
        println!("Feature vector: {:?}", self.feature_vector);
        println!("Label: {}", self.label);
    }
}

/* Encoded model point
 * m(X) : polynomial of degree f_size - 1
 * m'(X) : polynomial of degree f_size - 1
 * label : label of the model point
 */
#[allow(dead_code)]
#[derive(Clone)]
pub struct ModelPointEncoded {
    pub m: Poly,
    pub m_prime: u64,
    pub label: u64,
}

/* Model is a collection of model points
 * d : number of model points
 * f_size : number of features
 */
#[allow(dead_code)]
#[derive(Clone)]
pub struct Model {
    pub model_points: Vec<ModelPoint>,
    pub d: usize,
    pub gamma: usize,
    pub dist_modulus: u64,
}

impl Model {
    #[allow(dead_code)]
    pub fn print_first_point(&self) {
        self.model_points[0].print();
        println!("d: {}", self.d);
        println!("f_size: {}", self.gamma);
        let delta_dist = (1u64 << 63) / self.dist_modulus;
        println!("log2(delta_dist): {}", delta_dist.ilog2());
    }

    pub fn new(feature_vectors: Vec<Vec<u64>>, labels: Vec<u64>, dist_modulus: u64) -> Model {
        let d = feature_vectors.len();
        let gamma = feature_vectors[0].len();
        let model_points: Vec<ModelPoint> = feature_vectors
            .into_iter()
            .zip(labels)
            .map(|(feature_vector, label)| ModelPoint {
                feature_vector,
                label,
            })
            .collect();

        Model {
            model_points,
            d,
            gamma,
            dist_modulus,
        }
    }
}

// Function to generate random model points
#[allow(dead_code)]
pub fn generate_random_model(d: usize, f_size: usize, ctx: &Context) -> Model {
    let mut rng = rand::thread_rng();
    let mut model_points = Vec::new();
    let modulus = ctx.full_message_modulus() as u64;

    for _ in 0..d {
        let feature_vector: Vec<u64> = (0..f_size)
            .map(|_| rng.gen_range(0..10) % modulus)
            .collect();
        let label = rng.gen_range(0..5) % modulus;
        model_points.push(ModelPoint {
            feature_vector,
            label,
        });
    }

    Model {
        model_points,
        d,
        gamma: f_size,
        dist_modulus: ctx.full_message_modulus() as u64,
    }
}

#[allow(dead_code)]
pub fn model_test(d: usize, f_size: usize, dist_modulus: u64) -> Model {
    // Create test model points
    let model_points = vec![
        ModelPoint {
            feature_vector: vec![1, 2, 3],
            label: 2,
        },
        ModelPoint {
            feature_vector: vec![0, 1, 0],
            label: 4,
        },
        ModelPoint {
            feature_vector: vec![1, 0, 0],
            label: 3,
        },
        ModelPoint {
            feature_vector: vec![0, 2, 0],
            label: 5,
        },
        ModelPoint {
            feature_vector: vec![2, 3, 1],
            label: 2,
        },
    ];

    Model {
        model_points,
        d: d,
        gamma: f_size,
        dist_modulus: dist_modulus,
    }
}

pub fn parse_csv(
    file_path: &str,
    quantize_type: QuantizeType,
    d: usize,
    dist_modulus: u64,
) -> Model {
    let f_handle = File::open(Path::new(file_path)).unwrap();
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .from_reader(f_handle);

    let mut rows: Vec<_> = reader
        .records()
        .take(d)
        .map(|res| {
            let record = res.unwrap();
            record
                .iter()
                .map(|s| s.parse().unwrap())
                .collect::<Vec<_>>()
        })
        .collect();

    let mut max_row_len = 0;

    match quantize_type {
        QuantizeType::None => {
            rows.iter_mut().for_each(|row| {
                max_row_len = row.len().max(max_row_len);
            });
        }
        QuantizeType::Binary => {
            let threshold = MAX_MODEL / 2;
            let f = |x| {
                assert!(x <= MAX_MODEL);
                if x < threshold {
                    0
                } else {
                    1
                }
            };
            rows.iter_mut().for_each(|row| {
                row.iter_mut().rev().skip(1).for_each(|x| {
                    *x = f(*x);
                });
                max_row_len = row.len().max(max_row_len);
            });
        }
        QuantizeType::Ternary => {
            let third = (MAX_MODEL as f64 / 3.0).ceil() as u64;
            assert_eq!(third, 6);
            let f = |x| {
                if x < third {
                    0
                } else if x >= third && x < 2 * third {
                    1
                } else {
                    2
                }
            };
            rows.iter_mut().for_each(|row| {
                row.iter_mut().rev().skip(1).for_each(|x| {
                    *x = f(*x);
                });
                max_row_len = row.len().max(max_row_len);
            });
        }
    }

    let model_points: Vec<ModelPoint> = rows
        .into_iter()
        .map(|row| {
            let label = row.last().cloned().unwrap();
            let feature_vector = row[..row.len() - 1].to_vec();
            ModelPoint {
                feature_vector,
                label,
            }
        })
        .collect();

    Model {
        model_points,
        d,
        gamma: max_row_len - 1,
        dist_modulus,
    }
}

/* Encode the model points into two polynomials m(X) and m'(X) as explained in section 4.1 of https://eprint.iacr.org/2023/852.pdf */
pub fn encode_model_points(
    model_points: &Vec<ModelPoint>,
    ctx: &Context,
) -> Vec<ModelPointEncoded> {
    let mut encoded_points: Vec<ModelPointEncoded> = Vec::new();

    for point in model_points {
        let feature_vector = &point.feature_vector;
        let dim = feature_vector.len(); // f_size : dimension of the feature space

        // Create the polynomial m(X)
        let mut m_coeffs: Vec<u64> = vec![0; dim];
        for (i, &feature) in feature_vector.iter().rev().enumerate() {
            m_coeffs[i] = feature;
        }
        // padding with 0 to ctx.polynomial_size().0
        m_coeffs.resize(ctx.polynomial_size().0, 0);
        let m_polynomial = Polynomial::from_container(m_coeffs); // m(X) = sum_{i=0}^{f_size-1} feature_{f_size-1-i} * X^i

        // Calculate the sum of squares of features for m'(X)
        let m_prime: u64 = feature_vector.iter().map(|&feature| feature.pow(2)).sum();

        // Add the encoded point to the final vector
        encoded_points.push(ModelPointEncoded {
            m: m_polynomial,
            m_prime,
            label: point.label,
        });
    }
    encoded_points
}

pub fn parse_csv_dataset(file_path: &str, quantize_type: QuantizeType) -> Vec<Vec<u64>> {
    let f_handle = File::open(Path::new(file_path)).unwrap();
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .from_reader(f_handle);

    let mut rows: Vec<_> = reader
        .records()
        .map(|res| {
            let record = res.unwrap();
            record
                .iter()
                .map(|s| s.parse().unwrap())
                .collect::<Vec<_>>()
        })
        .collect();

    let mut max_row_len = 0;

    match quantize_type {
        QuantizeType::None => {
            rows.iter_mut().for_each(|row| {
                max_row_len = row.len().max(max_row_len);
            });
        }
        QuantizeType::Binary => {
            let threshold = MAX_MODEL / 2;
            let f = |x| {
                assert!(x <= MAX_MODEL);
                if x < threshold {
                    0
                } else {
                    1
                }
            };
            rows.iter_mut().for_each(|row| {
                row.iter_mut().rev().skip(1).for_each(|x| {
                    *x = f(*x);
                });
                max_row_len = row.len().max(max_row_len);
            });
        }
        QuantizeType::Ternary => {
            let third = (MAX_MODEL as f64 / 3.0).ceil() as u64;
            assert_eq!(third, 6);
            let f = |x| {
                if x < third {
                    0
                } else if x >= third && x < 2 * third {
                    1
                } else {
                    2
                }
            };
            rows.iter_mut().for_each(|row| {
                row.iter_mut().rev().skip(1).for_each(|x| {
                    *x = f(*x);
                });
                max_row_len = row.len().max(max_row_len);
            });
        }
    }

    rows
}
