use std::{error::Error, fs::File, path::Path};

use crate::{Context, Poly};
use csv::ReaderBuilder;
use rand::Rng;
use tfhe::core_crypto::prelude::*;

// Define the ModelPoint structure
pub struct ModelPoint {
    pub feature_vector: Vec<u64>,
    pub label: u64,
}

/* Encoded model point
 * m(X) : polynomial of degree f_size - 1
 * m'(X) : polynomial of degree f_size - 1
 * label : label of the model point
 */
#[allow(dead_code)]
pub struct ModelPointEncoded {
    pub m: Poly,
    pub m_prime: Poly,
    pub label: u64,
}

/* Model is a collection of model points
 * d : number of model points
 * f_size : number of features
 */
#[allow(dead_code)]
pub struct Model {
    pub model_points: Vec<ModelPoint>,
    pub d: usize,
    pub f_size: usize,
}

// Function to generate random model points
#[allow(dead_code)]
pub fn generate_random_model(model_size: usize, f_size: usize, ctx: &Context) -> Model {
    let mut rng = rand::thread_rng();
    let mut model_points = Vec::new();
    let modulus = ctx.full_message_modulus() as u64;

    for _ in 0..model_size {
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
        d: model_size,
        f_size: f_size,
    }
}

pub fn read_csv(file_path: &str, d: usize) -> Result<Model, Box<dyn Error>> {
    let mut data = Vec::new();
    let file = File::open(Path::new(file_path))?;
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);

    let mut f_size = 0;
    for (i, result) in reader.records().enumerate() {
        if i >= d {
            break;
        }
        let record = result?;
        let row: Vec<u64> = record.iter().map(|s| s.parse().unwrap()).collect();
        let label = row[row.len() - 1];
        let features_vector = row[..row.len() - 1].to_vec();
        f_size = features_vector.len().max(f_size);
        data.push(ModelPoint {
            feature_vector: features_vector,
            label: label,
        });
    }
    Ok(Model {
        model_points: data,
        d: d,
        f_size: f_size,
    })
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

        let n = ctx.full_message_modulus() as u64;

        // Create the polynomial m(X)
        let mut m_coeffs: Vec<u64> = vec![0; dim];
        for (i, &feature) in feature_vector.iter().rev().enumerate() {
            // m_coeffs[i] = feature;
            m_coeffs[i] = ((n - 2) * feature) % n;
        }
        // padding with 0 to ctx.polynomial_size().0
        m_coeffs.resize(ctx.polynomial_size().0, 0);
        let m_polynomial = Polynomial::from_container(m_coeffs); // m(X) = sum_{i=0}^{f_size-1} feature_i * X^i

        // Calculate the sum of squares of features for m'(X)
        let sum_squares_features: u64 = feature_vector.iter().map(|&feature| feature.pow(2)).sum();
        let mut m_prime_coeffs: Vec<u64> = vec![0; dim];
        m_prime_coeffs[dim - 1] = sum_squares_features;
        m_prime_coeffs.resize(ctx.polynomial_size().0, 0);
        let m_prime_polynomial = Polynomial::from_container(m_prime_coeffs); // m'(X) = sum(features^2) * X^dim

        // Add the encoded point to the final vector
        encoded_points.push(ModelPointEncoded {
            m: m_polynomial,
            m_prime: m_prime_polynomial,
            label: point.label,
        });
    }
    encoded_points
}
