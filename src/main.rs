use rand::Rng;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::time::Instant;

use revolut::*;

// Fast Fourier Transform
use concrete_fft::c64;
use dyn_stack::{GlobalPodBuffer, PodStack, ReborrowMut};
use tfhe::core_crypto::entities::FourierPolynomial;
use tfhe::core_crypto::fft_impl::fft64::math::fft::FftView;

// TFHE
use tfhe::core_crypto::prelude::*;
use tfhe::shortint::parameters::*;

pub struct ModelPoint {
    feature_vector: Vec<u64>,
    label: u64,
}

pub struct ModelPointEncoded {
    m: Polynomial<Vec<u64>>,
    m_prime: Polynomial<Vec<u64>>,
    _label: u64,
}

// TODO : function that encode the model points into two polynomial as explained in section 4.1 of https://eprint.iacr.org/2023/852.pdf
pub fn encode_model_points(
    model_points: &Vec<ModelPoint>,
    ctx: &Context,
) -> Vec<ModelPointEncoded> {
    let mut encoded_points: Vec<ModelPointEncoded> = Vec::new();

    for point in model_points {
        let feature_vector = &point.feature_vector;
        let dim = feature_vector.len(); // d : dimension de l'espace des features

        // Create the polynomial m(X)
        let mut m_coeffs: Vec<u64> = vec![0; dim];
        for (i, &feature) in feature_vector.iter().rev().enumerate() {
            // m_coeffs[i] = feature * ctx.delta() as u64;
            m_coeffs[i] = feature;
        }
        // padding with 0 to ctx.polynomial_size().0
        m_coeffs.resize(ctx.polynomial_size().0, 0);
        let m_polynomial = Polynomial::from_container(m_coeffs); // m(X) = sum_{i=0}^{d-1} feature_i * X^i

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
            _label: point.label,
        });
    }
    encoded_points
}

// Function to calculate the squared distance of a query vector to a model point
pub fn squared_distance(
    query: &Query,
    model_points: &ModelPointEncoded,
    vector_dimension: usize,
    public_key: &PublicKey,
    ctx: &Context,
) -> LweCiphertext<Vec<u64>> {
    let m = model_points.m.clone();
    let m_prime = model_points.m_prime.clone();

    // Step 1: Compute the inner product
    let mut inner_product = public_key.glwe_absorption_polynomial_with_fft(&query.ct, &m);

    // Step 2: Compute 2 * inner product and negate it
    let mut neg_double_inner_product = public_key
        .glwe_ciphertext_plaintext_mul(&mut inner_product, ctx.full_message_modulus() as u64 - 2);

    // Step 3: Compute -2*inner_product + m_prime
    let dist = public_key.glwe_sum_polynomial(&mut neg_double_inner_product, &m_prime, ctx);

    // Step 4: Add the second component of the query
    let dist = public_key.glwe_sum(&dist, &query.ct_second);

    // Step 5: Sample exctract at vector_dimension - 1
    let result = public_key.sample_extract_in_glwe(&dist, vector_dimension - 1, ctx);

    result
}

/* Sort the distances and apply the permutation to the identity to get the index_labels */
pub fn sort_distances(
    distances: &LUT,
    public_key: &PublicKey,
    ctx: &Context,
) -> Vec<LweCiphertext<Vec<u64>>> {
    let n = ctx.full_message_modulus() as u64;
    // let lut_distances = LUT::from_vec_of_lwe(distances.clone(), public_key, ctx);

    let sorted_lut = public_key.blind_counting_sort(&distances, ctx);
    let mut sorted_distances = Vec::new();
    for i in 0..n {
        let lwe = public_key.sample_extract(&sorted_lut, i as usize, ctx);
        sorted_distances.push(lwe);
    }
    sorted_distances
}

pub fn find_k_nearest_labels(
    sorted_distances: &Vec<LweCiphertext<Vec<u64>>>,
    distances: &LUT,
    model_points: &Vec<ModelPoint>,
    k: usize,
    public_key: &PublicKey,
    ctx: &Context,
) -> Vec<LweCiphertext<Vec<u64>>> {
    let label_lut = LUT::from_vec_trivially(
        &model_points.iter().map(|p| p.label).collect::<Vec<u64>>(),
        ctx,
    );

    let k_labels: Vec<LweCiphertext<Vec<u64>>> = (0..k)
        .into_par_iter() // Use parallel iterator
        .map(|i| {
            let index_distance = public_key.blind_index(distances, &sorted_distances[i], ctx); // index of the nearest distance
            public_key.blind_array_access(&index_distance, &label_lut, ctx) // label of the nearest distance using the index
        })
        .collect();

    k_labels
}

pub fn calculate_second(client_feature_vector: &Vec<u64>) -> u64 {
    client_feature_vector.iter().map(|&x| x.pow(2)).sum()
}

pub fn encrypt_second_in_glwe(
    client_feature_vector: &Vec<u64>,
    private_key: &PrivateKey,
    ctx: &mut Context,
) -> GlweCiphertext<Vec<u64>> {
    let dim = client_feature_vector.len();
    let second_value = calculate_second(client_feature_vector);

    // Create a polynomial of size ctx.polynomial_size() filled with zeros
    let mut poly_coeffs: Vec<u64> = vec![0; ctx.polynomial_size().0];

    // Set the value at slot dim - 1
    poly_coeffs[dim - 1] = second_value;

    // Encrypt the polynomial in GLWE
    let result = private_key.allocate_and_encrypt_glwe_from_vec(&poly_coeffs, ctx);
    result
}

pub struct Query {
    pub ct: GlweCiphertext<Vec<u64>>,
    pub ct_second: GlweCiphertext<Vec<u64>>,
}

impl Query {
    pub fn from_vec(
        feature_vector: &Vec<u64>,
        private_key: &PrivateKey,
        ctx: &mut Context,
    ) -> Query {
        let ct = private_key.allocate_and_encrypt_glwe_from_vec(feature_vector, ctx);
        let ct_second = encrypt_second_in_glwe(feature_vector, private_key, ctx);
        Query {
            ct: ct,
            ct_second: ct_second,
        }
    }
}

pub fn generate_random_model_points(
    s: usize,
    feature_vector_size: usize,
    ctx: &Context,
) -> Vec<ModelPoint> {
    let mut rng = rand::thread_rng();
    let mut model_points = Vec::new();
    let modulus = ctx.full_message_modulus() as u64;

    for _ in 0..s {
        let feature_vector: Vec<u64> = (0..feature_vector_size)
            .map(|_| rng.gen_range(0..10) % modulus)
            .collect();
        let label = rng.gen_range(0..5) % modulus;
        model_points.push(ModelPoint {
            feature_vector,
            label,
        });
    }

    model_points
}

fn main() {
    let pool = ThreadPoolBuilder::new().num_threads(4).build().unwrap();

    let mut ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);
    let private_key = key(PARAM_MESSAGE_4_CARRY_0);
    let public_key = &private_key.public_key;

    let dimension = 3;
    let s = 10;
    let model_points = generate_random_model_points(s, dimension, &ctx);

    // Create a query vector from a client
    let client_feature_vector = vec![1, 2, 3];
    let query = Query::from_vec(&client_feature_vector, &private_key, &mut ctx);

    // Encode the model points
    let encoded_points = encode_model_points(&model_points, &ctx);

    // Time here
    let start = Instant::now();
    // Calculate the squared distance of the query vector to each model point
    let distances: Vec<_> = encoded_points
        .par_iter()
        .map(|point| squared_distance(&query, point, dimension, public_key, &ctx))
        .collect();

    let lut_distances = LUT::from_vec_of_lwe(distances, public_key, &ctx);
    let sorted_distances = sort_distances(&lut_distances, public_key, &ctx);
    let k_labels = find_k_nearest_labels(
        &sorted_distances,
        &lut_distances,
        &model_points,
        3,
        public_key,
        &ctx,
    );
    let end = Instant::now();
    println!("Time taken: {:?}", end.duration_since(start));
    println!("k_labels:");
    for (i, label) in k_labels.iter().enumerate() {
        let label = private_key.decrypt_lwe(&label, &ctx);
        println!("label_{}: {:?}", i, label);
    }
}

#[cfg(test)]
mod tests {
    use dyn_stack::GlobalPodBuffer;
    use polynomial_algorithms::polynomial_karatsuba_wrapping_mul;

    use super::*;

    #[test]
    fn test_encode_model_points() {
        let ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);

        // Create test model points
        let model_points = vec![
            ModelPoint {
                feature_vector: vec![1, 2, 3],
                label: 0,
            },
            ModelPoint {
                feature_vector: vec![4, 5, 6],
                label: 1,
            },
        ];

        // Encode the model points
        let encoded_points = encode_model_points(&model_points, &ctx);

        // Test the first encoded point
        let mut expected_m = vec![3, 2, 1];
        expected_m.resize(ctx.polynomial_size().0, 0);
        assert_eq!(encoded_points[0].m.as_ref(), expected_m.as_slice());

        let mut expected_m_prime = vec![0, 0, 6];
        expected_m_prime.resize(ctx.polynomial_size().0, 0);
        assert_eq!(
            encoded_points[0].m_prime.as_ref(),
            expected_m_prime.as_slice()
        );
        assert_eq!(encoded_points[0]._label, 0);

        // Test the second encoded point
        expected_m = vec![6, 5, 4];
        expected_m.resize(ctx.polynomial_size().0, 0);
        assert_eq!(encoded_points[1].m.as_ref(), expected_m.as_slice());
        expected_m_prime = vec![0, 0, 15];
        expected_m_prime.resize(ctx.polynomial_size().0, 0);
        assert_eq!(
            encoded_points[1].m_prime.as_ref(),
            expected_m_prime.as_slice()
        );
        assert_eq!(encoded_points[1]._label, 1);
    }

    #[test]
    fn test_squared_distance() {
        let mut ctx = Context::from(PARAM_MESSAGE_4_CARRY_0);
        let private_key = key(PARAM_MESSAGE_4_CARRY_0);
        let public_key = &private_key.public_key;

        // Create a test feature vector
        let client_feature_vector: Vec<u64> = vec![1, 0, 0];
        let query = Query::from_vec(&client_feature_vector, &private_key, &mut ctx);

        // Create a test model point
        let model_point = ModelPoint {
            feature_vector: vec![1, 0, 0],
            label: 1,
        };

        println!("client_feature_vector: {:?}", client_feature_vector);
        println!("model_point: {:?}", model_point.feature_vector);

        let dim = model_point.feature_vector.len();

        // Encode the model point
        let encoded_points = encode_model_points(&vec![model_point], &ctx);

        // Calculate the squared distance

        let result = squared_distance(&query, &encoded_points[0], dim, public_key, &ctx);

        // Decrypt the result
        let decrypted_result = private_key.decrypt_lwe(&result, &ctx);

        println!("decrypted_result: {:?}", decrypted_result);
    }
}
