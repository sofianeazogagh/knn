use std::time::Instant;

use crate::model::*;
use revolut::*;
use tfhe::core_crypto::prelude::*;

use crate::Query;
use rayon::{prelude::*, ThreadPoolBuilder};
pub struct Server {
    public_key: PublicKey,
    model: Model,
}

impl Server {
    pub fn new(public_key: PublicKey, model: Model) -> Self {
        Server { public_key, model }
    }

    pub fn encode_model(&self, ctx: &Context) -> Vec<ModelPointEncoded> {
        let model_points = &self.model.model_points;
        encode_model_points(model_points, ctx)
    }

    // Function to calculate the squared distance of a query vector to a model point
    // TODO : optimization by encoding the model points as one GLWE M and one lwe m_prime as well as the query as one GLWE C and one lwe c_second (much less heavy in memory?)
    fn squared_distance(
        &self,
        query: &Query,
        point: &ModelPointEncoded,
        ctx: &Context,
    ) -> LweCiphertext<Vec<u64>> {
        let m = point.m.clone();
        let m_prime = point.m_prime.clone();

        let f_size = self.model.f_size;

        // Step 1: Compute the inner product m(X) * q(X)
        let mut inner_product = self
            .public_key
            .glwe_absorption_polynomial_with_fft(&query.ct, &m);

        // Step 2: Compute -2*inner_product + m_prime
        let dist = self
            .public_key
            .glwe_sum_polynomial(&mut inner_product, &m_prime, ctx);

        // Step 4: Add the second component of the query
        let dist = self.public_key.glwe_sum(&dist, &query.ct_second);

        // Step 5: Sample exctract at feature_vector_size - 1
        let result = self
            .public_key
            .sample_extract_in_glwe(&dist, f_size - 1, ctx);

        result
    }

    fn topk_labels(
        &self,
        many_lwes: &Vec<Vec<LweCiphertext<Vec<u64>>>>,
        k: usize,
        ctx: &Context,
    ) -> Vec<LweCiphertext<Vec<u64>>> {
        let mut topk_many_lut = self.public_key.blind_topk_many_lut_par(many_lwes, k, ctx);

        // The first lut is the distances, the second is the labels
        let k_labels = topk_many_lut.remove(1);
        k_labels
    }

    pub fn predict(
        &self,
        query: &Query,
        model_points: &Vec<ModelPointEncoded>,
        k: usize,
        ctx: &Context,
    ) -> Vec<LweCiphertext<Vec<u64>>> {
        let pool = ThreadPoolBuilder::new().num_threads(4).build().unwrap();

        // Step 0: Encrypt the labels as LWE ciphertexts trivially
        let labels = model_points
            .iter()
            .map(|p| {
                self.public_key
                    .allocate_and_trivially_encrypt_lwe(p.label, ctx)
            })
            .collect::<Vec<LweCiphertext<Vec<u64>>>>();

        let start = Instant::now();

        // Step 1Compute the distances
        let distances: Vec<LweCiphertext<Vec<u64>>> = pool.install(|| {
            model_points
                .par_iter()
                .map(|point| self.squared_distance(query, point, ctx))
                .collect()
        });
        let end_distances = Instant::now();
        println!(
            "Time taken to compute distances: {:?}",
            end_distances - start
        );

        // Step 2: Compute the topk labels
        let k_labels = self.topk_labels(&vec![distances, labels], k, ctx);
        let end_topk = Instant::now();
        println!(
            "Time taken to compute topk labels: {:?}",
            end_topk - end_distances
        );

        k_labels
    }
}
