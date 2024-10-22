use crate::model::*;
use revolut::*;
use tfhe::core_crypto::prelude::*;

use crate::Query;
use rayon::prelude::*;
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
    // TODO : optimization by encoding the model points as one GLWE M and one lwe m_prime as well as the query as one GLWE C and one lwe c_second
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

    /* Sort the distances and apply the permutation to the identity to get the index_labels */
    fn sort_distances(&self, distances: &LUT, ctx: &Context) -> Vec<LweCiphertext<Vec<u64>>> {
        let n = ctx.full_message_modulus() as u64;
        // let lut_distances = LUT::from_vec_of_lwe(distances.clone(), public_key, ctx);

        let sorted_lut = self.public_key.blind_counting_sort(&distances, ctx);
        let mut sorted_distances = Vec::new();
        for i in 0..n {
            let lwe = self.public_key.sample_extract(&sorted_lut, i as usize, ctx);
            sorted_distances.push(lwe);
        }
        sorted_distances
    }

    fn find_k_nearest_labels(
        &self,
        sorted_distances: &Vec<LweCiphertext<Vec<u64>>>,
        distances: &LUT,
        encoded_points: &Vec<ModelPointEncoded>,
        k: usize,
        ctx: &Context,
    ) -> Vec<LweCiphertext<Vec<u64>>> {
        let label_lut = LUT::from_vec_trivially(
            &encoded_points.iter().map(|p| p.label).collect::<Vec<u64>>(),
            ctx,
        );

        let k_labels: Vec<LweCiphertext<Vec<u64>>> = (0..k)
            .into_par_iter() // Use parallel iterator
            .map(|i| {
                let index_distance =
                    self.public_key
                        .blind_index(distances, &sorted_distances[i], ctx); // index of the nearest distance
                self.public_key
                    .blind_array_access(&index_distance, &label_lut, ctx) // label of the nearest distance using the index
            })
            .collect();

        k_labels
    }

    pub fn predict(
        &self,
        query: &Query,
        model_points: &Vec<ModelPointEncoded>,
        k: usize,
        ctx: &Context,
    ) -> Vec<LweCiphertext<Vec<u64>>> {
        // Compute the distances
        let distances: Vec<LweCiphertext<Vec<u64>>> = model_points
            .par_iter()
            .map(|point| self.squared_distance(query, point, ctx))
            .collect();

        let lut_distances = LUT::from_vec_of_lwe(distances, &self.public_key, ctx);
        let sorted_distances = self.sort_distances(&lut_distances, ctx);
        self.find_k_nearest_labels(&sorted_distances, &lut_distances, &model_points, k, ctx)
    }
}
