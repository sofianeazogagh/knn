use crate::model::*;
use revolut::*;
use tfhe::core_crypto::prelude::*;

use crate::Query;
use rayon::prelude::*;
pub struct Server<'a> {
    public_key: &'a PublicKey,
    ctx: &'a Context,
}

impl<'a> Server<'a> {
    pub fn new(public_key: &'a PublicKey, ctx: &'a Context) -> Self {
        Server { public_key, ctx }
    }

    pub fn encode_model_points(&self, model_points: &Vec<ModelPoint>) -> Vec<ModelPointEncoded> {
        encode_model_points(model_points, &self.ctx)
    }

    // Function to calculate the squared distance of a query vector to a model point
    // TODO : optimization by encoding the model points as one GLWE M and one lwe m_prime as well as the query as one GLWE C and one lwe c_second
    fn squared_distance(
        query: &Query,
        model_points: &ModelPointEncoded,
        vector_dimension: usize,
        public_key: &PublicKey,
        ctx: &Context,
    ) -> LweCiphertext<Vec<u64>> {
        let m = model_points.m.clone();
        let m_prime = model_points.m_prime.clone();

        // Step 1: Compute the inner product m(X) * q(X)
        let mut inner_product = public_key.glwe_absorption_polynomial_with_fft(&query.ct, &m);

        // Step 2: Compute -2*inner_product + m_prime
        let dist = public_key.glwe_sum_polynomial(&mut inner_product, &m_prime, ctx);

        // Step 4: Add the second component of the query
        let dist = public_key.glwe_sum(&dist, &query.ct_second);

        // Step 5: Sample exctract at vector_dimension - 1
        let result = public_key.sample_extract_in_glwe(&dist, vector_dimension - 1, ctx);

        result
    }

    /* Sort the distances and apply the permutation to the identity to get the index_labels */
    fn sort_distances(
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

    fn find_k_nearest_labels(
        sorted_distances: &Vec<LweCiphertext<Vec<u64>>>,
        distances: &LUT,
        encoded_points: &Vec<ModelPointEncoded>,
        k: usize,
        public_key: &PublicKey,
        ctx: &Context,
    ) -> Vec<LweCiphertext<Vec<u64>>> {
        let label_lut = LUT::from_vec_trivially(
            &encoded_points
                .iter()
                .map(|p| p._label)
                .collect::<Vec<u64>>(),
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

    pub fn predict(
        &self,
        query: &Query,
        model_points: &Vec<ModelPointEncoded>,
        k: usize,
    ) -> Vec<LweCiphertext<Vec<u64>>> {
        // Compute the distances
        let distances: Vec<LweCiphertext<Vec<u64>>> = model_points
            .par_iter()
            .map(|point| {
                Self::squared_distance(
                    query,
                    point,
                    query.ct.as_ref().len(),
                    &self.public_key,
                    &self.ctx,
                )
            })
            .collect();
        let lut_distances = LUT::from_vec_of_lwe(distances, &self.public_key, &self.ctx);

        let sorted_distances = Self::sort_distances(&lut_distances, &self.public_key, &self.ctx);
        Self::find_k_nearest_labels(
            &sorted_distances,
            &lut_distances,
            &model_points,
            k,
            &self.public_key,
            &self.ctx,
        )
    }
}
