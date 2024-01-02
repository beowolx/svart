use ndarray::{s, Array1};
use packed_simd::f32x4;
use pyo3::prelude::*;

use instant_distance::{Builder, HnswMap, Point, Search};

const BUCKET_SIZE: usize = 32;

/// The constant 768 is used for the embedding dimensionality in the BERT model.
/// For more details, check:  https://arxiv.org/abs/1810.04805
const BERT_EMBEDDING_DIM: usize = 768;

#[derive(Clone)]
struct EmbeddingPoint(Array1<f32>);

impl Point for EmbeddingPoint {
  fn distance(&self, other: &Self) -> f32 {
    let dot_product = simd_dot_product(&self.0, &other.0);

    let mag_self = simd_magnitude(&self.0);
    let mag_other = simd_magnitude(&other.0);

    if mag_self == 0.0 || mag_other == 0.0 {
      return f32::MAX;
    }

    1.0 - (dot_product / (mag_self * mag_other)).powi(2)
  }
}

fn simd_dot_product(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
  let mut sum = f32x4::splat(0.0);
  let chunks = a.len() / 4; // f32x4 has 4 lanes

  for i in 0..chunks {
    let a_slice = a.slice(s![4 * i..4 * (i + 1)]);
    let b_slice = b.slice(s![4 * i..4 * (i + 1)]);
    let a_chunk = f32x4::from_slice_unaligned(a_slice.as_slice().unwrap());
    let b_chunk = f32x4::from_slice_unaligned(b_slice.as_slice().unwrap());
    sum += a_chunk * b_chunk;
  }

  let mut dot_product = sum.sum();

  for i in (chunks * 4)..a.len() {
    dot_product += a[i] * b[i];
  }

  dot_product
}

fn simd_magnitude(a: &Array1<f32>) -> f32 {
  let mut sum = f32x4::splat(0.0);
  let chunks = a.len() / 4;

  for i in 0..chunks {
    let a_slice = a.slice(s![4 * i..4 * (i + 1)]);
    let a_chunk = f32x4::from_slice_unaligned(a_slice.as_slice().unwrap());
    sum += a_chunk * a_chunk;
  }

  let mut magnitude = sum.sum();

  for i in (chunks * 4)..a.len() {
    magnitude += a[i] * a[i];
  }

  magnitude.sqrt()
}

#[pyclass]
#[derive(FromPyObject)]
pub struct Data {
  pub text: String,
  pub embedding: Vec<f32>,
}

#[pymethods]
impl Data {
  #[new]
  fn new(text: String, embedding: Vec<f32>) -> Self {
    Data { text, embedding }
  }
}

#[pyclass]
#[derive(Clone)]
pub struct Node {
  #[pyo3(get)]
  text: String,
}

#[pymethods]
impl Node {
  #[new]
  fn new(text: String) -> Self {
    Node { text }
  }

  fn __repr__(&self) -> PyResult<String> {
    Ok(format!("Node(text: {})", self.text))
  }

  fn __str__(&self) -> PyResult<String> {
    Ok(self.text.clone())
  }
}

#[pyclass]
pub struct Svart {
  hnsw_map: HnswMap<EmbeddingPoint, Node>,
}

#[pymethods]
impl Svart {
  #[new]
  pub fn new() -> Self {
    Self {
      hnsw_map: Builder::default().build(Vec::new(), Vec::new()),
    }
  }

  pub fn index(&mut self, data: Vec<PyRef<Data>>) {
    let points: Vec<EmbeddingPoint> = data
      .iter()
      .map(|d| EmbeddingPoint(d.embedding.clone().into()))
      .collect();
    let nodes: Vec<Node> = data
      .into_iter()
      .map(|d| Node {
        text: d.text.clone(),
      })
      .collect();

    self.hnsw_map = Builder::default().build(points, nodes);
  }

  pub fn search(&self, query: Vec<f32>) -> Vec<Node> {
    let query_point = EmbeddingPoint(query.into());
    let mut search = Search::default();
    let results = self.hnsw_map.search(&query_point, &mut search);

    results.map(|item| item.value.clone()).collect()
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::core::embeddings_fixtures::{EMBEDDINGS, QUERY, TEXT};
  use pyo3::{prepare_freethreaded_python, Py, Python};

  fn create_data_instance(
    py: Python,
    text: &str,
    embedding: Vec<f32>,
  ) -> Py<Data> {
    Py::new(
      py,
      Data {
        text: text.to_string(),
        embedding,
      },
    )
    .unwrap()
  }

  #[test]
  fn it_correctly_indexes_the_data() {
    prepare_freethreaded_python();
    Python::with_gil(|py| {
      let mut svart = Svart::new();

      let data_instances = vec![
        create_data_instance(py, "text1", vec![1.0; 768]),
        create_data_instance(py, "text2", vec![2.0; 768]),
        create_data_instance(py, "text3", vec![3.0; 768]),
      ];

      let data_pyrefs: Vec<PyRef<Data>> = data_instances
        .iter()
        .map(|data_instance| data_instance.borrow(py))
        .collect();

      svart.index(data_pyrefs);

      assert_eq!(svart.hnsw_map.values.len(), 3);
    });
  }

  #[test]
  fn it_returns_all_search_results() {
    prepare_freethreaded_python();
    Python::with_gil(|py| {
      let mut svart = Svart::new();

      let data_instances = vec![
        create_data_instance(py, "text1", vec![1.0; 768]),
        create_data_instance(py, "text2", vec![2.0; 768]),
        create_data_instance(py, "text3", vec![3.0; 768]),
      ];

      let data_pyrefs: Vec<PyRef<Data>> = data_instances
        .iter()
        .map(|data_instance| data_instance.borrow(py))
        .collect();

      svart.index(data_pyrefs);

      let query = vec![1.0; 768];
      let results = svart.search(query);

      assert_eq!(results.len(), 3);
    });
  }

  #[test]
  fn it_returns_the_indexed_data() {
    prepare_freethreaded_python();
    Python::with_gil(|py| {
      let mut svart = Svart::new();

      let data_instances: Vec<Py<Data>> = EMBEDDINGS
        .iter()
        .enumerate()
        .map(|(i, x)| {
          create_data_instance(py, TEXT.get(i).unwrap(), x.to_vec())
        })
        .collect();

      let data_pyrefs: Vec<PyRef<Data>> = data_instances
        .iter()
        .map(|data_instance| data_instance.borrow(py))
        .collect();

      svart.index(data_pyrefs);

      let results = svart.search(QUERY.to_vec());

      assert_eq!(results.get(0).unwrap().text, TEXT[2]);
    });
  }
}
