use pyo3::prelude::*;

use kiddo::float::{distance::SquaredEuclidean, kdtree::KdTree};

const BUCKET_SIZE: usize = 32;

/// The constant 768 is used for the embedding dimensionality in the BERT model.
/// For more details, check:  https://arxiv.org/abs/1810.04805
const BERT_EMBEDDING_DIM: usize = 768;

const K_NEAREST_NEIGHBOURS: usize = 10;

type Tree = KdTree<f32, u32, BERT_EMBEDDING_DIM, BUCKET_SIZE, u16>;

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
  tree: Tree,
  data: Vec<Node>,
}

#[pymethods]
impl Svart {
  #[new]
  pub fn new() -> Self {
    Self {
      tree: Tree::new(),
      data: Vec::new(),
    }
  }
  /// Indexes the given `data` by adding it to the internal data structure and KdTree.
  ///
  /// This method takes ownership of the `data` vector and adds each element to the internal data
  /// structure. It also resizes the embeddings to `BERT_EMBEDDING_DIM` and adds them to the KdTree.
  ///
  /// # Arguments
  ///
  /// * `data` - A vector of `Data` elements to be indexed.
  ///
  /// # Examples
  ///
  /// ```
  /// use svart::{Svart, Data};
  ///
  /// let mut svart = Svart::new();
  /// let data = vec![Data {
  ///     text: "Hello, world!".to_string(),
  ///     embedding: vec![1.0, 2.0, 3.0],
  /// }];
  ///
  /// svart.index(data).unwrap();
  /// ```
  pub fn index(&mut self, mut data: Vec<PyRef<Data>>) {
    // Preallocate memory for self.data
    self.data.reserve(data.len());

    for d in data.iter_mut() {
      // Resize embeddings and add to KdTree
      let mut embedding = d.embedding.clone();
      embedding.resize(BERT_EMBEDDING_DIM, 0.0);

      let node = Node {
        text: d.text.clone(),
      };

      let index = self.data.len() as u32;
      self.data.push(node);

      // Convert the resized embedding to a fixed-size array
      let query: &[f32; BERT_EMBEDDING_DIM] =
        match embedding.as_slice().try_into() {
          Ok(arr) => arr,
          Err(_) => {
            eprintln!(
              "Error: Embedding vector has incorrect length after resizing"
            );
            continue;
          }
        };

      self.tree.add(query, index)
    }
  }

  /// Searches the KdTree for the nearest neighbors to the given query vector using squared Euclidean distance.
  ///
  /// This method employs the K-Nearest Neighbors (K-NN) algorithm with squared Euclidean distance as the distance metric.
  /// It first resizes the query vector to match the fixed dimensionality of the BERT model (768).
  /// Then, it searches the KdTree to find the nearest neighbors based on the squared Euclidean distance.
  ///
  /// The function returns a vector of references to `Node` objects, which are the nearest neighbors to the query vector.
  ///
  /// # Arguments
  ///
  /// * `query` - A vector of `f32` values representing the query vector.
  ///
  /// # Returns
  ///
  /// Returns a `Result` containing a vector of references to `Node` objects, which are the nearest neighbors to the query vector.
  ///
  /// # Errors
  ///
  /// - Returns an error if the query vector cannot be resized and converted into a fixed-size array of `BERT_EMBEDDING_DIM` elements.
  /// - Returns an error if a node corresponding to a particular index in the KdTree is not found.
  /// # Examples
  ///
  /// ```
  /// use svart::{Svart, Data};
  ///
  /// let mut svart = Svart::new();
  /// let data = vec![Data {
  ///     text: "Hello, world!".to_string(),
  ///     embedding: vec![1.0; 100],
  /// }];
  ///
  /// svart.index(data).unwrap();
  ///
  /// let query = vec![1.0; 768];
  ///
  /// let results = svart.search(query).unwrap();
  /// assert_eq!(results.len(), 1);
  /// ```
  pub fn search(&mut self, mut query: Vec<f32>) -> Vec<Node> {
    query.resize(BERT_EMBEDDING_DIM, 0.0);

    let query: &[f32; BERT_EMBEDDING_DIM] = &query.try_into().unwrap();

    // Search the KdTree
    let neighbours = self
      .tree
      .nearest_n::<SquaredEuclidean>(query, K_NEAREST_NEIGHBOURS);

    let mut data = Vec::with_capacity(neighbours.len());

    for n in neighbours.iter() {
      let index = n.item as usize;
      let node = self.data.get(index).unwrap();

      data.push(node.clone());
    }

    data
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
        create_data_instance(py, "text1", vec![1.0; 100]),
        create_data_instance(py, "text2", vec![1.0; 1000]),
        create_data_instance(py, "text3", vec![1.0; 768]),
      ];

      let data_pyrefs: Vec<PyRef<Data>> = data_instances
        .iter()
        .map(|data_instance| data_instance.borrow(py))
        .collect();

      svart.index(data_pyrefs);

      assert_eq!(svart.data.len(), 3);
      assert_eq!(svart.tree.size(), 3);
    });
  }

  #[test]
  fn it_returns_all_search_results() {
    prepare_freethreaded_python();
    Python::with_gil(|py| {
      let mut svart = Svart::new();

      let data_instances = vec![
        create_data_instance(py, "text1", vec![1.0; 100]),
        create_data_instance(py, "text2", vec![1.0; 1000]),
        create_data_instance(py, "text3", vec![1.0; 768]),
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
