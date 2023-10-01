use anyhow::Context;
use kiddo::float::{distance::squared_euclidean, kdtree::KdTree};

const BUCKET_SIZE: usize = 32;

/// The constant 768 is used for the embedding dimensionality in the BERT model.
/// For more details, check:  https://arxiv.org/abs/1810.04805
const BERT_EMBEDDING_DIM: usize = 768;

const K_NEAREST_NEIGHBOURS: usize = 10;

type Tree = KdTree<f32, u32, BERT_EMBEDDING_DIM, BUCKET_SIZE, u16>;

pub struct Data {
  pub text: String,
  pub embedding: Vec<f32>,
}

pub struct Node {
  text: String,
}

pub struct Svart {
  tree: Tree,
  data: Vec<Node>,
}

impl Svart {
  pub fn new() -> Self {
    Self {
      tree: Tree::new(),
      data: Vec::new(),
    }
  }
}

impl Svart {
  /// Indexes the given `data` by adding it to the internal data structure and KdTree.
  ///
  /// This method takes ownership of the `data` vector and adds each element to the internal data
  /// structure. It also resizes the embeddings to `BERT_EMBEDDING_DIM` and adds them to the KdTree.
  ///
  /// # Arguments
  ///
  /// * `data` - A vector of `Data` elements to be indexed.
  ///
  /// # Errors
  ///
  /// This method returns an `anyhow::Error` if it fails to convert the embeddings into a fixed-size
  /// array.
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
  pub fn index(&mut self, mut data: Vec<Data>) -> anyhow::Result<()> {
    // Preallocate memory for self.data
    self.data.reserve(data.len());

    for d in data.iter_mut() {
      let node = Node {
        // Consume the text field to avoid cloning
        text: std::mem::take(&mut d.text),
      };

      let index = self.data.len() as u32;
      self.data.push(node);

      // Resize embeddings and add to KdTree
      d.embedding.resize(BERT_EMBEDDING_DIM, 0.0);

      let query: &[f32; BERT_EMBEDDING_DIM] =
        d.embedding.as_slice().try_into().map_err(|_| {
          anyhow::anyhow!("Failed to convert embeddings into fixed-size array")
        })?;

      self.tree.add(query, index)
    }
    Ok(())
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
  pub fn search(&mut self, mut query: Vec<f32>) -> anyhow::Result<Vec<&Node>> {
    query.resize(BERT_EMBEDDING_DIM, 0.0);

    let query: &[f32; BERT_EMBEDDING_DIM] =
      &query.try_into().map_err(|_| {
        anyhow::anyhow!(
          "Failed to convert query embeddings into fixed-size array"
        )
      })?;

    // Search the KdTree
    let neighbours =
      self
        .tree
        .nearest_n(query, K_NEAREST_NEIGHBOURS, &squared_euclidean);

    let mut data = Vec::with_capacity(neighbours.len());

    for n in neighbours.iter() {
      let index = n.item as usize;
      let node = self
        .data
        .get(index)
        .context(format!("Node not found at index {}", index))?;

      data.push(node);
    }

    Ok(data)
  }
}

#[cfg(test)]
mod tests {

  use crate::core::embeddings_fixtures::{EMBEDDINGS, QUERY, TEXT};

  use super::*;

  #[test]
  fn it_correctly_indexes_the_data() {
    let mut svart = Svart::new();

    let data = vec![
      Data {
        text: "text1".to_string(),
        embedding: vec![1.0; 100],
      },
      Data {
        text: "text2".to_string(),
        embedding: vec![1.0; 1000],
      },
      Data {
        text: "text3".to_string(),
        embedding: vec![1.0; 768],
      },
    ];

    svart.index(data).unwrap();

    assert_eq!(svart.data.len(), 3);
    assert_eq!(svart.tree.size(), 3);
  }

  #[test]
  fn it_returns_all_search_results() {
    let mut svart = Svart::new();

    let data = vec![
      Data {
        text: "text1".to_string(),
        embedding: vec![1.0; 100],
      },
      Data {
        text: "text2".to_string(),
        embedding: vec![1.0; 1000],
      },
      Data {
        text: "text3".to_string(),
        embedding: vec![1.0; 768],
      },
    ];

    svart.index(data).unwrap();

    let query = vec![1.0; 768];

    let results = svart.search(query).unwrap();

    assert_eq!(results.len(), 3);
  }

  #[test]
  fn it_returns_the_indexed_data() {
    let mut svart = Svart::new();
    let data: Vec<Data> = EMBEDDINGS
      .iter()
      .enumerate()
      .map(|(i, x)| Data {
        text: TEXT.get(i).unwrap().to_string(),
        embedding: x.to_vec(),
      })
      .collect();

    svart.index(data).unwrap();

    // The query here is: "What are your vegan options?"
    let results = svart.search(QUERY.to_vec()).unwrap();

    assert_eq!(results.get(0).unwrap().text, TEXT[2]);
  }
}
