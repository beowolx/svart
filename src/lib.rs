mod embeddings_fixtures;

use anyhow::Context;
use kiddo::float::{distance::squared_euclidean, kdtree::KdTree};

const BUCKET_SIZE: usize = 32;

// The constant 768 is used for the embedding dimensionality in the BERT model.
// https://arxiv.org/abs/1810.04805
const BERT_EMBEDDING_DIM: usize = 768;

const K_NEAREST_NEIGHBOURS: usize = 10;

pub type Tree = KdTree<f32, u32, BERT_EMBEDDING_DIM, BUCKET_SIZE, u16>;

#[derive(Clone, Debug)]
pub struct Data {
    pub text: String,
    pub embedding: Vec<f32>,
}

pub struct Node {
    pub text: String,
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
    pub fn index(&mut self, data: Vec<Data>) -> anyhow::Result<()> {
        for d in data.iter() {
            let node = Node {
                text: d.text.clone(),
            };

            // Use the length as the index
            let index = self.data.len() as u32;
            self.data.push(node);

            // Resize embeddings and add to KdTree
            let mut embeddings = d.embedding.clone();
            embeddings.resize(BERT_EMBEDDING_DIM, 0.0);
            let query: &[f32; BERT_EMBEDDING_DIM] = &embeddings.try_into().map_err(|_| {
                anyhow::anyhow!("Failed to convert embeddings into fixed-size array")
            })?;

            self.tree.add(query, index)
        }
        Ok(())
    }

    pub fn search(&mut self, mut query: Vec<f32>) -> anyhow::Result<Vec<&Node>> {
        query.resize(BERT_EMBEDDING_DIM, 0.0);

        let query: &[f32; BERT_EMBEDDING_DIM] = &query.try_into().map_err(|_| {
            anyhow::anyhow!("Failed to convert query embeddings into fixed-size array")
        })?;

        // Search the KdTree
        let neighbours = self
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
    use crate::embeddings_fixtures::{EMBEDDINGS, QUERY, TEXT};

    use super::*;

    #[test]
    fn it_correctly_indexes_the_data() {
        let mut svart = Svart::new();

        // Create test Data objects
        let data = vec![
            Data {
                text: "text1".to_string(),
                embedding: vec![1.0; 100], // Fewer than 768 elements
            },
            Data {
                text: "text2".to_string(),
                embedding: vec![1.0; 1000], // More than 768 elements
            },
            Data {
                text: "text3".to_string(),
                embedding: vec![1.0; 768], // Exactly 768 elements
            },
        ];

        // Index the Data objects
        svart.index(data).unwrap();

        // Check the size of the HashMap and the KdTree
        assert_eq!(svart.data.len(), 3);
        assert_eq!(svart.tree.size(), 3);
    }

    #[test]
    fn it_returns_all_search_results() {
        let mut svart = Svart::new();

        // Create test Data objects
        let data = vec![
            Data {
                text: "text1".to_string(),
                embedding: vec![1.0; 100], // Fewer than 768 elements
            },
            Data {
                text: "text2".to_string(),
                embedding: vec![1.0; 1000], // More than 768 elements
            },
            Data {
                text: "text3".to_string(),
                embedding: vec![1.0; 768], // Exactly 768 elements
            },
        ];

        // Index the Data objects
        svart.index(data).unwrap();

        // Create a query
        let query = vec![1.0; 768];

        // Search the KdTree
        let results = svart.search(query).unwrap();

        // Check the size of the results
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
