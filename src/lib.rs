use kiddo::float::kdtree::KdTree;
use std::collections::HashMap;
use uuid::Uuid;

const BUCKET_SIZE: usize = 32;

// The constant 768 is used for the embedding dimensionality in the BERT model.
// https://arxiv.org/abs/1810.04805
const BERT_EMBEDDING_DIM: usize = 768;

pub type Tree = KdTree<f32, u32, BERT_EMBEDDING_DIM, BUCKET_SIZE, u16>;

pub struct Data {
    pub id: Uuid,
    pub text: String,
    pub embedding: Vec<f32>,
}

pub struct Node {
    pub id: Uuid,
    pub text: String,
}

pub struct Svart {
    tree: Tree,
    data: HashMap<Uuid, Node>,
}

impl Svart {
    pub fn new() -> Self {
        Self {
            tree: Tree::new(),
            data: HashMap::new(),
        }
    }
}

impl Svart {
    pub fn index(&mut self, data: Vec<Data>) -> () {
        for d in data.iter() {
            let node = Node {
                id: d.id,
                text: d.text.clone(),
            };

            self.data.insert(d.id, node);

            // Resize embeddings and add to KdTree
            let mut embeddings = d.embedding.clone();
            embeddings.resize(BERT_EMBEDDING_DIM, 0.0);
            let query: &[f32; BERT_EMBEDDING_DIM] =
                &embeddings.try_into().expect("Failed to convert embeddings");
            self.tree.add(query, d.id.as_u128() as u32)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_index() {
        let mut svart = Svart::new();

        // Create test Data objects
        let data = vec![
            Data {
                id: Uuid::new_v4(),
                text: "text1".to_string(),
                embedding: vec![1.0; 100], // Fewer than 768 elements
            },
            Data {
                id: Uuid::new_v4(),
                text: "text2".to_string(),
                embedding: vec![1.0; 1000], // More than 768 elements
            },
            Data {
                id: Uuid::new_v4(),
                text: "text3".to_string(),
                embedding: vec![1.0; 768], // Exactly 768 elements
            },
        ];

        // Index the Data objects
        svart.index(data);

        // Check the size of the HashMap and the KdTree
        assert_eq!(svart.data.len(), 3);
        assert_eq!(svart.tree.size(), 3);
    }
}
