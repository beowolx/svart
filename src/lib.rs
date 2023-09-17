use kiddo::float::kdtree::KdTree;

type Embeddings = Vec<Vec<f32>>;

pub type Tree = KdTree<f32, u32, 768, 32, u16>;

pub struct Svart {
    storage: Tree,
}

impl Svart {
    pub fn new() -> Self {
        Self {
            storage: Tree::new(),
        }
    }
}

impl Svart {
    pub fn index(&mut self, mut embeddings: Embeddings) {
        for (idx, embedding) in embeddings.iter_mut().enumerate() {
            embedding.resize(768, 0.0);
            let query: &[f32; 768] = embedding[0..768]
                .try_into()
                .expect("slice with incorrect length");
            self.storage.add(query, idx as u32);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index() {
        let mut svart = Svart::new();

        // Index multiple embeddings at once
        let embeddings: Embeddings = vec![
            vec![1.0; 100],  // Fewer than 768 elements
            vec![1.0; 1000], // More than 768 elements
            vec![1.0; 768],  // Exactly 768 elements
        ];

        svart.index(embeddings);

        // Check the size of the tree
        assert_eq!(svart.storage.size(), 3);
    }
}
