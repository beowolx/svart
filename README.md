# Svart: A Memory-Efficient Vector Store in Rust

## Overview

Svart is an innovative vector store built in Rust, offering unparalleled memory efficiency and speed for indexing and searching high-dimensional vectors. It is particularly tailored for use in Natural Language Processing (NLP) embeddings and similar complex data scenarios. With native Python bindings, Svart seamlessly integrates into Python environments, bridging the gap between Rust's performance and Python's ease of use.

**Note**: This project is under active development and is not yet considered stable.

## Features

- **Memory Efficiency**: Built with Rust's zero-cost abstractions to minimize memory footprint.
- **Efficient Indexing and Searching**: Utilizes [K-d Trees](https://en.wikipedia.org/wiki/K-d_tree) for efficient spatial searches.
- **Flexible**: Designed to work with [BERT](https://arxiv.org/abs/1810.04805) embeddings.
- **Python Integration**: Native Python bindings for effortless use in Python projects.
- **Focused and Minimalist**: Delivers essential features without unnecessary complexity.

## Technical Details

### Memory Efficiency

One of the primary goals of Svart is to be memory-efficient. Rust's ownership model and zero-cost abstractions enable Svart to manage resources effectively, thereby reducing the memory footprint. This is particularly beneficial for applications that require handling large datasets or running on resource-constrained environments.

### Data Structures

- **KdTree**: The core data structure used for indexing and searching vectors. It is parameterized to work with `f32` types and has a bucket size of 32.
- **Node**: Represents the data being indexed. It contains the text associated with the vector.
- **Data**: A struct that holds the text and its corresponding embedding vector.
- **Svart**: The main struct that holds the KdTree (`tree`) and a vector of Nodes (`data`).

### Constants

- **BUCKET_SIZE**: The bucket size for the KdTree, set to 32.
- **BERT_EMBEDDING_DIM**: The dimensionality of the BERT embeddings, set to 768.
- **K_NEAREST_NEIGHBOURS**: The number of nearest neighbors to search for, set to 10.

### Methods

- **index**: Takes a vector of `Data` and indexes it into the KdTree and the internal data structure.
- **search**: Takes a query vector and returns the K nearest neighbors from the KdTree.

## Usage

### Rust

```rust
use svart::{Svart, Data};

let mut svart = Svart::new();
let data = vec![Data {
    text: "Hello, world!".to_string(),
    embedding: vec![1.0; 100],
}];

svart.index(data).unwrap();

let query = vec![1.0; 768];

let results = svart.search(query).unwrap();
assert_eq!(results.len(), 1);
```

### Python

```python
import svart

svart_store = svart.Svart()
data = [svart.Data(text="Example", embedding=[...])]
svart_store.index(data)
results = svart_store.search([...])
```

## Limitations and Future Work

- The dimensionality is fixed to BERT's 768 dimensions, which may not be suitable for all use-cases.
- The project is under active development, and more features are planned for future releases.

## Contributing

Feel free to open issues and pull requests. Make sure to run all tests before submitting any code.

## License

This project is open-source and available under the MIT License.
