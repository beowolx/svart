#![allow(dead_code, clippy::new_without_default)]
mod core;

use pyo3::prelude::*;

pub use crate::core::{Data, Node, Svart};

#[pymodule]
fn svart(_py: Python, m: &PyModule) -> PyResult<()> {
  m.add_class::<Data>()?;
  m.add_class::<Node>()?;
  m.add_class::<Svart>()?;

  Ok(())
}
