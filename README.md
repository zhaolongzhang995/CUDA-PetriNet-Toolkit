# CUDA-PetriNet-Toolkit

This project provides a CUDA-accelerated framework for simulating and analyzing Petri nets. By utilizing GPU parallel computing, it significantly improves the performance of reachability graph computation in complex Petri net models.

## Features
- GPU-accelerated Petri net reachability analysis
- Customizable net model definitions
- Parallel state space exploration

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/zhaolongzhang995/CUDA-PetriNet-Toolkit.git
   ```

2. Navigate to the `src` directory:
   ```bash
   cd src
   ```
3. Compile the CUDA code. You can define the `HASH_TABLE_LENGTH` macro variable during compilation:
   ```bash
   nvcc -DHASH_TABLE_LENGTH=0xffffffU main.cu -o run
   ```
> **Note:** CUDA must be installed and configured on your machine.

## Usage
To run the reachability analysis on a Petri net file:
   ```bash
   ./run PntFileName.pnt
   ```
Alternatively, you can modify the `filename` parameter directly in the code if no file is provided.

The `src/pnt` directory contains sample `.pnt` files for testing. If you want to use other Petri net files, you can generate them using the **TINA** tool by exporting nets via the "net to ina (.pnt/.tim)" option from the export menu. Visit [TINA](https://projects.laas.fr/tina/index.php) for more information.

## Dependencies
- CUDA Toolkit 11.6+

## Changelog
- **V1.0.0**: Initial release with Petri net reachability graph calculation.

## Notes
You can uncomment line 537 in the code to print the reachability graph information. However, it is not recommended to do this for nets with more than 100,000 states, as it can consume significant time.