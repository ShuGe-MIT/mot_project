# MOT Solver Training Script

This README provides detailed instructions on how to use the `MOT_models.py` script to train models using the Monge-Ampere Optimal Transport solver. This script is designed to adjust transport plans based on user-defined cost functions and convergence criteria.

## Prerequisites

To run this script, you will need:
- Python 3.x

## Installation

Ensure that you have Python installed on your machine. You can download Python from [python.org](https://www.python.org/downloads/).
To install the required Python libraries, run the following command:

```bash
pip install argparse numpy scipy matplotlib torch torchvision torchaudio
```

## Data Generation
Use `data_generation.ipynb` to generate the MOT input from raw data stored in `/data`.

## MOT Training
The script `MOT_models.py` can be executed from the command line with several options that allow you to customize the training process. Below is a list of the available command-line arguments:

* `--target_epsilon`: Target epsilon for the convergence of the solver (default: 1e-3).
* `--start_epsilon`: Initial epsilon value (default: 1).
* `--epsilon_scale_num`: Factor to scale epsilon after each epsilon_scale_gap iterations (default: 0.99).
* `--epsilon_scale_gap`: Number of iterations between epsilon scaling (default: 100).
* `--cost_type`: Type of cost function used in transport plan (default: square, options: cov, square, normed_square, etc.).
* `--verbose`: Verbosity level of the output (default: 2).
* `--cost_scale`: Scaling factor for the cost (default: 1).
* `--max_iter`: Maximum number of iterations to run (default: 5000).
* `--iter_gap`: Iteration gap to check for convergence (default: 100).
* `--solver`: Type of MOT solver used (default: sinkhorn, options: sinkhorn, rrsinkhorn, greenkhorn, aam, etc.).
* `--data_file`: Path to the data file used for training (default: weight_loss).

During training, the results are logged inside the `/log` folder.

## Visualization
Use `view_results.ipynb` to view any results from `/log`.

## Example Command
To run the training script with custom settings, use the following command:

```bash
python MOT_models.py --start_epsilon=1 --target_epsilon=1e-3 --iter_gap=1 --epsilon_scale_num=0.99 --epsilon_scale_gap=5 --data_file=weight_loss --max_iter=5000 --cost_type=squared
```
This command sets the training parameters according to the user's specific needs, such as cost type and epsilon values.
