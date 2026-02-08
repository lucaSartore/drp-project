# Runner-Chaser project

This project is a distributed system where some "chasers" robot are trying to catch a "runner"
robot in a non deterministic scenario. To learn more about it check out the [report](./report/report.pdf)

## How to setup the environment

The environment is straightforward to setup, it requires *Python 3.14*
(you will be able to make it work with older python versions, but it may requires some 
fixes especially related to type annotations and co)
and have specific library requirements that are written in [requires.txt](./requirements.txt)


### Installation using conda

A simple installation of a virtual environment using conda can be done with:
```
conda create -n drp python=3.14
conda activate drp
pip install -r requirements.txt
python ./src/main.py
```

## A note on the GUI

The GUI provided with this project is intended to be used for development only,
it will crash some times due to a limitation of matplotlib (that refiuses to work
on threads different from the main one).

The "gui-free" execution for benchmarking has no known issue.

## CLI Benchmarking Tool Documentation

This document explains how to interface with the simulation runner via the command line. The tool is designed to be flexible, allowing for both large-scale data collection and quick visual debugging.


### 1. General Explanation of the CLI

The CLI (Command Line Interface) provides a bridge between the simulation logic and the user. Instead of modifying variables directly in `main.py`, you pass arguments when executing the script.

The tool supports four types of inputs:

* **Flags:** Toggles like `--display` (defaults to `False`), `--no-seed` (defaults to `True`), and `--no-logs` (defaults to saving results).
* **Integers:** Parameters for the environment like `--runners`, `--chasers`, and `--reps`.
* **Strings/Lists:** The `--testcases` argument accepts one or more of the valid controllers: `baseline`, `gaussian`, or `particle_filter`.
* **Help:** You can always run `python src/main.py --help` to see the full list of available commands and their default values.

---

### 2. How to Run Long Automated Benchmarks

For high-confidence data collection, you should run all controllers over a large number of iterations. In this mode, we disable the display to maximize processing speed and ensure the random seed is fixed for reproducibility.

To run the full suite with **100 repetitions** per test case:

```bash
python src/main.py --testcases baseline gaussian particle_filter --reps 100

```

---

### 3. How to Run Manual Benchmarks

Manual benchmarking is useful for "sanity checking" a specific controller's behavior visually. In this mode, we enable the display, run only a single iteration, and disable the fixed seed to see how the controller handles a truly random environment.

To run a single **visual test** of the `particle_filter`:

```bash
python src/main.py --testcases particle_filter --reps 1 --display --no-seed --no-logs

```
