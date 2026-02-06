# Runner-Chaser project

This project is a distributed system where some "chasers" robot are trying to catch a "runner"
robot in a non deterministic scenario. To learn more about it check out the [report](./report/report.pdf)

## How to setup the environment

The environment is straightforward to setup, it requires *Python 3.14*
(you are free to try older versions, but I haven't tested them)
and have specific library requirements that are written in [requires.txt](./requirements.txt)


### Installation using conda

A simple installation of a virtual environment using conda can be done with:
```
conda create -n drp python=3.14
conda activate drp
pip install -r requirements.txt
python ./src/main.py
```

## CLI Benchmarking Tool Documentation

This document explains how to interface with the simulation runner via the command line. The tool is designed to be flexible, allowing for both large-scale data collection and quick visual debugging.


### 1. General Explanation of the CLI

The CLI (Command Line Interface) provides a bridge between the simulation logic and the user. Instead of modifying variables directly in `main.py`, you pass arguments when executing the script.

The tool supports four types of inputs:

* **Flags:** Toggles like `--display` (defaults to `False`) and `--no-seed` (defaults to `True`).
* **Integers:** Parameters for the environment like `--runners`, `--chasers`, and `--reps`.
* **Strings/Lists:** The `--testcases` argument accepts one or more of the valid controllers: `baseline`, `gaussian`, or `particle_filter`.
* **Help:** You can always run `python main.py --help` to see the full list of available commands and their default values.

---

### 2. How to Run Long Automated Benchmarks

For high-confidence data collection, you should run all controllers over a large number of iterations. In this mode, we disable the display to maximize processing speed and ensure the random seed is fixed for reproducibility.

To run the full suite with **100 repetitions** per test case:

```bash
python main.py --testcases baseline gaussian particle_filter --reps 100

```

**Note:** Since `--display` defaults to `False` and `fix_seed` defaults to `True`, you do not need to specify those flags for this command.

---

### 3. How to Run Manual Benchmarks

Manual benchmarking is useful for "sanity checking" a specific controller's behavior visually. In this mode, we enable the display, run only a single iteration, and disable the fixed seed to see how the controller handles a truly random environment.

To run a single **visual test** of the `particle_filter`:

```bash
python main.py --testcases particle_filter --reps 1 --display --no-seed

```

| Parameter | Value | Description |
| --- | --- | --- |
| `--testcases` | `particle_filter` | Isolates a single controller. |
| `--reps` | `1` | Runs only one iteration. |
| `--display` | Enabled | Opens the GUI window to watch the simulation. |
| `--no-seed` | Disabled | Uses a non-deterministic seed for varied results. |
