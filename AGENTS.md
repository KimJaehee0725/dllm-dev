# Repository Guidelines

## Project Structure & Module Organization
This repo focuses on diffusion language model training and evaluation.
- `src/`: core Python packages (denoisers, backbones, datasets, custom Composer/Transformers).
- `configs/`: Hydra configs for training and eval entry points (see `configs/config.yaml` and `configs/eval_config.yaml`).
- `scripts/`: primary training and evaluation entry points.
- `bash_scripts/`: convenience shell scripts to reproduce experiments.
- `docs/` and `notebooks/`: supporting docs and exploratory work.
- `docker/` and `skypilot/`: container and cluster provisioning assets.

## Build, Test, and Development Commands
Environment setup:
```bash
micromamba env create -y -f requirements.yaml --channel-priority flexible
conda activate dllm-dev
source setup_env.sh
```
Common workflows:
```bash
python scripts/composer_scripts/train_discrete_denoiser.py  # training entry point
python scripts/eval/seq2seq_eval.py                         # seq2seq evals
python scripts/eval/lm_eval_harness.py                      # lm-eval-harness
```
Use `bash_scripts/` for full experiment runs and cluster wrappers.

## Coding Style & Naming Conventions
- Python with 4-space indentation.
- Formatting/linting: `black` and `ruff` (line length 88, Python 3.12 target).
- Names: `snake_case` for functions/vars, `PascalCase` for classes, `SCREAMING_SNAKE_CASE` for constants.
- Install hooks: `pre-commit install` to run formatters/linters on commit.

## Testing Guidelines
There is no dedicated `tests/` directory yet. Use `pytest` for new tests and place them under `tests/` with `test_*.py` naming. For functional validation, run the eval scripts in `scripts/eval/` against small configs.

## Commit & Pull Request Guidelines
Recent commits use short, imperative summaries and occasionally include issue/PR numbers (e.g., `Fix encoder attn mask shape`, `Noise level annealing (#88)`).
- Keep commits focused and descriptive.
- PRs should link an issue (per README), describe the change, and include relevant experiment logs or eval results when applicable.

## Worktree Usage
- For any non-small change or feature, create a dedicated `git worktree` instead of working in the main tree.

## Configuration & Secrets
W&B and HuggingFace tokens are expected via environment variables (see `setup_env.sh`).
Never commit credentials; use local setup scripts in `/home/<USER>/setup_discdiff.sh`.
