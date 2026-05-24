# Noncommutative Probability – Free Probability for ETF Correlations

Applies free probability theory (Voiculescu) to the empirical spectral distribution of ETF correlation matrices. Computes free cumulants (κ₁…κ₄) and derives a per‑ETF score by comparing eigenvector loadings against the Marcenko‑Pastur benchmark.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63 to 5040 days)
- Free cumulants via moment‑cumulant formula for non‑crossing partitions
- Theoretical Marcenko‑Pastur free cumulants for the null hypothesis
- Score = weighted eigenvector projection of free cumulant deviations
- Best window selected automatically (largest absolute raw signal)
- Results stored in Hugging Face: `P2SAMAPA/p2-etf-free-prob-results`
- Streamlit dashboard with refresh button

## Usage

1. Set `HF_TOKEN` environment variable.
2. Run training: `python train.py`
3. Launch dashboard: `streamlit run streamlit_app.py`
4. GitHub Actions runs daily.

## Interpretation

- **Free cumulants** capture non‑Gaussian, non‑commutative structure in the eigenvalue distribution.
- A high score indicates that an ETF's correlation pattern deviates strongly from the random matrix baseline – potential source of alpha.
- The method generalises classical RMT filtering by allowing exact asymptotic spectral calculation for sums/products of large random matrices.

## Requirements

See `requirements.txt`.
