import numpy as np
from scipy.linalg import eigh

def sample_covariance(returns):
    """Return sample covariance matrix (n x n) from log returns."""
    return np.cov(returns.T, ddof=1)

def empirical_spectral_distribution(cov):
    """Eigenvalues of covariance matrix."""
    eigvals = eigh(cov, eigvals_only=True)
    return np.sort(eigvals)[::-1]   # descending

def moments_from_eigvals(eigvals, max_moment):
    """Compute moments m_k = (1/n) Σ λ_i^k for k=1..max_moment."""
    n = len(eigvals)
    moments = []
    for k in range(1, max_moment+1):
        m = np.mean(eigvals ** k)
        moments.append(m)
    return moments

def free_cumulants_from_moments(moments):
    """
    Convert moments to free cumulants using the moment‑cumulant formula for
    non‑crossing partitions (free probability). Returns list κ1, κ2, ... κ_K.
    """
    K = len(moments)
    cum = [0.0] * K
    # κ1 = m1
    cum[0] = moments[0]
    if K >= 2:
        # κ2 = m2 - m1^2
        cum[1] = moments[1] - moments[0]**2
    if K >= 3:
        # κ3 = m3 - 3*m1*m2 + 2*m1^3
        cum[2] = moments[2] - 3*moments[0]*moments[1] + 2*moments[0]**3
    if K >= 4:
        # κ4 = m4 - 4*m1*m3 - 3*m2^2 + 12*m1^2*m2 - 6*m1^4
        m1, m2, m3, m4 = moments[0], moments[1], moments[2], moments[3]
        cum[3] = m4 - 4*m1*m3 - 3*m2**2 + 12*m1**2*m2 - 6*m1**4
    return cum

def marcenko_pastur_free_cumulants(q):
    """
    Theoretical free cumulants of Marcenko‑Pastur distribution (variance = 1).
    q = p/n (number of assets / number of observations).
    Returns κ1, κ2, κ3, κ4 as per MP law.
    """
    κ1 = 1.0
    κ2 = q
    κ3 = q
    κ4 = q + 2*q**2
    return [κ1, κ2, κ3, κ4]

def per_etf_scores(cov, eigvals, cumulants, mp_cumulants):
    """
    Compute a score for each ETF using the eigenvectors weighted by
    the deviation of free cumulants from the Marcenko‑Pastur benchmark.
    Score_i = Σ_k (Δκ_k) * (Σ_j |v_ij|^2 * λ_j^{k-1})
    where Δκ_k = κ_k_sample - κ_k_MP.
    """
    Δκ = np.array(cumulants) - np.array(mp_cumulants)
    n = cov.shape[0]
    eigvals, eigvecs = eigh(cov)           # eigvecs[:, i] column i
    # sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    scores = np.zeros(n)
    for i in range(n):
        vec = eigvecs[:, i]
        for k in range(1, len(Δκ)+1):
            score_contrib = Δκ[k-1] * np.sum(vec**2 * (eigvals[i]**(k-1)))
            scores += score_contrib   # accumulate per ETF across eigenvalues
    # The above loop accumulates incorrectly; we need per ETF sum over eigenvalues.
    # Fix: For each ETF j, sum over all eigenvalues i.
    scores = np.zeros(n)
    for j in range(n):
        total = 0.0
        for i in range(n):
            vi = eigvecs[j, i]   # component of ETF j on eigenvector i
            for k in range(1, len(Δκ)+1):
                total += Δκ[k-1] * (vi**2) * (eigvals[i]**(k-1))
        scores[j] = total
    return scores

def compute_free_prob_scores(returns):
    """
    Main function: given return DataFrame (T x n), compute per‑ETF free probability score.
    """
    # Drop rows with NaN
    returns_clean = returns.dropna()
    n = returns_clean.shape[1]
    T = returns_clean.shape[0]
    if T < n:
        raise ValueError("More assets than observations; cannot compute stable covariance.")
    cov = sample_covariance(returns_clean)
    eigvals = empirical_spectral_distribution(cov)
    moments = moments_from_eigvals(eigvals, max_moment=config.NUM_FREE_CUMULANTS)
    free_cum = free_cumulants_from_moments(moments)
    q = n / T
    mp_cum = marcenko_pastur_free_cumulants(q)
    scores = per_etf_scores(cov, eigvals, free_cum, mp_cum)
    tickers = returns_clean.columns
    return {ticker: scores[i] for i, ticker in enumerate(tickers)}
