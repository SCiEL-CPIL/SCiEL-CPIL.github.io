"""Rewritten QAOA angle design: per-graph optimization via analytic expectation + Nelder-Mead."""
import numpy as np
from scipy.optimize import minimize


def _build_adj(n, edges):
    adj = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    return adj


def _expected_cut(n, edges, gammas, betas):
    """
    Compute the expected cut value for a fixed-depth QAOA circuit on the given graph.
    """
    dim = 1 << n
    # Initial state: uniform superposition
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute cut_size for each basis state
    cut_sizes = np.zeros(dim, dtype=np.float64)
    for x in range(dim):
        cs = 0
        for u, v in edges:
            bu = (x >> u) & 1
            bv = (x >> v) & 1
            if bu != bv:
                cs += 1
        cut_sizes[x] = cs
    
    for k in range(len(gammas)):
        gamma = gammas[k]
        beta = betas[k]
        # Phase from C
        phases = np.exp(-1j * gamma * cut_sizes)
        psi = psi * phases
        # Apply exp(-i beta X_total)
        cb = np.cos(beta)
        sb = np.sin(beta)
        for q in range(n):
            mask = 1 << q
            idx0 = [x for x in range(dim) if (x & mask) == 0]
            idx1 = [x | mask for x in idx0]
            vals0 = psi[idx0].copy()
            vals1 = psi[idx1].copy()
            psi[idx0] = cb * vals0 - 1j * sb * vals1
            psi[idx1] = -1j * sb * vals0 + cb * vals1
    
    # Compute expected cut
    probs = np.abs(psi) ** 2
    expected_cut = 0.0
    for u, v in edges:
        mask_u = 1 << u
        mask_v = 1 << v
        signs = np.ones(dim, dtype=np.float64)
        for x in range(dim):
            bu = (x & mask_u) != 0
            bv = (x & mask_v) != 0
            if bu != bv:
                signs[x] = -1.0
        expected_cut += (1.0 - np.sum(probs * signs)) / 2.0
    return expected_cut


def _exact_maxcut(n, edges):
    """Brute-force maximum cut for small graphs."""
    best = 0
    for x in range(1 << n):
        cs = 0
        for u, v in edges:
            bu = (x >> u) & 1
            bv = (x >> v) & 1
            if bu != bv:
                cs += 1
        if cs > best:
            best = cs
    return best


def angles(n, edges, p):
    # Optimize angles using Nelder-Mead on the negative expected cut
    def objective(params):
        gammas = params[:p]
        betas = params[p:]
        return -_expected_cut(n, edges, gammas, betas)
    
    # Initial guess
    x0 = np.array([0.4, 0.3, 0.2, 0.15])
    
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
    
    gammas = result.x[:p]
    betas = result.x[p:]
    
    return np.column_stack([gammas, betas])
