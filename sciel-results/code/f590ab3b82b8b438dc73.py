import numpy as np
from scipy.optimize import minimize


def _maxcut_optimum(n, edges):
    """Compute exact MaxCut optimum via brute force."""
    best = 0
    for mask in range(1 << n):
        cut = 0
        for (i, j) in edges:
            if (mask >> i) & 1 != (mask >> j) & 1:
                cut += 1
        if cut > best:
            best = cut
    return best


def _apply_x_rotation(psi, n, beta, qubit):
    """Apply exp(-i beta X_q) to state vector."""
    # X_q flips bit q
    # exp(-i beta X) = cos(beta) I - i sin(beta) X
    # So (psi_new)[s] = cos(beta)*psi[s] - i*sin(beta)*psi[s^(1<<qubit)]
    c = np.cos(beta)
    s = np.sin(beta)
    psi_new = c * psi.copy()
    for idx in range(len(psi)):
        idx2 = idx ^ (1 << qubit)
        psi_new[idx] -= 1j * s * psi[idx2]
    return psi_new


def _expected_cut(n, edges, gamma, beta):
    """Compute expected cut for QAOA with given angles on the graph."""
    dim = 1 << n
    
    # Initial state: |+>^n
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    
    # Precompute c_vals: number of cut edges for each state
    edge_list = list(set((min(i, j), max(i, j)) for (i, j) in edges))
    c_vals = np.zeros(dim, dtype=float)
    for s in range(dim):
        cnt = 0
        for (i, j) in edge_list:
            si = (s >> i) & 1
            sj = (s >> j) & 1
            if si != sj:
                cnt += 1
        c_vals[s] = cnt
    
    # Apply layer 1: exp(-i gamma C) then exp(-i beta sum X)
    psi = psi * np.exp(-1j * gamma * c_vals)
    for q in range(n):
        psi = _apply_x_rotation(psi, n, beta, q)
    
    # Apply layer 2: exp(-i gamma C) then exp(-i beta sum X)
    psi = psi * np.exp(-1j * gamma * c_vals)
    for q in range(n):
        psi = _apply_x_rotation(psi, n, beta, q)
    
    # Compute expected cut
    prob = np.abs(psi) ** 2
    expected = np.sum(prob * c_vals)
    
    return float(expected)


def angles(n, edges, p):
    """Design graph-dependent two-layer QAOA angles."""
    opt = _maxcut_optimum(n, edges)
    if opt == 0:
        return np.tile([0.35, 0.25], (p, 1))
    
    def neg_ratio(gb):
        g, b = gb
        ec = _expected_cut(n, edges, g, b)
        return -ec / opt
    
    # Coarse grid
    best_val = -np.inf
    best_gb = (0.35, 0.25)
    grid_g = np.linspace(0.1, np.pi - 0.1, 6)
    grid_b = np.linspace(0.1, np.pi - 0.1, 6)
    
    for g in grid_g:
        for b in grid_b:
            val = -neg_ratio((g, b))
            if val > best_val:
                best_val = val
                best_gb = (g, b)
    
    # Refine with Nelder-Mead
    try:
        result = minimize(neg_ratio, best_gb, method='Nelder-Mead',
                          options={'maxiter': 30, 'xatol': 1e-4, 'fatol': 1e-6})
        if -result.fun > best_val:
            best_gb = (result.x[0], result.x[1])
    except Exception:
        pass
    
    g, b = best_gb
    g = float(np.clip(g, 0.01, np.pi - 0.01))
    b = float(np.clip(b, 0.01, np.pi - 0.01))
    
    return np.array([[g, b], [g, b]], dtype=float)
