"""QAOA parameter design: grid search + Nelder-Mead refinement."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Global deadline guard
_DEADLINE = None
_TIME_LIMIT = 140.0  # seconds, safety margin under 150s

def _check_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _TIME_LIMIT
    return time.time() > _DEADLINE


def _build_cost_matrix(n, edges):
    """Build the MaxCut cost matrix C where C[i,j] = 1 if (i,j) in edges else 0."""
    C = np.zeros((n, n))
    for (u, v) in edges:
        C[u, v] = 1.0
        C[v, u] = 1.0
    return C


def _expected_cut(gamma_beta, n, C):
    """
    Compute the expected cut value for a given (gamma, beta) pair on a single layer.
    For p=2, we have two layers: (gamma1, beta1), (gamma2, beta2).
    
    The expected cut = sum_{i<j, (i,j) in E} <sigma_z_i sigma_z_j>
    
    We use the standard QAOA expectation calculation.
    """
    gamma1, beta1, gamma2, beta2 = gamma_beta
    return _expected_cut_full(n, C, [(gamma1, beta1), (gamma2, beta2)])


def _expected_cut_full(n, C, layers):
    """
    Compute expected cut for full p-layer QAOA.
    layers: list of (gamma, beta) pairs.
    
    We compute <sigma_z_i sigma_z_j> for each edge using the state evolution.
    For small n (<=10), we can compute the full state vector.
    """
    dim = 1 << n
    # Initial state: |+>^n = (1/sqrt(2^n)) * ones(2^n)
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Apply layers
    for gamma, beta in layers:
        # Apply exp(-i gamma C)
        # C = sum_{(i,j) in E} sigma_z_i sigma_z_j / 2  (for MaxCut, the cost is number of cut edges)
        # Actually for MaxCut, C = sum_{(i,j) in E} (1 - sigma_z_i sigma_z_j)/2
        # But in QAOA literature, typically C = sum_{(i,j) in E} sigma_z_i sigma_z_j
        # Let's use: the cost Hamiltonian for MaxCut is H_C = sum_{(i,j) in E} (I - Z_i Z_j)/2
        # But the standard QAOA uses H_C = sum_{(i,j) in E} Z_i Z_j (up to constant)
        # The expected cut = (|E| - sum_{(i,j)} <Z_i Z_j>) / 2
        
        # For efficiency, we apply the unitary directly.
        # exp(-i gamma * H_C) where H_C = sum_{(i,j) in E} Z_i Z_j / 2 (or similar)
        # Let's use H_C = sum_{(i,j) in E} (I - Z_i Z_j)/2 so that eigenvalues are 0 (not cut) or 1 (cut)
        # Then exp(-i gamma H_C)
        
        # For small n, we can construct the full matrix or apply term by term.
        # Since n <= 10, dim <= 1024, we can construct H_C as a sparse-like structure.
        
        # Actually, let's apply the cost Hamiltonian term by term using the fact that
        # Z_i Z_j is diagonal.
        
        # Apply cost layer: exp(-i gamma * H_C)
        # H_C = sum_{(i,j) in E} (I - Z_i Z_j)/2
        # This is diagonal in the computational basis.
        # For each basis state, the phase is exp(-i gamma * sum_{(i,j) in E} (1 - z_i z_j)/2)
        # where z_i = +/-1 depending on the bit.
        
        # Let's compute the diagonal of exp(-i gamma H_C)
        diag_cost = np.ones(dim, dtype=np.complex128)
        # For each basis state, compute the cost
        # Precompute: for each edge (i,j), the contribution
        edges_list = []
        for i in range(n):
            for j in range(i + 1, n):
                if C[i, j] > 0.5:
                    edges_list.append((i, j))
        
        for idx in range(dim):
            # Get bits for this state
            # bit k of idx gives qubit k (let's say qubit 0 is LSB)
            cost = 0.0
            for (i, j) in edges_list:
                bit_i = (idx >> i) & 1
                bit_j = (idx >> j) & 1
                z_i = 1 - 2 * bit_i
                z_j = 1 - 2 * bit_j
                # (1 - z_i z_j)/2
                cost += (1.0 - z_i * z_j) / 2.0
            diag_cost[idx] = np.exp(-1j * gamma * cost)
        
        state = diag_cost * state
        
        # Apply mixing layer: exp(-i beta * sum_i X_i)
        # sum_i X_i is not diagonal, so we need to apply it.
        # For small n, we can construct the full matrix.
        # X_i flips qubit i.
        # The matrix for exp(-i beta * sum_i X_i)
        
        # Actually, sum_i X_i doesn't have a simple closed form for the exponential.
        # For n up to 10, dim up to 1024, we can use matrix exponential or
        # apply it via the fact that it's a sum of Pauli X.
        
        # Let's use: the operator H_M = sum_i X_i
        # We need to apply exp(-i beta H_M) to the state.
        
        # For efficiency, note that H_M = X_0 + X_1 + ... + X_{n-1}
        # This is a sparse matrix. We can use scipy.sparse or just construct it.
        
        # Construct H_M as a dense matrix for n <= 10 (dim <= 1024)
        H_M = np.zeros((dim, dim), dtype=np.complex128)
        for i in range(n):
            for idx in range(dim):
                flipped = idx ^ (1 << i)
                H_M[flipped, idx] += 1.0
        
        # Now compute exp(-i beta H_M) @ state
        # Use scipy.linalg.expm or diagonalize
        # For dim up to 1024, expm might be slow. Let's use eigh.
        eigvals, eigvecs = np.linalg.eigh(H_M)
        # H_M is real symmetric, so this is fine.
        phase = np.exp(-1j * beta * eigvals)
        # state = eigvecs @ (phase * (eigvecs.T @ state))  -- wait, eigvecs is unitary
        # state_new = eigvecs @ diag(phase) @ eigvecs^H @ state
        # Since eigvecs is unitary, eigvecs^H = eigvecs.T.conj()
        coeff = eigvecs.conj().T @ state
        state = eigvecs @ (phase * coeff)
    
    # Compute expected cut
    # Expected cut = (|E| - sum_{(i,j) in E} <Z_i Z_j>) / 2
    # <Z_i Z_j> = sum_k state[k]^* * Z_i Z_j |k> * state[k]
    # Z_i Z_j |k> = z_i(k) z_j(k) |k> where z_i(k) = +/-1
    # So <Z_i Z_j> = sum_k |state[k]|^2 * z_i(k) z_j(k)
    
    probs = np.abs(state) ** 2
    total_zz = 0.0
    for (i, j) in edges_list:
        zz = 0.0
        for idx in range(dim):
            bit_i = (idx >> i) & 1
            bit_j = (idx >> j) & 1
            z_i = 1 - 2 * bit_i
            z_j = 1 - 2 * bit_j
            zz += probs[idx] * z_i * z_j
        total_zz += zz
    
    num_edges = len(edges_list)
    expected_cut = (num_edges - total_zz) / 2.0
    return expected_cut


def _exact_maxcut(n, edges):
    """Compute exact MaxCut by brute force for small n."""
    best = 0
    for mask in range(1 << n):
        cut = 0
        for (u, v) in edges:
            bit_u = (mask >> u) & 1
            bit_v = (mask >> v) & 1
            if bit_u != bit_v:
                cut += 1
        if cut > best:
            best = cut
    return best


def _objective(params, n, C, exact_opt):
    """Objective to minimize: negative of the ratio."""
    gamma1, beta1, gamma2, beta2 = params
    ec = _expected_cut_full(n, C, [(gamma1, beta1), (gamma2, beta2)])
    ratio = ec / exact_opt if exact_opt > 0 else 0.0
    return -ratio


def angles(n, edges, p):
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT
    
    if p != 2:
        # Fallback for unexpected p
        return np.tile([0.35, 0.25], (p, 1))
    
    C = _build_cost_matrix(n, edges)
    exact_opt = _exact_maxcut(n, edges)
    
    if exact_opt == 0:
        return np.zeros((p, 2))
    
    # Grid search over parameter space
    # gamma in [0, pi], beta in [0, pi] typically
    # Coarse grid
    gamma_vals = np.linspace(0.1, np.pi, 8)
    beta_vals = np.linspace(0.1, np.pi, 8)
    
    best_params = None
    best_ratio = -1
    
    for g1 in gamma_vals:
        for b1 in beta_vals:
            for g2 in gamma_vals:
                for b2 in beta_vals:
                    if _check_deadline():
                        break
                    params = [g1, b1, g2, b2]
                    ec = _expected_cut_full(n, C, [(g1, b1), (g2, b2)])
                    ratio = ec / exact_opt
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_params = params
    
    if best_params is None:
        best_params = [0.5, 0.5, 0.5, 0.5]
    
    # Nelder-Mead refinement
    if not _check_deadline():
        try:
            result = minimize(
                _objective,
                x0=best_params,
                args=(n, C, exact_opt),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if result.fun < 0:  # means ratio > 0
                refined_ratio = -result.fun
                if refined_ratio > best_ratio:
                    best_params = result.x
        except Exception:
            pass
    
    # Return as (p, 2) array
    result_array = np.array([best_params[0:2], best_params[2:4]])
    return result_array
# EVOLVE-BLOCK-END
