"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

_DEADLINE = 140.0  # wall-clock safety margin (evaluator kills at 150s)
_START = time.time()


def _check_deadline():
    """Return True if we're running out of time."""
    return (time.time() - _START) > _DEADLINE


def _maxcut_exact(n, edges):
    """Brute-force MaxCut for small n."""
    if n == 0:
        return 0
    # Build adjacency matrix
    adj = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        if u < n and v < n:
            adj[u, v] = 1
            adj[v, u] = 1
    
    best = 0
    # For n <= 20, we can use bitmasks
    for mask in range(1 << n):
        cut = 0
        for i in range(n):
            if (mask >> i) & 1:
                # Count edges from i to j where j < i and j is 0
                for j in range(i):
                    if not ((mask >> j) & 1):
                        cut += adj[i, j]
        if cut > best:
            best = cut
    return best


def _qaoa_expectation(n, edges, gammas, betas):
    """
    Compute the expected cut size for a QAOA circuit with p layers.
    C = sum_{(i,j) in E} (1 - Z_i Z_j) / 2
    U = exp(-i gamma C) exp(-i beta sum X)
    Starting state: |+>^n
    
    For p=2, we can compute this efficiently using the fact that the state
    after each layer is a superposition of basis states. For n up to 10 and p=2,
    we can track the full state vector (2^n amplitudes).
    """
    dim = 1 << n
    
    # Initial state: |+>^n = (1/sqrt(2^n)) * sum_x |x>
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Build C matrix (MaxCut Hamiltonian)
    # C = sum_{(i,j) in E} (1 - Z_i Z_j) / 2
    # We need to apply exp(-i gamma C) for each gamma
    
    # Precompute the action of C on basis states
    # Z_i Z_j |x> = (-1)^{x_i + x_j} |x>
    # (1 - Z_i Z_j)/2 |x> = (1 - (-1)^{x_i + x_j})/2 |x>
    # This is 1 if x_i != x_j, 0 otherwise
    
    # For efficiency, precompute the diagonal of C for each basis state
    c_diag = np.zeros(dim, dtype=np.float64)
    for mask in range(dim):
        cut_val = 0
        for u, v in edges:
            if u < n and v < n:
                # Check if u and v are on different sides
                if ((mask >> u) & 1) != ((mask >> v) & 1):
                    cut_val += 1
        c_diag[mask] = cut_val
    
    # Apply p layers
    for layer in range(len(gammas)):
        gamma = gammas[layer]
        beta = betas[layer]
        
        # Apply exp(-i gamma C): diagonal in computational basis
        phase = np.exp(-1j * gamma * c_diag)
        state = state * phase
        
        # Apply exp(-i beta sum_i X_i)
        # sum_i X_i flips all bits
        # exp(-i beta sum_i X_i) = cos(beta) I - i sin(beta) sum_i X_i
        # But wait, this is not quite right. Let's think more carefully.
        # 
        # The operator is exp(-i beta * S) where S = sum_i X_i
        # S^2 = sum_i X_i^2 + sum_{i!=j} X_i X_j = n + sum_{i!=j} X_i X_j
        # This doesn't simplify to a simple form for n > 1.
        #
        # Actually, for the full flip operator F = X_1 X_2 ... X_n, we have F^2 = I
        # But S = sum_i X_i is different.
        #
        # Let's use the fact that we can apply exp(-i beta S) by diagonalizing S.
        # For n=6, dim=64, this is feasible.
        
        # Build S matrix (sum of X_i)
        # X_i flips bit i
        S = np.zeros((dim, dim), dtype=np.complex128)
        for i in range(n):
            for mask in range(dim):
                flipped = mask ^ (1 << i)
                S[flipped, mask] += 1
        
        # Diagonalize S
        eigvals, eigvecs = np.linalg.eigh(S)
        
        # Apply exp(-i beta S) = V diag(exp(-i beta eigvals)) V^H
        phase_vec = np.exp(-1j * beta * eigvals)
        state = eigvecs @ (phase_vec * (eigvecs.conj().T @ state))
    
    # Compute expected cut: <state| C |state>
    # C is diagonal with entries c_diag
    expected_cut = np.real(np.vdot(state, c_diag * state))
    return expected_cut


def _optimize_angles_for_graph(n, edges, p, max_iters=500):
    """Optimize QAOA angles for a given graph using gradient-free optimization."""
    from scipy.optimize import minimize
    
    # Get exact optimum
    opt_cut = _maxcut_exact(n, edges)
    if opt_cut == 0:
        return np.zeros((p, 2))
    
    best_angles = None
    best_ratio = -1
    
    # Try multiple random initializations
    n_starts = 20
    for start in range(n_starts):
        if _check_deadline():
            break
        
        # Random initial angles
        if start == 0:
            init = np.array([[0.5, 0.3], [0.4, 0.2]])[:p]
        else:
            init = np.random.uniform(0, np.pi, size=(p, 2))
        
        def objective(params):
            if _check_deadline():
                return 0.0
            gammas = params[:p]
            betas = params[p:2*p]
            exp_cut = _qaoa_expectation(n, edges, gammas, betas)
            ratio = exp_cut / opt_cut
            return -ratio  # Minimize negative ratio
        
        try:
            result = minimize(objective, init.flatten(), method='Nelder-Mead',
                             options={'maxiter': max_iters, 'xatol': 1e-4, 'fatol': 1e-4})
            if not _check_deadline():
                ratio = -result.fun
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_angles = result.x.reshape(p, 2)
        except Exception:
            pass
    
    if best_angles is not None:
        return best_angles
    
    # Fallback: return reasonable defaults
    return np.tile([0.4, 0.3], (p, 1))


def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles.
    
    Args:
        n: number of qubits
        edges: list of (u, v) tuples
        p: number of QAOA layers (p=2)
    
    Returns:
        numpy array of shape (p, 2), rows are (gamma, beta) in radians
    """
    # Ensure p=2
    if p != 2:
        # For generality, but task specifies p=2
        return np.tile([0.4, 0.3], (p, 1))
    
    # Optimize angles for this specific graph
    opt_angles = _optimize_angles_for_graph(n, edges, p)
    
    return opt_angles
# EVOLVE-BLOCK-END
