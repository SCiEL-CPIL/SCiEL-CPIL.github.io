"""
QAOA angle optimization for MaxCut on small graphs.
Design: multi-restart local search with graph-dependent initialization,
density-aware parameter ranges, and a final high-precision polish.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import itertools

_DEADLINE_SECONDS = 110.0  # safety margin before 120s kill
_start_time = None


def _get_elapsed():
    if _start_time is None:
        return 0.0
    return time.time() - _start_time


def _build_adjacency(n, edges):
    """Build adjacency matrix from edge list."""
    adj = np.zeros((n, n), dtype=np.float64)
    for u, v in edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    return adj


def _expected_cut_value(n, edges, gammas, betas):
    """
    Compute the expected cut value for a given QAOA circuit.
    
    Circuit: |+>^n, then for each layer i:
        exp(-i gamma_i * C) then exp(-i beta_i * sum_j X_j)
    
    Where C = sum_{(u,v) in E} (I - Z_u Z_v) / 2  (MaxCut cost)
    
    We compute <C> = sum_{(u,v) in E} (1 - <Z_u Z_v>) / 2
    
    For two-layer QAOA, we can compute <Z_u Z_v> analytically.
    """
    m = len(edges)
    if m == 0:
        return 0.0
    
    # For each edge (u,v), compute <Z_u Z_v>
    # The state after the full circuit is |psi> = U_2 U_1 |+>^n
    # where U_i = exp(-i beta_i * B) exp(-i gamma_i * C)
    # B = sum_j X_j
    
    # For two layers, <Z_u Z_v> can be computed via:
    # |psi> = U_2 U_1 |+>^n
    # 
    # A key insight: for unweighted MaxCut with the standard QAOA ansatz,
    # <Z_u Z_v> depends only on whether u,v are adjacent and on the parameters.
    # Specifically, by symmetry, all edges have the same <Z_u Z_v> value
    # ONLY if the graph is edge-transitive. In general, they differ.
    #
    # Let's use a more direct approach: simulate the state evolution.
    # For n up to 10, we can represent the state as a vector of size 2^n.
    
    n_qubits = n
    dim = 1 << n_qubits
    
    # Initial state: |+>^n = (1/sqrt(2^n)) * ones(2^n)
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute Z and X operators (as sparse or dense for small n)
    # For n <= 10, dim <= 1024, so dense is fine.
    
    # Z operator: diagonal with +1 for even parity, -1 for odd parity
    # Actually Z = diag(1, -1, 1, -1, ...) in computational basis
    # Z_u = I ⊗ ... ⊗ Z ⊗ ... ⊗ I
    # Z_u Z_v = I ⊗ ... ⊗ Z ⊗ ... ⊗ Z ⊗ ... ⊗ I
    
    # Build Z_u Z_v for each edge
    # Z in 2D: [[1,0],[0,-1]]
    # X in 2D: [[0,1],[1,0]]
    
    # For efficiency, let's build the operators
    Z_single = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    X_single = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    I_single = np.eye(2, dtype=np.complex128)
    
    def _tensor_ops(ops_list):
        """Tensor product of a list of 2x2 matrices."""
        result = ops_list[0]
        for op in ops_list[1:]:
            result = np.kron(result, op)
        return result
    
    # Build Z_u Z_v for each edge
    ZZ_ops = []
    for u, v in edges:
        ops = [I_single] * n_qubits
        ops[u] = Z_single
        ops[v] = Z_single
        ZZ_ops.append(_tensor_ops(ops))
    
    # Build B = sum_j X_j
    B = np.zeros((dim, dim), dtype=np.complex128)
    for j in range(n_qubits):
        ops = [I_single] * n_qubits
        ops[j] = X_single
        B += _tensor_ops(ops)
    
    # Build C = sum_{(u,v) in E} (I - Z_u Z_v) / 2
    C = np.zeros((dim, dim), dtype=np.complex128)
    for ZZ in ZZ_ops:
        C += (np.eye(dim) - ZZ) / 2.0
    
    # Apply layers
    for i in range(len(gammas)):
        # Apply exp(-i gamma_i * C)
        # Diagonalize C or use matrix exponential
        # C is Hermitian, so use eigendecomposition
        # For small dim, this is fine
        eigvals_C, eigvecs_C = np.linalg.eigh(C)
        # exp(-i gamma C) = V diag(exp(-i gamma * eigvals)) V^dagger
        phase_C = np.exp(-1j * gammas[i] * eigvals_C)
        U_C = (eigvecs_C * phase_C) @ eigvecs_C.conj().T
        psi = U_C @ psi
        
        # Apply exp(-i beta_i * B)
        eigvals_B, eigvecs_B = np.linalg.eigh(B)
        phase_B = np.exp(-1j * betas[i] * eigvals_B)
        U_B = (eigvecs_B * phase_B) @ eigvecs_B.conj().T
        psi = U_B @ psi
    
    # Compute <C>
    expected_C = np.vdot(psi, C @ psi).real
    
    # Expected cut = <C> since C = sum (I - ZZ)/2 and each term contributes 1 if cut
    # Actually <C> = sum_{edges} (1 - <Z_u Z_v>)/2 = expected number of cut edges
    return expected_C


def _expected_cut_fast(n, edges, gammas, betas):
    """
    Faster computation using the structure of the QAOA circuit.
    For two-layer QAOA, we can use a recursive/probabilistic approach.
    
    Actually, let's use a different approach: track the probability distribution
    over the computational basis, or use the fact that the state remains
    in a structured form.
    
    For n <= 10, the dense approach above should be fast enough.
    Let's optimize it by precomputing eigendecompositions.
    """
    dim = 1 << n
    if dim > 2048:
        # Fallback for larger n (shouldn't happen for this task)
        return _expected_cut_value(n, edges, gammas, betas)
    
    # Precompute once per (n, edges) - we'll cache these
    key = (n, tuple(sorted(edges)))
    if not hasattr(_expected_cut_fast, '_cache'):
        _expected_cut_fast._cache = {}
    
    if key not in _expected_cut_fast._cache:
        Z_single = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        X_single = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        I_single = np.eye(2, dtype=np.complex128)
        
        def _tensor_ops(ops_list):
            result = ops_list[0]
            for op in ops_list[1:]:
                result = np.kron(result, op)
            return result
        
        # Build B = sum_j X_j
        B = np.zeros((dim, dim), dtype=np.complex128)
        for j in range(n):
            ops = [I_single] * n
            ops[j] = X_single
            B += _tensor_ops(ops)
        
        # Build C
        C = np.zeros((dim, dim), dtype=np.complex128)
        for u, v in edges:
            ops = [I_single] * n
            ops[u] = Z_single
            ops[v] = Z_single
            ZZ = _tensor_ops(ops)
            C += (np.eye(dim) - ZZ) / 2.0
        
        # Precompute eigendecompositions
        eigvals_C, eigvecs_C = np.linalg.eigh(C)
        eigvals_B, eigvecs_B = np.linalg.eigh(B)
        
        _expected_cut_fast._cache[key] = (
            eigvals_C, eigvecs_C, eigvals_B, eigvecs_B, C
        )
    
    eigvals_C, eigvecs_C, eigvals_B, eigvecs_B, C = _expected_cut_fast._cache[key]
    
    # Initial state
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    for i in range(len(gammas)):
        phase_C = np.exp(-1j * gammas[i] * eigvals_C)
        U_C = (eigvecs_C * phase_C) @ eigvecs_C.conj().T
        psi = U_C @ psi
        
        phase_B = np.exp(-1j * betas[i] * eigvals_B)
        U_B = (eigvecs_B * phase_B) @ eigvecs_B.conj().T
        psi = U_B @ psi
    
    expected_C = np.vdot(psi, C @ psi).real
    return expected_C


def _exact_optimum(n, edges):
    """Compute the exact MaxCut optimum by brute force."""
    if n > 20:
        # Fallback: use a heuristic
        return _heuristic_maxcut(n, edges)
    
    best = 0
    m = len(edges)
    for mask in range(1 << n):
        cut = 0
        for u, v in edges:
            if ((mask >> u) & 1) != ((mask >> v) & 1):
                cut += 1
        if cut > best:
            best = cut
    return best


def _heuristic_maxcut(n, edges):
    """Simple greedy MaxCut heuristic."""
    assignment = [0] * n
    # Random restarts
    best = 0
    for _ in range(100):
        np.random.seed()
        assignment = np.random.randint(0, 2, n)
        cut = 0
        for u, v in edges:
            if assignment[u] != assignment[v]:
                cut += 1
        best = max(best, cut)
        # Local search
        improved = True
        while improved:
            improved = False
            for v in range(n):
                flip_gain = 0
                for u, w in [(e[0], e[1]) for e in edges if e[1] == v] + \
                            [(e[1], e[0]) for e in edges if e[0] == v]:
                    if assignment[u] == assignment[v]:
                        flip_gain += 1
                    else:
                        flip_gain -= 1
                if flip_gain > 0:
                    assignment[v] = 1 - assignment[v]
                    improved = True
            cut = 0
            for u, v in edges:
                if assignment[u] != assignment[v]:
                    cut += 1
            best = max(best, cut)
    return best


def _ratio_objective(params, n, edges, opt_val):
    """Negative ratio for minimization."""
    gammas = params[:2]
    betas = params[2:]
    exp_cut = _expected_cut_fast(n, edges, gammas, betas)
    return -exp_cut / opt_val


def _optimize_for_graph(n, edges, p, deadline):
    """
    Optimize QAOA angles for a specific graph.
    Uses multi-restart local search with density-aware ranges.
    """
    opt_val = _exact_optimum(n, edges)
    if opt_val == 0:
        return np.zeros((p, 2))
    
    m = len(edges)
    density = 2.0 * m / (n * (n - 1)) if n > 1 else 0.0
    
    # Density-aware initial ranges
    # For sparse graphs, angles tend to be smaller
    gamma_range = (0.0, 1.5)
    beta_range = (0.0, 1.5)
    
    if density < 0.3:
        gamma_range = (0.0, 1.0)
        beta_range = (0.0, 1.0)
    elif density > 0.6:
        gamma_range = (0.0, 2.0)
        beta_range = (0.0, 2.0)
    
    best_params = None
    best_ratio = -np.inf
    
    # Multiple restarts
    n_restarts = 10
    rng = np.random.RandomState(42)
    
    for r in range(n_restarts):
        if _get_elapsed() > deadline:
            break
        
        # Generate initial guess
        if r == 0:
            # Start from a reasonable default
            gammas = np.array([0.35, 0.25])
            betas = np.array([0.25, 0.15])
        else:
            gammas = rng.uniform(gamma_range[0], gamma_range[1], p)
            betas = rng.uniform(beta_range[0], beta_range[1], p)
        
        params0 = np.concatenate([gammas, betas])
        
        # Local optimization
        try:
            result = minimize(
                _ratio_objective,
                params0,
                args=(n, edges, opt_val),
                method='L-BFGS-B',
                bounds=[(0, 2*np.pi)] * (2 * p),
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.success or result.fun < best_ratio:
                candidate = result.x
                cand_ratio = -result.fun
                if cand_ratio > best_ratio:
                    best_ratio = cand_ratio
                    best_params = candidate.copy()
        except Exception:
            continue
    
    if best_params is None:
        # Fallback
        return np.array([[0.35, 0.25], [0.25, 0.15]])
    
    # High-precision polish
    try:
        result = minimize(
            _ratio_objective,
            best_params,
            args=(n, edges, opt_val),
            method='L-BFGS-B',
            bounds=[(0, 2*np.pi)] * (2 * p),
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        if result.fun < best_ratio:
            best_params = result.x
    except Exception:
        pass
    
    gammas = best_params[:p]
    betas = best_params[p:]
    return np.column_stack([gammas, betas])


def angles(n, edges, p):
    """
    Return QAOA angles for the given graph.
    
    Args:
        n: number of vertices
        edges: list of (u, v) edges
        p: number of QAOA layers
    
    Returns:
        np.array of shape (p, 2) with rows (gamma, beta) in radians.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    deadline = _start_time + _DEADLINE_SECONDS
    
    # Clear cache for this call to avoid memory issues across calls
    if hasattr(_expected_cut_fast, '_cache'):
        _expected_cut_fast._cache.clear()
    
    result = _optimize_for_graph(n, edges, p, deadline)
    
    # Ensure valid output
    if result.shape != (p, 2):
        result = np.zeros((p, 2))
    
    return result
# EVOLVE-BLOCK-END
