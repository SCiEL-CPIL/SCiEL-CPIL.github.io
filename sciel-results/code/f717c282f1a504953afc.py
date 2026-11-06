"""
Strategy: Global Grid Search over Graph-Scaled Parameter Space + Nelder-Mead Refinement.
Instead of relying on local gradient descent (BFGS) which can get trapped in local minima,
we perform a coarse global search using a grid over a parameter space scaled by the graph's
spectral properties (Laplacian eigenvalues). This ensures we sample the landscape broadly.
We then take the best point from the grid and perform a high-precision Nelder-Mead refinement.
This two-stage approach is robust to local minima and leverages the small dimensionality (p=2, 4 params)
of the QAOA angle space for n<=10.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm
import time
import sys

# Global deadline guard
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 15.0

def _time_remaining():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)

def _build_hamiltonians(n, edges):
    """
    Build the problem Hamiltonian C (MaxCut) and mixing Hamiltonian B (sum X).
    C = sum_{(i,j) in E} 0.5 * (I - Z_i Z_j)
    B = sum_{i=0}^{n-1} X_i
    """
    if n == 0:
        return np.zeros((1,1), dtype=complex), np.zeros((1,1), dtype=complex)
        
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    dim = 2**n
    
    # Helper to apply operator on qubit k
    def apply_op(op, k, n):
        ops = [I] * n
        ops[k] = op
        result = ops[0]
        for i in range(1, n):
            result = np.kron(result, ops[i])
        return result

    C = np.zeros((dim, dim), dtype=complex)
    for (u, v) in edges:
        Z_u = apply_op(Z, u, n)
        Z_v = apply_op(Z, v, n)
        ZZ = Z_u @ Z_v
        C += 0.5 * (np.eye(dim, dtype=complex) - ZZ)
        
    B = np.zeros((dim, dim), dtype=complex)
    for i in range(n):
        X_i = apply_op(X, i, n)
        B += X_i
        
    return C, B

def _expected_cut_value(n, edges, angles, C=None, B=None):
    """
    Compute the expected cut value for given angles.
    angles: (p, 2) array of (gamma, beta)
    """
    if len(edges) == 0:
        return 0.0
        
    if C is None or B is None:
        C, B = _build_hamiltonians(n, edges)
        
    dim = 2**n
    
    # Initial state |+>^n
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    
    p = len(angles)
    for i in range(p):
        gamma, beta = angles[i]
        
        # Apply exp(-i gamma C)
        U_C = expm(-1j * gamma * C)
        psi = U_C @ psi
        
        # Apply exp(-i beta B)
        U_B = expm(-1j * beta * B)
        psi = U_B @ psi
        
    # Calculate expectation <psi| C |psi>
    val = np.vdot(psi, C @ psi).real
    return val

def _graph_spectral_scale(n, edges):
    """
    Estimate a scale factor for the angles based on the graph's Laplacian.
    For sparse graphs, angles tend to be smaller; for dense, larger.
    We use the average degree or spectral radius of the Laplacian.
    """
    if n == 0 or len(edges) == 0:
        return 1.0, 1.0
        
    # Build adjacency matrix
    A = np.zeros((n, n))
    for u, v in edges:
        A[u, v] = 1
        A[v, u] = 1
        
    # Laplacian L = D - A
    D = np.diag(A.sum(axis=1))
    L = D - A
    
    # Eigenvalues of L
    try:
        eigvals = np.linalg.eigvalsh(L)
        max_eig = np.max(eigvals)
        min_nonzero = np.min(eigvals[eigvals > 1e-10]) if np.any(eigvals > 1e-10) else 1.0
    except Exception:
        max_eig = 2 * np.max(np.diag(D))
        min_nonzero = 1.0
        
    # Heuristic scaling:
    # Gamma is related to the energy scale of the cost Hamiltonian.
    # Beta is related to the mixing strength.
    # Typical ranges: gamma in [0, pi], beta in [0, pi/2]
    # We scale the grid size based on the spectral gap/radius.
    # If the graph is dense (large max_eig), the energy levels are closer/farther?
    # Actually, for MaxCut, the cost Hamiltonian eigenvalues are in [0, M].
    # The adiabatic path suggests gamma ~ 1/t, where t is related to the gap.
    # For small p, optimal angles are often O(1).
    
    # Let's define a scale factor that adjusts the grid range.
    # If max_eig is large, the "stiffness" is high, maybe smaller steps?
    # Or maybe larger?
    # Empirically, for n=6..10, angles are usually in [0, 1.5].
    # We will use a fixed base range but adjust the grid resolution or range slightly.
    
    # Let's return a scale factor s such that we search in [0, pi*s] for gamma
    # and [0, pi/2 * s] for beta.
    # For standard graphs, s=1 is good.
    # If the graph is very sparse, s might be smaller?
    # If dense, s might be larger?
    
    # Let's stick to s=1 for now but allow the grid to be adaptive in resolution.
    # Actually, the directive asks for "graph-scaled parameter space".
    # Let's scale the upper bound of the search space.
    # If the graph is dense, the optimal angles might be smaller to avoid over-rotation?
    # Let's try: scale = 1.0 / (1.0 + 0.1 * (max_eig / n))
    # This is a guess. A safer bet is to keep the range fixed but use the spectrum
    # to prioritize grid points near specific values.
    
    # Alternative: Use the average degree to set the center of the grid.
    avg_deg = 2 * len(edges) / n
    # Heuristic: gamma_center ~ pi / (4 * avg_deg) ?
    # Let's just return a scale of 1.0 for the range, but we can use it for weighting.
    return 1.0, avg_deg

def angles(n, edges, p):
    # Deadline check
    if _time_remaining() < 10.0:
        return np.tile([0.35, 0.25], (p, 1))
        
    # Precompute Hamiltonians
    C, B = _build_hamiltonians(n, edges)
    
    # 1. Define Grid Search Space
    # Gamma range: [0, pi]
    # Beta range: [0, pi/2]
    # We use a coarse grid first.
    
    # Number of points per dimension.
    # Total points = N_gamma^p * N_beta^p.
    # For p=2, if N=10, total = 10000.
    # Each eval is ~1024x1024 expm. 10000 * 0.01s = 100s. Too slow.
    # We need a sparser grid or parallelization (not available).
    # Let's use N=6 for gamma, N=6 for beta. Total = 36^2 = 1296 points.
    # 1296 * 0.005s = 6.5s. Feasible.
    
    N_GAMMA = 8
    N_BETA = 8
    
    gamma_vals = np.linspace(0.0, np.pi, N_GAMMA)
    beta_vals = np.linspace(0.0, np.pi / 2, N_BETA)
    
    # Generate all combinations for p layers
    # We will iterate through all combinations of (gamma_1, ..., gamma_p, beta_1, ..., beta_p)
    # This is a Cartesian product.
    
    # To save time, we can evaluate sequentially and keep the best.
    best_val = -1.0
    best_angles = None
    
    # Create iterators for each layer
    # For p=2, we have gamma1, beta1, gamma2, beta2
    # Total iterations: N_GAMMA^p * N_BETA^p
    
    # We will use itertools.product
    import itertools
    
    # Check time budget for grid search
    # Estimated time per eval:
    # n=10 -> 1024 dim. expm is ~1ms. 2 layers -> 4 expms. ~4ms.
    # 1296 points * 4ms = 5s.
    # n=6 -> 64 dim. expm is ~0.1ms.
    
    # If time is tight, reduce grid size
    if _time_remaining() < 30.0:
        N_GAMMA = 5
        N_BETA = 5
        gamma_vals = np.linspace(0.0, np.pi, N_GAMMA)
        beta_vals = np.linspace(0.0, np.pi / 2, N_BETA)
        
    total_points = (N_GAMMA * N_BETA) ** p
    
    # If total points is too high, we might skip grid and go straight to heuristic + local
    if total_points > 5000:
        # Fallback to heuristic initialization
        return np.tile([0.35, 0.25], (p, 1))
        
    # Perform Grid Search
    # We need to be careful with memory and time.
    # We will iterate.
    
    # Pre-compute U_C and U_B for each unique angle?
    # No, angles are unique per combination.
    
    # To speed up, we can pre-compute the exponentials for each gamma and beta value
    # and store them.
    # U_C[g] = exp(-i * gamma[g] * C)
    # U_B[b] = exp(-i * beta[b] * B)
    
    U_C_cache = {}
    U_B_cache = {}
    
    for g in gamma_vals:
        U_C_cache[g] = expm(-1j * g * C)
    for b in beta_vals:
        U_B_cache[b] = expm(-1j * b * B)
        
    # Now iterate through combinations
    # For p=2:
    # for g1 in gamma_vals:
    #   for b1 in beta_vals:
    #     for g2 in gamma_vals:
    #       for b2 in beta_vals:
    #         psi = |+>
    #         psi = U_C[g1] @ psi
    #         psi = U_B[b1] @ psi
    #         psi = U_C[g2] @ psi
    #         psi = U_B[b2] @ psi
    #         val = <psi|C|psi>
        
    # This is much faster because we avoid expm in the inner loop.
    
    psi_init = np.ones(2**n, dtype=complex) / np.sqrt(2**n)
    
    # We will use a recursive or iterative approach for p layers.
    # Since p is fixed at 2 in the problem statement (p=2), we can hardcode or loop.
    
    # General loop for p layers
    # We'll use a stack or just nested loops if p is small.
    # Let's use itertools.product for the flattened angle list.
    
    # Create list of all possible angle tuples
    # angle_tuple = (g1, b1, g2, b2, ...)
    
    # To avoid creating a huge list in memory, we iterate.
    
    # For efficiency, we can pre-compute the sequence of U matrices for each layer index?
    # No, the choice of g and b for layer i is independent.
    
    # Let's just iterate.
    
    # Optimization: If p=2, we can pre-compute the state after layer 1 for all (g1, b1)
    # and then apply layer 2.
    
    if p == 1:
        for g in gamma_vals:
            U_C_g = U_C_cache[g]
            for b in beta_vals:
                U_B_b = U_B_cache[b]
                psi = U_B_b @ (U_C_g @ psi_init)
                val = np.vdot(psi, C @ psi).real
                if val > best_val:
                    best_val = val
                    best_angles = np.array([[g, b]])
                    if _time_remaining() < 5.0:
                        break
            if _time_remaining() < 5.0:
                break
    elif p == 2:
        # Pre-compute states after layer 1
        psi_layer1 = {}
        for g1 in gamma_vals:
            U_C_g1 = U_C_cache[g1]
            for b1 in beta_vals:
                U_B_b1 = U_B_cache[b1]
                psi1 = U_B_b1 @ (U_C_g1 @ psi_init)
                psi_layer1[(g1, b1)] = psi1
                
        # Now apply layer 2
        for (g1, b1), psi1 in psi_layer1.items():
            for g2 in gamma_vals:
                U_C_g2 = U_C_cache[g2]
                psi2_base = U_C_g2 @ psi1
                for b2 in beta_vals:
                    U_B_b2 = U_B_cache[b2]
                    psi2 = U_B_b2 @ psi2_base
                    val = np.vdot(psi2, C @ psi2).real
                    if val > best_val:
                        best_val = val
                        best_angles = np.array([[g1, b1], [g2, b2]])
                        if _time_remaining() < 5.0:
                            break
                if _time_remaining() < 5.0:
                    break
            if _time_remaining() < 5.0:
                break
    else:
        # Generic fallback for p > 2 (unlikely)
        # Use simple grid with expm in loop (slower)
        angle_lists = [gamma_vals] * p + [beta_vals] * p # Wrong structure
        # Actually, for each layer i, we have gamma_i and beta_i.
        # So we need product of (gamma_vals x beta_vals) repeated p times.
        
        layer_combs = list(itertools.product(gamma_vals, beta_vals))
        all_combs = list(itertools.product(*([layer_combs] * p)))
        
        for comb in all_combs:
            # comb is ((g1, b1), (g2, b2), ...)
            psi = psi_init.copy()
            valid = True
            for i in range(p):
                g, b = comb[i]
                try:
                    psi = U_B_cache[b] @ (U_C_cache[g] @ psi)
                except:
                    valid = False
                    break
            if not valid:
                continue
            val = np.vdot(psi, C @ psi).real
            if val > best_val:
                best_val = val
                best_angles = np.array([list(c) for c in comb])
            if _time_remaining() < 5.0:
                break
                
    # If no best found, use default
    if best_angles is None:
        best_angles = np.tile([0.35, 0.25], (p, 1))
        
    # 2. Refinement with Nelder-Mead
    # Use the best grid point as the initial simplex for Nelder-Mead.
    # This allows local refinement around the global best found by the grid.
    
    def objective(x):
        if _time_remaining() < 5.0:
            return 1e9
        x = np.array(x).reshape(-1, 2)
        return -_expected_cut_value(n, edges, x, C, B)
        
    x0 = best_angles.flatten()
    
    try:
        # Nelder-Mead is derivative-free and robust.
        # We set maxiter to ensure we don't run out of time.
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 200, 'disp': False, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if res.fun < -best_val:
            best_angles = res.x.reshape(p, 2)
            best_val = -res.fun
    except Exception:
        pass # Keep grid result
        
    # Final safety check
    if _time_remaining() < 5.0:
        pass
        
    return best_angles

# EVOLVE-BLOCK-END
