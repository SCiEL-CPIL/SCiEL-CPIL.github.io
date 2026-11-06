"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools

def _exhaustive_maxcut(n, edges):
    """Compute the exact maximum cut value for a small graph."""
    if not edges:
        return 0.0
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if u == v:
            continue
        adj[u].append(v)
        adj[v].append(u)
        edge_set.add((min(u, v), max(u, v)))
    
    best_val = 0.0
    # For n <= 10, 2^10 = 1024 assignments is trivial
    for mask in range(1 << n):
        val = 0
        for u, v in edge_set:
            # Check if u and v are in different partitions
            if ((mask >> u) & 1) != ((mask >> v) & 1):
                val += 1
        if val > best_val:
            best_val = val
    
    return float(best_val)

def _expected_cut_qaoa(n, edges, gammas, betas, p):
    """
    Compute the expected cut value for a p-layer QAOA circuit.
    Uses the exact quantum mechanical expectation value.
    
    For MaxCut with C = sum_{(u,v) in E} (1 - Z_u Z_v)/2,
    the expectation value can be computed efficiently.
    
    We use the fact that for p-layer QAOA, the state is:
    |psi> = U_p ... U_1 |+>^n
    
    where U_k = exp(-i beta_k X^{⊗n}) exp(-i gamma_k C)
    
    The expected cut is (1/2) * sum_{(u,v) in E} (1 - <Z_u Z_v>)
    
    Computing <Z_u Z_v> exactly for general p is expensive, but for p=2 and small n,
    we can use the full state vector approach.
    """
    dim = 1 << n
    # Start in |+>^n
    psi = np.ones(dim) / np.sqrt(dim)
    
    # Precompute Z_u Z_v operators as diagonal matrices (or just their diagonal)
    # For efficiency, we'll track the state and compute expectations at the end
    # But for p=2, we can simulate layer by layer
    
    # Helper: apply exp(-i * theta * A) to state
    # For diagonal A (like C), this is element-wise multiplication
    # For X^{⊗n}, this is a non-diagonal unitary
    
    # Precompute C diagonal: C = sum_{(u,v) in E} (1 - Z_u Z_v)/2
    # The diagonal of Z_u Z_v in computational basis:
    # For a basis state |b>, Z_u Z_v |b> = (-1)^{b_u + b_v} |b>
    # So (1 - Z_u Z_v)/2 has diagonal entry (1 - (-1)^{b_u + b_v})/2
    # which is 1 if b_u != b_v, 0 otherwise
    
    c_diag = np.zeros(dim)
    for u, v in edges:
        if u == v:
            continue
        for state in range(dim):
            bu = (state >> u) & 1
            bv = (state >> v) & 1
            if bu != bv:
                c_diag[state] += 0.5
            else:
                c_diag[state] += 0.0
        # Wait, let me reconsider. C = sum (1 - Z_u Z_v)/2
        # Diagonal of (1 - Z_u Z_v)/2: (1 - (-1)^{bu+bv})/2
        # If bu != bv: (1 - (-1))/2 = 1
        # If bu == bv: (1 - 1)/2 = 0
        # So c_diag[state] = number of edges cut by state / 2? No...
        # Let me recompute properly.
    
    # Recompute c_diag correctly
    c_diag = np.zeros(dim)
    edge_list = [(min(u,v), max(u,v)) for u, v in edges if u != v]
    # Remove duplicates
    edge_list = list(set(edge_list))
    for state in range(dim):
        cut_val = 0
        for u, v in edge_list:
            bu = (state >> u) & 1
            bv = (state >> v) & 1
            if bu != bv:
                cut_val += 1
        c_diag[state] = cut_val / 2.0
    
    # X^{⊗n} matrix: this is the tensor product of X matrices
    # X = [[0, 1], [1, 0]]
    # X^{⊗n} flips all bits
    # So X^{⊗n} |b> = |~b> (bitwise NOT)
    # exp(-i beta X^{⊗n}) = cos(beta) I - i sin(beta) X^{⊗n}
    
    def apply_x_all(psi, beta):
        """Apply exp(-i beta X^{⊗n}) to psi."""
        # X^{⊗n} maps state |b> to |b_flipped> where b_flipped = b XOR (2^n - 1)
        mask = dim - 1
        psi_flipped = np.zeros_like(psi)
        for i in range(dim):
            psi_flipped[i ^ mask] = psi[i]
        return np.cos(beta) * psi - 1j * np.sin(beta) * psi_flipped
    
    def apply_c(psi, gamma):
        """Apply exp(-i gamma C) to psi. C is diagonal."""
        return psi * np.exp(-1j * gamma * c_diag)
    
    # Apply p layers: for each layer, first apply C, then X
    for k in range(p):
        psi = apply_c(psi, gammas[k])
        psi = apply_x_all(psi, betas[k])
    
    # Compute expected cut: (1/2) sum_{(u,v)} (1 - <Z_u Z_v>)
    # = (1/2) * |E| - (1/2) sum_{(u,v)} <Z_u Z_v>
    
    # <Z_u Z_v> = sum_b psi[b]^* (-1)^{b_u + b_v} psi[b]
    # Since psi is complex in general, we need to be careful.
    
    exp_cut = 0.0
    for u, v in edge_list:
        zz = 0.0
        for state in range(dim):
            bu = (state >> u) & 1
            bv = (state >> v) & 1
            sign = (-1) ** (bu + bv)
            zz += np.conj(psi[state]) * sign * psi[state]
        exp_cut += (1 - zz.real) / 2.0
    
    return exp_cut.real

def _objective(params, n, edges, p):
    """Objective function to minimize: negative expected cut."""
    gammas = params[:p]
    betas = params[p:]
    # Keep angles in [0, pi] range for stability
    gammas = np.mod(gammas, np.pi)
    betas = np.mod(betas, np.pi)
    
    val = _expected_cut_qaoa(n, edges, gammas, betas, p)
    return -val

def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles maximizing the mean ratio
    of the independently simulated expected cut to the exact classical optimum.
    """
    # Get the exact optimum
    opt_val = _exhaustive_maxcut(n, edges)
    
    if opt_val == 0:
        return np.zeros((p, 2))
    
    # Optimize angles for this specific graph
    # Initial guess
    x0 = np.concatenate([np.full(p, 0.5), np.full(p, 0.5)])
    
    # Try multiple random starts to avoid local minima
    best_params = x0
    best_obj = _objective(x0, n, edges, p)
    
    # A few random restarts
    for trial in range(5):
        x_try = np.concatenate([
            np.random.uniform(0, np.pi, p),
            np.random.uniform(0, np.pi, p)
        ])
        res = minimize(_objective, x_try, args=(n, edges, p), 
                      method='Nelder-Mead', 
                      options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
        if res.fun < best_obj:
            best_obj = res.fun
            best_params = res.x
    
    gammas = np.mod(best_params[:p], np.pi)
    betas = np.mod(best_params[p:], np.pi)
    
    result = np.column_stack([gammas, betas])
    return result
# EVOLVE-BLOCK-END
