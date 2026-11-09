import numpy as np
from scipy.optimize import minimize
import time

# Cache for graph-specific optimizations
_angle_cache = {}

def _get_graph_key(n, edges):
    # Create a hashable key from n and edges
    # Sort edges to make them independent of order
    sorted_edges = sorted(tuple(sorted(e)) for e in edges)
    return (n, tuple(sorted_edges))

def _compute_cut_values(n, edges):
    """
    Compute the cut value for every possible bitstring (0 to 2^n - 1).
    Return an array of size 2^n.
    """
    num_states = 1 << n
    cut_vals = np.zeros(num_states)
    
    # Precompute bit masks for edges? 
    # For each state, for each edge, check if bits differ.
    # This is O(2^n * |E|). For n=10, 2^10 * 45 ~ 45k ops. Very fast.
    
    # Optimization: use bit manipulation
    # For each state s:
    #   count = 0
    #   for (u, v) in edges:
    #       if ((s >> u) & 1) != ((s >> v) & 1):
    #           count += 1
    #   cut_vals[s] = count
    
    # Vectorized approach might be tricky with arbitrary edges, loop is fine.
    for s in range(num_states):
        c = 0
        for u, v in edges:
            if ((s >> u) & 1) != ((s >> v) & 1):
                c += 1
        cut_vals[s] = c
    return cut_vals

def _apply_x_rotation(v, beta, n):
    """
    Apply exp(-i * beta * sum_j X_j) to state vector v of length 2^n.
    This is equivalent to applying the rotation matrix for X to each qubit independently.
    R_X = [[cos(beta), -i sin(beta)], [-i sin(beta), cos(beta)]]
    """
    c = np.cos(beta)
    s = np.sin(beta)
    # The rotation matrix acts on the subspace of qubit j.
    # We can reshape v into a 2D array where the last axis is the qubit index?
    # No, standard layout: index is sum b_k 2^k.
    # If we treat v as a tensor of shape (2, 2, ..., 2), we can use tensordot or simply reshape.
    
    # Efficient way:
    # Reshape v to shape (2^j, 2, 2^(n-1-j)) for qubit j?
    # Actually, since all qubits get the SAME rotation, we can just apply the 2x2 matrix to the "blocks" corresponding to each qubit bit.
    
    # Let's do it sequentially for each qubit. It's O(n * 2^n).
    # For n=10, 10 * 1024 = 10k ops.
    
    v_new = v.copy()
    for j in range(n):
        # We need to apply the 2x2 matrix to the pairs of elements that differ only in bit j.
        # Step size is 2^(j+1). In each block of size 2^(j+1), the first 2^j elements are bit j=0, next 2^j are bit j=1.
        # We can use reshaping to extract these pairs.
        
        # Reshape v_new to (-1, 2, 2^(n-1-j))? No, that's not quite right for standard memory layout.
        # Let's use a simpler indexing trick.
        # indices 0..2^n-1.
        # For a given j, the bit j is 1 for indices where (i >> j) & 1 == 1.
        
        # We can reshape v_new into a 2D array of shape (-1, 2) if we permute dimensions?
        # Easier: use numpy stride tricks or just simple slicing.
        
        # Let's reshape to (2^j, 2, 2^(n-1-j)) ? 
        # If n=3, j=1. Shape (2, 2, 2).
        # v.reshape(2, 2, 2). 
        # Axis 1 is the bit we want to rotate.
        # We can moveaxis to put the qubit axis last, apply 2x2 matmul, move back.
        
        shape = [1] * n
        shape[j] = 2
        # This is getting complex. Let's stick to the explicit block loop for clarity and safety, it's fast enough.
        
        block_size = 1 << (j + 1)
        half_block = 1 << j
        
        # Iterate over blocks
        for start in range(0, (1 << n), block_size):
            # Indices for bit j = 0
            idx0 = np.arange(start, start + half_block)
            # Indices for bit j = 1
            idx1 = np.arange(start + half_block, start + block_size)
            
            # Apply rotation
            # v_new[idx0] = c * v_old[idx0] - i*s * v_old[idx1]
            # v_new[idx1] = -i*s * v_old[idx0] + c * v_old[idx1]
            
            v0 = v[idx0]
            v1 = v[idx1]
            
            v_new[idx0] = c * v0 - 1j * s * v1
            v_new[idx1] = -1j * s * v0 + c * v1
            
    return v_new

def _expected_cut(gamma1, beta1, gamma2, beta2, n, cut_vals):
    """
    Calculate expected cut for given angles.
    """
    num_states = 1 << n
    # Initial state |+>^n
    v = np.ones(num_states, dtype=np.complex128)
    v /= np.sqrt(num_states)
    
    # Layer 1
    # exp(-i gamma1 C)
    v *= np.exp(-1j * gamma1 * cut_vals)
    # exp(-i beta1 X_total)
    v = _apply_x_rotation(v, beta1, n)
    
    # Layer 2
    # exp(-i gamma2 C)
    v *= np.exp(-1j * gamma2 * cut_vals)
    # exp(-i beta2 X_total)
    v = _apply_x_rotation(v, beta2, n)
    
    # Expected cut = sum |v_i|^2 * cut_vals[i]
    probs = np.abs(v)**2
    exp_cut = np.sum(probs * cut_vals)
    return exp_cut

def _max_cut_classical(n, edges):
    """
    Compute exact max cut value.
    """
    num_states = 1 << n
    max_cut = 0
    for s in range(num_states):
        c = 0
        for u, v in edges:
            if ((s >> u) & 1) != ((s >> v) & 1):
                c += 1
        if c > max_cut:
            max_cut = c
    return max_cut

def angles(n, edges, p):
    global _angle_cache
    start_time = time.time()
    deadline = start_time + 100.0  # Leave 20s buffer
    
    key = _get_graph_key(n, edges)
    if key in _angle_cache:
        return _angle_cache[key]
        
    # Precompute cut values
    cut_vals = _compute_cut_values(n, edges)
    opt_cut = np.max(cut_vals)
    
    if opt_cut == 0:
        # Graph has no edges or is trivial, any angles work
        result = np.zeros((p, 2))
        _angle_cache[key] = result
        return result
        
    # Define objective function for optimizer (maximize)
    def objective(theta):
        g1, b1, g2, b2 = theta
        return -_expected_cut(g1, b1, g2, b2, n, cut_vals)
        
    # Initial guesses
    # Common good starts for QAOA
    starts = [
        [0.35, 0.25, 0.35, 0.25], # Seed
        [np.pi/4, np.pi/4, np.pi/4, np.pi/4],
        [0.2, 0.5, 0.8, 0.3],
        [0.5, 0.1, 0.1, 0.5],
        [0.0, 0.0, 0.0, 0.0],
        [np.pi/2, np.pi/4, np.pi/4, np.pi/2]
    ]
    
    best_theta = None
    best_val = -np.inf
    
    # Try a few random starts as well
    np.random.seed(42)
    for _ in range(5):
        starts.append(np.random.uniform(0, np.pi, 4).tolist())
        
    # Run optimization from each start
    for i, x0 in enumerate(starts):
        if time.time() > deadline - 10:
            break
            
        try:
            # Use L-BFGS-B for speed and robustness
            # Bounds: gamma in [0, pi], beta in [0, pi] (symmetries allow this range)
            bounds = [(0, np.pi), (0, np.pi), (0, np.pi), (0, np.pi)]
            
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 50, 'ftol': 1e-8})
            
            if res.success:
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_theta = res.x
        except Exception:
            continue
            
    if best_theta is None:
        # Fallback to seed if optimization failed
        best_theta = np.array([0.35, 0.25, 0.35, 0.25])
    else:
        # Final polish: use Nelder-Mead for higher precision if time allows
        if time.time() < deadline - 20:
            try:
                res2 = minimize(objective, best_theta, method='Nelder-Mead', 
                                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
                if -res2.fun > best_val:
                    best_val = -res2.fun
                    best_theta = res2.x
            except Exception:
                pass
        
    # Reshape to (p, 2)
    # Row 1: gamma1, beta1
    # Row 2: gamma2, beta2
    result = np.array([best_theta[:2], best_theta[2:4]])
    
    _angle_cache[key] = result
    return result
