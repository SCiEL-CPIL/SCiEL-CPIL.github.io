import numpy as np
import time
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _compute_cost_vector(n, edges):
    """
    Compute the cost for all 2^n states.
    state k is an integer. Bit j of k is the value of qubit j.
    Cost is number of cut edges.
    """
    num_states = 1 << n
    costs = np.zeros(num_states, dtype=np.float64)
    
    # Pre-compute bit masks for edges?
    # For each edge (u, v), if bit u != bit v, add 1.
    # We can iterate through all states and all edges.
    # n <= 10, edges <= 45. 2^10 * 45 = 45000 ops. Very fast.
    
    for k in range(num_states):
        cut_count = 0
        for u, v in edges:
            bit_u = (k >> u) & 1
            bit_v = (k >> v) & 1
            if bit_u != bit_v:
                cut_count += 1
        costs[k] = cut_count
    return costs

def _apply_mixer(v, beta, n):
    """
    Apply the mixer U = exp(-i beta sum X_j) to state vector v.
    This is equivalent to applying exp(-i beta X_j) to each qubit j.
    exp(-i beta X) = [[cos(beta), -i sin(beta)], [-i sin(beta), cos(beta)]]
    """
    cos_b = np.cos(beta)
    sin_b = np.sin(beta)
    # U = [[cos, -i sin], [-i sin, cos]]
    # We apply this to each qubit.
    
    v_tensor = v.reshape([2] * n)
    
    # We need to apply the gate to each qubit.
    # Qubit j corresponds to axis n-1-j in the tensor (assuming C-order reshape from integer index).
    # Let's verify:
    # Index 0 -> 00...0
    # Index 1 -> 00...1 (LSB is 1)
    # In reshape([2]*n), the last index changes fastest.
    # So axis n-1 is LSB (qubit 0).
    # Axis 0 is MSB (qubit n-1).
    
    for j in range(n):
        axis = n - 1 - j
        # Move axis to front
        v_moved = np.moveaxis(v_tensor, axis, 0)
        
        # Apply matrix
        # v_new[0] = cos * v_moved[0] - i sin * v_moved[1]
        # v_new[1] = -i sin * v_moved[0] + cos * v_moved[1]
        
        v_new = np.empty_like(v_moved)
        v_new[0] = cos_b * v_moved[0] - 1j * sin_b * v_moved[1]
        v_new[1] = -1j * sin_b * v_moved[0] + cos_b * v_moved[1]
        
        # Move axis back
        v_tensor = np.moveaxis(v_new, 0, axis)
        
    return v_tensor.reshape(-1)

def _expected_cut(params, n, edges, costs):
    """
    Compute expected cut for given parameters.
    params: [gamma_1, beta_1, ..., gamma_p, beta_p]
    """
    p = len(params) // 2
    num_states = 1 << n
    v = np.ones(num_states, dtype=np.complex128) / np.sqrt(num_states)
    
    for l in range(p):
        gamma = params[2*l]
        beta = params[2*l+1]
        
        # Apply cost phase: v_k *= exp(-i gamma * cost_k)
        # costs are real
        phase = np.exp(-1j * gamma * costs)
        v = v * phase
        
        # Apply mixer
        v = _apply_mixer(v, beta, n)
        
    # Compute expected cut: sum |v_k|^2 * cost_k
    probs = np.abs(v)**2
    exp_cut = np.sum(probs * costs)
    return exp_cut

def angles(n, edges, p):
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    costs = _compute_cost_vector(n, edges)
    
    # Define objective to minimize (negative expected cut)
    def objective(params):
        return -_expected_cut(params, n, edges, costs)
    
    # Initial guesses
    # Heuristic: gamma ~ pi/4, beta ~ pi/8
    best_params = np.array([np.pi/4, np.pi/8] * p)
    best_val = objective(best_params)
    
    # Try a few random restarts
    rng = np.random.default_rng(42)
    
    # Number of restarts depends on time, but we can try a fixed number and check time
    max_restarts = 20
    
    for i in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Generate random start
        # gammas in [0, pi], betas in [0, pi/2]
        gammas = rng.uniform(0, np.pi, p)
        betas = rng.uniform(0, np.pi/2, p)
        
        # Interleave
        init_params = np.zeros(2*p)
        init_params[0::2] = gammas
        init_params[1::2] = betas
        
        # Also try the heuristic start once
        if i == 0:
             init_params = np.array([np.pi/4, np.pi/8] * p)
        
        try:
            # Use L-BFGS-B
            # Bounds: gamma in [0, pi], beta in [0, pi/2]
            bounds = []
            for _ in range(p):
                bounds.append((0, np.pi))
                bounds.append((0, np.pi/2))
                
            res = minimize(objective, init_params, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 100, 'ftol': 1e-8})
            
            if res.success or res.fun < best_val:
                best_params = res.x
                best_val = res.fun
        except Exception:
            pass

    # Extract final angles
    # Return shape (p, 2)
    result = np.zeros((p, 2))
    for l in range(p):
        result[l, 0] = best_params[2*l]
        result[l, 1] = best_params[2*l+1]
        
    return result

# EVOLVE-BLOCK-END
