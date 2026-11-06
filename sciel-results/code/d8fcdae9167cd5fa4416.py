import numpy as np
from scipy.optimize import minimize
import time

def angles(n, edges, p):
    """
    Compute optimal QAOA angles for a given graph.
    
    Args:
        n: Number of nodes.
        edges: List of edges (u, v).
        p: Number of layers (fixed to 2).
        
    Returns:
        np.array of shape (p, 2) containing (gamma, beta) pairs.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute cost vector
    num_states = 1 << n
    cost_vec = np.zeros(num_states, dtype=np.float64)
    indices = np.arange(num_states)
    
    for u, v in edges:
        bu = (indices >> u) & 1
        bv = (indices >> v) & 1
        mask = bu != bv
        cost_vec[mask] += 1
        
    # Initial state |+>^n
    # state is a complex vector of size num_states
    # We can represent it as a tensor of shape (2, 2, ..., 2) for easier mixer application
    # But for cost application, vector is easier.
    # Let's stick to vector and use a helper for mixer.
    
    # Precompute the R_x matrix for reuse? No, beta changes.
    # But we can precompute the indices for moving axes?
    
    # Helper to apply mixer U_B = exp(-i beta sum X)
    # This is tensor product of R_x(beta)
    def apply_mixer(state_vec, beta):
        # state_vec is 1D complex array of size 2^n
        # Reshape to tensor
        shape = [2] * n
        tensor = state_vec.reshape(shape)
        
        c = np.cos(beta / 2.0)
        s = np.sin(beta / 2.0)
        # R_x = [[c, -i*s], [-i*s, c]]
        
        for q in range(n):
            # Move axis q to front
            tensor = np.moveaxis(tensor, q, 0)
            orig_shape = tensor.shape
            tensor = tensor.reshape(2, -1)
            
            v0 = tensor[0, :]
            v1 = tensor[1, :]
            
            new_v0 = c * v0 - 1j * s * v1
            new_v1 = -1j * s * v0 + c * v1
            
            tensor = np.vstack((new_v0, new_v1))
            tensor = tensor.reshape(orig_shape)
            tensor = np.moveaxis(tensor, 0, q)
            
        return tensor.reshape(num_states)
        
    def evaluate(params):
        """
        Compute expected cut for given parameters [g1, b1, g2, b2].
        """
        g1, b1, g2, b2 = params
        
        # Start with |+>^n
        # Uniform superposition
        state = np.ones(num_states, dtype=np.complex128) / np.sqrt(num_states)
        
        # Layer 1
        # Cost: state * exp(-i g1 * cost_vec)
        state = state * np.exp(-1j * g1 * cost_vec)
        # Mixer
        state = apply_mixer(state, b1)
        
        # Layer 2
        # Cost
        state = state * np.exp(-1j * g2 * cost_vec)
        # Mixer
        state = apply_mixer(state, b2)
        
        # Expected cut
        probs = np.abs(state) ** 2
        return np.dot(probs, cost_vec)

    # Objective function for minimization (negate expected cut)
    def objective(params):
        if time.time() > deadline:
            return 1e10
        return -evaluate(params)

    # Strategy: Grid Search + Local Refinement
    
    # 1. Coarse Grid Search
    # Range [0, pi] for both gamma and beta
    # Use 12 points per dimension
    grid_res = 12
    grid_vals = np.linspace(0, np.pi, grid_res)
    
    best_val = -np.inf
    best_params = np.array([0.35, 0.25, 0.35, 0.25]) # Fallback
    
    # Iterate through grid
    # To save time, we can break early if we exceed deadline
    # Total evaluations: 12^4 = 20736
    # Each evaluation is ~ O(n 2^n) = 10 * 1024 = 10240 ops (roughly)
    # 20736 * 10240 ~ 2e8 ops. Should take < 1 sec in NumPy? Maybe 2-5 secs.
    
    eval_count = 0
    for g1 in grid_vals:
        if time.time() > deadline - 20.0: # Leave 20s for refinement
             break
        for b1 in grid_vals:
            if time.time() > deadline - 20.0:
                 break
            for g2 in grid_vals:
                if time.time() > deadline - 20.0:
                     break
                for b2 in grid_vals:
                    if time.time() > deadline - 20.0:
                         break
                    params = np.array([g1, b1, g2, b2])
                    val = evaluate(params)
                    eval_count += 1
                    if val > best_val:
                        best_val = val
                        best_params = params
                        
    # 2. Local Refinement
    # Use the best point from grid as initial guess for Nelder-Mead
    # Also try a few other good points from the grid?
    # Let's just use the best one, and maybe its neighbors?
    # Nelder-Mead is robust enough.
    
    if time.time() < deadline - 5.0:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            if -res.fun > best_val:
                best_val = -res.fun
                best_params = res.x
        except Exception:
            pass
            
    # Return in required shape (p, 2)
    # best_params is [g1, b1, g2, b2]
    # We want rows (gamma, beta)
    return best_params.reshape(p, 2)
