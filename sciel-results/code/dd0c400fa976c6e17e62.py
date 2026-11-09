"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
import time

def _exhaustive_maxcut(n, edges):
    """Compute the exact maximum cut value for small graphs via brute force."""
    if n == 0:
        return 0.0
    best = 0.0
    # For n up to ~12, 2^n is feasible. For larger, we might need heuristics,
    # but the task specifies n=6,8,10 for evaluation and similar for dev.
    # 2^10 = 1024, 2^8=256, 2^6=64. Very fast.
    for mask in range(1 << n):
        cut_val = 0.0
        for u, v in edges:
            # Check if u and v are in different partitions
            # Bit u is set if node u is in partition 1
            bit_u = (mask >> u) & 1
            bit_v = (mask >> v) & 1
            if bit_u != bit_v:
                cut_val += 1.0
        if cut_val > best:
            best = cut_val
    return best

def _qaoa_expectation(n, edges, gammas, betas):
    """
    Compute the expected cut size for a given set of QAOA angles.
    Uses the standard formula for expectation value of MaxCut Hamiltonian
    under the QAOA state.
    
    For a 2-layer QAOA (p=2):
    State |psi> = e^{-i beta_2 B} e^{-i gamma_2 C} e^{-i beta_1 B} e^{-i gamma_1 C} |+>^n
    
    The expectation of the cut operator C = sum_{(u,v) in E} (1 - Z_u Z_v)/2
    can be computed by summing over edges the probability that u and v are in
    different partitions.
    
    P(u != v) = 0.5 * (1 - <Z_u Z_v>)
    
    So Expected Cut = 0.5 * |E| - 0.5 * sum_{(u,v) in E} <Z_u Z_v>
    
    We need to compute <Z_u Z_v> for each edge.
    This requires propagating the state or using the correlation tensor.
    For small n (<=10), we can represent the state as a vector of size 2^n.
    """
    dim = 1 << n
    # Initialize state |+>^n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute Z operator action? Or just apply unitaries directly.
    # Applying exp(-i gamma C) and exp(-i beta B) directly on the vector is O(n * 2^n) per layer.
    # Total O(p * n * 2^n). For n=10, 2^10=1024, n=10, p=2 -> ~20k ops. Very fast.
    
    # Operator B = sum_i X_i
    # exp(-i beta B) = product_i exp(-i beta X_i)
    # exp(-i gamma C) = product_{(u,v) in E} exp(-i gamma (1 - Z_u Z_v)/2) = product_{(u,v) in E} exp(-i gamma/2) exp(i gamma/2 Z_u Z_v)
    # The global phase exp(-i gamma/2) per edge doesn't matter for probabilities, but let's be precise or ignore it.
    # Actually, C = sum_{(u,v)} (I - Z_u Z_v)/2.
    # exp(-i gamma C) = prod_{(u,v)} exp(-i gamma (I - Z_u Z_v)/2)
    # = prod_{(u,v)} exp(-i gamma/2) exp(i gamma/2 Z_u Z_v)
    # = exp(-i gamma |E|/2) * prod_{(u,v)} exp(i gamma/2 Z_u Z_v)
    # The global phase factor is exp(-i gamma |E|/2). It doesn't affect <Z_u Z_v> relative to norm, but let's keep it for correctness if we were measuring other things. For Z_u Z_v expectation, global phase cancels.
    
    # Let's implement the unitaries step by step.
    
    def apply_x_layer(state, beta):
        """Apply exp(-i beta sum X_i) to state."""
        # exp(-i beta X_i) = cos(beta) I - i sin(beta) X_i
        # Since X_i acts on bit i, we can decompose the state.
        # It's easier to iterate over bits.
        # However, for n=10, we can just build the matrix or apply qubits one by one.
        # Let's apply qubit by qubit.
        # State is a vector of length 2^n.
        # Applying X on qubit k:
        # The state can be reshaped to (2, 2^(n-1)) if we consider qubit k as the most significant or least significant?
        # Let's assume standard ordering: bit 0 is least significant (LSB).
        # Reshape state to (2, 2^(n-1)) is tricky if bit 0 is LSB.
        # Let's use a more direct approach: for each basis state |b>, the amplitude moves to |b xor (1<<k)> with factor -i sin(beta) and stays with cos(beta).
        
        new_state = np.zeros_like(state)
        cos_b = np.cos(beta)
        sin_b = -1j * np.sin(beta)
        
        # For each qubit k
        for k in range(n):
            mask_k = 1 << k
            # We need to update new_state based on current state (which is being modified).
            # It's safer to apply all qubits sequentially to a temporary buffer or in-place carefully.
            # Since exp(-i beta sum X) = prod exp(-i beta X_k), we can apply them one by one.
            # Let's do it in-place with a copy for each qubit to avoid interference.
            pass
        
        # Better: apply all at once? No, they don't commute with each other? 
        # Wait, X_i and X_j commute for i!=j. So exp(-i beta sum X) = prod exp(-i beta X_i).
        # So we can apply each single-qubit rotation sequentially.
        
        current_state = state.copy()
        for k in range(n):
            mask_k = 1 << k
            next_state = np.zeros_like(current_state)
            # For each index idx:
            # Amplitude at idx contributes to idx and idx^mask_k
            # Actually, the operator is cos(beta) I - i sin(beta) X_k
            # So new[idx] += cos(beta) * old[idx]
            # new[idx^mask_k] += (-i sin(beta)) * old[idx]
            
            # Vectorized approach:
            # Split indices into those with bit k=0 and bit k=1.
            # This is efficient.
            # Let's just loop over the smaller dimension if possible, or use slicing.
            # For n=10, looping over 1024 elements is fine.
            for idx in range(len(current_state)):
                target_idx = idx ^ mask_k
                next_state[idx] += cos_b * current_state[idx]
                next_state[target_idx] += sin_b * current_state[idx]
            current_state = next_state
        return current_state

    def apply_c_layer(state, gamma):
        """Apply exp(-i gamma C) to state."""
        # C = sum_{(u,v) in E} (I - Z_u Z_v)/2
        # exp(-i gamma C) = prod_{(u,v) in E} exp(-i gamma (I - Z_u Z_v)/2)
        # Note: edges in C are distinct.
        # The operators for different edges (u,v) and (x,y) do NOT commute generally if they share a vertex.
        # So we must apply them sequentially.
        
        current_state = state.copy()
        # We can ignore the global phase exp(-i gamma |E|/2) for now, or include it.
        # Let's include the local part: exp(i gamma/2 Z_u Z_v) * exp(-i gamma/2)
        # The term exp(-i gamma/2) is a scalar multiplication of the whole state.
        # We can accumulate the global phase separately.
        global_phase = 1.0 - 0j
        # Actually, exp(-i gamma (I - Z_u Z_v)/2) = exp(-i gamma/2) * exp(i gamma/2 Z_u Z_v)
        # So for each edge, multiply state by exp(-i gamma/2) and apply exp(i gamma/2 Z_u Z_v).
        
        # Let's just apply the full 2-qubit gate for each edge.
        # Gate: U_edge = exp(-i gamma/2 (I - Z_u Z_v))
        # Matrix in basis |u,v>:
        # |00> -> exp(-i gamma/2 (1 - 1)) |00> = 1 |00> ? No.
        # Z_u Z_v eigenvalues:
        # |00>: Z_u=1, Z_v=1 -> ZZ=1. (1-1)/2 = 0. exp(0)=1.
        # |01>: Z_u=1, Z_v=-1 -> ZZ=-1. (1-(-1))/2 = 1. exp(-i gamma).
        # |10>: Z_u=-1, Z_v=1 -> ZZ=-1. (1-(-1))/2 = 1. exp(-i gamma).
        # |11>: Z_u=-1, Z_v=-1 -> ZZ=1. (1-1)/2 = 0. exp(0)=1.
        
        # So the gate is diagonal:
        # diag(1, exp(-i gamma), exp(-i gamma), 1) in basis |00>, |01>, |10>, |11>
        
        for u, v in edges:
            # Apply gate on qubits u and v.
            # We need to update the state.
            # Loop over all basis states.
            next_state = np.zeros_like(current_state)
            for idx in range(len(current_state)):
                # Extract bits u and v
                bit_u = (idx >> u) & 1
                bit_v = (idx >> v) & 1
                if bit_u == bit_v:
                    # |00> or |11>
                    phase = 1.0
                else:
                    # |01> or |10>
                    phase = np.exp(-1j * gamma)
                
                next_state[idx] = current_state[idx] * phase
            
            current_state = next_state
            
        return current_state

    # Main loop
    for i in range(len(gammas)):
        # Apply C layer
        state = apply_c_layer(state, gammas[i])
        # Apply B layer
        state = apply_x_layer(state, betas[i])
    
    # Now compute expectation of C
    # <C> = sum_{(u,v) in E} P(u != v)
    # P(u != v) = 0.5 * (1 - <Z_u Z_v>)
    # <Z_u Z_v> = sum_{idx} |psi_idx|^2 * z_u(idx) * z_v(idx)
    # where z_k(idx) = 1 if bit k is 0, -1 if bit k is 1.
    
    probs = np.abs(state) ** 2
    # Normalize just in case
    total_prob = np.sum(probs)
    if total_prob > 0:
        probs = probs / total_prob
    
    expected_cut = 0.0
    for u, v in edges:
        corr = 0.0
        for idx in range(len(probs)):
            bit_u = (idx >> u) & 1
            bit_v = (idx >> v) & 1
            z_u = 1 if bit_u == 0 else -1
            z_v = 1 if bit_v == 0 else -1
            corr += probs[idx] * z_u * z_v
        p_diff = 0.5 * (1 - corr)
        expected_cut += p_diff
        
    return expected_cut

def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles.
    """
    # Time limit is 120s.
    # We need to find gammas, betas that maximize the ratio:
    # E[Cut] / OptCut
    # Since OptCut is constant for a given graph, we just maximize E[Cut].
    
    # Get exact optimum
    opt_cut = _exhaustive_maxcut(n, edges)
    if opt_cut == 0:
        return np.zeros((p, 2))
    
    # Define search space
    # Heuristics suggest gamma in [0, pi/2] or [0, 1] and beta in [0, 1] or [0, pi/2].
    # Let's use a grid search followed by local optimization.
    
    # Grid search parameters
    # For n=10, 2^10=1024 states.
    # apply_c_layer: O(|E| * 2^n). |E| ~ 25 for n=10, d=0.5. 25*1024 ~ 25k ops.
    # apply_x_layer: O(n * 2^n). 10*1024 ~ 10k ops.
    # Total per eval ~ 35k ops.
    # If we do 1000 evaluations, that's 35M ops. In Python, this might be slow (seconds).
    # 120s limit is generous, but we should be efficient.
    
    # Let's try a coarse grid first.
    # gamma in [0, 1.5], beta in [0, 1.5]
    # Step size 0.1? 15x15 grid = 225 points.
    # 225 * 35k = 7.8M ops. Should be fast enough.
    
    # Even finer: step 0.05? 30x30 = 900 points. 31M ops. Still likely ok.
    
    # Let's use scipy minimize for refinement.
    from scipy.optimize import minimize
    
    def neg_expected_cut(params):
        gammas = params[:p]
        betas = params[p:2*p]
        try:
            ec = _qaoa_expectation(n, edges, gammas, betas)
            return -ec
        except:
            return 0.0 # Or large positive
    
    # Initial guess from grid search
    best_params = None
    best_val = -1.0
    
    # Grid search
    # Ranges: gamma [0, 1.5], beta [0, 1.5]
    # Let's try [0, 2.0] to be safe.
    g_range = np.linspace(0, 2.0, 20)
    b_range = np.linspace(0, 2.0, 20)
    
    # For p=2, we have 4 parameters. Grid search 20^4 = 160,000 points.
    # 160k * 35k ops = 5.6 billion ops. Too slow in Python.
    
    # Strategy:
    # 1. Optimize gamma1, beta1, gamma2, beta2 sequentially or use a smaller grid.
    # 2. Use random restarts with local optimization.
    
    # Let's do random restarts with Nelder-Mead.
    # Each optimization takes ~100-200 function evals.
    # 200 * 35k = 7M ops per optimization.
    # 50 restarts = 350M ops. Might be tight in 120s if Python overhead is high.
    # Let's reduce grid or use smarter initialization.
    
    # Alternative: Analytical approximation for initial guess?
    # Or just use a few random restarts with a tighter range.
    
    # Let's try 10 random restarts.
    np.random.seed(42) # For reproducibility in dev, but evaluator might vary.
    
    best_params_global = None
    best_val_global = -1.0
    
    # Time check
    start_time = time.time()
    
    for _ in range(10):
        # Random initial guess
        # gamma typically in [0, 1], beta in [0, 1]
        init_params = np.random.uniform(0, 1.5, size=2*p)
        
        # Bounds
        bounds = [(0, 2.0)] * (2*p)
        
        try:
            res = minimize(neg_expected_cut, init_params, method='Nelder-Mead', bounds=bounds, options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
            
            if res.success or res.fun < best_val_global:
                val = -res.fun
                if val > best_val_global:
                    best_val_global = val
                    best_params_global = res.x
        except Exception as e:
            pass
        
        # Check time
        if time.time() - start_time > 60:
            break
            
    if best_params_global is None:
        # Fallback
        return np.tile([0.5, 0.5], (p, 1))
        
    gammas = best_params_global[:p]
    betas = best_params_global[p:2*p]
    
    result = np.zeros((p, 2))
    result[:, 0] = gammas
    result[:, 1] = betas
    
    return result

# EVOLVE-BLOCK-END
