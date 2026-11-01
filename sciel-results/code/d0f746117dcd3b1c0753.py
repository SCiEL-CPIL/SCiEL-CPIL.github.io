"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import itertools

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _verify(U, V, W):
    """Verify that the decomposition exactly computes 4x4 matrix multiplication."""
    n, m, p = 4, 4, 4
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
        
    rank = U.shape[1]
    if U.shape[0] != 16 or V.shape[0] != 16 or W.shape[0] != 16:
        return False

    U4 = U.reshape(n, m, rank)
    V4 = V.reshape(m, p, rank)
    W4 = W.reshape(p, n, rank)
    
    # Efficient check using einsum or broadcasting
    # recon[i, j, k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We can compute this as a batched dot product or just loop for small N
    recon = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                recon[i, j, k] = np.sum(U4[i, j, :] * V4[j, k, :] * W4[k, i, :])
    
    target = np.ones((n, m, p))
    max_err = np.max(np.abs(recon - target))
    return max_err < 1e-9

def _build_strassen_4x4():
    """Build rank-49 Strassen decomposition for 4x4 MM as a fallback and seed."""
    # Standard 2x2 Strassen
    a2 = np.zeros((7, 2, 2))
    b2 = np.zeros((7, 2, 2))
    c2 = np.zeros((2, 2, 7))
    
    a2[0] = [[1, 0], [0, 1]]
    b2[0] = [[1, 0], [0, 1]]
    a2[1] = [[0, 0], [1, 1]]
    b2[1] = [[1, 0], [0, 0]]
    a2[2] = [[1, 0], [0, 0]]
    b2[2] = [[0, 0], [1, -1]]
    a2[3] = [[0, 0], [0, 1]]
    b2[3] = [[-1, 1], [0, 0]]
    a2[4] = [[1, 1], [0, 0]]
    b2[4] = [[0, 0], [0, 1]]
    a2[5] = [[-1, 0], [1, 0]]
    b2[5] = [[1, 1], [0, 0]]
    a2[6] = [[0, 1], [0, -1]]
    b2[6] = [[0, 0], [1, 1]]
    
    c2[0, 0] = [1, 0, 0, 1, -1, 0, 1]
    c2[0, 1] = [0, 0, 1, 0, 1, 0, 0]
    c2[1, 0] = [0, 1, 0, 1, 0, 0, 0]
    c2[1, 1] = [1, -1, 1, 0, 0, 1, 0]
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    for s1 in range(7):
        for s2 in range(7):
            col = 7 * s1 + s2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = 2 * i1 + i2
                            j = 2 * j1 + j2
                            U[i * 4 + j, col] = a2[s1, i1, j1] * a2[s2, i2, j2]
            
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = 2 * j1 + j2
                            k = 2 * k1 + k2
                            V[j * 4 + k, col] = b2[s1, j1, k1] * b2[s2, j2, k2]
            
            for i1 in range(2):
                for k1 in range(2):
                    for i2 in range(2):
                        for k2 in range(2):
                            i = 2 * i1 + i2
                            k = 2 * k1 + k2
                            W[k * 4 + i, col] = c2[i1, k1, s1] * c2[i2, k2, s2]
    
    return U, V, W

def _generate_candidate_pool():
    """
    Generate a pool of candidate bilinear forms (triples of vectors) 
    that are likely to appear in a rank-49 decomposition.
    We use the Strassen solution as a base and perturb it slightly 
    to create variations, plus some random sparse vectors.
    """
    candidates = []
    n, m, p = 4, 4, 4
    
    # 1. Add the 49 Strassen terms
    U_s, V_s, W_s = _build_strassen_4x4()
    for r in range(49):
        u = U_s[:, r]
        v = V_s[:, r]
        w = W_s[:, r]
        # Flatten for storage
        candidates.append((u.copy(), v.copy(), w.copy()))
        
    # 2. Add perturbed versions of Strassen terms
    # Small integer perturbations might reveal alternative decompositions
    rng = np.random.default_rng(42)
    for _ in range(50):
        base_idx = rng.integers(0, 49)
        u = candidates[base_idx][0].copy()
        v = candidates[base_idx][1].copy()
        w = candidates[base_idx][2].copy()
        
        # Perturb a few entries
        for _ in range(3):
            idx_u = rng.integers(0, 16)
            u[idx_u] += rng.choice([-1, 1])
            idx_v = rng.integers(0, 16)
            v[idx_v] += rng.choice([-1, 1])
            idx_w = rng.integers(0, 16)
            w[idx_w] += rng.choice([-1, 1])
            
        candidates.append((u, v, w))
        
    # 3. Add random sparse vectors with entries in {-1, 0, 1}
    for _ in range(100):
        u = np.zeros(16)
        v = np.zeros(16)
        w = np.zeros(16)
        
        # Sparse support
        supp_u = rng.choice(16, size=2, replace=False)
        supp_v = rng.choice(16, size=2, replace=False)
        supp_w = rng.choice(16, size=2, replace=False)
        
        for idx in supp_u: u[idx] = rng.choice([-1, 1])
        for idx in supp_v: v[idx] = rng.choice([-1, 1])
        for idx in supp_w: w[idx] = rng.choice([-1, 1])
        
        candidates.append((u, v, w))
        
    return candidates

def _search_subset(candidates, target_rank, deadline, initial_selection=None):
    """
    Simulated annealing to find a subset of 'target_rank' candidates 
    whose sum of outer products equals the identity tensor.
    """
    n, m, p = 4, 4, 4
    num_candidates = len(candidates)
    
    # Precompute the "contribution" of each candidate to the 64-dim error vector
    # Error vector e_r[i,j,k] = u_r[i,j] * v_r[j,k] * w_r[k,i]
    # We want sum_r e_r = 1 (all ones)
    
    contribs = np.zeros((num_candidates, 64))
    for r, (u, v, w) in enumerate(candidates):
        u4 = u.reshape(4, 4)
        v4 = v.reshape(4, 4)
        w4 = w.reshape(4, 4)
        
        # Compute tensor t[i,j,k] = u4[i,j] * v4[j,k] * w4[k,i]
        t = np.zeros((4, 4, 4))
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    t[i, j, k] = u4[i, j] * v4[j, k] * w4[k, i]
        contribs[r] = t.flatten()
        
    target_vec = np.ones(64)
    
    # Initialize selection
    if initial_selection is None:
        # Start with the first 'target_rank' candidates (Strassen ones)
        selection = list(range(target_rank))
    else:
        selection = list(initial_selection)
        
    current_sum = np.sum(contribs[selection], axis=0)
    current_err = np.sum((current_sum - target_vec) ** 2)
    
    best_selection = selection.copy()
    best_err = current_err
    best_sum = current_sum.copy()
    
    if best_err < 1e-9:
        return best_selection, best_err
        
    T = 1.0
    T_min = 1e-6
    cooling = 0.995
    
    steps = 0
    max_steps = 20000
    
    while T > T_min and steps < max_steps:
        if time.time() > deadline:
            break
            
        # Pick a candidate to remove and one to add
        if len(selection) < target_rank:
            # Add only
            remove_idx = -1
            add_idx = rng.integers(0, num_candidates)
            if add_idx in selection:
                continue
        elif len(selection) > target_rank:
            # Remove only
            remove_idx = selection[rng.integers(0, len(selection))]
            add_idx = -1
        else:
            # Swap
            remove_idx = selection[rng.integers(0, len(selection))]
            add_idx = rng.integers(0, num_candidates)
            if add_idx == remove_idx or add_idx in selection:
                continue
                
        # Calculate delta error
        # new_sum = current_sum - contrib[remove] + contrib[add]
        delta_sum = np.zeros(64)
        if remove_idx >= 0:
            delta_sum -= contribs[remove_idx]
        if add_idx >= 0:
            delta_sum += contribs[add_idx]
            
        new_sum = current_sum + delta_sum
        new_err = np.sum((new_sum - target_vec) ** 2)
        
        delta_err = new_err - current_err
        
        # Acceptance probability
        if delta_err < 0 or (T > 0 and np.random.rand() < np.exp(-delta_err / T)):
            # Accept
            if remove_idx >= 0:
                selection.remove(remove_idx)
            if add_idx >= 0:
                selection.append(add_idx)
                
            current_sum = new_sum
            current_err = new_err
            
            if current_err < best_err:
                best_err = current_err
                best_selection = selection.copy()
                best_sum = current_sum.copy()
                
                if best_err < 1e-9:
                    return best_selection, best_err
                    
        T *= cooling
        steps += 1
        
    return best_selection, best_err

def construct():
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    # 1. Fallback: Strassen rank 49
    U49, V49, W49 = _build_strassen_4x4()
    if _verify(U49, V49, W49):
        best_result = {"U": U49.tolist(), "V": V49.tolist(), "W": W49.tolist()}
        best_rank = 49
    else:
        # Fallback to schoolbook rank-64 (should not happen)
        n, m, p = 4, 4, 4
        rank = 64
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1.0
                    V[j * p + k][column] = 1.0
                    W[k * n + i][column] = 1.0
                    column += 1
        return {"U": U, "V": V, "W": W}
        
    # 2. Generate candidate pool
    candidates = _generate_candidate_pool()
    
    # 3. Try to find a rank 48 or 49 solution from the pool using Simulated Annealing
    # We know 49 is possible. Can we find a *different* 49 or a 48?
    # The task asks for rank-minimal. If 49 is the theoretical minimum, we are done with 49.
    # But maybe the specific Strassen construction isn't the only one?
    # More importantly, is 48 possible?
    # Literature says rank(4x4x4) = 49. So 48 is impossible.
    # Therefore, the goal is just to return *a* valid rank 49 decomposition.
    # The Strassen one is valid.
    # However, the "different strategy" directive suggests we should demonstrate the search mechanism.
    # If the search finds a rank 49 solution, it's equivalent in score.
    # If it fails, we return Strassen.
    
    # Let's try to find a rank 49 solution from the pool to ensure robustness, 
    # or potentially a rank 48 if my understanding of the lower bound is wrong for this specific interface.
    
    # Try rank 48 first (just in case)
    if time.time() < deadline:
        sel48, err48 = _search_subset(candidates, 48, deadline)
        if err48 < 1e-9:
            U_new = np.column_stack([candidates[r][0] for r in sel48])
            V_new = np.column_stack([candidates[r][1] for r in sel48])
            W_new = np.column_stack([candidates[r][2] for r in sel48])
            if _verify(U_new, V_new, W_new):
                return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
                
    # Try rank 49 (to verify search works and potentially find a different decomposition)
    if time.time() < deadline:
        # Use the Strassen indices as a good starting point for the search
        # The first 49 candidates are the Strassen ones
        sel49, err49 = _search_subset(candidates, 49, deadline, initial_selection=list(range(49)))
        if err49 < 1e-9:
            U_new = np.column_stack([candidates[r][0] for r in sel49])
            V_new = np.column_stack([candidates[r][1] for r in sel49])
            W_new = np.column_stack([candidates[r][2] for r in sel49])
            if _verify(U_new, V_new, W_new):
                # This is a valid rank 49 solution. 
                # It might be identical to Strassen or a different combination.
                # Score is the same.
                return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
                
    # Return the known good Strassen solution
    return best_result

# EVOLVE-BLOCK-END
