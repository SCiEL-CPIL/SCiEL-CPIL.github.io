import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Generate candidates with squared norm 4
    candidates = []

    # Type 1: One 2, rest 0: ±2e_i
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)

    # Type 2: Four 1s, rest 0: ±e_i ± e_j ± e_k ± e_l
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)

    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)

    # Compute dot product matrix M[i,j] = candidates[i] . candidates[j]
    # Constraint: for i != j, M[i,j] <= 2
    M = candidates @ candidates.T

    # Build compatibility matrix: compat[i,j] = True if M[i,j] <= 2
    # Note: M[i,i] = 4, so we must exclude self or handle it.
    # We want i != j.
    compat = (M <= 2)
    np.fill_diagonal(compat, False)
    
    # Helper function to extend a set greedily
    def extend_set(selected_indices, rng):
        selected = set(selected_indices)
        available = np.ones(N, dtype=bool)
        available[list(selected)] = False
        
        while True:
            addable = np.ones(N, dtype=bool)
            for v in selected:
                addable &= compat[v]
            
            for v in selected:
                addable[v] = False
                
            if not np.any(addable):
                break
            
            candidates_to_add = np.where(addable)[0]
            if len(candidates_to_add) == 0:
                break
            idx = candidates_to_add[rng.integers(len(candidates_to_add))]
            selected.add(idx)
            
        return selected

    # Initial Greedy to get a good starting point
    deg = (~compat).sum(axis=1)
    
    best_set = None
    best_count = 0
    rng = np.random.default_rng(42)
    
    # Run a few greedy initializations
    for trial in range(10):
        if time.time() > deadline - 30:
            break
        if trial == 0:
            order = np.argsort(deg, kind='stable')
        else:
            order = np.random.permutation(N)
            
        selected = set()
        available = np.ones(N, dtype=bool)
        for idx in order:
            if available[idx]:
                selected.add(idx)
                available &= compat[idx]
                if not np.any(available):
                    break
        
        extended = extend_set(list(selected), rng)
        if len(extended) > best_count:
            best_count = len(extended)
            best_set = extended.copy()

    if best_set is None:
        return candidates[:22].astype(np.int64)

    current_set = best_set.copy()
    current_size = len(current_set)
    
    while time.time() < deadline - 10:
        extended = extend_set(list(current_set), rng)
        if len(extended) > current_size:
            current_set = extended
            current_size = len(current_set)
            if current_size > best_count:
                best_count = current_size
                best_set = current_set.copy()
        
        if time.time() > deadline - 10:
            break
            
        if len(current_set) > 0:
            elements = list(current_set)
            remove_idx = elements[rng.integers(len(elements))]
            
            new_set = current_set.copy()
            new_set.remove(remove_idx)
            
            new_extended = extend_set(list(new_set), rng)
            
            if len(new_extended) > current_size:
                current_set = new_extended
                current_size = len(current_set)
                if current_size > best_count:
                    best_count = current_size
                    best_set = current_set.copy()

    # Structural Change: Augment with D11 Roots
    # 1. Generate D11 roots (vectors with two non-zero entries ±1)
    d11_roots = []
    for i in range(D):
        for j in range(i + 1, D):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0] * D
                    v[i] = s1
                    v[j] = s2
                    d11_roots.append(v)
    
    d11_roots = np.array(d11_roots, dtype=np.int32)
    N_roots = len(d11_roots)
    
    # 2. Filter roots compatible with current_set (norm 2 vectors)
    # Condition: root . v <= 1 for all v in current_set
    # Note: current_set indices refer to 'candidates' array
    selected_vectors = candidates[list(best_set)]
    
    # Dot products between roots and selected vectors
    # Shape: (N_roots, len(selected_vectors))
    dot_roots_selected = d11_roots @ selected_vectors.T
    
    # Compatible roots have max dot product <= 1
    # Since we need dot <= 1 for ALL selected vectors
    compatible_roots_mask = np.all(dot_roots_selected <= 1, axis=1)
    
    compatible_roots = d11_roots[compatible_roots_mask]
    N_compat = len(compatible_roots)
    
    # 3. Select maximal subset of compatible roots with pairwise dot product <= 0
    # Condition for two roots r1, r2: r1 . r2 <= 0
    if N_compat > 0:
        # Compute dot product matrix for compatible roots
        M_roots = compatible_roots @ compatible_roots.T
        # We need off-diagonal elements <= 0
        # Diagonal is 2
        compat_roots = (M_roots <= 0)
        np.fill_diagonal(compat_roots, False)
        
        # Greedy selection on this smaller graph
        # Use degree heuristic
        deg_roots = (~compat_roots).sum(axis=1)
        order_roots = np.argsort(deg_roots, kind='stable')
        
        selected_roots = set()
        available_roots = np.ones(N_compat, dtype=bool)
        
        for idx in order_roots:
            if available_roots[idx]:
                selected_roots.add(idx)
                available_roots &= compat_roots[idx]
                if not np.any(available_roots):
                    break
        
        # Add selected roots to the result
        if selected_roots:
            selected_root_vectors = compatible_roots[list(selected_roots)]
            final_vectors = np.vstack([selected_vectors, selected_root_vectors])
            return final_vectors.astype(np.int64)
        else:
            return selected_vectors.astype(np.int64)
    else:
        return selected_vectors.astype(np.int64)
