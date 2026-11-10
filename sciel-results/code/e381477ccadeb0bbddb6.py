import numpy as np
import itertools
import time

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 600  # Safety margin
    
    # Generate candidates
    # Norm squared 1: 1 non-zero
    # Norm squared 2: 2 non-zeros
    # Norm squared 3: 3 non-zeros
    # Norm squared 4: 4 non-zeros
    
    candidates = []
    # Use itertools to generate positions
    for n_nonzero in range(1, 5):
        for pos in itertools.combinations(range(d), n_nonzero):
            # Generate all sign combinations
            for signs in itertools.product([1, -1], repeat=n_nonzero):
                v = np.zeros(d, dtype=np.int64)
                for p, s in zip(pos, signs):
                    v[p] = s
                candidates.append(v)
    
    candidates = np.array(candidates)
    n_cand = len(candidates)
    
    # Precompute norms
    norms = np.sum(candidates * candidates, axis=1)
    
    # Greedy function
    def greedy_search(order):
        selected = []
        selected_norms = []
        current_max_norm = 0
        
        for idx in order:
            v = candidates[idx]
            norm_v = norms[idx]
            
            # New max norm
            new_max_norm = max(current_max_norm, norm_v)
            
            # Check compatibility with all selected
            compatible = True
            for j, v_j in enumerate(selected):
                # Dot product
                dot = np.dot(v, v_j)
                # Condition: ||v - v_j||^2 > new_max_norm
                # ||v||^2 + ||v_j||^2 - 2 dot > new_max_norm
                # 2 dot < ||v||^2 + ||v_j||^2 - new_max_norm
                rhs = norm_v + selected_norms[j] - new_max_norm
                if 2 * dot >= rhs:
                    compatible = False
                    break
            
            if compatible:
                selected.append(v)
                selected_norms.append(norm_v)
                current_max_norm = new_max_norm
                
        return np.array(selected) if selected else np.array([])

    best_points = np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)
    
    # Try different orders
    # 1. Sorted by norm
    sorted_indices = np.argsort(norms)
    pts = greedy_search(sorted_indices)
    if len(pts) > len(best_points):
        best_points = pts
        
    # 2. Random restarts
    np.random.seed(42)
    for i in range(50):
        if time.time() > deadline - 10:
            break
        random_indices = np.random.permutation(n_cand)
        pts = greedy_search(random_indices)
        if len(pts) > len(best_points):
            best_points = pts
            
    # Final check and conversion
    best_points = np.around(best_points).astype(np.int64)
    
    # Validate
    if len(best_points) < 2:
        return np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)
        
    return best_points
