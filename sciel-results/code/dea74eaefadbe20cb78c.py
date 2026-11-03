"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time


def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Generate candidates with squared norm 4
    candidates = []
    
    # Type 1: One 2, rest 0
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = s * 2
            candidates.append(v)
    
    # Type 2: Four ±1's with even sum
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([-1, 1], repeat=4):
            if sum(signs) % 2 == 0:
                v = [0] * D
                for pos, sign in zip(support, signs):
                    v[pos] = sign
                candidates.append(v)
    
    n_cands = len(candidates)
    cand_arr = np.array(candidates, dtype=np.float64)
    
    # Precompute conflict matrix: conflict[i,j] = True if dot(cand[i], cand[j]) > 2
    # For kissing: min pairwise dist >= max norm
    # If all have norm^2=4, then dist^2 = 8 - 2*dot >= 4 => dot <= 2
    dot_matrix = cand_arr @ cand_arr.T
    conflict = dot_matrix > 2  # True means incompatible
    
    # Precompute adjacency lists for faster access
    adj = [np.where(conflict[i])[0] for i in range(n_cands)]
    
    best_vecs = None
    best_count = 0
    
    # Multiple random greedy passes
    for trial in range(500):
        if time.time() > deadline - 5:
            break
        
        rng_t = np.random.RandomState(trial * 137 + 42)
        
        # Random order
        order = list(range(n_cands))
        rng_t.shuffle(order)
        
        selected = []
        selected_set = set()
        remaining = set(range(n_cands))
        
        for idx in order:
            if time.time() > deadline - 5:
                break
            if idx not in remaining:
                continue
            
            # Check if idx is compatible with all selected
            # Optimization: Check against selected_set
            compatible = True
            for s in selected_set:
                if conflict[idx, s]:
                    compatible = False
                    break
            
            if compatible:
                selected.append(idx)
                selected_set.add(idx)
                # Remove incompatible vertices from remaining
                for u in adj[idx]:
                    if u in remaining:
                        remaining.remove(u)
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_vecs = np.array([cand_arr[i] for i in selected], dtype=np.int64)
    
    # Local search: try to swap out one vector to allow adding two others, or just add
    if best_vecs is not None and best_count > 0:
        # Reconstruct selected set from best_vecs
        # Map vector content to index
        vec_to_idx = {}
        for i in range(n_cands):
            vec_to_idx[tuple(cand_arr[i].astype(int))] = i
        
        current_selected = set()
        for v in best_vecs:
            current_selected.add(vec_to_idx[tuple(v)])
            
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # 1. Try to add any vector
            for idx in range(n_cands):
                if idx in current_selected:
                    continue
                compatible = True
                for s in current_selected:
                    if conflict[idx, s]:
                        compatible = False
                        break
                if compatible:
                    current_selected.add(idx)
                    improved = True
                    break
            
            # 2. Try to swap: remove one, add two
            if not improved:
                # Simple swap heuristic is expensive. 
                # Instead, try to find a vector that conflicts with exactly 1 selected vector,
                # and if we remove that one, we can add this one plus another?
                # This is complex. Let's stick to simple addition first.
                pass
        
        if len(current_selected) > best_count:
            best_count = len(current_selected)
            best_vecs = np.array([cand_arr[i] for i in sorted(current_selected)], dtype=np.int64)
    
    if best_vecs is None or len(best_vecs) == 0:
        # Fallback: return norm^2=2 vectors
        base_vecs = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i], v[j] = si, sj
                    base_vecs.append(v)
        best_vecs = np.array(base_vecs, dtype=np.int64)
    
    return best_vecs
# EVOLVE-BLOCK-END
