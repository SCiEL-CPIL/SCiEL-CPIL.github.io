import numpy as np
import time
from itertools import product

def construct():
    D = 11
    deadline = time.time() + 140.0  # 10s safety margin
    
    # Generate candidate vectors: all integer vectors with small norm
    # We want vectors where min pairwise distance >= max norm
    # Strategy: generate candidates with norm <= some bound, then find max independent set
    
    # Generate candidate vectors with L2 norm between 1 and some max
    # For kissing number in 11D, known bounds suggest we need vectors with small norms
    # Let's generate all vectors with entries in {-2,-1,0,1} and norm >= 1
    # This gives a manageable candidate set
    
    candidates = []
    # Include vectors with entries in {-1, 0, 1}
    for combo in product([-1, 0, 1], repeat=D):
        v = np.array(combo, dtype=np.int64)
        norm_sq = np.dot(v, v)
        if norm_sq >= 1:  # nonzero
            candidates.append(v)
    
    # Also include some vectors with entries in {-2, -1, 0, 1, 2} but only if they help
    # For now, start with the smaller set
    
    n_cand = len(candidates)
    candidates = np.array(candidates, dtype=np.int64)
    
    # Compute norms
    norms = np.sqrt(np.sum(candidates ** 2, axis=1))
    max_norm = np.max(norms)
    
    # Build conflict graph: two vectors conflict if distance < max_norm
    # dist(v_i, v_j)^2 = ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j
    # They conflict if dist < max_norm, i.e., dist^2 < max_norm^2
    # i.e., ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j < max_norm^2
    
    # For efficiency, use bit-packed adjacency
    # n_cand can be large (3^11 - 1 = 177146), so we need to be careful
    # Let's filter candidates first: only keep those that could potentially be in a good set
    
    # Actually, let's think about this differently.
    # The kissing number problem: we want max number of unit vectors with pairwise distance >= 2
    # (after normalizing to norm 1, distance >= 2 means angle >= 60 degrees, i.e., dot <= 0.5)
    # But we're working with integer vectors, so we normalize.
    # 
    # Two integer vectors v, w with norms ||v||, ||w|| are compatible if:
    # ||v - w|| >= max(||v||, ||w||)
    # This is the condition from the task.
    
    # Let's use a greedy approach with local search on the conflict graph
    # But first, let's reduce the candidate set by removing dominated vectors
    
    # A vector v is dominated if there's another vector w with ||w|| <= ||v|| and w is "better"
    # For simplicity, let's just work with the full candidate set but use efficient methods
    
    # Use bit-packed representation for the conflict graph
    # n_cand = 177146, so we need ~2800 bytes per row for bit packing
    # Total memory: 177146 * 2800 bytes ~ 496 MB - too much
    
    # Instead, let's use a different approach:
    # 1. Start with a good initial solution using greedy
    # 2. Use local search to improve
    
    # Greedy: sort candidates by norm (prefer smaller norms), add if compatible with all selected
    # This is O(n^2) which might be slow for 177k candidates
    
    # Let's reduce candidates: only use vectors with norm^2 <= 4 (i.e., entries in {-1,0,1} with at most 4 nonzeros)
    # or norm^2 <= 3
    
    # Actually, let me reconsider. The seed program uses vectors with exactly 2 nonzeros (±1).
    # That gives C(11,2) * 4 = 220 vectors.
    # Let's expand to include vectors with 1, 2, or 3 nonzeros.
    
    # Regenerate candidates more carefully
    candidates = []
    # 1-nonzero vectors
    for i in range(D):
        for s in [1, -1]:
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
    
    # 2-nonzero vectors
    for i in range(D):
        for j in range(i + 1, D):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    candidates.append(v)
    
    # 3-nonzero vectors
    for i in range(D):
        for j in range(i + 1, D):
            for k in range(j + 1, D):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        for s3 in [1, -1]:
                            v = np.zeros(D, dtype=np.int64)
                            v[i] = s1
                            v[j] = s2
                            v[k] = s3
                            candidates.append(v)
    
    # 4-nonzero vectors (maybe too many? C(11,4)*16 = 330*16 = 5280)
    for i in range(D):
        for j in range(i + 1, D):
            for k in range(j + 1, D):
                for l in range(k + 1, D):
                    for s1 in [1, -1]:
                        for s2 in [1, -1]:
                            for s3 in [1, -1]:
                                for s4 in [1, -1]:
                                    v = np.zeros(D, dtype=np.int64)
                                    v[i] = s1
                                    v[j] = s2
                                    v[k] = s3
                                    v[l] = s4
                                    candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    # Compute norms
    norms_sq = np.sum(candidates ** 2, axis=1)
    norms = np.sqrt(norms_sq.astype(np.float64))
    max_norm = np.max(norms)
    
    # Build conflict matrix: conflict[i,j] = True if dist(v_i, v_j) < max_norm
    # dist^2 = norm_sq[i] + norm_sq[j] - 2 * dot(v_i, v_j)
    # Conflict if dist^2 < max_norm^2
    
    # For memory efficiency, compute in chunks
    # n_cand is now 22 + 220 + 1320 + 5280 = 6842, which is manageable
    
    # Compute dot product matrix
    dots = candidates @ candidates.T
    
    # dist_sq[i,j] = norms_sq[i] + norms_sq[j] - 2*dots[i,j]
    norms_sq_col = norms_sq[:, np.newaxis]
    norms_sq_row = norms_sq[np.newaxis, :]
    dist_sq = norms_sq_col + norms_sq_row - 2 * dots
    
    # Conflict if dist_sq < max_norm^2 (strictly less, since we need dist >= max_norm)
    conflict = dist_sq < (max_norm ** 2)
    np.fill_diagonal(conflict, False)
    
    # Now find maximum independent set in this graph
    # Use greedy + local search
    
    def greedy_independent_set(conflict, n):
        """Greedy: pick vertices in order of increasing degree"""
        degrees = conflict.sum(axis=1)
        order = np.argsort(degues) if False else np.argsort(degrees)
        selected = []
        remaining = set(range(n))
        for v in order:
            if v not in remaining:
                continue
            # Check if v is compatible with all selected
            compatible = True
            for u in selected:
                if conflict[v, u]:
                    compatible = False
                    break
            if compatible:
                selected.append(v)
                remaining.discard(v)
        return selected
    
    def local_search(selected, conflict, n, max_iters=1000):
        """Local search: try to add vertices, or swap to improve"""
        selected = list(selected)
        selected_set = set(selected)
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            
            # Try to add any unselected vertex that's compatible
            for v in range(n):
                if v in selected_set:
                    continue
                compatible = True
                for u in selected:
                    if conflict[v, u]:
                        compatible = False
                        break
                if compatible:
                    selected.append(v)
                    selected_set.add(v)
                    improved = True
                    break
            
            if not improved:
                # Try swap: remove one selected, add two unselected
                for i, u in enumerate(selected):
                    # Try removing u
                    temp_selected = selected[:i] + selected[i+1:]
                    temp_set = set(temp_selected)
                    
                    # Find unselected vertices compatible with temp_selected
                    added = []
                    for v in range(n):
                        if v in temp_set:
                            continue
                        compatible = True
                        for w in temp_selected:
                            if conflict[v, w]:
                                compatible = False
                                break
                        if compatible:
                            added.append(v)
                            if len(added) >= 2:
                                break
                    
                    if len(added) >= 2:
                        # Check if added vertices are compatible with each other
                        if not conflict[added[0], added[1]]:
                            selected = temp_selected + added
                            selected_set = set(selected)
                            improved = True
                            break
            
            if not improved:
                break
        
        return selected
    
    # Initial greedy solution
    selected = greedy_independent_set(conflict, n_cand)
    best_size = len(selected)
    
    # Local search
    selected = local_search(selected, conflict, n_cand)
    best_size = len(selected)
    
    # Try multiple restarts with different orderings
    np.random.seed(42)
    for restart in range(50):
        if time.time() > deadline - 10.0:
            break
        
        # Random perturbation of initial solution
        if selected:
            # Remove a random subset
            remove_count = min(len(selected) // 3, 10)
            remove_indices = np.random.choice(len(selected), size=remove_count, replace=False)
            remove_set = set(selected[i] for i in remove_indices)
            new_selected = [v for v in selected if v not in remove_set]
        else:
            new_selected = []
        
        # Greedy fill
        remaining = [v for v in range(n_cand) if v not in set(new_selected)]
        for v in remaining:
            if time.time() > deadline - 5.0:
                break
            compatible = True
            for u in new_selected:
                if conflict[v, u]:
                    compatible = False
                    break
            if compatible:
                new_selected.append(v)
        
        # Local search
        new_selected = local_search(new_selected, conflict, n_cand)
        
        if len(new_selected) > best_size:
            best_size = len(new_selected)
            selected = new_selected
    
    # Extract the best vectors
    if selected:
        result = candidates[selected]
    else:
        # Fallback to seed
        vectors = []
        for i in range(D):
            for j in range(i + 1, D):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        v = [0] * D
                        v[i] = s1
                        v[j] = s2
                        vectors.append(v)
        result = np.array(vectors, dtype=np.int64)
    
    return result
