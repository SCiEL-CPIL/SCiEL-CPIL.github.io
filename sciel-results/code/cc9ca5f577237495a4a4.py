import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 150s kill
    
    # Generate Weight 3 and Weight 4 vectors only (proven effective)
    candidates = []
    norms_sq = []
    
    # Weight 3: ±e_i ± e_j ± e_k
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            norms_sq.append(3)
            
    # Weight 4: ±e_i ± e_j ± e_k ± e_l
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            norms_sq.append(4)
            
    candidates = np.array(candidates, dtype=np.int64)
    norms_sq = np.array(norms_sq, dtype=np.int64)
    n_candidates = len(candidates)
    
    # Compute dot products
    dots = np.dot(candidates, candidates.T)
    
    # Determine conflicts: dist_sq[i,j] < max_norm_sq (4)
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    conflicts = dist_sq < 4
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbors as frozensets for fast intersection
    neighbors = [set() for _ in range(n_candidates)]
    for i in range(n_candidates):
        neigh_indices = np.where(conflicts[i])[0]
        neighbors[i] = set(neigh_indices.tolist())
    
    # Precompute conflict matrix as numpy bool array for fast batch checks
    # conflicts[i] is already available
    
    def local_search_1swap2add(selected_list):
        """
        Local search: try 1-for-2 swaps (remove one vertex, add two non-conflicting ones).
        Also try simple additions first.
        Returns improved list.
        """
        selected = list(selected_list)
        selected_set = set(selected)
        
        # Phase 1: Add any compatible vertex
        improved = True
        while improved:
            improved = False
            for v in range(n_candidates):
                if v not in selected_set:
                    if not (neighbors[v] & selected_set):
                        selected.append(v)
                        selected_set.add(v)
                        improved = True
                        break
        
        # Phase 2: Try 1-for-2 swaps
        # For each u in selected, try removing u and adding two vertices v, w
        # that don't conflict with each other or with (selected - {u})
        swapped = True
        max_swap_iters = 50
        for _ in range(max_swap_iters):
            if time.time() > deadline - 3.0:
                break
            swapped = False
            for u in list(selected):
                if time.time() > deadline - 3.0:
                    break
                remaining = selected_set - {u}
                # Find candidates that don't conflict with remaining
                # and don't conflict with each other
                candidates_to_add = []
                for v in range(n_candidates):
                    if v not in selected_set and v != u:
                        if not (neighbors[v] & remaining):
                            candidates_to_add.append(v)
                
                # Try to find a pair (v, w) in candidates_to_add that don't conflict
                # Sample to keep it fast
                if len(candidates_to_add) > 50:
                    sampled = random.sample(candidates_to_add, 50)
                else:
                    sampled = candidates_to_add
                
                found_pair = False
                for i_idx, v in enumerate(sampled):
                    if found_pair:
                        break
                    for w in sampled[i_idx+1:]:
                        if v == w:
                            continue
                        if not (neighbors[v] & neighbors[w]) and (neighbors[v] & neighbors[w]) == set():
                            # v and w don't conflict with each other
                            if not (neighbors[v] & neighbors[w]):
                                # Check they don't conflict with each other
                                if not (v in neighbors[w] or w in neighbors[v]):
                                    # Do the swap
                                    selected.remove(u)
                                    selected_set.discard(u)
                                    selected.append(v)
                                    selected_set.add(v)
                                    selected.append(w)
                                    selected_set.add(w)
                                    swapped = True
                                    found_pair = True
                                    break
            
            if not swapped:
                break
        
        return selected
    
    def grasp_construct(alpha, rng_seed):
        """
        GRASP construction with restricted candidate list.
        """
        random.seed(rng_seed)
        available = set(range(n_candidates))
        selected = []
        
        while available:
            if time.time() > deadline - 2.0:
                break
            
            # Sample for degree computation
            if len(available) > 300:
                sample = random.sample(list(available), 300)
            else:
                sample = list(available)
                
            degs = []
            for v in sample:
                deg = len(neighbors[v] & available)
                degs.append(deg)
            
            if not degs:
                break
                
            min_deg = min(degs)
            max_deg = max(degs)
            
            threshold = min_deg + alpha * (max_deg - min_deg)
            rcl = [v for v, d in zip(sample, degs) if d <= threshold]
            
            if rcl:
                chosen = random.choice(rcl)
            else:
                chosen = sample[0]
                
            selected.append(chosen)
            available.discard(chosen)
            for n in neighbors[chosen]:
                if n in available:
                    available.discard(n)
                    
        return selected
    
    def greedy_min_degree(seed_val):
        """Pure min-degree greedy."""
        random.seed(seed_val)
        available = set(range(n_candidates))
        selected = []
        
        while available:
            if time.time() > deadline - 2.0:
                break
            
            if len(available) > 300:
                sample = random.sample(list(available), 300)
            else:
                sample = list(available)
                
            best_v = None
            min_deg = float('inf')
            for v in sample:
                deg = len(neighbors[v] & available)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            
            if best_v is None:
                break
                
            selected.append(best_v)
            available.discard(best_v)
            for n in neighbors[best_v]:
                if n in available:
                    available.discard(n)
                    
        return selected
    
    best_selected = []
    np.random.seed(42)
    random.seed(42)
    
    # Phase 1: GRASP with multiple alphas
    alphas = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    n_iter_per_alpha = 40
    
    for alpha in alphas:
        if time.time() > deadline - 15.0:
            break
        for it in range(n_iter_per_alpha):
            if time.time() > deadline - 5.0:
                break
            sol = grasp_construct(alpha, 42 + int(alpha * 1000) + it)
            sol = local_search_1swap2add(sol)
            if len(sol) > len(best_selected):
                best_selected = sol.copy()
    
    # Phase 2: Min-degree greedy with multiple seeds
    for trial in range(150):
        if time.time() > deadline - 5.0:
            break
        sol = greedy_min_degree(1000 + trial)
        sol = local_search_1swap2add(sol)
        if len(sol) > len(best_selected):
            best_selected = sol.copy()
    
    # Phase 3: Final improvement on best solution
    if best_selected:
        best_selected = local_search_1swap2add(best_selected)
        
        # Try to add any remaining compatible vectors
        best_set = set(best_selected)
        for v in range(n_candidates):
            if v not in best_set:
                if not (neighbors[v] & best_set):
                    best_selected.append(v)
                    best_set.add(v)
    
    if not best_selected:
        v = [0] * D
        v[0] = 1
        v[1] = 1
        v[2] = 1
        return np.array([v], dtype=np.int64)
    
    return candidates[best_selected]
