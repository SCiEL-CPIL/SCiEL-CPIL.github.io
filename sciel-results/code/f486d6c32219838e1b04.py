import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Generate Weight 2, 3, and 4 vectors
    # Max norm squared is 4 (from Weight 4)
    # Condition: dist_sq >= 4
    candidates = []
    norms_sq = []
    
    # Weight 2
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            v[positions[0]] = signs[0]
            v[positions[1]] = signs[1]
            candidates.append(v)
            norms_sq.append(2)
            
    # Weight 3
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            norms_sq.append(3)
            
    # Weight 4
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
    
    # Determine conflicts
    # dist_sq[i, j] = norms_sq[i] + norms_sq[j] - 2*dots[i, j]
    # Conflict if dist_sq[i, j] < 4
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    conflicts = dist_sq < 4
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbors as sets for fast intersection
    neighbors = [set() for _ in range(n_candidates)]
    for i in range(n_candidates):
        neigh_indices = np.where(conflicts[i])[0]
        neighbors[i] = set(neigh_indices.tolist())
        
    def greedy_min_degree():
        """
        Greedy selection picking the vertex with the minimum number of 
        conflicts with the *remaining* available vertices.
        """
        available = set(range(n_candidates))
        selected = []
        
        while available:
            if time.time() > deadline - 2.0:
                break
            
            # Find vertex with min degree in available subgraph
            # Sample if large to save time
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
            # Remove neighbors of best_v from available
            for n in neighbors[best_v]:
                if n in available:
                    available.discard(n)
                    
        return selected

    def local_search(S):
        """
        Improve an independent set S using addition and 1-for-k swaps.
        """
        S_set = set(S)
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # 1. Try to add any vertex that fits
            # Iterate over a shuffled list of candidates not in S
            # To avoid O(N) check every time, we can just check all N.
            # N is ~5000. Checking 5000 sets of size ~2000 is fast enough if done a few times.
            # But we do this in a loop until no improvement.
            # Optimization: Only check candidates that are NOT in S.
            # We can maintain a list of 'free' candidates?
            # Simpler: just loop over range(n_candidates).
            
            # To speed up, we can shuffle the order of checking candidates
            # so we don't always check 0, 1, 2...
            order = random.sample(range(n_candidates), n_candidates)
            
            for v in order:
                if time.time() > deadline - 5.0:
                    break
                if v not in S_set:
                    if not (neighbors[v] & S_set):
                        S_set.add(v)
                        improved = True
            
            # 2. Try 1-for-k swaps
            if not improved:
                # Iterate over a copy of S to allow modification
                # Shuffled order
                S_list = list(S_set)
                random.shuffle(S_list)
                
                for v in S_list:
                    if time.time() > deadline - 5.0:
                        break
                    
                    # Remove v temporarily
                    test_set = S_set - {v}
                    
                    # Find all vertices that are compatible with test_set
                    # These are candidates for the swap
                    candidates_for_swap = []
                    for u in range(n_candidates):
                        if u not in test_set and u != v:
                            if not (neighbors[u] & test_set):
                                candidates_for_swap.append(u)
                    
                    if not candidates_for_swap:
                        continue
                        
                    # Greedily pick an independent subset from candidates_for_swap
                    # We want at least 2 to improve size (remove 1, add 2 => net +1)
                    
                    available_swap = set(candidates_for_swap)
                    added_swap = []
                    
                    while available_swap:
                        if time.time() > deadline - 5.0:
                            break
                        if len(available_swap) > 50:
                            sample = random.sample(list(available_swap), 50)
                        else:
                            sample = list(available_swap)
                            
                        best_u = None
                        min_deg = float('inf')
                        for u in sample:
                            deg = len(neighbors[u] & available_swap)
                            if deg < min_deg:
                                min_deg = deg
                                best_u = u
                                
                        if best_u is None:
                            break
                            
                        added_swap.append(best_u)
                        available_swap.discard(best_u)
                        for n in neighbors[best_u]:
                            if n in available_swap:
                                available_swap.discard(n)
                        
                        # Early stop if we have enough to justify the swap
                        # We need > 1 added to replace 1 removed for net gain
                        if len(added_swap) >= 3: 
                             break

                    if len(added_swap) > 1:
                        S_set = test_set.union(added_swap)
                        improved = True

        return list(S_set)

    best_selected = []
    np.random.seed(42)
    random.seed(42)
    
    # Phase 1: Randomized Greedy Restarts
    # Allocate 60% of time to greedy restarts
    greedy_deadline = start_time + (deadline - start_time) * 0.6
    
    num_trials = 0
    while time.time() < greedy_deadline:
        num_trials += 1
        random.seed(42 + num_trials)
        sel = greedy_min_degree()
        if len(sel) > len(best_selected):
            best_selected = sel
            
    # Phase 2: Local Search on the best greedy result
    if best_selected and time.time() < deadline - 5.0:
        # Run local search on the best result found so far
        refined = local_search(best_selected)
        if len(refined) > len(best_selected):
            best_selected = refined
        
        # If time permits, try local search on a few other good greedy results
        # We didn't store them, so let's just run local search on the current best again?
        # Or restart greedy once more and local search?
        # Given the time budget, one solid local search pass is usually best.
        # But let's add a small phase 3: if time remains, do another greedy + local search
        # and keep the best.
        
        if time.time() < deadline - 20.0:
            # One more restart + local search
            random.seed(42 + num_trials + 100)
            sel2 = greedy_min_degree()
            if len(sel2) > len(best_selected) * 0.9: # Only if it's competitive
                refined2 = local_search(sel2)
                if len(refined2) > len(best_selected):
                    best_selected = refined2

    # Construct final array
    if not best_selected:
        # Fallback: just return empty or a trivial valid set
        # But we should always find something.
        return np.array([], dtype=np.int64).reshape(0, D)
        
    result = candidates[best_selected]
    return result
