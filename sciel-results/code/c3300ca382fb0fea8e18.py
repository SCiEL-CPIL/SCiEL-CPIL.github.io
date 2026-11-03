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
    conflicts = (dist_sq < 4).astype(np.uint8)
    np.fill_diagonal(conflicts, 0)
    
    # Precompute neighbors as lists for fast intersection
    neighbors = []
    for i in range(n_candidates):
        neighbors.append(np.where(conflicts[i] == 1)[0])
        
    # Convert to sets for faster intersection in local search
    neighbor_sets = [set(n.tolist()) for n in neighbors]

    def greedy_min_degree():
        """
        Greedy selection picking the vertex with the minimum number of 
        conflicts with the *remaining* available vertices.
        """
        available = np.ones(n_candidates, dtype=bool)
        selected = []
        
        while np.any(available):
            if time.time() > deadline - 2.0:
                break
            
            # Calculate degree in the available subgraph
            # deg[i] = sum(conflicts[i, j] for j where available[j] is True)
            deg = (conflicts * available[np.newaxis, :]).sum(axis=1)
            deg[~available] = n_candidates + 1
            
            min_deg = deg.min()
            if min_deg > n_candidates:
                break
                
            # Pick randomly from those with min degree to add diversity
            candidates_to_pick = np.where((deg == min_deg) & available)[0]
            if len(candidates_to_pick) == 0:
                break
                
            pick_idx = random.choice(candidates_to_pick)
            
            selected.append(pick_idx)
            available[pick_idx] = False
            # Remove neighbors of pick_idx from available
            available[neighbors[pick_idx]] = False
            
        return selected

    def local_search(S):
        """
        Improve an independent set S using addition and 1-for-k swaps.
        """
        S_set = set(S)
        S_list = list(S_set)
        improved = True
        max_iterations = 500
        iter_count = 0
        
        while improved and iter_count < max_iterations and time.time() < deadline - 5.0:
            improved = False
            iter_count += 1
            
            # 1. Try to add any vertex that fits
            # To speed up, we can shuffle the order of checking candidates
            order = random.sample(range(n_candidates), n_candidates)
            
            for v in order:
                if time.time() > deadline - 5.0:
                    break
                if v not in S_set:
                    if not (neighbor_sets[v] & S_set):
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
                            if not (neighbor_sets[u] & test_set):
                                candidates_for_swap.append(u)
                    
                    if len(candidates_for_swap) < 2:
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
                            deg = len(neighbor_sets[u] & available_swap)
                            if deg < min_deg:
                                min_deg = deg
                                best_u = u
                                
                        if best_u is None:
                            break
                            
                        added_swap.append(best_u)
                        available_swap.discard(best_u)
                        for n in neighbor_sets[best_u]:
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

    def simulated_annealing(S):
        """
        Simulated Annealing on the Independent Set.
        Moves:
        1. Add a random compatible vertex.
        2. Remove a random vertex.
        3. Swap: Remove v, add u if compatible with S\{v}.
        Objective: Maximize |S|.
        """
        S_set = set(S)
        current_size = len(S_set)
        best_set = set(S_set)
        best_size = current_size
        
        # Annealing parameters
        T_start = 10.0
        T_end = 0.1
        cooling_rate = 0.995
        T = T_start
        
        while T > T_end and time.time() < deadline - 5.0:
            if not S_set:
                break
            
            # Choose a move type
            move_type = random.random()
            
            if move_type < 0.3:
                # Try to add a vertex
                # Sample a few random candidates to avoid O(N)
                candidates_to_add = [v for v in random.sample(range(n_candidates), min(100, n_candidates)) 
                                     if v not in S_set and not (neighbor_sets[v] & S_set)]
                if candidates_to_add:
                    v = random.choice(candidates_to_add)
                    S_set.add(v)
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_set = set(S_set)
            
            elif move_type < 0.6:
                # Try to remove a vertex (to escape local optima)
                v = random.choice(list(S_set))
                S_set.remove(v)
                current_size -= 1
            
            else:
                # Try a swap
                # Pick a vertex to remove
                v = random.choice(list(S_set))
                test_set = S_set - {v}
                
                # Find compatible candidates for swap
                # Sample random candidates
                swap_candidates = []
                for _ in range(50):
                    u = random.randint(0, n_candidates - 1)
                    if u not in test_set and u != v and not (neighbor_sets[u] & test_set):
                        swap_candidates.append(u)
                
                if swap_candidates:
                    u = random.choice(swap_candidates)
                    # Accept swap if it improves size (always +1 in this simple swap? No, it's 1-for-1)
                    # Size remains same. We accept with probability based on T?
                    # For pure size maximization, 1-for-1 is neutral. 
                    # However, it can change the structure to allow future additions.
                    # Let's always accept 1-for-1 swaps that are valid, or accept with high probability.
                    # Actually, if we just want to maximize size, neutral moves help exploration.
                    S_set = test_set.union({u})
                    # Size unchanged
                    
            # Cooling
            T *= cooling_rate
            
        return list(best_set)

    best_selected = []
    np.random.seed(42)
    random.seed(42)
    
    # Phase 1: Randomized Greedy Restarts
    # Allocate 50% of time to greedy restarts
    greedy_deadline = start_time + (deadline - start_time) * 0.5
    
    num_trials = 0
    while time.time() < greedy_deadline:
        num_trials += 1
        random.seed(42 + num_trials)
        sel = greedy_min_degree()
        if len(sel) > len(best_selected):
            best_selected = sel
            
    # Phase 2: Local Search on the best greedy result
    if best_selected and time.time() < deadline - 10.0:
        # Run local search on the best result found so far
        refined = local_search(best_selected)
        if len(refined) > len(best_selected):
            best_selected = refined
        
        # Phase 3: Simulated Annealing on the refined result
        if time.time() < deadline - 5.0:
            sa_result = simulated_annealing(best_selected)
            if len(sa_result) > len(best_selected):
                best_selected = sa_result
                
                # Run local search again on the SA result to polish
                if time.time() < deadline - 2.0:
                    final_refined = local_search(best_selected)
                    if len(final_refined) > len(best_selected):
                        best_selected = final_refined

    # Construct final array
    if not best_selected:
        # Fallback: just return empty or a trivial valid set
        # But we should always find something.
        return np.array([], dtype=np.int64).reshape(0, D)
        
    result = candidates[best_selected]
    return result
