import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Generate Weight 2, 3, and 4 vectors
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
    
    # Compute conflict matrix
    dots = np.dot(candidates, candidates.T)
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    conflicts = dist_sq < 4
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbors as sets
    neighbors = [set() for _ in range(n_candidates)]
    for i in range(n_candidates):
        neigh_indices = np.where(conflicts[i])[0]
        neighbors[i] = set(neigh_indices.tolist())
    
    def greedy_select(available, selected_set):
        """Greedy selection from available vertices, adding to selected_set."""
        while available:
            if time.time() > deadline - 3.0:
                break
            # Pick vertex with min conflicts with remaining available
            if len(available) > 200:
                sample = random.sample(list(available), 200)
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
            
            selected_set.add(best_v)
            available.discard(best_v)
            for n in neighbors[best_v]:
                if n in available:
                    available.discard(n)
    
    def local_search(S):
        """Strong local search with addition and multi-swaps."""
        S_set = set(S)
        improved = True
        max_iter = 200
        iter_count = 0
        
        while improved and time.time() < deadline - 5.0 and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            # Phase 1: Add any compatible vertex
            order = random.sample(range(n_candidates), n_candidates)
            for v in order:
                if time.time() > deadline - 5.0:
                    break
                if v not in S_set and not (neighbors[v] & S_set):
                    S_set.add(v)
                    improved = True
            
            # Phase 2: Try 1-for-k swaps (remove 1, add up to 3)
            if not improved:
                S_list = list(S_set)
                random.shuffle(S_list)
                
                for v in S_list:
                    if time.time() > deadline - 5.0:
                        break
                    
                    test_set = S_set - {v}
                    
                    # Find all vertices compatible with test_set
                    compatible = []
                    for u in range(n_candidates):
                        if u not in test_set and u != v:
                            if not (neighbors[u] & test_set):
                                compatible.append(u)
                    
                    if len(compatible) <= 1:
                        continue
                    
                    # Greedily select up to 3 from compatible
                    available_swap = set(compatible)
                    added = []
                    while available_swap and len(added) < 3:
                        if time.time() > deadline - 5.0:
                            break
                        if len(available_swap) > 100:
                            sample = random.sample(list(available_swap), 100)
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
                        
                        added.append(best_u)
                        available_swap.discard(best_u)
                        for n in neighbors[best_u]:
                            if n in available_swap:
                                available_swap.discard(n)
                    
                    if len(added) > 1:
                        S_set = test_set.union(added)
                        improved = True
            
            # Phase 3: Try 2-for-k swaps (remove 2, add up to 4)
            if not improved and len(S_set) >= 2:
                S_list = list(S_set)
                random.shuffle(S_list)
                tried = 0
                for i in range(min(len(S_list), 50)):
                    if time.time() > deadline - 5.0 or tried >= 20:
                        break
                    tried += 1
                    
                    v1, v2 = S_list[i], S_list[i + 1] if i + 1 < len(S_list) else None
                    if v2 is None or v2 == v1:
                        continue
                    
                    test_set = S_set - {v1, v2}
                    
                    compatible = []
                    for u in range(n_candidates):
                        if u not in test_set and u != v1 and u != v2:
                            if not (neighbors[u] & test_set):
                                compatible.append(u)
                    
                    if len(compatible) <= 2:
                        continue
                    
                    available_swap = set(compatible)
                    added = []
                    while available_swap and len(added) < 4:
                        if time.time() > deadline - 5.0:
                            break
                        if len(available_swap) > 100:
                            sample = random.sample(list(available_swap), 100)
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
                        
                        added.append(best_u)
                        available_swap.discard(best_u)
                        for n in neighbors[best_u]:
                            if n in available_swap:
                                available_swap.discard(n)
                    
                    if len(added) > 2:
                        S_set = test_set.union(added)
                        improved = True
        
        return list(S_set)
    
    def greedy_restart():
        """Single greedy restart."""
        available = set(range(n_candidates))
        selected = set()
        greedy_select(available, selected)
        return list(selected)
    
    best_selected = []
    np.random.seed(42)
    random.seed(42)
    
    # Phase 1: Multiple greedy restarts
    greedy_deadline = start_time + (deadline - start_time) * 0.35
    
    num_trials = 0
    while time.time() < greedy_deadline:
        num_trials += 1
        random.seed(42 + num_trials * 7)
        sel = greedy_restart()
        if len(sel) > len(best_selected):
            best_selected = sel
    
    # Phase 2: Local search on best
    if best_selected and time.time() < deadline - 15.0:
        refined = local_search(best_selected)
        if len(refined) > len(best_selected):
            best_selected = refined
    
    # Phase 3: More restarts with local search
    ls_deadline = deadline - 10.0
    while time.time() < ls_deadline:
        random.seed(random.randint(1, 100000))
        sel = greedy_restart()
        if len(sel) + 2 > len(best_selected):  # Only do LS if promising
            refined = local_search(sel)
            if len(refined) > len(best_selected):
                best_selected = refined
    
    # Phase 4: Final local search
    if best_selected and time.time() < deadline - 3.0:
        refined = local_search(best_selected)
        if len(refined) > len(best_selected):
            best_selected = refined
    
    # Construct final array
    if not best_selected:
        return np.array([], dtype=np.int64).reshape(0, D)
        
    result = candidates[best_selected]
    return result
