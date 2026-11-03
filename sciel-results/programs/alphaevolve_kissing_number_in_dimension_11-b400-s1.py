"""Kissing number in dimension 11: support-level MIS + SA + Type A vectors."""
import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Generate all C(11,4) = 330 supports
    supports = list(itertools.combinations(range(D), 4))
    n_sup = len(supports)  # 330

    # Build conflict matrix: edge if intersection size == 3
    # (intersection 4 means same support, intersection <=2 means no conflict)
    adj_mask = np.zeros((n_sup, n_sup), dtype=np.uint8)
    support_sets = [set(s) for s in supports]
    for i in range(n_sup):
        for j in range(i + 1, n_sup):
            if len(support_sets[i] & support_sets[j]) == 3:
                adj_mask[i, j] = 1
                adj_mask[j, i] = 1

    # Precompute adjacency lists for faster operations
    adj_lists = [np.where(adj_mask[i] == 1)[0] for i in range(n_sup)]
    adj_sizes = np.array([len(a) for a in adj_lists])

    def greedy_is(seed=0):
        """Greedy independent set using minimum-degree heuristic with random tie-breaking."""
        rng = np.random.default_rng(seed)
        remaining = np.ones(n_sup, dtype=np.uint8)
        selected = np.zeros(n_sup, dtype=np.uint8)
        
        while True:
            if time.time() > deadline - 15.0:
                break
            rem_idx = np.where(remaining > 0)[0]
            if len(rem_idx) == 0:
                break
            
            # Compute degree within remaining set
            deg = (adj_mask * remaining[np.newaxis, :]).sum(axis=1)
            deg[remaining == 0] = n_sup + 1
            
            min_deg = deg.min()
            candidates = np.where((deg == min_deg) & (remaining > 0))[0]
            if len(candidates) == 0:
                break
            
            pick = rng.choice(candidates)
            selected[pick] = 1
            remaining[pick] = 0
            remaining[adj_mask[pick] == 1] = 0
        
        return selected

    def local_search_sa(selected, n_steps=50000, seed=0):
        """Simulated annealing with 1-for-k moves on the support-level problem."""
        rng = np.random.default_rng(seed)
        rand = random.Random(seed)
        
        S = selected.copy()
        current_size = int(S.sum())
        best_size = current_size
        best_S = S.copy()
        
        T = 2.0
        cooling = 0.9997
        min_T = 0.05
        
        step = 0
        while step < n_steps and time.time() < deadline - 10.0:
            step += 1
            v = rng.integers(0, n_sup)
            
            if S[v]:
                # Removal move (probabilistic)
                if rand.random() < 0.03:
                    S[v] = 0
                    current_size -= 1
                continue
            
            # v not in S: try to add it
            conflicts = np.where(adj_mask[v] & S)[0]
            n_conf = len(conflicts)
            
            if n_conf == 0:
                S[v] = 1
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_S = S.copy()
            else:
                # 1-for-n move: remove n_conf conflicting supports, add v
                delta = 1 - n_conf
                if delta >= 0:
                    S[conflicts] = 0
                    S[v] = 1
                    current_size += delta
                    if current_size > best_size:
                        best_size = current_size
                        best_S = S.copy()
                else:
                    if T > min_T:
                        p = np.exp(delta / T)
                        if rand.random() < p:
                            S[conflicts] = 0
                            S[v] = 1
                            current_size += delta
                            if current_size > best_size:
                                best_size = current_size
                                best_S = S.copy()
            
            T = max(min_T, T * cooling)
        
        return best_S, best_size

    def polish_2for2(S, max_rounds=20):
        """Try all 2-for-2 moves: remove 2 supports, add 2 others."""
        improved = True
        rounds = 0
        while improved and rounds < max_rounds and time.time() < deadline - 8.0:
            improved = False
            rounds += 1
            
            sel = np.where(S == 1)[0]
            k = len(sel)
            if k < 2:
                break
            
            # For each pair of selected supports, try to replace with a better pair
            # This is O(k^2 * n^2) which is too slow, so use sampling
            n_trials = min(5000, k * k // 2)
            for _ in range(n_trials):
                if time.time() > deadline - 8.0:
                    break
                i1, i2 = rng_pair = np.random.default_rng().integers(0, k, size=2)
                if i1 == i2:
                    continue
                v1, v2 = sel[i1], sel[i2]
                
                # Find all supports compatible with both v1 and v2 (not conflicting)
                # and not currently selected
                compat = (adj_mask[v1] == 0) & (adj_mask[v2] == 0) & (S == 0)
                # Also need the two new supports to be compatible with each other
                candidates = np.where(compat)[0]
                if len(candidates) < 2:
                    continue
                
                # Sample pairs from candidates
                for trial in range(min(10, len(candidates) - 1)):
                    u_idx = np.random.choice(len(candidates))
                    w_idx = np.random.choice(len(candidates) - 1)
                    if w_idx >= u_idx:
                        w_idx += 1
                    u, w = candidates[u_idx], candidates[w_idx]
                    if adj_mask[u, w] == 1:
                        continue
                    
                    # Check: removing v1, v2 and adding u, w improves
                    # u and w must not conflict with any other selected support
                    others = np.delete(sel, [i1, i2])
                    if len(others) > 0:
                        if np.any(adj_mask[u][others] == 1) or np.any(adj_mask[w][others] == 1):
                            continue
                    
                    S[v1] = 0
                    S[v2] = 0
                    S[u] = 1
                    S[w] = 1
                    improved = True
                    break
                if improved:
                    break
        
        return S

    # Main search: multiple restarts
    best_S = np.zeros(n_sup, dtype=np.uint8)
    best_size = 0
    
    # Initial greedy starts with different seeds
    n_restarts = 30
    for i in range(n_restarts):
        if time.time() > deadline - 25.0:
            break
        
        # Greedy initialization
        S_init = greedy_is(seed=1000 + i)
        init_size = int(S_init.sum())
        
        # SA refinement
        S_sa, sa_size = local_search_sa(S_init, n_steps=30000, seed=2000 + i)
        
        # Polish
        S_polished = polish_2for2(S_sa.copy(), max_rounds=10)
        pol_size = int(S_polished.sum())
        
        if pol_size > best_size:
            best_size = pol_size
            best_S = S_polished.copy()
        
        # Also try SA with longer run every few restarts
        if i % 5 == 0 and time.time() < deadline - 30.0:
            S_sa2, sa2_size = local_search_sa(S_init, n_steps=80000, seed=5000 + i)
            S_pol2 = polish_2for2(S_sa2.copy(), max_rounds=15)
            pol2_size = int(S_pol2.sum())
            if pol2_size > best_size:
                best_size = pol2_size
                best_S = S_pol2.copy()

    # Construct vectors from best support set
    # For each selected support, add all 16 sign patterns
    vectors = []
    for idx in range(n_sup):
        if best_S[idx]:
            sup = supports[idx]
            for signs in itertools.product([1, -1], repeat=4):
                v = [0] * D
                for pos, s in zip(sup, signs):
                    v[pos] = s
                vectors.append(v)
    
    # Add Type A vectors: +/- 2e_i (22 vectors)
    for i in range(D):
        for sign in (1, -1):
            v = [0] * D
            v[i] = sign * 2
            vectors.append(v)
    
    result = np.array(vectors, dtype=np.int64)
    
    # Verify: check that minimum pairwise distance >= max norm
    # (This is a sanity check; the construction guarantees it by design)
    # Norms: Type A have norm 2, Type B have norm 2
    # Distance between Type A vectors: 4 (different coords) or 0 (same)
    #   Actually +/-2e_i vs +/-2e_j: if i!=j, dist^2 = 4+4=8, dist=2*sqrt(2) > 2
    #   +/-2e_i vs -/+2e_i: dist^2 = 16 or 0
    #   Wait: 2e_i vs -2e_i: dist^2 = (4)^2 = 16, dist = 4 > 2
    #   2e_i vs 2e_j (i!=j): dist^2 = 4+4 = 8, dist = 2*sqrt(2) > 2 ✓
    # Distance between Type B vectors: supports intersect in <=2 coords
    #   If intersect in 0: min dist^2 >= 4 (each has 4 nonzeros)
    #   If intersect in 1: min dist^2 >= 2*1^2 + 3*1^2 = 5... let me think
    #   Actually with |intersection| <= 2, we can always choose signs to make dot <= 0
    #   So dist^2 >= 4+4-2*dot >= 8 if dot <= 0
    #   Wait, we need dist >= max_norm = 2, so dist^2 >= 4
    #   With |intersection| <= 2 and appropriate signs, dot can be made <= 0
    #   So dist^2 >= 4+4 = 8 > 4 ✓
    # Distance between Type A (2e_i) and Type B (support containing i or not):
    #   If i in support: dot = 2 * (+/-1) = +/-2, dist^2 = 4+4-2*(+/-2) = 0 or 8
    #   Wait, that's a problem! 2e_i vs e_i+e_j+e_k+e_l: dot = 2, dist^2 = 4+4-4 = 4, dist = 2 ✓
    #   2e_i vs -e_i+e_j+e_k+e_l: dot = -2, dist^2 = 4+4+4 = 12, dist > 2 ✓
    #   So minimum is exactly 2, which equals max_norm = 2. That's OK (>= required).
    #   Actually wait: the requirement is dist >= max_norm. max_norm = 2. dist = 2 >= 2 ✓
    
    return result

# EVOLVE-BLOCK-END
