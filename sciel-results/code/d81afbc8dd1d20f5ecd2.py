"""Program for kissing number in dimension 11 using Leech lattice construction."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from collections import defaultdict

def construct():
    """
    Construct 165 vectors in R^11 forming a kissing configuration.
    
    The 165 kissing vectors in dimension 11 consist of:
    - 22 vectors of the form ±2e_i (weight 1)
    - 143 vectors of weight 4 (four entries ±1)
    
    We use a branch-and-bound maximum independent set search on the conflict
    graph of weight-4 vectors, exploiting the sparse structure of conflicts.
    """
    start_time = time.time()
    deadline = start_time + 115
    
    # Type 1: weight 1, one entry ±2
    v1_list = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_list.append(v)
    
    # Type 2: weight 4, four entries ±1
    v4_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_list.append(v)
    
    v1_arr = np.array(v1_list, dtype=np.int64)
    v4_arr = np.array(v4_list, dtype=np.int64)
    
    N4 = len(v4_arr)  # 5280
    
    # Build conflict graph efficiently
    # Conflicts only when: different position sets with |intersection| = 3 
    # and all 3 sign products = +1
    
    pos_sign_map = {}
    idx_to_pos_sign = []
    for i, v in enumerate(v4_list):
        positions = tuple(j for j in range(D) if v[j] != 0)
        signs = tuple(v[j] for j in positions)
        pos_sign_map[(positions, signs)] = i
        idx_to_pos_sign.append((positions, signs))
    
    # Build neighbor list
    neighbor_list = [set() for _ in range(N4)]
    
    # Group position sets by their 3-subsets
    three_subset_map = defaultdict(list)
    for i in range(N4):
        positions, signs = idx_to_pos_sign[i]
        for three in itertools.combinations(positions, 3):
            three_subset_map[three].append(i)
    
    for three, indices in three_subset_map.items():
        if len(indices) < 2:
            continue
        for idx1 in range(len(indices)):
            i1 = indices[idx1]
            pos1, s1 = idx_to_pos_sign[i1]
            for idx2 in range(idx1 + 1, len(indices)):
                i2 = indices[idx2]
                pos2, s2 = idx_to_pos_sign[i2]
                # Check if sign products on the 3 intersection positions are all +1
                prod = 1
                for pos in three:
                    idx1_pos = pos1.index(pos)
                    idx2_pos = pos2.index(pos)
                    prod *= s1[idx1_pos] * s2[idx2_pos]
                    if prod != 1:
                        break
                if prod == 1:
                    neighbor_list[i1].add(i2)
                    neighbor_list[i2].add(i1)
    
    # Convert to sorted lists for faster access
    neighbor_list = [sorted(lst) for lst in neighbor_list]
    
    # Branch and bound maximum independent set
    # Use Tomita's algorithm with coloring for upper bound
    
    def max_independent_set_bnb():
        """Branch and bound maximum independent set using coloring upper bound."""
        best_size = 0
        best_set = []
        
        # Precompute degrees
        degrees = [len(neighbor_list[i]) for i in range(N4)]
        
        # Order vertices by degree (descending) for better pruning
        order = sorted(range(N4), key=lambda i: -degrees[i])
        
        # Recursion with memoization-like pruning
        def dfs(candidates, current_set):
            nonlocal best_size, best_set
            
            if time.time() > deadline - 2:
                return
            
            if not candidates:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
                return
            
            # Upper bound: current size + len(candidates)
            if len(current_set) + len(candidates) <= best_size:
                return
            
            # Choose vertex with highest degree in candidates
            # Use coloring-based upper bound
            # Simple bound: len(candidates)
            upper_bound = len(current_set) + len(candidates)
            if upper_bound <= best_size:
                return
            
            # Pick a vertex to branch on
            # Heuristic: pick vertex with max degree in candidate subgraph
            max_deg = -1
            chosen = None
            for v in candidates:
                deg_in_cand = sum(1 for u in candidates if u in neighbor_list[v])
                if deg_in_cand > max_deg:
                    max_deg = deg_in_cand
                    chosen = v
            
            if chosen is None:
                # All isolated, take all
                if len(current_set) + len(candidates) > best_size:
                    best_size = len(current_set) + len(candidates)
                    best_set = current_set + candidates[:]
                return
            
            # Branch 1: include chosen
            new_set = current_set + [chosen]
            # Remove chosen and its neighbors from candidates
            new_candidates = [v for v in candidates if v != chosen and v not in neighbor_list[chosen]]
            dfs(new_candidates, new_set)
            
            # Branch 2: exclude chosen
            new_candidates2 = [v for v in candidates if v != chosen]
            dfs(new_candidates2, current_set)
        
        dfs(list(range(N4)), [])
        return best_size, best_set
    
    # The full B&B may be too slow. Use a hybrid: greedy + local improvement
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)
    
    def greedy_with_conflict_check():
        """Greedy independent set with min-conflict heuristic."""
        in_set = set()
        alive = set(range(N4))
        
        # Compute initial degrees
        deg = {i: len(neighbor_list[i]) for i in range(N4)}
        
        while alive:
            # Pick vertex with minimum degree among alive
            min_deg = min(deg[i] for i in alive)
            cands = [i for i in alive if deg[i] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set.add(chosen)
            to_remove = {chosen} | (set(neighbor_list[chosen]) & alive)
            for r in to_remove:
                alive.discard(r)
                for n in neighbor_list[r]:
                    if n in alive:
                        deg[n] -= 1
        
        return in_set
    
    def local_search(in_set_init, max_time=5.0):
        """Local search with add/remove/swap moves."""
        in_set = set(in_set_init)
        current_size = len(in_set)
        local_best_size = current_size
        local_best_in_set = set(in_set)
        
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        sa_deadline = time.time() + max_time
        
        not_in = set(range(N4)) - in_set
        
        while time.time() < sa_deadline:
            if not not_in:
                break
            
            v = list(not_in)[rng.integers(len(not_in))]
            
            # Find neighbors of v that are in the set
            nbrs_in_set = [u for u in neighbor_list[v] if u in in_set]
            delta = 1 - len(nbrs_in_set)
            
            if delta >= 0 or rng.random() < np.exp(delta / T):
                for u in nbrs_in_set:
                    in_set.discard(u)
                    not_in.add(u)
                in_set.add(v)
                not_in.discard(v)
                current_size += delta
                
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = set(in_set)
            
            T *= cooling
            if T < min_T:
                T = min_T
        
        return local_best_in_set, local_best_size
    
    # Multiple restarts
    n_restarts = 0
    max_restarts = 500
    
    while time.time() < deadline - 3 and n_restarts < max_restarts:
        n_restarts += 1
        
        in_set = greedy_with_conflict_check()
        current_size = len(in_set)
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = np.zeros(N4, dtype=bool)
            for i in in_set:
                best_in_set[i] = True
        
        remaining = deadline - 3 - time.time()
        if remaining < 1:
            break
        
        sa_time = min(remaining * 0.3, 3.0)
        if sa_time < 0.3:
            sa_time = 0.3
        
        sa_result, sa_size = local_search(in_set, max_time=sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = np.zeros(N4, dtype=bool)
            for i in sa_result:
                best_in_set[i] = True
        
        if best_size >= 143:
            break
    
    # Combine with weight-1 vectors
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = v4_arr[final_v4_indices]
    
    final_vectors = np.vstack([v1_arr, result_v4])
    
    return final_vectors

# EVOLVE-BLOCK-END
