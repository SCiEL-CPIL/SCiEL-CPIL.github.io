"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import itertools
import time
from itertools import combinations, product

D = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Strategy:
    # 1. Formulate support selection as a packing problem in H(11, 4) with min Hamming distance 6.
    #    This is equivalent to finding a Maximum Independent Set (MIS) in the graph where
    #    nodes are weight-4 supports and edges connect supports with intersection size >= 3.
    # 2. Use a vectorized greedy construction with randomized tie-breaking to find a large independent set.
    # 3. Use iterative 1-swap local search to optimize the code size.
    # 4. Expand each selected support into all 16 sign patterns.
    # 5. Verify the resulting vector set satisfies the distance constraint.
    
    supports = list(combinations(range(D), 4))
    N_sup = len(supports)
    
    # Precompute intersection sizes or incompatibility matrix
    # incompat[i, j] = 1 if |sup_i ∩ sup_j| >= 3
    incompat = np.zeros((N_sup, N_sup), dtype=np.int8)
    
    # Efficiently compute incompatibility
    # Represent supports as bitmasks for fast intersection
    masks = []
    for s in supports:
        m = 0
        for idx in s:
            m |= (1 << idx)
        masks.append(m)
        
    for i in range(N_sup):
        if time.time() > deadline - 20.0:
            break
        for j in range(i + 1, N_sup):
            inter = masks[i] & masks[j]
            # Count bits in inter
            if bin(inter).count('1') >= 3:
                incompat[i, j] = 1
                incompat[j, i] = 1
                
    # Precompute neighbors as lists for faster access in local search
    neighbors = [np.where(incompat[i] == 1)[0] for i in range(N_sup)]
    
    def greedy_min_degree(available, rng):
        """
        Greedy MIS using minimum degree heuristic.
        """
        selected = []
        avail = set(available)
        
        while avail:
            if time.time() > deadline - 10.0:
                break
            
            # Find node in avail with minimum degree in the induced subgraph on avail
            min_deg = float('inf')
            candidates = []
            for u in avail:
                # Count neighbors in avail
                deg = 0
                for v in neighbors[u]:
                    if v in avail:
                        deg += 1
                if deg < min_deg:
                    min_deg = deg
                    candidates = [u]
                elif deg == min_deg:
                    candidates.append(u)
            
            # Pick random from candidates to break symmetry
            u = rng.choice(candidates)
            selected.append(u)
            
            # Remove u and its neighbors from avail
            to_remove = {u}
            for v in neighbors[u]:
                if v in avail:
                    to_remove.add(v)
            avail -= to_remove
            
        return selected

    def local_search_1swap(selected, rng):
        """
        1-2 swap: remove one node v, add two compatible nodes u1, u2.
        Optimized with bitset operations.
        """
        selected_set = set(selected)
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            selected_list = list(selected_set)
            rng.shuffle(selected_list)
            
            for v in selected_list:
                if time.time() > deadline - 10.0:
                    break
                
                # Build incompatibility mask for selected \ {v}
                # A node x is a potential add if it's not in selected and not incompatible with any y in selected\{v}
                blocked = set()
                for y in selected_set:
                    if y == v:
                        continue
                    for nb in neighbors[y]:
                        blocked.add(nb)
                
                potential_adds = [x for x in range(N_sup) if x not in selected_set and x not in blocked]
                
                if len(potential_adds) >= 2:
                    found = False
                    for i in range(len(potential_adds)):
                        u1 = potential_adds[i]
                        for j in range(i + 1, len(potential_adds)):
                            u2 = potential_adds[j]
                            if not incompat[u1, u2]:
                                selected_set.discard(v)
                                selected_set.add(u1)
                                selected_set.add(u2)
                                improved = True
                                found = True
                                break
                        if found:
                            break
                if improved:
                    break
                    
        return list(selected_set)

    best_iset = None
    best_size = 0
    
    rng = np.random.default_rng(42)
    
    # Run multiple restarts
    n_restarts = 40
    for r in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
            
        avail = set(range(N_sup))
        
        # Randomize the order slightly? Or just use min-degree greedy.
        # The min-degree greedy is deterministic given the tie-breaking.
        # We use rng.choice for tie-breaking.
        
        iset = greedy_min_degree(avail, rng)
        
        # Local search
        iset = local_search_1swap(iset, rng)
        
        size = len(iset)
        if size > best_size:
            best_size = size
            best_iset = iset.copy()
            
    if best_iset is None:
        # Fallback: return a single vector
        v = np.zeros(D, dtype=np.int64)
        v[0] = 2
        return v.reshape(1, -1)
        
    # Generate vectors
    vectors = []
    for i in best_iset:
        sup = supports[i]
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(sup, signs):
                v[idx] = s
            vectors.append(v)
    
    # Add Type A vectors: ±2e_i for all i in 0..D-1
    # These have norm 2 and are universally compatible with Type D vectors
    for i in range(D):
        v = np.zeros(D, dtype=np.int64)
        v[i] = 2
        vectors.append(v)
        v2 = np.zeros(D, dtype=np.int64)
        v2[i] = -2
        vectors.append(v2)
    
    if not vectors:
        return np.array([[]], dtype=np.int64)

    result = np.array(vectors, dtype=np.int64)
    
    # Optional: Verify
    # M = result @ result.T
    # np.fill_diagonal(M, 0)
    # assert np.all(M <= 2)
    
    return result

# EVOLVE-BLOCK-END
