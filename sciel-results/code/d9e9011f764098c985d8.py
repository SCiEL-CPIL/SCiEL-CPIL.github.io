import numpy as np
import time
import random
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11

    # Strategy: Use weight-4 supports.
    # For a support of size w=4, there are 2^4=16 sign patterns.
    # The minimum distance squared between two distinct sign patterns in the same support is:
    # dist^2 = ||u - v||^2 = ||u||^2 + ||v||^2 - 2 u.v = 4 + 4 - 2 u.v = 8 - 2 u.v
    # u.v is the dot product. For binary signs, u.v = sum(s_i t_i).
    # The minimum distance occurs when they differ in the most positions? No.
    # u.v is maximized when signs are most similar.
    # If u and v differ in k positions, u.v = 4 - 2k.
    # dist^2 = 8 - 2(4 - 2k) = 8 - 8 + 4k = 4k.
    # Minimum k=1 (differ in 1 position) -> dist^2 = 4.
    # So if we take ALL sign patterns for a support, the min intra-support distance is 4.
    # This means we can set max_norm^2 = 4.
    # Then we need all pairwise distances >= 4.
    
    # For two different supports S, T:
    # dist^2 = 4 + 4 - 2 u.v = 8 - 2 u.v.
    # We need 8 - 2 u.v >= 4 => 4 >= 2 u.v => u.v <= 2.
    # u.v = sum_{i in S∩T} s_i t_i.
    # Let I = S∩T, |I| = m.
    # u.v can range from -m to m in steps of 2.
    # We need u.v <= 2.
    
    # If m=0: u.v=0. OK.
    # If m=1: u.v=±1. OK.
    # If m=2: u.v=-2,0,2. OK (max 2).
    # If m=3: u.v=-3,-1,1,3. We need u.v <= 2. So u.v=3 is forbidden.
    #    u.v=3 means all signs in intersection are identical.
    # If m=4: same support. Handled above.
    
    # So, two supports S, T with |S∩T|=3 are compatible ONLY IF we do not choose sign patterns
    # that agree on all 3 intersection indices.
    # This complicates things. We can't just take all sign patterns for arbitrary supports.
    
    # However, if we restrict to a set of supports such that |S∩T| <= 2 for all pairs,
    # then we can include ALL 16 sign patterns for each support.
    # This reduces to finding a Maximum Independent Set (MIS) in the graph where
    # nodes are 4-subsets of [11], and edges connect subsets with intersection size >= 3.
    
    # This is exactly the strategy from Inspiration 1, which scored 2.5455.
    # The current program scores 1.0 (220 vectors).
    # Inspiration 1 found ~560 vectors (2.5455 * 220 approx? No, seed is 220. 2.5455 * 220 = 560).
    # Let's verify the score logic.
    # Score = k / seed_k.
    # If seed is 220, and score is 2.5455, then k = 220 * 2.5455 ≈ 560.
    # 16 * 35 = 560. So it found 35 supports.
    
    # Let's implement the MIS search on 4-subsets with intersection >= 3.
    
    supports = list(combinations(range(D), 4))
    N_sup = len(supports)
    sup_sets = [set(s) for s in supports]
    
    # Build incompatibility graph
    # incompat[i, j] = True if |sup_i ∩ sup_j| >= 3
    # Use bitmasks for fast intersection checking
    sup_masks = []
    for s in supports:
        mask = 0
        for idx in s:
            mask |= (1 << idx)
        sup_masks.append(mask)
    
    neighbors = [set() for _ in range(N_sup)]
    
    for i in range(N_sup):
        if time.time() > deadline - 10.0:
            break
        for j in range(i + 1, N_sup):
            # Count bits in intersection
            inter = sup_masks[i] & sup_masks[j]
            # Count set bits
            c = 0
            t = inter
            while t:
                c += 1
                t &= t - 1
            if c >= 3:
                neighbors[i].add(j)
                neighbors[j].add(i)
                
    def greedy_min_degree(available, rng):
        selected = set()
        avail = set(available)
        
        while avail:
            if time.time() > deadline - 5.0:
                break
            
            min_deg = float('inf')
            candidates = []
            for u in avail:
                deg = len(neighbors[u] & avail)
                if deg < min_deg:
                    min_deg = deg
                    candidates = [u]
                elif deg == min_deg:
                    candidates.append(u)
            
            u = rng.choice(candidates)
            selected.add(u)
            
            to_remove = {u} | (neighbors[u] & avail)
            avail -= to_remove
            
        return selected

    def local_search(selected, rng):
        """
        Try to improve the independent set using 1-2 swaps and 2-opt moves.
        """
        selected = set(selected)
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # Try 1-2 swap: remove v, add u1, u2
            for v in list(selected):
                if time.time() > deadline - 5.0:
                    break
                
                sel_minus_v = selected - {v}
                potential_adds = []
                for x in range(N_sup):
                    if x in selected or x == v:
                        continue
                    if not (neighbors[x] & sel_minus_v):
                        potential_adds.append(x)
                        
                L = len(potential_adds)
                found = False
                for i in range(L):
                    u1 = potential_adds[i]
                    for j in range(i + 1, L):
                        u2 = potential_adds[j]
                        if u2 not in neighbors[u1]:
                            selected.remove(v)
                            selected.add(u1)
                            selected.add(u2)
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            
            # Try adding a single node (1-0 swap)
            if not improved:
                for x in range(N_sup):
                    if time.time() > deadline - 5.0:
                        break
                    if x not in selected and not (neighbors[x] & selected):
                        selected.add(x)
                        improved = True
                        break

        return selected

    def simulated_annealing(selected, rng):
        """
        Simulated annealing to escape local optima.
        """
        selected = set(selected)
        T = 1.0
        cooling = 0.995
        min_T = 0.01
        
        current_size = len(selected)
        best_selected = set(selected)
        best_size = current_size
        
        steps = 0
        max_steps = 2000
        
        while T > min_T and steps < max_steps:
            if time.time() > deadline - 5.0:
                break
            
            steps += 1
            new_selected = set(selected)
            
            # Random move: remove a random node
            if new_selected and rng.random() < 0.5:
                v = rng.choice(list(new_selected))
                new_selected.remove(v)
                delta = -1
            # Random move: add a random node (even if it conflicts, to allow annealing)
            elif rng.random() < 0.8:
                x = rng.randint(0, N_sup - 1)
                if x not in new_selected:
                    new_selected.add(x)
                    delta = 1
                else:
                    delta = 0
            else:
                delta = 0
            
            if delta == 0:
                continue
                
            # Check if new_selected is a valid independent set
            valid = True
            for v in new_selected:
                if neighbors[v] & new_selected:
                    valid = False
                    break
            
            if valid:
                new_size = len(new_selected)
                delta_size = new_size - current_size
                if delta_size >= 0 or rng.random() < np.exp(delta_size / T):
                    selected = new_selected
                    current_size = new_size
                    if current_size > best_size:
                        best_size = current_size
                        best_selected = set(selected)
            else:
                # Penalty for invalid states
                # Count conflicts
                conflict_count = 0
                for v in new_selected:
                    conflict_count += len(neighbors[v] & new_selected)
                # This is expensive, so skip penalty for now
                pass
                
            T *= cooling
            
        return best_selected

    best_iset = None
    best_size = 0
    
    rng = np.random.default_rng(42)
    
    # Run multiple restarts
    n_restarts = 0
    while time.time() < deadline - 10.0:
        n_restarts += 1
        avail = set(range(N_sup))
        iset = greedy_min_degree(avail, rng)
        iset = local_search(iset, rng)
        iset = simulated_annealing(iset, rng)
        
        size = len(iset)
        if size > best_size:
            best_size = size
            best_iset = iset.copy()
            
        # Occasionally try a different initialization
        if n_restarts % 10 == 0:
            # Random initialization
            iset = set()
            for i in range(N_sup):
                if time.time() > deadline - 10.0:
                    break
                if rng.random() < 0.3:
                    if not (neighbors[i] & iset):
                        iset.add(i)
            iset = local_search(iset, rng)
            iset = simulated_annealing(iset, rng)
            size = len(iset)
            if size > best_size:
                best_size = size
                best_iset = iset.copy()

    if best_iset is None:
        return np.array([[]], dtype=np.int8)
        
    # Generate vectors
    vectors = []
    for i in best_iset:
        sup = supports[i]
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(sup, signs):
                v[idx] = s
            vectors.append(v)
            
    if not vectors:
        return np.array([[]], dtype=np.int8)

    return np.array(vectors, dtype=np.int8)
