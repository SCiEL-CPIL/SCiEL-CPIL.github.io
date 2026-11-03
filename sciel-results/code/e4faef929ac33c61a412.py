"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _generate_supports():
    """Generate all 4-element subsets of {0,...,10} as bitmasks."""
    supports = []
    indices = range(D)
    for comb in combinations(indices, 4):
        mask = 0
        for idx in comb:
            mask |= (1 << idx)
        supports.append(mask)
    return np.array(supports, dtype=np.int64)

def _compute_conflict_matrix(supports):
    """
    Compute conflict matrix where conflict[i,j]=True if supports i and j share 3 indices.
    Two supports conflict if their intersection has size 3 (since 4+4-3=5 > 4, they can't both be in a valid set
    if we require pairwise intersection <= 2 for the sign assignments to work out).
    Actually, the condition for two weight-4 vectors u, v to have |u.v| <= 1 is that their supports intersect in <= 2 elements.
    If they intersect in 3 elements, we can choose signs to make u.v = -1 or 1 (by flipping one sign in the overlap),
    but we cannot make it 0. The requirement is min pairwise distance >= max norm.
    Norm^2 = 4. Distance^2 = ||u||^2 + ||v||^2 - 2u.v = 8 - 2u.v.
    We need 8 - 2u.v >= 4 => 2u.v <= 4 => u.v <= 2.
    Since max |u.v| is 4 (if identical), and we want distinct vectors, max u.v for distinct is 3 (if 3 overlap with same sign).
    Wait, if 3 overlap, max u.v is 3. 8 - 2(3) = 2 < 4. So distance^2 = 2 < 4. Invalid.
    So we MUST have u.v <= 1.
    If intersection is 3, min |u.v| is 1 (by choosing signs to oppose one overlap). So u.v can be 1 or -1.
    If u.v = 1, distance^2 = 8 - 2 = 6 >= 4. Valid!
    If u.v = -1, distance^2 = 8 + 2 = 10 >= 4. Valid!
    So intersection of 3 is OK as long as we pick signs carefully?
    No, the constraint is on the SET of vectors. We need ALL pairs to satisfy distance >= norm.
    For a pair with intersection 3, we can always choose signs such that u.v = 1 or -1.
    But we have to choose signs for ALL vectors consistently.
    This becomes a constraint satisfaction problem on the signs.
    
    However, the current program's strategy was:
    1. Find MIS on supports where edge = intersection 3.
    This ensures NO two supports in the set intersect in 3.
    So all pairs have intersection <= 2.
    If intersection <= 2, max |u.v| <= 2.
    If u.v = 2, distance^2 = 8 - 4 = 4 >= 4. Valid.
    If u.v = -2, distance^2 = 8 + 4 = 12 >= 4. Valid.
    So any sign assignment works for supports with intersection <= 2.
    
    So the MIS on the "intersection 3" graph is a valid sufficient condition.
    Is it necessary? No, we could potentially include supports with intersection 3 if we carefully assign signs.
    But that's much harder. The MIS approach is clean and yields a large set.
    
    Let's stick with the MIS approach but improve the solver.
    """
    n = len(supports)
    # Compute pairwise intersection sizes
    # supports is (n, D) but we use bitmasks.
    # Let's use the bitmask representation for fast intersection counting.
    # Actually, the input supports are already bitmasks in the list.
    # Let's assume supports is a list of ints (bitmasks).
    
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            inter = supports[i] & supports[j]
            if inter.bit_count() == 3:
                adj[i].add(j)
                adj[j].add(i)
    return adj

def _greedy_mis(adj):
    """Greedy Maximum Independent Set heuristic."""
    n = len(adj)
    alive = set(range(n))
    mis = []
    while alive:
        # Pick node with min degree in alive
        v = min(alive, key=lambda x: len(adj[x].intersection(alive)))
        mis.append(v)
        alive.difference_update(adj[v])
        alive.remove(v)
    return mis

def _local_search_mis(adj, initial_mis):
    """
    Local search to improve MIS.
    Try to add nodes one by one, or swap.
    """
    mis = set(initial_mis)
    alive = set(range(len(adj))) - mis
    improved = True
    while improved:
        improved = False
        # Try to add a node from alive
        for v in list(alive):
            # Check if v can be added (no neighbors in mis)
            if not adj[v].intersection(mis):
                mis.add(v)
                alive.remove(v)
                # Remove neighbors of v from alive (they can't be added now)
                # Actually, if we add v, its neighbors can still be in alive, but they can't be added
                # unless we remove v. But since we're doing greedy add, we just mark them as not addable?
                # No, if we add v, we must remove its neighbors from the *potential* set, but they are already in alive.
                # We just need to ensure we don't add two adjacent nodes.
                # So when we add v, we should remove its neighbors from alive?
                # No, that's for building a MIS from scratch.
                # For local search, if we add v, we can't add its neighbors.
                # So we should remove neighbors from alive to prevent adding them later in this pass.
                for u in adj[v]:
                    if u in alive:
                        alive.remove(u)
                improved = True
                break # Restart scan
        if not improved:
            # Try swaps? Too complex for now.
            break
    return list(mis)

def _max_independent_set(n, adj, deadline):
    """
    Combine greedy and local search.
    """
    start = time.time()
    mis = _greedy_mis(adj)
    while time.time() < deadline - 5:
        new_mis = _local_search_mis(adj, mis)
        if len(new_mis) > len(mis):
            mis = new_mis
        else:
            # Try a different greedy order?
            # Shuffle alive set?
            break
    return mis

def construct():
    deadline = time.time() + 110
    
    # 1. Generate supports for m=4
    supports_list = []
    indices = range(D)
    for comb in combinations(indices, 4):
        mask = 0
        for idx in comb:
            mask |= (1 << idx)
        supports_list.append(mask)
    
    n_sup = len(supports_list) # 330
    
    # 2. Build adjacency list for conflict graph (intersection size 3)
    adj = [set() for _ in range(n_sup)]
    for i in range(n_sup):
        for j in range(i + 1, n_sup):
            inter = supports_list[i] & supports_list[j]
            if inter.bit_count() == 3:
                adj[i].add(j)
                adj[j].add(i)
    
    # 3. Find Maximum Independent Set
    mis_nodes = _max_independent_set(n_sup, adj, deadline)
    
    # 4. Generate vectors for MIS supports
    vectors = []
    for idx in mis_nodes:
        mask = supports_list[idx]
        inds = [i for i in range(D) if (mask >> i) & 1]
        for sign_mask in range(16):
            v = np.zeros(D, dtype=np.int64)
            for j, idx_val in enumerate(inds):
                if sign_mask & (1 << j):
                    v[idx_val] = 1
                else:
                    v[idx_val] = -1
            vectors.append(v)
    
    S4 = np.array(vectors, dtype=np.int64)
    
    # 5. Try to add Shell 2 vectors
    shell2_vecs = []
    for i in range(D):
        for j in range(i+1, D):
            # e_i - e_j and e_j - e_i
            v1 = np.zeros(D, dtype=np.int64)
            v1[i] = 1; v1[j] = -1
            shell2_vecs.append(v1)
            v2 = np.zeros(D, dtype=np.int64)
            v2[i] = -1; v2[j] = 1
            shell2_vecs.append(v2)
    shell2 = np.array(shell2_vecs, dtype=np.int64)
    
    if len(S4) > 0:
        dots_24 = shell2 @ S4.T
        max_ip_24 = dots_24.max(axis=1)
        valid_2_mask = max_ip_24 <= 1
        S2_candidates = shell2[valid_2_mask]
        
        n2 = len(S2_candidates)
        if n2 > 0:
            dots_22 = S2_candidates @ S2_candidates.T
            selected2 = []
            alive2 = list(range(n2))
            while alive2:
                v_idx = alive2[0]
                selected2.append(v_idx)
                conflicts = np.where(dots_22[v_idx] > 0)[0]
                alive2 = [i for i in alive2 if i not in conflicts and i != v_idx]
            
            S2 = S2_candidates[selected2]
            result = np.vstack([S4, S2])
        else:
            result = S4
    else:
        result = S4
        
    return result

# EVOLVE-BLOCK-END
