"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _triangle_free_check(n, S_set, neg_S_set):
    """Check if S is triangle-free. Returns True if no triangle exists."""
    for x in S_set:
        for y in S_set:
            if x == y:
                continue
            z = (x + y) % n
            if z in S_set or z in neg_S_set:
                return False
    return True

def _add_element_safe(n, S_set, neg_S_set, x):
    """Check if adding x (and -x) to S preserves triangle-freeness.
    A triangle is formed if there exists y in S such that (x+y) mod n is in S or -S.
    Also need to check that x itself doesn't create issues with existing pairs.
    """
    neg_x = (-x) % n
    # Check: for each y in S, is (x+y) mod n in S or -S?
    # Also: for each y in S, is (neg_x+y) mod n in S or -S?
    # Also: is x in S or -S already? (shouldn't be)
    # Also: is neg_x in S or -S already? (shouldn't be)
    if x in S_set or x in neg_S_set:
        return False
    if neg_x in S_set or neg_x in neg_S_set:
        return False
    
    for y in S_set:
        z1 = (x + y) % n
        if z1 in S_set or z1 in neg_S_set:
            return False
        z2 = (neg_x + y) % n
        if z2 in S_set or z2 in neg_S_set:
            return False
    return True

def _remove_element(n, S_set, neg_S_set, x):
    """Remove x and -x from S. Returns new sets."""
    new_S = set(S_set)
    new_neg = set(neg_S_set)
    if x in new_S:
        new_S.discard(x)
    if x in new_neg:
        new_neg.discard(x)
    neg_x = (-x) % n
    if neg_x in new_S:
        new_S.discard(neg_x)
    if neg_x in new_neg:
        new_neg.discard(neg_x)
    return new_S, new_neg

def _max_independent_set(n, S_set):
    """Compute alpha of circulant graph using branch and bound with bitmasks."""
    adj_mask = [0] * n
    for i in range(n):
        for s in S_set:
            j = (i + s) % n
            adj_mask[i] |= (1 << j)
    
    full_mask = (1 << n) - 1
    comp_mask = [0] * n
    for i in range(n):
        comp_mask[i] = full_mask ^ (1 << i) ^ adj_mask[i]
    
    best = [0]
    
    def dfs(candidates, current_size):
        if current_size + candidates.bit_count() <= best[0]:
            return
        if candidates == 0:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Pick vertex with fewest neighbors in candidate set (for independence, we want vertices with few neighbors)
        # Actually, for finding large independent set, pick vertex that allows most other vertices
        # Heuristic: pick vertex with minimum degree in the induced subgraph on candidates
        cands_list = []
        tmp = candidates
        while tmp:
            lsb = tmp & (-tmp)
            v = lsb.bit_length() - 1
            cands_list.append(v)
            tmp ^= lsb
        
        # Pick vertex with fewest neighbors among candidates
        best_v = -1
        min_deg = n + 1
        for v in cands_list:
            deg = (candidates & adj_mask[v]).bit_count()
            if deg < min_deg:
                min_deg = deg
                best_v = v
        
        v = best_v
        # Include v: candidates become non-neighbors of v
        dfs(candidates & comp_mask[v] & ~(1 << v), current_size + 1)
        # Exclude v
        dfs(candidates & ~(1 << v), current_size)
    
    dfs(full_mask, 0)
    return best[0]

def _build_edges(n, S_set):
    """Build edge list for circulant graph."""
    edges = []
    for i in range(n):
        for s in S_set:
            j = (i + s) % n
            if j > i:
                edges.append([i, j])
    return edges

def _local_search_circulant(n, initial_S, time_limit):
    """Local search over symmetric connection sets to maximize n while keeping alpha < 10."""
    start_time = time.time()
    
    # Initialize
    S = set(initial_S)
    neg_S = {(-x) % n for x in S}
    
    best_alpha = float('inf')
    best_S = set(S)
    
    # Current state
    cur_S = set(S)
    cur_neg_S = set(neg_S)
    
    # Try to minimize alpha by adding elements (making graph denser reduces independence number)
    # But we must maintain triangle-freeness
    
    half = n // 2
    all_candidates = list(range(1, half + 1))
    
    # Start with a sparse set, try to add elements to reduce alpha
    # First, compute initial alpha
    if len(cur_S) > 0:
        cur_alpha = _max_independent_set(n, cur_S)
        if cur_alpha < best_alpha:
            best_alpha = cur_alpha
            best_S = set(cur_S)
    
    # Local search: try adding elements to reduce alpha
    for iteration in range(2000):
        if time.time() > time_limit:
            break
        
        # Try adding a random candidate element
        if len(cur_S) < n - 1:
            available = [x for x in all_candidates if x not in cur_S and x not in cur_neg_S]
            if available:
                x = random.choice(available)
                if _add_element_safe(n, cur_S, cur_neg_S, x):
                    new_S = cur_S | {x, (-x) % n}
                    new_neg = cur_neg_S | {x, (-x) % n}
                    new_alpha = _max_independent_set(n, new_S)
                    if new_alpha < best_alpha:
                        best_alpha = new_alpha
                        best_S = set(new_S)
                        cur_S = new_S
                        cur_neg_S = new_neg
                    elif new_alpha <= best_alpha and random.random() < 0.3:
                        # Accept with some probability to escape local optima
                        cur_S = new_S
                        cur_neg_S = new_neg
        
        # Try removing an element (to allow future additions)
        if len(cur_S) > 2:
            x = random.choice(list(cur_S))
            cur_S, cur_neg_S = _remove_element(n, cur_S, cur_neg_S, x)
    
    return best_S, best_alpha

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    best_n = 0
    best_edges = []
    
    # Try to find largest valid graph
    for n in range(40, 25, -1):
        if time.time() > deadline:
            break
        
        found = False
        
        # Strategy: Local search over circulant connection sets
        for seed_offset in range(30):
            if time.time() > deadline:
                break
            if found:
                break
            
            random.seed(n * 10000 + seed_offset)
            
            # Initialize with a small random set
            half = n // 2
            available = list(range(1, half + 1))
            random.shuffle(available)
            
            # Start with 2-4 random elements
            init_size = random.choice([2, 3, 4])
            init_S = set(available[:init_size])
            init_neg = {(-x) % n for x in init_S}
            
            # Verify initial set is triangle-free
            if not _triangle_free_check(n, init_S, init_neg):
                continue
            
            # Run local search
            best_S, best_alpha = _local_search_circulant(n, init_S, time.time() + 3.0)
            
            if best_alpha < 10 and len(best_S) > 0:
                best_n = n
                best_edges = _build_edges(n, best_S)
                found = True
                break
        
        if found:
            break
    
    # Fallback: known valid small graphs
    if best_n < 26:
        # Use a simple construction
        n0 = 25
        best_edges = []
        for i in range(n0):
            for j in range(i + 1, n0):
                if (j - i) % 5 in [1, 4]:
                    best_edges.append([i, j])
        best_n = n0
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
