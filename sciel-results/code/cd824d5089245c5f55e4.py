"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import itertools

def _popcount(x):
    return x.bit_count()

def _is_regular_6(adj_mask, sub_mask, verts6):
    """Check if 6-vertex subset induces regular subgraph."""
    first_deg = -1
    for v in verts6:
        deg = _popcount(adj_mask[v] & sub_mask)
        if first_deg < 0:
            first_deg = deg
        elif deg != first_deg:
            return False
    return True

def _is_regular_k(adj_mask, sub_mask, verts):
    """Check if k-vertex subset induces regular subgraph."""
    k = len(verts)
    if k < 6:
        return False
    first_deg = -1
    for v in verts:
        deg = _popcount(adj_mask[v] & sub_mask)
        if first_deg < 0:
            first_deg = deg
        elif deg != first_deg:
            return False
    return True

def _find_any_violation(n, adj_mask):
    """Find any regular induced subgraph on 6+ vertices. Returns (verts, k) or (None, 0)."""
    for k in range(6, n + 1):
        for comb in itertools.combinations(range(n), k):
            sub_mask = 0
            for v in comb:
                sub_mask |= (1 << v)
            if _is_regular_k(adj_mask, sub_mask, comb):
                return (comb, k)
    return (None, 0)

def _local_search_fast(n, seed, max_iters=50000):
    """Fast local search with incremental violation tracking for size-6 subsets."""
    rng = random.Random(seed)
    
    # Initialize random graph
    adj_mask = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
    
    # Precompute size-6 subsets
    subsets6 = []
    for comb in itertools.combinations(range(n), 6):
        mask = 0
        for v in comb:
            mask |= (1 << v)
        subsets6.append((comb, mask))
    num_subsets = len(subsets6)
    
    # Track which size-6 subsets are regular
    regular_flags = [False] * num_subsets
    regular_count = 0
    for idx, (verts, mask) in enumerate(subsets6):
        if _is_regular_6(adj_mask, mask, verts):
            regular_flags[idx] = True
            regular_count += 1
    
    # Check larger subsets
    def _check_larger():
        for k in range(7, n + 1):
            for comb in itertools.combinations(range(n), k):
                sub_mask = 0
                for v in comb:
                    sub_mask |= (1 << v)
                if _is_regular_k(adj_mask, sub_mask, comb):
                    return True
        return False
    
    if regular_count == 0 and not _check_larger():
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj_mask[i] & (1 << j):
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
    
    # Build index: for each pair (u,v), which size-6 subsets contain both
    subsets_containing = {}
    for idx, (verts, mask) in enumerate(subsets6):
        for i in range(6):
            for j in range(i + 1, 6):
                u, v = verts[i], verts[j]
                key = (min(u, v), max(u, v))
                if key not in subsets_containing:
                    subsets_containing[key] = []
                subsets_containing[key].append(idx)
    
    for it in range(max_iters):
        if regular_count == 0:
            # No size-6 violations, check larger
            if not _check_larger():
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj_mask[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
            # There's a larger violation; pick random edge to flip
            u, v = rng.sample(range(n), 2)
        else:
            # Pick a random regular size-6 subset and flip an edge within it
            reg_idx = rng.randrange(num_subsets)
            while not regular_flags[reg_idx]:
                reg_idx = (reg_idx + 1) % num_subsets
            verts = subsets6[reg_idx][0]
            u, v = rng.sample(verts, 2)
        
        # Flip edge (u, v)
        if adj_mask[u] & (1 << v):
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
        else:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        # Update regularity of size-6 subsets containing both u and v
        key = (min(u, v), max(u, v))
        if key in subsets_containing:
            for idx in subsets_containing[key]:
                old_flag = regular_flags[idx]
                verts, mask = subsets6[idx]
                new_flag = _is_regular_6(adj_mask, mask, verts)
                regular_flags[idx] = new_flag
                if old_flag and not new_flag:
                    regular_count -= 1
                elif not old_flag and new_flag:
                    regular_count += 1
    
    return None

def _try_random(n, seed, num_tries=20):
    """Try random graphs with fast validation."""
    rng = random.Random(seed)
    for attempt in range(num_tries):
        adj_mask = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.5:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        
        # Check size-6 subsets
        valid = True
        for comb in itertools.combinations(range(n), 6):
            sub_mask = 0
            for v in comb:
                sub_mask |= (1 << v)
            first_deg = -1
            reg = True
            for v in comb:
                deg = _popcount(adj_mask[v] & sub_mask)
                if first_deg < 0:
                    first_deg = deg
                elif deg != first_deg:
                    reg = False
                    break
            if reg:
                valid = False
                break
        if not valid:
            continue
        
        # Check larger subsets
        for k in range(7, n + 1):
            found = False
            for comb in itertools.combinations(range(n), k):
                sub_mask = 0
                for v in comb:
                    sub_mask |= (1 << v)
                first_deg = -1
                reg = True
                for v in comb:
                    deg = _popcount(adj_mask[v] & sub_mask)
                    if first_deg < 0:
                        first_deg = deg
                    elif deg != first_deg:
                        reg = False
                        break
                if reg:
                    found = True
                    break
            if found:
                valid = False
                break
        
        if valid:
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj_mask[i] & (1 << j):
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
    return None

# EVOLVE-BLOCK-END

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s kill
    
    best_result = {'n': 5, 'edges': []}
    
    # Try n=21 first, then n=20
    for n in [21, 20, 19, 18]:
        if time.time() > deadline - 10:
            break
        
        # Phase 1: Quick random tries
        result = _try_random(n, seed=n * 7919, num_tries=10)
        if result is not None:
            best_result = result
            continue
        
        # Phase 2: Local search with multiple restarts
        if time.time() > deadline - 15:
            break
        for attempt in range(30):
            if time.time() > deadline - 10:
                break
            seed = n * 10000 + attempt * 997
            result = _local_search_fast(n, seed, max_iters=30000)
            if result is not None:
                best_result = result
                break
        
        if best_result['n'] >= n:
            continue
    
    return best_result
# EVOLVE-BLOCK-END
