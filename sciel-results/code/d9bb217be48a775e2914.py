import time
import random

N = 40
_DEADLINE = time.time() + 85

def _find_diff_basis(target_size, max_val, seed=42):
    """Find a set S of size target_size from 1..max_val-1 such that
    pairwise differences cover 1..max_val-2. Must include 1 and max_val-1."""
    target = set(range(1, max_val - 1))  # 1..38
    rng = random.Random(seed)
    
    # Bitmask for differences 1..38
    # Bit i represents difference i+1? Or just use a set for clarity and switch to int if slow.
    # 38 bits fits in an int.
    # Let's use an integer bitmask where bit d is set if difference d is covered.
    
    def compute_covered(S):
        s_list = sorted(S)
        mask = 0
        for i in range(len(s_list)):
            for j in range(i + 1, len(s_list)):
                d = s_list[j] - s_list[i]
                if 1 <= d <= max_val - 2:
                    mask |= (1 << d)
        return mask
    
    target_mask = 0
    for d in range(1, max_val - 1):
        target_mask |= (1 << d)
        
    # Always include 1 and max_val-1
    fixed = {1, max_val - 1}
    remaining_size = target_size - len(fixed)
    
    best = None
    best_score = -1
    best_mask = 0
    
    for attempt in range(2000):
        if time.time() > _DEADLINE - 5:
            break
        # Choose remaining elements
        candidates = [x for x in range(2, max_val - 1) if x not in fixed]
        S = set(fixed)
        S.update(rng.sample(candidates, remaining_size))
        mask = compute_covered(S)
        score = bin(mask & target_mask).count('1')
        
        if score > best_score:
            best_score = score
            best = set(S)
            best_mask = mask
        
        if (mask & target_mask) == target_mask:
            return sorted(S)
        
        # Local search
        improved = True
        current_mask = mask
        current_score = score
        while improved and time.time() < _DEADLINE - 5:
            improved = False
            s_list = sorted(S)
            missing = target_mask & ~current_mask
            if not missing:
                return sorted(S)
            
            # Try to add a new element that covers the most missing diffs
            # Or swap an existing element
            # Swap is better for fixed size
            
            for idx in range(len(s_list)):
                old = s_list[idx]
                if old in fixed:
                    continue
                for nv in range(2, max_val - 1):
                    if nv in S and nv != old:
                        continue
                    S.remove(old)
                    S.add(nv)
                    new_mask = compute_covered(S)
                    new_score = bin(new_mask & target_mask).count('1')
                    if new_score > current_score:
                        current_score = new_score
                        current_mask = new_mask
                        improved = True
                        break
                    else:
                        S.remove(nv)
                        S.add(old)
                if (current_mask & target_mask) == target_mask:
                    return sorted(S)
                if improved:
                    break # Restart loop with new state
    
    if best is not None and (best_mask & target_mask) == target_mask:
        return sorted(best)
    return None

def construct():
    # Find difference basis of size 10
    S = _find_diff_basis(10, N, seed=42)
    if S is None:
        S = _find_diff_basis(10, N, seed=123)
    if S is None:
        S = _find_diff_basis(10, N, seed=999)
    if S is None:
        # Try size 11
        S = _find_diff_basis(11, N, seed=42)
    if S is None:
        # Last resort: use all of 1..39
        S = list(range(1, N))
    
    S_set = set(S)
    edges = []
    edges_set = set()
    
    def add_edge(a, b):
        if a == b:
            return
        if a > b:
            a, b = b, a
        if (a, b) not in edges_set:
            edges_set.add((a, b))
            edges.append((a, b))
    
    # Path edges 1-2, ..., 38-39
    for i in range(1, N - 1):
        add_edge(i, i + 1)
    # Spoke edges 0-s
    for s in S:
        add_edge(0, s)
    
    cycles = []
    valid = True
    for L in range(3, N + 1):
        diff = L - 2
        found = False
        for u in sorted(S_set):
            v = u + diff
            if v in S_set:
                # Cycle: 0, u, u+1, ..., v
                cycle = [0] + list(range(u, v + 1))
                # Verify all edges exist
                ok = True
                for i in range(len(cycle) - 1):
                    a, b = cycle[i], cycle[i + 1]
                    if a > b:
                        a, b = b, a
                    if (a, b) not in edges_set:
                        ok = False
                        break
                if ok:
                    a, b = cycle[-1], cycle[0]
                    if a > b:
                        a, b = b, a
                    if (a, b) not in edges_set:
                        ok = False
                if ok:
                    cycles.append(cycle)
                    found = True
                    break
        if not found:
            valid = False
            break
    
    if not valid:
        # Fallback: full star + path (guaranteed valid)
        e2 = [(0, i) for i in range(1, N)] + [(i, i + 1) for i in range(1, N - 1)]
        c2 = []
        for L in range(3, N + 1):
            c2.append(list(range(L)))
        return {'edges': e2, 'cycles': c2}
    
    return {'edges': edges, 'cycles': cycles}
