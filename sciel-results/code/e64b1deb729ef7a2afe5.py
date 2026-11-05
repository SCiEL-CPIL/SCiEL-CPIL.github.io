import time
import random

N = 40
_DEADLINE = time.time() + 85

def _find_diff_basis(target_size, max_val, seed=42):
    """Find a set S of size target_size from 1..max_val-1 such that
    pairwise differences cover 1..max_val-2."""
    if time.time() > _DEADLINE - 5:
        return None
    
    target_mask = (1 << (max_val - 2)) - 1  # bits 0..(max_val-3) represent diffs 1..(max_val-2)
    
    def compute_covered(S):
        s_list = sorted(S)
        mask = 0
        for i in range(len(s_list)):
            for j in range(i + 1, len(s_list)):
                mask |= 1 << (s_list[j] - s_list[i] - 1)
        return mask
    
    def count_covered(mask):
        return bin(mask & target_mask).count('1')
    
    fixed = {1, max_val - 1}
    remaining_size = target_size - len(fixed)
    candidates = [x for x in range(2, max_val - 1) if x not in fixed]
    
    best = None
    best_score = -1
    
    rng = random.Random(seed)
    
    # Greedy initialization
    S = set(fixed)
    covered = compute_covered(S)
    for _ in range(remaining_size):
        best_candidate = None
        best_gain = -1
        for c in candidates:
            if c in S:
                continue
            S.add(c)
            new_covered = compute_covered(S)
            gain = count_covered(new_covered) - count_covered(covered)
            if gain > best_gain:
                best_gain = gain
                best_candidate = c
            S.remove(c)
        if best_candidate is not None:
            S.add(best_candidate)
            covered = compute_covered(S)
    
    # Local search
    improved = True
    while improved and time.time() < _DEADLINE - 5:
        improved = False
        s_list = sorted(S)
        covered = compute_covered(S)
        score = count_covered(covered)
        if covered & target_mask == target_mask:
            return sorted(S)
        
        for idx in range(len(s_list)):
            old = s_list[idx]
            if old in fixed:
                continue
            for nv in range(2, max_val - 1):
                if nv in S and nv != old:
                    continue
                S.remove(old)
                S.add(nv)
                new_covered = compute_covered(S)
                new_score = count_covered(new_covered)
                if new_score > score:
                    score = new_score
                    covered = new_covered
                    improved = True
                    break
                else:
                    S.remove(nv)
                    S.add(old)
            if covered & target_mask == target_mask:
                return sorted(S)
        
        # 2-swap move
        for i in range(len(s_list)):
            if s_list[i] in fixed: continue
            for j in range(i+1, len(s_list)):
                if s_list[j] in fixed: continue
                old1, old2 = s_list[i], s_list[j]
                for nv1 in range(2, max_val - 1):
                    if nv1 in S: continue
                    for nv2 in range(2, max_val - 1):
                        if nv2 in S or nv2 == nv1: continue
                        S.remove(old1)
                        S.remove(old2)
                        S.add(nv1)
                        S.add(nv2)
                        new_covered = compute_covered(S)
                        new_score = count_covered(new_covered)
                        if new_score > score:
                            score = new_score
                            covered = new_covered
                            improved = True
                            break
                        else:
                            S.remove(nv1)
                            S.remove(nv2)
                            S.add(old1)
                            S.add(old2)
                    if improved: break
                if improved: break
            if improved: break

    if best is None or score > best_score:
        best = set(S)
        best_score = score
        
    if best is not None and compute_covered(best) & target_mask == target_mask:
        return sorted(best)
    return None

def construct():
    # Try smaller sizes first
    for size in [7, 8, 9, 10]:
        S = _find_diff_basis(size, N, seed=42)
        if S is None:
            S = _find_diff_basis(size, N, seed=123)
        if S is not None:
            break
    
    if S is None:
        # Fallback to larger size
        S = _find_diff_basis(11, N, seed=42)
    if S is None:
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
            
    # Path edges
    for i in range(1, N - 1):
        add_edge(i, i + 1)
    # Spoke edges
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
                cycle = [0] + list(range(u, v + 1))
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
        # Fallback: full star + path
        e2 = [(0, i) for i in range(1, N)] + [(i, i + 1) for i in range(1, N - 1)]
        c2 = []
        for L in range(3, N + 1):
            c2.append(list(range(L)))
        return {'edges': e2, 'cycles': c2}
        
    return {'edges': edges, 'cycles': cycles}
