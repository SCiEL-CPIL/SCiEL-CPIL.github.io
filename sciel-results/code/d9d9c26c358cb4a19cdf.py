import time
import random
import numpy as np

N = 16
DEADLINE = 325.0
ALL_POINTS = [(x, y) for y in range(N) for x in range(N)]
N_TOTAL = N * N

def _idx(x, y):
    return y * N + x

def _xy(idx):
    return (idx % N, idx // N)

# Precompute squared distances
dist2 = np.zeros((N_TOTAL, N_TOTAL), dtype=np.int64)
for i in range(N_TOTAL):
    xi, yi = _xy(i)
    for j in range(i + 1, N_TOTAL):
        xj, yj = _xy(j)
        d = (xi - xj) ** 2 + (yi - yj) ** 2
        dist2[i, j] = d
        dist2[j, i] = d

# For each point p, precompute the set of pairs (a,b) with a<b such that {p,a,b} is isosceles
# Store as a list of (a,b) tuples
isos_pairs = [[] for _ in range(N_TOTAL)]
for p in range(N_TOTAL):
    for a in range(N_TOTAL):
        if a == p:
            continue
        for b in range(a + 1, N_TOTAL):
            if b == p:
                continue
            d_pa = dist2[p, a]
            d_pb = dist2[p, b]
            d_ab = dist2[a, b]
            if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                isos_pairs[p].append((a, b))

# For each pair (a,b), precompute the set of points c such that {a,b,c} is isosceles
# This is used for the no-isosceles check
pair_to_conflicts = {}
for a in range(N_TOTAL):
    for b in range(a + 1, N_TOTAL):
        conflicts = set()
        for c in range(N_TOTAL):
            if c == a or c == b:
                continue
            d_ab = dist2[a, b]
            d_ac = dist2[a, c]
            d_bc = dist2[b, c]
            if d_ab == d_ac or d_ab == d_bc or d_ac == d_bc:
                conflicts.add(c)
        pair_to_conflicts[(a, b)] = conflicts

def _can_add(p, S_set):
    """Check if point p can be added to S_set without creating an isosceles triangle."""
    S_list = list(S_set)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            key = (a, b) if a < b else (b, a)
            if p in pair_to_conflicts[key]:
                return False
    return True

def _is_no_isosceles(S_set):
    S_list = list(S_set)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            key = (a, b) if a < b else (b, a)
            conflicts = pair_to_conflicts[key]
            for k in range(j + 1, n):
                if S_list[k] in conflicts:
                    return False
    return True

def _is_blocked(p, S_set):
    """Check if p is blocked by S_set."""
    for a, b in isos_pairs[p]:
        if a in S_set and b in S_set:
            return True
    return False

def _is_maximal(S_set):
    for p in range(N_TOTAL):
        if p in S_set:
            continue
        if not _is_blocked(p, S_set):
            return False
    return True

def _reduce_to_minimal(S_set):
    S_list = sorted(S_set)
    changed = True
    while changed:
        changed = False
        for i in range(len(S_list)):
            p = S_list[i]
            S_test = set(S_list[:i] + S_list[i+1:])
            if len(S_test) < 2:
                continue
            if _is_maximal(S_test):
                S_list = sorted(S_test)
                changed = True
                break
    return set(S_list)

def _build_maximal_from_core(core, deadline_check=None):
    """Greedily extend core to a maximal no-isosceles set."""
    S = set(core)
    if not _is_no_isosceles(S):
        return None
    
    if _is_maximal(S):
        return S
    
    while True:
        if deadline_check and deadline_check():
            return None
        
        # Find addable points
        addable = []
        for p in range(N_TOTAL):
            if p in S:
                continue
            if _can_add(p, S):
                addable.append(p)
        
        if not addable:
            if _is_maximal(S):
                return S
            return None
        
        # Find unblocked points
        unblocked = []
        for p in range(N_TOTAL):
            if p in S:
                continue
            if not _is_blocked(p, S):
                unblocked.append(p)
        
        if not unblocked:
            return S
        
        # Pick point that covers most unblocked
        best_p = None
        best_gain = -1
        for p in addable:
            gain = 0
            for q in unblocked:
                if q == p:
                    gain += 1
                    continue
                # Check if {q, p, a} is isosceles for some a in S
                blocked = False
                for a in S:
                    d_qp = dist2[q, p]
                    d_qa = dist2[q, a]
                    d_pa = dist2[p, a]
                    if d_qp == d_qa or d_qp == d_pa or d_qa == d_pa:
                        blocked = True
                        break
                if blocked:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_p = p
        
        if best_p is None:
            break
        
        S.add(best_p)
    
    if _is_maximal(S):
        return S
    return None

def construct():
    start_time = time.time()
    rng = random.Random(99999)
    
    best_set = None
    best_size = N_TOTAL + 1
    
    def time_check():
        return time.time() - start_time > DEADLINE - 20
    
    # Strategy 1: Random greedy + reduce
    for attempt in range(2000):
        if time_check():
            break
        
        order = list(range(N_TOTAL))
        rng.shuffle(order)
        
        S = set()
        for p in order:
            if _can_add(p, S):
                S.add(p)
        
        # Try to make maximal
        if not _is_maximal(S):
            S_ext = _build_maximal_from_core(S, time_check)
            if S_ext is None:
                continue
            S = S_ext
        
        if not _is_maximal(S):
            continue
        
        S_min = _reduce_to_minimal(S)
        if len(S_min) < best_size:
            best_size = len(S_min)
            best_set = S_min
        
        if best_size <= 5:
            break
    
    # Strategy 2: Try small cores (pairs, triples)
    if time.time() - start_time < DEADLINE - 60:
        # Try all pairs
        for a in range(N_TOTAL):
            for b in range(a + 1, N_TOTAL):
                if time_check():
                    break
                
                S = {a, b}
                S_ext = _build_maximal_from_core(S, time_check)
                if S_ext is None:
                    continue
                if not _is_maximal(S_ext):
                    continue
                
                S_min = _reduce_to_minimal(S_ext)
                if len(S_min) < best_size:
                    best_size = len(S_min)
                    best_set = S_min
                    if best_size <= 5:
                        break
            if best_size <= 5:
                break
    
    # Strategy 3: Local search
    if best_set is not None and best_size > 5 and time.time() - start_time < DEADLINE - 30:
        for iteration in range(300):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            improved = False
            S_list = sorted(best_set)
            for i in range(len(S_list)):
                p_remove = S_list[i]
                S_without = set(S_list[:i] + S_list[i+1:])
                
                for p_add in range(N_TOTAL):
                    if p_add in S_without or p_add == p_remove:
                        continue
                    if not _can_add(p_add, S_without):
                        continue
                    
                    S_new = S_without | {p_add}
                    if not _is_maximal(S_new):
                        continue
                    
                    S_new_min = _reduce_to_minimal(S_new)
                    if len(S_new_min) < best_size:
                        best_size = len(S_new_min)
                        best_set = S_new_min
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                break
    
    if best_set is None:
        S = set()
        for p in range(N_TOTAL):
            if _can_add(p, S):
                S.add(p)
        best_set = _reduce_to_minimal(S)
    
    return [_xy(idx) for idx in sorted(best_set)]
