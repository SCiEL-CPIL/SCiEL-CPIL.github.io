import time
from collections import deque
import random

N = 12
FULL_SET = (1 << N) - 1

def _is_singleton(mask):
    return mask != 0 and (mask & (mask - 1)) == 0

def _bfs_reset_length_fast(a, b):
    """
    Fast BFS using precomputed transition tables for all subsets.
    """
    trans_a = [0] * (1 << N)
    trans_b = [0] * (1 << N)
    
    # Build transition tables incrementally
    for subset in range(1, 1 << N):
        # Find lowest set bit
        lsb = subset & -subset
        idx = lsb.bit_length() - 1
        prev = subset ^ lsb
        trans_a[subset] = trans_a[prev] | (1 << a[idx])
        trans_b[subset] = trans_b[prev] | (1 << b[idx])
    
    # BFS from full_set
    visited = [-1] * (1 << N)
    visited[FULL_SET] = 0
    queue = deque([FULL_SET])
    
    while queue:
        curr = queue.popleft()
        dist = visited[curr]
        
        # Apply 'a'
        next_a = trans_a[curr]
        if visited[next_a] == -1:
            if next_a != 0 and (next_a & (next_a - 1)) == 0:
                return dist + 1
            visited[next_a] = dist + 1
            queue.append(next_a)
            
        # Apply 'b'
        next_b = trans_b[curr]
        if visited[next_b] == -1:
            if next_b != 0 and (next_b & (next_b - 1)) == 0:
                return dist + 1
            visited[next_b] = dist + 1
            queue.append(next_b)
            
    return -1  # Not synchronizing

def _local_search(a, b, timeout, seed=42):
    """
    Hill-climbing local search on both a and b transitions with restarts.
    Returns best (score, a_copy, b_copy) found.
    """
    rng = random.Random(seed)
    start_time = time.time()
    
    current_a = a[:]
    current_b = b[:]
    current_best = _bfs_reset_length_fast(current_a, current_b)
    best_a = current_a[:]
    best_b = current_b[:]
    
    if current_best < 0:
        return current_best, best_a, best_b
    
    while time.time() - start_time < timeout:
        improved = False
        # Try modifying a transitions
        for i in range(N):
            if time.time() - start_time > timeout:
                break
            old_val = current_a[i]
            for j in range(N):
                if j == old_val:
                    continue
                current_a[i] = j
                val = _bfs_reset_length_fast(current_a, current_b)
                if val > current_best:
                    current_best = val
                    best_a = current_a[:]
                    best_b = current_b[:]
                    improved = True
                    break
            if not improved:
                current_a[i] = old_val
            if improved:
                break
        
        # Try modifying b transitions
        if not improved:
            for i in range(N):
                if time.time() - start_time > timeout:
                    break
                old_val = current_b[i]
                for j in range(N):
                    if j == old_val:
                        continue
                    current_b[i] = j
                    val = _bfs_reset_length_fast(current_a, current_b)
                    if val > current_best:
                        current_best = val
                        best_a = current_a[:]
                        best_b = current_b[:]
                        improved = True
                        break
                if not improved:
                    current_b[i] = old_val
                if improved:
                    break
        
        if not improved:
            # Restart with perturbation
            if time.time() - start_time < timeout - 2.0:
                # Perturb current best with 2-3 random changes
                perturbed_a = best_a[:]
                perturbed_b = best_b[:]
                for _ in range(rng.randint(2, 3)):
                    if rng.random() < 0.5:
                        idx = rng.randint(0, N - 1)
                        perturbed_a[idx] = rng.randint(0, N - 1)
                    else:
                        idx = rng.randint(0, N - 1)
                        perturbed_b[idx] = rng.randint(0, N - 1)
                
                val = _bfs_reset_length_fast(perturbed_a, perturbed_b)
                if val >= 0:
                    current_a = perturbed_a
                    current_b = perturbed_b
                    current_best = val
                else:
                    break
            else:
                break
                
    return current_best, best_a, best_b

def construct():
    """
    Systematically enumerate structural families and apply local search.
    """
    start_time = time.time()
    total_timeout = 85.0  # Safety margin
    
    best_score = -1
    best_a = None
    best_b = None
    
    def _update(a, b):
        nonlocal best_score, best_a, best_b
        score = _bfs_reset_length_fast(a, b)
        if score > best_score:
            best_score = score
            best_a = a[:]
            best_b = b[:]
    
    # Family 1: Pinwheel / cycle + break
    for break_pos in range(N):
        a = [(i + 1) % N for i in range(N)]
        b = [i for i in range(N)]
        b[break_pos] = (break_pos + N - 1) % N
        _update(a, b)
    
    # Family 2: Cycle + clamp
    for lo in range(N):
        for hi in range(lo, N):
            a = [(i + 1) % N for i in range(N)]
            b = [min(max(i, lo), hi) for i in range(N)]
            _update(a, b)
    
    # Family 3: Vorel halving variants
    a = [(i + 1) % N for i in range(N)]
    b = [i // 2 for i in range(N)]
    _update(a, b)
    
    # Family 4: Rank-2 constructions
    for mod in [2, 3, 4, 5, 6]:
        a = [(i + 1) % N for i in range(N)]
        b = []
        for i in range(N):
            if i % mod == 0:
                b.append(0)
            elif i % mod == 1:
                b.append(1)
            else:
                b.append(i)
        _update(a, b)
    
    # Family 5: Systematic sweep of b for fixed a=cycle
    a = [(i + 1) % N for i in range(N)]
    
    b = [0] * N
    _update(a, b)
    
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            b = list(range(N))
            b[i] = j
            _update(a, b)
    
    # Family 6: Two-break pinwheel (fixed bug: use separate variables)
    for i1 in range(N):
        for j1 in range(i1 + 1, N):
            a = [(k + 1) % N for k in range(N)]
            b = [k for k in range(N)]
            b[i1] = (i1 + N - 1) % N
            b[j1] = (j1 + N - 1) % N
            _update(a, b)
    
    # Family 7: Vorel with offset
    for offset in range(N):
        a = [(i + 1) % N for i in range(N)]
        b = [(i + offset) // 2 for i in range(N)]
        _update(a, b)
    
    # Family 8: Non-cycle a - two-cycle based
    # a: maps even states forward, odd states forward (two 6-cycles)
    for perm_type in range(3):
        a = []
        if perm_type == 0:
            # Two 6-cycles
            for i in range(N):
                if i < 6:
                    a.append((i + 1) % 6)
                else:
                    a.append(6 + (i - 6 + 1) % 6)
        elif perm_type == 1:
            # One 4-cycle + one 8-cycle
            for i in range(N):
                if i < 4:
                    a.append((i + 1) % 4)
                else:
                    a.append(4 + (i - 4 + 1) % 8)
        else:
            # One 3-cycle + one 9-cycle
            for i in range(N):
                if i < 3:
                    a.append((i + 1) % 3)
                else:
                    a.append(3 + (i - 3 + 1) % 9)
        
        # Try various b patterns with non-cycle a
        b_patterns = [
            [0] * N,
            [i // 2 for i in range(N)],
            [i for i in range(N)],
            [(i + 1) % N for i in range(N)],
        ]
        for b in b_patterns:
            _update(a, b)
    
    # Family 9: Systematic b with two targets
    a = [(i + 1) % N for i in range(N)]
    for t1 in range(N):
        for t2 in range(N):
            if t1 == t2:
                continue
            b = [t1 if i % 2 == 0 else t2 for i in range(N)]
            _update(a, b)
    
    # Check time and do local search on best candidate
    elapsed = time.time() - start_time
    remaining = total_timeout - elapsed
    
    if remaining > 3.0 and best_score > 0:
        # Do local search on the best candidate found
        local_timeout = min(remaining - 2.0, 40.0)
        local_score, local_a, local_b = _local_search(best_a, best_b, local_timeout)
        if local_score > best_score:
            best_score = local_score
            best_a = local_a
            best_b = local_b
    
    # If still have time, try a few more random restarts of local search
    elapsed = time.time() - start_time
    remaining = total_timeout - elapsed
    
    if remaining > 5.0 and best_score > 0:
        for seed in [123, 456, 789]:
            if time.time() - start_time > total_timeout - 3.0:
                break
            local_timeout = min(remaining - 3.0, 10.0)
            local_score, local_a, local_b = _local_search(best_a, best_b, local_timeout, seed=seed)
            if local_score > best_score:
                best_score = local_score
                best_a = local_a
                best_b = local_b
            remaining = total_timeout - (time.time() - start_time)
    
    if best_score < 0:
        # Fallback: return a known synchronizing automaton
        a = [(i + 1) % N for i in range(N)]
        b = [i // 2 for i in range(N)]
        return [a, b]
    
    return [best_a, best_b]
