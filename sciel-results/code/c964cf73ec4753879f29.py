import time
import random
from collections import deque

N = 12
ALL_STATES = (1 << N) - 1

def calculate_threshold(a_trans, b_trans):
    """
    Computes the exact shortest reset word length (threshold) for a binary automaton.
    Uses BFS on the subset automaton with precomputed transition tables for speed.
    """
    # Precompute transitions for all 2^N subsets
    # next_a[mask] = apply_trans(mask, a_trans)
    # next_b[mask] = apply_trans(mask, b_trans)
    
    # This is fast enough for N=12 (4096 entries)
    next_a = [0] * (1 << N)
    next_b = [0] * (1 << N)
    
    for mask in range(1 << N):
        res_a = 0
        res_b = 0
        m = mask
        while m:
            lsb = m & (-m)
            i = lsb.bit_length() - 1
            res_a |= (1 << a_trans[i])
            res_b |= (1 << b_trans[i])
            m ^= lsb
        next_a[mask] = res_a
        next_b[mask] = res_b

    start = ALL_STATES
    
    # Single state check
    if (start & (start - 1)) == 0:
        return 0

    visited = [-1] * (1 << N)
    visited[start] = 0
    q = deque([start])
    
    while q:
        curr = q.popleft()
        curr_dist = visited[curr]
        
        # Check if singleton
        if (curr & (curr - 1)) == 0:
            return curr_dist
        
        # Apply 'a'
        next_a_curr = next_a[curr]
        if visited[next_a_curr] == -1:
            visited[next_a_curr] = curr_dist + 1
            q.append(next_a_curr)
            
        # Apply 'b'
        next_b_curr = next_b[curr]
        if visited[next_b_curr] == -1:
            visited[next_b_curr] = curr_dist + 1
            q.append(next_b_curr)
            
    return -1  # Not synchronizing

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_score = -1
    best_a = None
    best_b = None
    
    # Strategy: Systematic search over structured families
    # Family 1: a = cycle, b = identity with one or two defects
    a_cycle = [(i + 1) % N for i in range(N)]
    
    # Family 1: Single defect in b (b[i]=i except b[k]=j)
    for k in range(N):
        for j in range(N):
            if j == k:
                continue
            b = [i for i in range(N)]
            b[k] = j
            score = calculate_threshold(a_cycle, b)
            if score > best_score:
                best_score = score
                best_a = list(a_cycle)
                best_b = list(b)
            if time.time() > deadline:
                return [best_a, best_b]
            if best_score == (N - 1) ** 2:
                return [best_a, best_b] # Found theoretical max

    # Family 2: Double defect in b
    for k1 in range(N):
        for k2 in range(k1+1, N):
            for j1 in range(N):
                for j2 in range(N):
                    if j1 == k1 and j2 == k2:
                        continue
                    b = [i for i in range(N)]
                    b[k1] = j1
                    b[k2] = j2
                    score = calculate_threshold(a_cycle, b)
                    if score > best_score:
                        best_score = score
                        best_a = list(a_cycle)
                        best_b = list(b)
                    if time.time() > deadline:
                        return [best_a, best_b]
                    if best_score == (N - 1) ** 2:
                        return [best_a, best_b]

    # Family 3: Random search with cycle 'a'
    random.seed(42)
    for _ in range(5000):
        if time.time() > deadline:
            break
        b = [random.randint(0, N-1) for _ in range(N)]
        # Ensure non-permutation: force a collision
        b[1] = b[0]
        score = calculate_threshold(a_cycle, b)
        if score > best_score:
            best_score = score
            best_a = list(a_cycle)
            best_b = list(b)
            if best_score == (N - 1) ** 2:
                return [best_a, best_b]

    # Family 4: Try different cycle structures for 'a'
    for d in range(1, N):
        a_shift = [(i + d) % N for i in range(N)]
        for k in range(N):
            for j in range(N):
                if j == k:
                    continue
                b = [i for i in range(N)]
                b[k] = j
                score = calculate_threshold(a_shift, b)
                if score > best_score:
                    best_score = score
                    best_a = list(a_shift)
                    best_b = list(b)
                if time.time() > deadline:
                    return [best_a, best_b]
                if best_score == (N - 1) ** 2:
                    return [best_a, best_b]

    # Family 5: More random search with varied 'a'
    for _ in range(3000):
        if time.time() > deadline:
            break
        # Random 'a' that is a permutation (cycle or product of cycles)
        perm = list(range(N))
        random.shuffle(perm)
        a_trans = [0] * N
        curr = perm[0]
        for step in range(N):
            a_trans[curr] = perm[(step + 1) % N]
            curr = perm[(step + 1) % N]
        
        b = [random.randint(0, N-1) for _ in range(N)]
        b[1] = b[0]
        
        score = calculate_threshold(a_trans, b)
        if score > best_score:
            best_score = score
            best_a = list(a_trans)
            best_b = list(b)
            if best_score == (N - 1) ** 2:
                return [best_a, best_b]

    if best_a is None:
        best_a = a_cycle
        best_b = [0] * N
    
    return [best_a, best_b]

# EVOLVE-BLOCK-START
# The function construct() is defined above.
# EVOLVE-BLOCK-END
