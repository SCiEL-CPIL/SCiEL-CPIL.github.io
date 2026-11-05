import time
import random
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1

def _apply_mask(mask, trans):
    """Apply transition 'trans' to bitmask 'mask'. Returns new bitmask."""
    result = 0
    m = mask
    while m:
        i = (m & -m).bit_length() - 1
        result |= 1 << trans[i]
        m &= m - 1
    return result

def _threshold(a_trans, b_trans):
    """
    BFS on subset automaton using bitmasks.
    Returns the shortest reset word length, or -1 if not synchronizing.
    """
    start = FULL_MASK
    if start == 1:
        return 0
    visited = {start: 0}
    q = deque([start])
    while q:
        curr = q.popleft()
        curr_dist = visited[curr]
        if curr & (curr - 1) == 0:  # single bit
            return curr_dist
        next_a = _apply_mask(curr, a_trans)
        if next_a not in visited:
            visited[next_a] = curr_dist + 1
            q.append(next_a)
        next_b = _apply_mask(curr, b_trans)
        if next_b not in visited:
            visited[next_b] = curr_dist + 1
            q.append(next_b)
    return -1

def _pinsker_a():
    """Standard Pinsker 'a' transition: 0->0, i->i+1 for i>=1."""
    a = list(range(N))
    for i in range(1, N):
        a[i] = i + 1
    a[N-1] = N - 1  # fix: in Pinsker, a(N-1)=N-1? Actually standard Pinsker: a(0)=0, a(i)=i+1 for 1<=i<N-1, a(N-1)=0
    # Let me use the classic Pinsker construction:
    # a: 0->0, 1->2, 2->3, ..., N-2->N-1, N-1->0
    a = [0] * N
    for i in range(1, N):
        a[i] = i + 1
    a[N-1] = 0
    return a

def construct():
    deadline = time.time() + 80  # 80s safety margin before 90s kill
    best_score = 0
    best_a = _pinsker_a()
    best_b = list(range(N))  # identity as fallback (not synchronizing, but we'll find better)
    
    # Initialize best with a known synchronizing automaton
    # Simple: a = cycle, b = constant 0
    a_cycle = [(i + 1) % N for i in range(N)]
    b_const = [0] * N
    t = _threshold(a_cycle, b_const)
    if t >= 0:
        best_score = t
        best_a = a_cycle
        best_b = b_const
    
    # Strategy 1: Pinsker-like 'a' with random 'b'
    a_pinsker = _pinsker_a()
    
    # Strategy 2: Random 'a' with random 'b'
    
    # We'll do local search: start with Pinsker a, randomize b, then hill-climb b
    rng = random.Random(42)
    
    # Try multiple random restarts on b given fixed Pinsker a
    for restart in range(200):
        if time.time() > deadline - 5:
            break
        
        # Random b
        b = [rng.randint(0, N-1) for _ in range(N)]
        score = _threshold(a_pinsker, b)
        if score >= 0:
            if score > best_score:
                best_score = score
                best_b = b[:]
                best_a = a_pinsker[:]
            
            # Local search: mutate one position of b
            improved = True
            while improved and time.time() < deadline - 3:
                improved = False
                # Try all single-position mutations
                for i in range(N):
                    old_val = b[i]
                    for new_val in range(N):
                        if new_val == old_val:
                            continue
                        b[i] = new_val
                        s = _threshold(a_pinsker, b)
                        if s >= 0 and s > best_score:
                            best_score = s
                            best_b = b[:]
                            best_a = a_pinsker[:]
                            improved = True
                        else:
                            b[i] = old_val
    
    # Strategy 3: Also try varying 'a' slightly around Pinsker
    # Perturb Pinsker a: change one or two transitions
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        a = a_pinsker[:]
        # Mutate 1-2 positions of a
        num_mut = rng.randint(1, 2)
        for _ in range(num_mut):
            i = rng.randint(0, N-1)
            a[i] = rng.randint(0, N-1)
        
        b = [rng.randint(0, N-1) for _ in range(N)]
        score = _threshold(a, b)
        if score >= 0:
            if score > best_score:
                best_score = score
                best_b = b[:]
                best_a = a[:]
            
            # Local search on both a and b
            improved = True
            while improved and time.time() < deadline - 3:
                improved = False
                # Mutate b
                for i in range(N):
                    old_val = b[i]
                    for new_val in range(N):
                        if new_val == old_val:
                            continue
                        b[i] = new_val
                        s = _threshold(a, b)
                        if s >= 0 and s > best_score:
                            best_score = s
                            best_b = b[:]
                            best_a = a[:]
                            improved = True
                        else:
                            b[i] = old_val
                # Mutate a
                for i in range(N):
                    old_val = a[i]
                    for new_val in range(N):
                        if new_val == old_val:
                            continue
                        a[i] = new_val
                        s = _threshold(a, b)
                        if s >= 0 and s > best_score:
                            best_score = s
                            best_b = b[:]
                            best_a = a[:]
                            improved = True
                        else:
                            a[i] = old_val
    
    # Strategy 4: Pure random search with local refinement
    for restart in range(100):
        if time.time() > deadline - 5:
            break
        
        a = [rng.randint(0, N-1) for _ in range(N)]
        b = [rng.randint(0, N-1) for _ in range(N)]
        score = _threshold(a, b)
        if score >= 0:
            if score > best_score:
                best_score = score
                best_b = b[:]
                best_a = a[:]
            
            # Quick local search
            improved = True
            while improved and time.time() < deadline - 3:
                improved = False
                for i in range(N):
                    old_val = b[i]
                    for new_val in range(N):
                        if new_val == old_val:
                            continue
                        b[i] = new_val
                        s = _threshold(a, b)
                        if s >= 0 and s > best_score:
                            best_score = s
                            best_b = b[:]
                            best_a = a[:]
                            improved = True
                        else:
                            b[i] = old_val
                for i in range(N):
                    old_val = a[i]
                    for new_val in range(N):
                        if new_val == old_val:
                            continue
                        a[i] = new_val
                        s = _threshold(a, b)
                        if s >= 0 and s > best_score:
                            best_score = s
                            best_b = b[:]
                            best_a = a[:]
                            improved = True
                        else:
                            a[i] = old_val
    
    # Ensure we return a valid synchronizing automaton
    if best_score <= 0:
        # Fallback to the simple cycle + constant
        return a_cycle, b_const
    
    return best_a, best_b
