import time
import random
from collections import deque

N = 12

def calculate_threshold_fast(a_trans, b_trans):
    """
    Computes the exact shortest reset word length using bitmask representation.
    """
    start = (1 << N) - 1  # All states set
    
    # Check for single state: power of 2
    if (start & (start - 1)) == 0:
        return 0
    
    visited = {start: 0}
    q = deque([start])
    
    # Precompute transitions for speed
    # For each mask, we need to know the result of applying a and b.
    # However, precomputing for all 2^12 masks is 4096 * 2 ops, which is fast.
    # But we can also compute on the fly.
    
    def apply_transition(mask, trans):
        result = 0
        temp = mask
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            result |= (1 << trans[idx])
            temp ^= lsb
        return result

    while q:
        curr = q.popleft()
        curr_dist = visited[curr]
        
        # Check if curr is a single state (power of 2)
        if (curr & (curr - 1)) == 0:
            return curr_dist
        
        # Apply letter 'a'
        next_a = apply_transition(curr, a_trans)
        if next_a not in visited:
            visited[next_a] = curr_dist + 1
            q.append(next_a)
            
        # Apply letter 'b'
        next_b = apply_transition(curr, b_trans)
        if next_b not in visited:
            visited[next_b] = curr_dist + 1
            q.append(next_b)
            
    return -1  # Not synchronizing

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_score = 0
    best_a = None
    best_b = None
    
    # The theoretical maximum for N=12 is (12-1)^2 = 121.
    # If we find 121, we can stop immediately.
    MAX_SCORE = 121
    
    def try_automaton(a, b):
        nonlocal best_score, best_a, best_b
        score = calculate_threshold_fast(a, b)
        if score > best_score:
            best_score = score
            best_a = a[:]
            best_b = b[:]
            if score >= MAX_SCORE:
                return True # Found optimal
        return False

    # Strategy 1: Try known structured families that are likely to be optimal.
    # Family 1: a = cycle, b = identity with one defect.
    a_cycle = [(i + 1) % N for i in range(N)]
    
    # Try all single-defect b's
    for k in range(N):
        for v in range(N):
            if v == k:
                continue
            b = list(range(N))
            b[k] = v
            if try_automaton(a_cycle, b):
                return [best_a, best_b]
            if time.time() > deadline:
                break
    
    # Family 2: a = cycle, b = "Pinwheel" like.
    # b(0)=0, b(i)=i-1 for i=1, b(i)=i for i>1?
    # Let's try a few specific patterns.
    patterns = [
        lambda k, v: (lambda b: (b.__setitem__(k, v), b)[1])(list(range(N))),
    ]
    
    # Strategy 2: Targeted Random Search
    # The current program found it with random search.
    # We will run a fast random search.
    random.seed(42)
    
    # Precompute a_cycle
    a = a_cycle
    
    for iteration in range(10000):
        if time.time() > deadline:
            break
        
        # Generate random b
        b = [random.randint(0, N-1) for _ in range(N)]
        
        # Ensure b is not a permutation (perm + cycle often doesn't sync or syncs fast)
        if len(set(b)) == N:
            b[random.randint(0, N-1)] = b[random.randint(0, N-1)]
        
        if try_automaton(a, b):
            return [best_a, best_b]
        
        # Local search improvement if we found a good score
        if best_score > 100:
            # Try flipping one transition in the best b found so far
            b_current = best_b[:]
            improved = False
            for i in range(N):
                for new_val in range(N):
                    if new_val == b_current[i]:
                        continue
                    b_old = b_current[i]
                    b_current[i] = new_val
                    if try_automaton(a, b_current):
                        return [best_a, best_b]
                    if time.time() > deadline:
                        b_current[i] = b_old
                        break
                    b_current[i] = b_old
                if time.time() > deadline:
                    break
    
    # Fallback
    if best_a is None:
        best_a = a_cycle
        best_b = [0] * N
    
    return [best_a, best_b]

# EVOLVE-BLOCK-START
# The function construct() is defined above.
# EVOLVE-BLOCK-END
