import time
import random
import sys
from collections import deque

N = 12
FULL_STATE = (1 << N) - 1
DEADLINE = 85.0  # Safety margin under 90s kill

def calculate_threshold(a_trans, b_trans):
    """
    Returns the shortest reset word length for the automaton defined by a_trans and b_trans,
    or -1 if not synchronizing. Uses precomputed transition tables for speed.
    """
    a_bits = [1 << a_trans[i] for i in range(N)]
    b_bits = [1 << b_trans[i] for i in range(N)]
    
    # Precompute transition tables for all possible subsets
    trans_a = [0] * (1 << N)
    trans_b = [0] * (1 << N)
    
    for mask in range(1, 1 << N):
        res_a = 0
        res_b = 0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            res_a |= a_bits[idx]
            res_b |= b_bits[idx]
            m ^= lsb
        trans_a[mask] = res_a
        trans_b[mask] = res_b
        
    # BFS
    visited = bytearray(1 << N)
    visited[FULL_STATE] = 1
    queue = deque()
    queue.append((FULL_STATE, 0))
    
    while queue:
        state, dist = queue.popleft()
        if state.bit_count() == 1:
            return dist
            
        next_a = trans_a[state]
        if not visited[next_a]:
            visited[next_a] = 1
            queue.append((next_a, dist + 1))
            
        next_b = trans_b[state]
        if not visited[next_b]:
            visited[next_b] = 1
            queue.append((next_b, dist + 1))
            
    return -1

def construct():
    start_time = time.time()
    
    best_score = 0
    best_a = None
    best_b = None
    
    def try_automaton(a, b):
        nonlocal best_score, best_a, best_b
        score = calculate_threshold(a, b)
        if score > best_score:
            best_score = score
            best_a = a[:]
            best_b = b[:]
            return True
        return False
    
    # Phase 1: Exhaustive search over "Pinsker-like" structures
    # a is a cycle, b has a single defect
    a_cycle = [(i + 1) % N for i in range(N)]
    
    for k in range(N):
        for v in range(N):
            if v == k:
                continue
            if time.time() - start_time > DEADLINE - 5:
                break
            b = list(range(N))
            b[k] = v
            try_automaton(a_cycle, b)
        if time.time() - start_time > DEADLINE - 5:
            break
    
    # Phase 2: Random restart local search
    # We optimize BOTH a and b, not just b with a fixed cycle.
    # This allows escaping local optima of the Pinsker family.
    
    def random_automaton():
        return [random.randint(0, N-1) for _ in range(N)]
    
    def mutate(a, b, rate=0.3):
        new_a = a[:]
        new_b = b[:]
        for i in range(N):
            if random.random() < rate:
                new_a[i] = random.randint(0, N-1)
            if random.random() < rate:
                new_b[i] = random.randint(0, N-1)
        return new_a, new_b
    
    # Start from the best found so far, or a random one if none
    if best_a:
        current_a = best_a[:]
        current_b = best_b[:]
        current_score = best_score
    else:
        current_a = random_automaton()
        current_b = random_automaton()
        current_score = calculate_threshold(current_a, current_b)
        if current_score > best_score:
            best_score = current_score
            best_a = current_a[:]
            best_b = current_b[:]
    
    # Local search loop
    while time.time() - start_time < DEADLINE:
        # Generate a neighbor
        neighbor_a, neighbor_b = mutate(current_a, current_b, rate=0.2)
        
        # Evaluate
        neighbor_score = calculate_threshold(neighbor_a, neighbor_b)
        
        if neighbor_score > current_score:
            # Accept the move
            current_a = neighbor_a
            current_b = neighbor_b
            current_score = neighbor_score
            
            if current_score > best_score:
                best_score = current_score
                best_a = current_a[:]
                best_b = current_b[:]
        else:
            # Reject the move, but occasionally accept worse moves to escape local optima
            # Simulated annealing-like acceptance
            if random.random() < 0.1:
                current_a = neighbor_a
                current_b = neighbor_b
                current_score = neighbor_score
        
        # Occasionally restart from a random point to explore different regions
        if random.random() < 0.05:
            current_a = random_automaton()
            current_b = random_automaton()
            current_score = calculate_threshold(current_a, current_b)
            if current_score > best_score:
                best_score = current_score
                best_a = current_a[:]
                best_b = current_b[:]
    
    # Fallback if no valid automaton was found (should not happen)
    if best_a is None:
        best_a = [(i + 1) % N for i in range(N)]
        best_b = [0] * N
    
    return [best_a, best_b]
