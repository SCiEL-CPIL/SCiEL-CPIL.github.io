import time
import random
import sys
from collections import deque

N = 12
DEADLINE = 85.0 # Safety margin before 90s kill

def compute_threshold(a_map, b_map):
    """
    Computes the exact shortest reset word length using BFS on the subset automaton.
    Returns the length or None if not synchronizing.
    Uses precomputed transitions for speed.
    """
    # Precompute transition tables for all 2^N subsets
    # trans_a[subset] -> next_subset after applying 'a'
    # trans_b[subset] -> next_subset after applying 'b'
    size = 1 << N
    trans_a = [0] * size
    trans_b = [0] * size
    
    for s in range(size):
        na = 0
        nb = 0
        for i in range(N):
            if s & (1 << i):
                na |= (1 << a_map[i])
                nb |= (1 << b_map[i])
        trans_a[s] = na
        trans_b[s] = nb
        
    start = size - 1
    visited = [False] * size
    visited[start] = True
    queue = deque([(start, 0)])
    
    while queue:
        state, dist = queue.popleft()
        
        # Check if singleton
        if state & (state - 1) == 0:
            return dist
            
        # Try 'a'
        next_a = trans_a[state]
        if not visited[next_a]:
            visited[next_a] = True
            queue.append((next_a, dist + 1))
            
        # Try 'b'
        next_b = trans_b[state]
        if not visited[next_b]:
            visited[next_b] = True
            queue.append((next_b, dist + 1))
            
    return None

def construct():
    """
    Returns the transition tables for a synchronizing automaton with N=12 states,
    maximizing the shortest reset word length.
    """
    start_time = time.time()
    
    best_a = [0] * N
    best_b = [0] * N
    best_score = 0
    
    # Strategy:
    # 1. Try the standard Černý automaton and its variants.
    # 2. Systematic search over "cycle + fold" structures.
    # 3. Randomized local search with fast evaluation if time permits.
    
    # Known good structure: Černý Automaton
    # a: i -> (i+1) % N
    # b: i -> i-1 if i>0, 0->0
    # This often has a high threshold.
    
    def evaluate_candidate(a, b):
        nonlocal best_a, best_b, best_score
        score = compute_threshold(a, b)
        if score is not None and score > best_score:
            best_score = score
            best_a = a[:]
            best_b = b[:]
            return True
        return False

    # 1. Standard Černý
    a_cz = [(i + 1) % N for i in range(N)]
    b_cz = [i - 1 if i > 0 else 0 for i in range(N)]
    evaluate_candidate(a_cz, b_cz)
    
    # 2. Variations of b for the cycle a
    # The optimal automata often have a as a cycle and b as a "nearly" decrement with a specific fold.
    # Let's iterate through different "folds" for b.
    # b(i) = i-1 for i != k, b(k) = some j.
    # Specifically, the Pawlik automata for N=12 might be reachable here.
    
    for k in range(N):
        for j in range(N):
            b_var = [i - 1 if i > 0 else 0 for i in range(N)]
            b_var[k] = j
            # Only evaluate if it's not the standard one (already checked)
            if b_var != b_cz:
                if time.time() < DEADLINE:
                    evaluate_candidate(a_cz, b_var)
    
    # 3. Try other cycle directions
    a_cz_rev = [(i - 1) % N for i in range(N)]
    evaluate_candidate(a_cz_rev, b_cz)
    
    for k in range(N):
        for j in range(N):
            b_var = [i - 1 if i > 0 else 0 for i in range(N)]
            b_var[k] = j
            if b_var != b_cz:
                if time.time() < DEADLINE:
                    evaluate_candidate(a_cz_rev, b_var)

    # 4. Randomized Local Search with Fast BFS
    # Start from the best found so far
    if best_score == 0:
        # Fallback if standard ones failed (unlikely)
        best_a = list(range(N))
        best_b = [0] * N
        evaluate_candidate(best_a, best_b)

    current_a = best_a[:]
    current_b = best_b[:]
    
    # Local search: Perturb one or two transitions
    while time.time() < DEADLINE - 2.0:
        a = current_a[:]
        b = current_b[:]
        
        # Random mutation
        # 1. Change one transition in b
        idx = random.randint(0, N-1)
        b[idx] = random.randint(0, N-1)
        
        # 2. Occasionally swap two transitions in a (keeping it a permutation is good for high thresholds)
        if random.random() < 0.5:
            i, j = random.sample(range(N), 2)
            a[i], a[j] = a[j], a[i]
            
        score = compute_threshold(a, b)
        
        if score is not None and score > best_score:
            best_score = score
            best_a = a
            best_b = b
            current_a = a
            current_b = b
            
            # If we hit the theoretical max for N=12 (121), we can stop early
            if best_score >= 121:
                break

    return [best_a, best_b]

if __name__ == "__main__":
    # Quick self-test
    a, b = construct()
    print(f"Best score found: {compute_threshold(a, b)}")
    print(f"Time: {time.time()}")
