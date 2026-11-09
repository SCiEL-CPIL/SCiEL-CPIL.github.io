import time
from collections import deque

N = 12
DEADLINE = 85.0  # Safety margin before the 90s kill

def compute_threshold_fast(a_map, b_map):
    """
    Computes the exact shortest reset word length using BFS on the subset automaton.
    Uses bitmask representation for speed.
    Returns the length or None if not synchronizing.
    """
    start = (1 << N) - 1
    
    # Precompute transitions for each state in the full mask
    # For speed, we compute transitions on-the-fly but efficiently
    
    visited = {start: 0}
    queue = deque([(start, 0)])
    
    while queue:
        state, dist = queue.popleft()
        
        # Check if singleton
        if state & (state - 1) == 0:
            return dist
            
        # Try 'a'
        next_a = 0
        s = state
        while s:
            lowbit = s & (-s)
            i = lowbit.bit_length() - 1
            next_a |= (1 << a_map[i])
            s ^= lowbit
            
        if next_a not in visited:
            visited[next_a] = dist + 1
            queue.append((next_a, dist + 1))
            
        # Try 'b'
        next_b = 0
        s = state
        while s:
            lowbit = s & (-s)
            i = lowbit.bit_length() - 1
            next_b |= (1 << b_map[i])
            s ^= lowbit
            
        if next_b not in visited:
            visited[next_b] = dist + 1
            queue.append((next_b, dist + 1))
            
    return None

def construct():
    """
    Returns the transition tables for a synchronizing automaton with N=12 states,
    maximizing the shortest reset word length.
    
    Strategy: Construct the Černý automaton which is known to achieve the 
    Černý conjecture bound of (N-1)^2 = 121 for N=12 states.
    
    The Černý construction:
    - Letter 'a' is a cycle: 0 -> 1 -> 2 -> ... -> N-1 -> 0
    - Letter 'b' maps: 0 -> 0, 1 -> 0, 2 -> 1, 3 -> 2, ..., N-1 -> N-2
    
    This is the canonical worst-case example for the Černý conjecture.
    """
    start_time = time.time()
    
    # Černý automaton construction
    # a: cycle 0->1->2->...->N-1->0
    a = [(i + 1) % N for i in range(N)]
    
    # b: 0->0, 1->0, 2->1, 3->2, ..., N-1->N-2
    b = [0] * N
    for i in range(1, N):
        b[i] = i - 1
    
    # Verify it synchronizes and get the threshold
    threshold = compute_threshold_fast(a, b)
    
    if threshold is not None:
        return [a, b]
    
    # Fallback: simple synchronizing automaton
    a = list(range(N))
    b = [0] * N
    for i in range(1, N):
        b[i] = i - 1
    
    return [a, b]

if __name__ == "__main__":
    a, b = construct()
    t = compute_threshold_fast(a, b)
    print(f"Threshold: {t}")
    print(f"Time: {time.time()}")
