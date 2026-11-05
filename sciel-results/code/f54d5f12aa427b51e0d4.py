import time
import random
from collections import deque

N = 12

def calculate_threshold(a_trans, b_trans):
    """
    Computes the exact shortest reset word length (threshold) for a binary automaton.
    Uses BFS on the subset automaton.
    """
    start = frozenset(range(N))
    if len(start) == 1:
        return 0
    
    visited = {start: 0}
    q = deque([start])
    
    while q:
        curr = q.popleft()
        curr_dist = visited[curr]
        
        if len(curr) == 1:
            return curr_dist
        
        # Apply letter 'a'
        next_a = frozenset(a_trans[i] for i in curr)
        if next_a not in visited:
            visited[next_a] = curr_dist + 1
            q.append(next_a)
            
        # Apply letter 'b'
        next_b = frozenset(b_trans[i] for i in curr)
        if next_b not in visited:
            visited[next_b] = curr_dist + 1
            q.append(next_b)
            
    return -1  # Not synchronizing

def construct():
    # Deadline guard: 90s wall-clock limit. 
    # We reserve 10s for safety/overhead, so we search for 80s.
    start_time = time.time()
    deadline = start_time + 80.0
    
    best_score = 0
    best_a = None
    best_b = None
    
    # Strategy: Search over "Pinsker-like" automata.
    # Letter 'a' is a cyclic shift: a[i] = (i + 1) % N
    # Letter 'b' is a permutation that is mostly identity or shift, 
    # but has some "defect" that causes synchronization.
    # A common high-threshold structure:
    # a: 0->1, 1->2, ..., N-1->0
    # b: i->i for most i, but one or two states map differently.
    
    # We will randomize the "defect" positions and targets.
    # To ensure synchronization, 'b' must not be a permutation that commutes with 'a' 
    # in a way that prevents collapse. 
    # Actually, any non-permutation 'b' in a cyclic 'a' setup usually synchronizes 
    # if the graph is connected appropriately.
    
    # Let's try a specific family known to be good:
    # a[i] = (i + 1) % N
    # b[i] = i for i != k, b[k] = (k + offset) % N
    
    # We'll iterate through different 'k' and 'offset' values, 
    # and also try slightly more complex 'b' structures if time permits.
    
    # 1. Try simple "single defect" permutations first
    for k in range(N):
        for offset in range(1, N):
            if time.time() > deadline:
                break
                
            a = [(i + 1) % N for i in range(N)]
            b = [i for i in range(N)]
            b[k] = (k + offset) % N
            
            # Check if it's a permutation (if yes, it might not sync or sync slowly)
            # Actually, if 'b' is a permutation, the automaton is a group, 
            # and it only synchronizes if the group action is trivial or similar.
            # Wait, Černý's conjecture applies to *synchronizing* automata.
            # If both are permutations, the automaton is a group action. 
            # A group action is synchronizing iff the group is trivial? 
            # No, a group action on a set is synchronizing iff the set has size 1.
            # So 'b' MUST be a non-permutation.
            
            # My above 'b' construction IS a permutation (just swapping k and k+offset? No, 
            # if b[k] = k+offset and b[k+offset] is still k+offset (if k+offset != k), 
            # then it's not a permutation because two states map to k+offset? 
            # No: b = [0, 1, ..., k-1, k+offset, k+1, ..., k+offset-1, k+offset, ...]
            # This is NOT a permutation. The value k is missing from the image? 
            # Let's trace: 
            # b[0]=0, ..., b[k-1]=k-1, b[k]=k+offset, b[k+1]=k+1, ..., b[N-1]=N-1.
            # The value 'k' is not in the image. So it is a non-permutation. Good.
            
            score = calculate_threshold(a, b)
            if score > best_score:
                best_score = score
                best_a = a
                best_b = b
                
    # 2. If time remains, try "double defect" or random non-permutations
    if time.time() < deadline - 5.0:
        # Random search for non-permutation 'b' with 'a' fixed as cycle
        a = [(i + 1) % N for i in range(N)]
        
        for _ in range(1000):
            if time.time() > deadline:
                break
                
            # Generate a random function b: {0..N-1} -> {0..N-1}
            # Ensure it's not a permutation by forcing a collision
            b = [random.randint(0, N-1) for _ in range(N)]
            # Force a collision to ensure non-permutation (increases sync chance)
            # e.g., make b[0] == b[1]
            b[1] = b[0]
            
            score = calculate_threshold(a, b)
            if score > best_score:
                best_score = score
                best_a = a
                best_b = b
                
    # If no valid synchronizing automaton was found (should not happen with above logic),
    # fall back to the seed's simple automaton (which is known to sync).
    if best_a is None:
        best_a = [(i + 1) % N for i in range(N)]
        best_b = [0] * N
        
    return [best_a, best_b]

# EVOLVE-BLOCK-START
# The function construct() is defined above. 
# We keep the block markers for compatibility if the evaluator expects them, 
# but the logic is inside the function.
# EVOLVE-BLOCK-END
