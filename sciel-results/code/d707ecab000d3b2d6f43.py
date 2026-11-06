import time
import random

N = 12

def calculate_threshold(trans_a, trans_b):
    """
    Computes the reset threshold of the automaton defined by trans_a and trans_b.
    Returns the length of the shortest reset word, or -1 if not synchronizing.
    """
    start_set = frozenset(range(N))
    if len(start_set) == 1:
        return 0
        
    # BFS on subsets
    visited = {start_set: 0}
    queue = [start_set]
    
    while queue:
        current_set = queue.pop(0)
        current_dist = visited[current_set]
        
        # Try applying trans_a
        next_set_a = frozenset(trans_a[i] for i in current_set)
        if next_set_a not in visited:
            if len(next_set_a) == 1:
                return current_dist + 1
            visited[next_set_a] = current_dist + 1
            queue.append(next_set_a)
            
        # Try applying trans_b
        next_set_b = frozenset(trans_b[i] for i in current_set)
        if next_set_b not in visited:
            if len(next_set_b) == 1:
                return current_dist + 1
            visited[next_set_b] = current_dist + 1
            queue.append(next_set_b)
            
    return -1

def construct():
    """
    Constructs a synchronizing automaton with 12 states and 2 letters.
    Uses a search to maximize the reset threshold.
    """
    start_time = time.time()
    deadline = 50.0  # Safety margin
    
    best_threshold = -1
    best_trans_a = None
    best_trans_b = None
    
    # Initial candidate: The known N-1 threshold automaton
    init_a = [(i + 1) % N for i in range(N)]
    init_b = [0] * N
    for i in range(1, N):
        init_b[i] = i - 1
        
    t = calculate_threshold(init_a, init_b)
    if t > best_threshold:
        best_threshold = t
        best_trans_a = init_a
        best_trans_b = init_b
        
    # Random search for better candidates
    # We fix trans_a as the cycle, as it provides the "mixing" needed for long thresholds
    # when combined with a "slowing" trans_b.
    trans_a = init_a
    
    random.seed(42) # For reproducibility
    
    while time.time() - start_time < deadline:
        # Generate a random trans_b
        # Strategy: Biased towards "decrement-like" behavior to encourage long paths
        # but with some randomness to break the simple structure.
        trans_b = [0] * N
        # Ensure 0 maps to 0 to have a fixed point (helps synchronization)
        # Or map to a small set.
        
        # Heuristic: Create a "staircase" with jumps
        # trans_b[i] is usually i-1, but sometimes jumps to 0 or random
        for i in range(1, N):
            if random.random() < 0.8:
                trans_b[i] = i - 1
            else:
                # Random jump to a smaller state
                trans_b[i] = random.randint(0, i - 1)
        
        # Ensure it's not a permutation (which would prevent synchronization if a is a cycle)
        # Actually, if a is a cycle and b is a permutation, the automaton is a group, 
        # and it only synchronizes if the group is trivial? No, a group automaton 
        # with a cycle and another permutation does not synchronize unless they 
        # generate a non-transitive group? 
        # If both are permutations, the automaton is a group action. 
        # It synchronizes only if the group is not transitive? 
        # No, a group action on a set synchronizes to a singleton iff the group 
        # has an orbit of size 1? No, it reduces to the orbit. 
        # If the group is transitive, it never synchronizes. 
        # So we MUST ensure b is NOT a permutation.
        
        # Check if b is a permutation
        if len(set(trans_b)) == N:
            # Make it non-permutation by mapping two states to the same
            trans_b[random.randint(0, N-1)] = 0
            
        t = calculate_threshold(trans_a, trans_b)
        if t > best_threshold:
            best_threshold = t
            best_trans_a = trans_a
            best_trans_b = trans_b
            
            # If we hit the theoretical max, we can stop early
            if best_threshold >= (N-1)**2:
                break
                
    # If search failed to find anything better, return the initial one
    if best_trans_a is None:
        return [init_a, init_b]
        
    return [best_trans_a, best_trans_b]
