import time
import random

N = 12

def calculate_threshold(trans_0, trans_1):
    """
    Calculates the exact shortest reset word length using BFS on the subset automaton.
    Returns -1 if not synchronizing.
    """
    start = frozenset(range(N))
    if len(start) == 1:
        return 0
        
    visited = {start: 0}
    queue = [(start, 0)]
    
    while queue:
        subset, dist = queue.pop(0)
        
        # Try adding '0'
        next_subset_0 = frozenset(trans_0[i] for i in subset)
        if next_subset_0 not in visited:
            if len(next_subset_0) == 1:
                return dist + 1
            visited[next_subset_0] = dist + 1
            queue.append((next_subset_0, dist + 1))
            
        # Try adding '1'
        next_subset_1 = frozenset(trans_1[i] for i in subset)
        if next_subset_1 not in visited:
            if len(next_subset_1) == 1:
                return dist + 1
            visited[next_subset_1] = dist + 1
            queue.append((next_subset_1, dist + 1))
            
    return -1

def construct():
    # Start with the pinwheel automaton (score 11)
    best_trans_0 = [(i + 1) % N for i in range(N)]
    best_trans_1 = [0] * N
    for i in range(1, N):
        best_trans_1[i] = (i - 1) % N
        
    best_score = calculate_threshold(best_trans_0, best_trans_1)
    
    # Heuristic search to improve
    # We want to maximize the reset length.
    # Randomly mutate trans_1 while keeping trans_0 as a cycle (often optimal for this class).
    # Or mutate both.
    
    start_time = time.time()
    max_time = 50.0 # Safety margin
    
    attempts = 0
    random.seed(42)
    
    while time.time() - start_time < max_time:
        attempts += 1
        # Generate a random automaton biased towards structure
        # Keep trans_0 as cycle
        trans_0 = best_trans_0[:]
        
        # Mutate trans_1
        trans_1 = best_trans_1[:]
        # Change 1-2 random mappings
        for _ in range(random.randint(1, 3)):
            idx = random.randint(0, N - 1)
            trans_1[idx] = random.randint(0, N - 1)
            
        score = calculate_threshold(trans_0, trans_1)
        if score > best_score:
            best_score = score
            best_trans_1 = trans_1[:]
            # If we hit the theoretical max, stop
            if best_score >= (N - 1) ** 2:
                break
                
    return [best_trans_0, best_trans_1]
