import random
import time
from collections import deque

N = 12

# EVOLVE-BLOCK-START
def _reset_length(a, b):
    """
    Compute the shortest reset word length for the automaton defined by a and b.
    Returns -1 if not synchronizing.
    """
    start = tuple(range(N))
    if len(set(a)) == 1 or len(set(b)) == 1:
        return 1
    
    visited = {start}
    queue = deque([(start, 0)])
    
    while queue:
        subset, dist = queue.popleft()
        if len(subset) == 1:
            return dist
        
        # Apply a
        next_a_set = set()
        for i in subset:
            next_a_set.add(a[i])
        next_a = tuple(sorted(next_a_set))
        
        if next_a not in visited:
            visited.add(next_a)
            queue.append((next_a, dist + 1))
            
        # Apply b
        next_b_set = set()
        for i in subset:
            next_b_set.add(b[i])
        next_b = tuple(sorted(next_b_set))
        
        if next_b not in visited:
            visited.add(next_b)
            queue.append((next_b, dist + 1))
            
    return -1

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    best_a = None
    best_b = None
    best_score = -1
    
    # We will run multiple SA restarts
    while time.time() < deadline:
        # Initialize
        a = [(i + 1) % N for i in range(N)]
        b = [0] * N
        # Randomize b to start search
        for i in range(N):
            b[i] = random.randint(0, N-1)
        
        current_a = a[:]
        current_b = b[:]
        current_score = _reset_length(current_a, current_b)
        
        if current_score < 0:
            # If initial random is not synchronizing, try a "safe" start
            # Map most to 0, one to 1
            b = [0] * N
            b[0] = 1
            b[N-1] = N-2
            current_b = b[:]
            current_score = _reset_length(current_a, current_b)
            if current_score < 0:
                continue # Skip this restart if still not synchronizing

        temp = 5.0
        cool_rate = 0.99
        
        # Local SA loop
        while time.time() < deadline:
            new_a = current_a[:]
            new_b = current_b[:]
            
            # Mutation
            if random.random() < 0.8:
                idx = random.randint(0, N-1)
                new_b[idx] = random.randint(0, N-1)
            else:
                idx = random.randint(0, N-1)
                new_a[idx] = random.randint(0, N-1)
            
            new_score = _reset_length(new_a, new_b)
            
            if new_score > current_score:
                current_a = new_a
                current_b = new_b
                current_score = new_score
                
                if current_score > best_score:
                    best_a = current_a[:]
                    best_b = current_b[:]
                    best_score = current_score
            else:
                if new_score < 0:
                    p = 0.005
                else:
                    delta = current_score - new_score
                    p = pow(2.71828, -delta / temp)
                
                if random.random() < p:
                    current_a = new_a
                    current_b = new_b
                    current_score = new_score
            
            temp *= cool_rate
            if temp < 0.5:
                break # Restart SA with new random init
        
        # If we have a good score, we might do a finer search
        # But the outer loop handles restarts.
        
    if best_a is None:
        # Fallback: Simple synchronizing automaton
        return [[(i + 1) % N for i in range(N)], [0] * N]
        
    return best_a, best_b
# EVOLVE-BLOCK-END
