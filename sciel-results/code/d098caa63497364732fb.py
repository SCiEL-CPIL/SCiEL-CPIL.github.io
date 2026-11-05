import random
import time
from collections import deque

N = 12

# Precompute bits for each mask to speed up BFS
BITS_FOR_MASK = []
for mask in range(1 << N):
    bits = []
    for i in range(N):
        if mask & (1 << i):
            bits.append(i)
    BITS_FOR_MASK.append(bits)

def reset_threshold(trans_a, trans_b):
    """
    Compute the exact shortest reset word length using BFS on the subset automaton.
    trans_a, trans_b: lists of length N, where trans_a[i] is the destination of state i under letter 'a'.
    Returns the length of the shortest reset word, or -1 if not synchronizing.
    """
    full_mask = (1 << N) - 1
    
    visited = {full_mask: 0}
    queue = deque([full_mask])
    
    while queue:
        current = queue.popleft()
        dist = visited[current]
        
        if current & (current - 1) == 0:
            return dist
        
        # Get indices of states in current subset
        bits = BITS_FOR_MASK[current]
        
        # Calculate next states
        next_a = 0
        next_b = 0
        for i in bits:
            next_a |= (1 << trans_a[i])
            next_b |= (1 << trans_b[i])
        
        for nxt in (next_a, next_b):
            if nxt not in visited:
                visited[nxt] = dist + 1
                queue.append(nxt)
    
    return -1

def construct():
    start_time = time.time()
    timeout = 55.0
    
    best_trans_a = None
    best_trans_b = None
    best_score = -1
    
    def evaluate(trans_a, trans_b):
        return reset_threshold(trans_a, trans_b)
    
    def steepest_ascent(trans_a, trans_b, max_rounds=20):
        """
        Perform steepest ascent hill climbing.
        For each letter and state, try all possible destinations.
        Accept the single best change if it improves the score.
        """
        current_a = list(trans_a)
        current_b = list(trans_b)
        current_score = evaluate(current_a, current_b)
        
        if current_score < 0:
            return current_a, current_b, current_score

        improved = True
        rounds = 0
        while improved and rounds < max_rounds:
            improved = False
            rounds += 1
            best_delta = 0
            best_change = None
            
            # Try changing trans_a[i] to all possible values
            for i in range(N):
                old_val = current_a[i]
                for new_val in range(N):
                    if new_val == old_val:
                        continue
                    current_a[i] = new_val
                    new_score = evaluate(current_a, current_b)
                    delta = new_score - current_score
                    if delta > best_delta:
                        best_delta = delta
                        best_change = ('a', i, new_val)
                    current_a[i] = old_val
            
            # Try changing trans_b[i] to all possible values
            for i in range(N):
                old_val = current_b[i]
                for new_val in range(N):
                    if new_val == old_val:
                        continue
                    current_b[i] = new_val
                    new_score = evaluate(current_a, current_b)
                    delta = new_score - current_score
                    if delta > best_delta:
                        best_delta = delta
                        best_change = ('b', i, new_val)
                    current_b[i] = old_val
            
            if best_change:
                letter, idx, val = best_change
                if letter == 'a':
                    current_a[idx] = val
                else:
                    current_b[idx] = val
                current_score += best_delta
                improved = True
        
        return current_a, current_b, current_score

    def random_automaton():
        ta = [random.randint(0, N - 1) for _ in range(N)]
        tb = [random.randint(0, N - 1) for _ in range(N)]
        return ta, tb

    def cycle_automaton():
        # a is a cycle, b is random or constant
        ta = [(i + 1) % N for i in range(N)]
        tb = [random.randint(0, N - 1) for _ in range(N)]
        # Sometimes make b a constant map
        if random.random() < 0.3:
            c = random.randint(0, N - 1)
            tb = [c] * N
        return ta, tb

    def local_search_with_restarts(init_a, init_b, max_restarts=5):
        best_a = list(init_a)
        best_b = list(init_b)
        best_score = evaluate(best_a, best_b)
        
        current_a = list(init_a)
        current_b = list(init_b)
        
        for r in range(max_restarts):
            # Run steepest ascent
            new_a, new_b, score = steepest_ascent(current_a, current_b)
            
            if score > best_score:
                best_score = score
                best_a = list(new_a)
                best_b = list(new_b)
            
            # If we hit a local optimum, jump to a new random point or mutate
            if score < 0 or random.random() < 0.5:
                # Random restart
                current_a, current_b = random_automaton()
            else:
                # Mutation
                for _ in range(random.randint(1, 3)):
                    if random.random() < 0.5:
                        idx = random.randint(0, N-1)
                        current_a[idx] = random.randint(0, N-1)
                    else:
                        idx = random.randint(0, N-1)
                        current_b[idx] = random.randint(0, N-1)
        
        return best_a, best_b, best_score

    # Main loop
    while time.time() - start_time < timeout:
        # Mix of random and cycle-based initializations
        if random.random() < 0.5:
            init_a, init_b = random_automaton()
        else:
            init_a, init_b = cycle_automaton()
            
        best_a, best_b, score = local_search_with_restarts(init_a, init_b)
        
        if score > best_score:
            best_score = score
            best_trans_a = best_a
            best_trans_b = best_b

    if best_trans_a is None or best_score < 0:
        best_trans_a = [(i + 1) % N for i in range(N)]
        best_trans_b = [0] * N
        best_score = reset_threshold(best_trans_a, best_trans_b)
    
    return best_trans_a, best_trans_b

if __name__ == "__main__":
    ta, tb = construct()
    score = reset_threshold(ta, tb)
    print(f"Reset threshold: {score}")
    print(f"Trans_a: {ta}")
    print(f"Trans_b: {tb}")
