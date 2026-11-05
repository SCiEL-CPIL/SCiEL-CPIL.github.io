import random
import time

N = 12
FULL_MASK = (1 << N) - 1

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
    # Use a list of -1 for visited distances, indexed by mask
    # This is faster than a dict for dense searches
    visited = [-1] * (1 << N)
    visited[FULL_MASK] = 0
    
    queue = [FULL_MASK]
    head = 0
    
    while head < len(queue):
        current = queue[head]
        head += 1
        dist = visited[current]
        
        if current & (current - 1) == 0:
            return dist
        
        bits = BITS_FOR_MASK[current]
        
        next_a = 0
        next_b = 0
        for i in bits:
            next_a |= (1 << trans_a[i])
            next_b |= (1 << trans_b[i])
        
        if visited[next_a] == -1:
            visited[next_a] = dist + 1
            queue.append(next_a)
            
        if visited[next_b] == -1:
            visited[next_b] = dist + 1
            queue.append(next_b)
    
    return -1

def construct():
    start_time = time.time()
    timeout = 55.0
    
    best_trans_a = None
    best_trans_b = None
    best_score = -1
    
    def evaluate(trans_a, trans_b):
        return reset_threshold(trans_a, trans_b)
    
    def hill_climb(trans_a, trans_b, max_iterations=50):
        """
        Perform best-improvement hill climbing.
        Try all single-bit changes, accept the best one.
        """
        current_a = list(trans_a)
        current_b = list(trans_b)
        current_score = evaluate(current_a, current_b)
        
        if current_score < 0:
            return current_a, current_b, current_score

        for _ in range(max_iterations):
            best_delta = 0
            best_change = None
            
            # Try changing trans_a[i] for all i and all new values
            for i in range(N):
                old_val = current_a[i]
                for new_val in range(N):
                    if new_val == old_val:
                        continue
                    current_a[i] = new_val
                    new_score = evaluate(current_a, current_b)
                    current_a[i] = old_val
                    if new_score > current_score + best_delta:
                        best_delta = new_score - current_score
                        best_change = ('a', i, new_val)
            
            # Try changing trans_b[i] for all i and all new values
            for i in range(N):
                old_val = current_b[i]
                for new_val in range(N):
                    if new_val == old_val:
                        continue
                    current_b[i] = new_val
                    new_score = evaluate(current_a, current_b)
                    current_b[i] = old_val
                    if new_score > current_score + best_delta:
                        best_delta = new_score - current_score
                        best_change = ('b', i, new_val)
            
            if best_change is None:
                break
            
            letter, i, new_val = best_change
            if letter == 'a':
                current_a[i] = new_val
            else:
                current_b[i] = new_val
            current_score += best_delta
        
        return current_a, current_b, current_score

    def random_automaton():
        ta = [random.randint(0, N - 1) for _ in range(N)]
        tb = [random.randint(0, N - 1) for _ in range(N)]
        return ta, tb

    def structured_automaton():
        # Generate a random permutation for 'a' (cycle-like but not necessarily one cycle)
        # Or just a random function that is likely to be synchronizing with a good 'b'
        ta = [random.randint(0, N - 1) for _ in range(N)]
        # For 'b', use a "reset-like" structure: most map to a single state, some random
        # This increases the chance of synchronization
        target = random.randint(0, N - 1)
        tb = [target] * N
        num_random = random.randint(0, 3)
        indices = random.sample(range(N), num_random)
        for idx in indices:
            tb[idx] = random.randint(0, N - 1)
        return ta, tb

    # Main loop: Large-scale random restart local search
    # Each iteration: generate a candidate, hill climb it, update best.
    # The key is that reset_threshold is fast (bitmask BFS), so we can do many iterations.
    
    while time.time() - start_time < timeout:
        # Alternate between random and structured initializations
        # Structured ones are more likely to be synchronizing
        if random.random() < 0.3:
            init_a, init_b = random_automaton()
        else:
            init_a, init_b = structured_automaton()
        
        # Perform hill climbing
        new_a, new_b, score = hill_climb(init_a, init_b)
        
        if score > best_score:
            best_score = score
            best_trans_a = new_a
            best_trans_b = new_b

    if best_trans_a is None or best_score < 0:
        # Fallback: Known good automaton or simple cycle
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
