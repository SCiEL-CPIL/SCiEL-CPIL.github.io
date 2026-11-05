import random
import time
import sys

N = 12

def reset_threshold(trans_a, trans_b):
    """
    Compute the exact shortest reset word length using BFS on the subset automaton.
    Optimized for speed using list-based queue and integer array for visited.
    trans_a, trans_b: lists of length N, where trans_a[i] is the destination of state i under letter 'a'.
    Returns the length of the shortest reset word, or -1 if not synchronizing.
    """
    full_mask = (1 << N) - 1
    max_states = 1 << N
    
    # Use a list for the queue and an array for visited distances
    # visited[mask] = distance, -1 if unvisited
    visited = [-1] * max_states
    visited[full_mask] = 0
    
    # Use list as queue (index pointer for O(1) pop)
    queue = [full_mask]
    head = 0
    
    while head < len(queue):
        current = queue[head]
        head += 1
        dist = visited[current]
        
        # Check if single state (reset)
        if current & (current - 1) == 0:
            return dist
        
        # Calculate next states for 'a' and 'b'
        next_a = 0
        next_b = 0
        
        # Iterate through bits in current mask
        # This is faster than iterating 0..N-1 and checking bit
        m = current
        while m:
            # Get lowest set bit
            lsb = m & (-m)
            # Index is log2(lsb), but we can precompute or use bit_length
            # bit_length - 1 gives the index
            i = lsb.bit_length() - 1
            next_a |= (1 << trans_a[i])
            next_b |= (1 << trans_b[i])
            m ^= lsb
            
        # Check next_a
        if visited[next_a] == -1:
            visited[next_a] = dist + 1
            queue.append(next_a)
            
        # Check next_b
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
    
    def random_automaton():
        ta = [random.randint(0, N - 1) for _ in range(N)]
        tb = [random.randint(0, N - 1) for _ in range(N)]
        return ta, tb

    def structured_automaton():
        # Start with a cycle for 'a' and random/constant for 'b'
        # This structure is known to yield high thresholds
        ta = [(i + 1) % N for i in range(N)]
        # 50% chance constant, 50% random
        if random.random() < 0.5:
            c = random.randint(0, N - 1)
            tb = [c] * N
        else:
            tb = [random.randint(0, N - 1) for _ in range(N)]
        return ta, tb

    def local_search(init_a, init_b, max_steps=1000):
        current_a = list(init_a)
        current_b = list(init_b)
        current_score = reset_threshold(current_a, current_b)
        
        if current_score < 0:
            return current_a, current_b, current_score

        steps = 0
        no_improve_count = 0
        max_no_improve = 50
        
        while steps < max_steps and no_improve_count < max_no_improve:
            improved = False
            steps += 1
            
            # Randomly select a letter and a state to modify
            letter = random.choice(['a', 'b'])
            state_idx = random.randint(0, N - 1)
            
            old_val = current_a[state_idx] if letter == 'a' else current_b[state_idx]
            
            # Try a few random new values for this state
            # Instead of trying all N, try a random subset or just one random one
            # To be effective, we try all N but only if it's cheap? 
            # No, BFS is the bottleneck. Let's try 3 random changes.
            candidates = random.sample(range(N), min(N, 3))
            
            for new_val in candidates:
                if new_val == old_val:
                    continue
                
                if letter == 'a':
                    current_a[state_idx] = new_val
                    new_score = reset_threshold(current_a, current_b)
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                        break
                    current_a[state_idx] = old_val
                else:
                    current_b[state_idx] = new_val
                    new_score = reset_threshold(current_a, current_b)
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                        break
                    current_b[state_idx] = old_val
            
            if not improved:
                no_improve_count += 1
            else:
                no_improve_count = 0
                
        return current_a, current_b, current_score

    # Main search loop
    while time.time() - start_time < timeout:
        # Choose initialization strategy
        if random.random() < 0.7:
            init_a, init_b = structured_automaton()
        else:
            init_a, init_b = random_automaton()
            
        # Run local search
        # Adjust max_steps based on remaining time? 
        # Keep it fixed but allow many restarts.
        new_a, new_b, score = local_search(init_a, init_b, max_steps=200)
        
        if score > best_score:
            best_score = score
            best_trans_a = new_a
            best_trans_b = new_b
            
            # Optional: If we find a very good score, do a deeper search on it
            if best_score > 60:
                # Do a few more local searches starting from this best
                for _ in range(3):
                    if time.time() - start_time > timeout - 5:
                        break
                    # Mutate best
                    mut_a = list(best_trans_a)
                    mut_b = list(best_trans_b)
                    for _ in range(2):
                        if random.random() < 0.5:
                            mut_a[random.randint(0, N-1)] = random.randint(0, N-1)
                        else:
                            mut_b[random.randint(0, N-1)] = random.randint(0, N-1)
                    deep_a, deep_b, deep_score = local_search(mut_a, mut_b, max_steps=100)
                    if deep_score > best_score:
                        best_score = deep_score
                        best_trans_a = deep_a
                        best_trans_b = deep_b

    if best_trans_a is None or best_score < 0:
        # Fallback: Cycle + Constant is always synchronizing
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
