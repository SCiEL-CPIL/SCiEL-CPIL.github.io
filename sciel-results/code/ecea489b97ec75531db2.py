import time
import random

def construct():
    start = time.time()
    deadline = start + 55.0

    def elapsed():
        return time.time() - start

    N = 256
    neighbors = [0] * N
    
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            if (a * b) % (a + b) == 0:
                neighbors[i] |= (1 << j)
                neighbors[j] |= (1 << i)

    def greedy_init():
        remaining = list(range(N))
        random.shuffle(remaining)
        chosen_mask = 0
        for v in remaining:
            if (neighbors[v] & chosen_mask) == 0:
                chosen_mask |= (1 << v)
        return chosen_mask

    def mask_to_set(mask):
        return [i for i in range(N) if (mask >> i) & 1]

    def local_search(initial_mask, max_iters=5000):
        chosen_mask = initial_mask
        best_mask = initial_mask
        best_size = bin(initial_mask).count('1')
        
        # Precompute list of vertices for iteration
        vertices = list(range(N))
        
        for _ in range(max_iters):
            if elapsed() > deadline:
                break
            
            # 1. Try to add a vertex
            # Find a vertex v not in chosen_mask such that neighbors[v] & chosen_mask == 0
            # We can iterate through all v
            added = False
            # Optimization: iterate through unchosen vertices
            # To speed up, we can maintain a list of "candidates" but it's complex.
            # Just loop 0..N-1
            for v in vertices:
                if not (chosen_mask >> v) & 1:
                    if (neighbors[v] & chosen_mask) == 0:
                        chosen_mask |= (1 << v)
                        added = True
                        break
            
            if added:
                new_size = bin(chosen_mask).count('1')
                if new_size > best_size:
                    best_size = new_size
                    best_mask = chosen_mask
                continue

            # 2. Stuck. Perform a kick.
            # Remove k random vertices
            k = random.randint(2, 5)
            # Get list of chosen vertices
            chosen_list = [i for i in range(N) if (chosen_mask >> i) & 1]
            if len(chosen_list) > k:
                to_remove = random.sample(chosen_list, k)
                for r in to_remove:
                    chosen_mask &= ~(1 << r)
                
                # Greedy re-add
                # Shuffle vertices to break symmetry
                random.shuffle(vertices)
                for v in vertices:
                    if not (chosen_mask >> v) & 1:
                        if (neighbors[v] & chosen_mask) == 0:
                            chosen_mask |= (1 << v)
                
                new_size = bin(chosen_mask).count('1')
                if new_size > best_size:
                    best_size = new_size
                    best_mask = chosen_mask

        return best_mask

    best_mask = 0
    best_size = 0
    
    # Run multiple restarts
    while elapsed() < deadline - 2.0:
        init_mask = greedy_init()
        res_mask = local_search(init_mask, max_iters=2000)
        
        if bin(res_mask).count('1') > best_size:
            best_size = bin(res_mask).count('1')
            best_mask = res_mask

    # Final polish: try to add any remaining
    for v in range(N):
        if not (best_mask >> v) & 1:
            if (neighbors[v] & best_mask) == 0:
                best_mask |= (1 << v)
                best_size += 1

    return mask_to_set(best_mask)
