import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    N = 512
    K = 32
    
    # Initialize random set
    # Using 0-based indexing for bits, 1-based for output
    current_set = set(random.sample(range(N), K))
    current_mask = 0
    for x in current_set:
        current_mask |= (1 << x)
        
    def score_mask(M):
        count = 0
        # d must be such that 2*d < 512
        # Max d is 255
        for d in range(1, 256):
            if (M & (M >> d) & (M >> (2 * d))) != 0:
                count += 1
        return count

    current_score = score_mask(current_mask)
    best_score = current_score
    best_mask = current_mask
    best_set = set(current_set)
    
    # Simulated Annealing parameters
    T = 10.0
    cooling_rate = 0.99999  # Adjust based on expected iterations
    # Let's estimate iterations.
    # Score function ~ 255 big int ops. Very fast.
    # Let's aim for 500,000 iterations.
    # With cooling rate, we need to balance.
    # Alternatively, fixed number of iterations at each T.
    
    iterations = 0
    while time.time() < deadline:
        # Generate neighbor
        # Pick an element to remove
        # We can pick a random index from the current set
        # Using list for faster random access?
        # current_set is a set. list(current_set) is O(K).
        # K=32 is small.
        elem_to_remove = random.choice(list(current_set))
        
        # Pick an element to add
        # Rejection sampling
        while True:
            elem_to_add = random.randint(0, N - 1)
            if not (current_mask & (1 << elem_to_add)):
                break
        
        # Calculate new mask
        new_mask = current_mask
        new_mask ^= (1 << elem_to_remove)
        new_mask ^= (1 << elem_to_add)
        
        # Score
        new_score = score_mask(new_mask)
        
        # Acceptance
        if new_score > current_score:
            # Accept
            current_set.remove(elem_to_remove)
            current_set.add(elem_to_add)
            current_mask = new_mask
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask
                best_set = set(current_set)
                
        else:
            # Accept with probability exp((new_score - current_score) / T)
            # new_score <= current_score, so delta <= 0
            delta = new_score - current_score
            if delta == 0:
                # Always accept neutral moves?
                # Helps escape plateaus
                current_set.remove(elem_to_remove)
                current_set.add(elem_to_add)
                current_mask = new_mask
                # current_score unchanged
            else:
                prob = 2.718281828459045 ** (delta / T)
                if random.random() < prob:
                    current_set.remove(elem_to_remove)
                    current_set.add(elem_to_add)
                    current_mask = new_mask
                    current_score = new_score
        
        iterations += 1
        # Cool down
        T *= cooling_rate
        if T < 0.01:
            T = 0.01
            
    # Convert best_set to list of ints 1..512
    result = sorted([x + 1 for x in best_set])
    return result

# Test locally
if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} elements")
    print(res)
    # Quick check score
    M = 0
    for x in res:
        M |= (1 << (x-1))
    cnt = 0
    for d in range(1, 256):
        if (M & (M >> d) & (M >> (2*d))) != 0:
            cnt += 1
    print(f"Score: {cnt}")
