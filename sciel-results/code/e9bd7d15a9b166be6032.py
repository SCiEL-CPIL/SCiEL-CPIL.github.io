import numpy as np
import time

N = 50

def find_sidon_set(n):
    """
    Find a Sidon set of maximum size in {0, ..., n-1}.
    A Sidon set has distinct pairwise sums.
    """
    # Greedy approach often works well for small n
    # Or simple backtracking
    best_set = []
    
    def is_sidon(candidate):
        sums = set()
        for i in range(len(candidate)):
            for j in range(i, len(candidate)):
                s = candidate[i] + candidate[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    # Try to build a set of size k
    # Max size for n=50 is 7
    for k in range(int(np.sqrt(n)) + 2, 0, -1):
        # Simple search for a set of size k
        # This might be slow if done naively, but n=50 is small
        # We can use a heuristic: start with 0, then pick next smallest that works
        # Better: Random restarts
        
        found = False
        for _ in range(1000): # Random restarts
            # Start with 0
            current = [0]
            # Pick remaining k-1 elements
            for _ in range(k - 1):
                # Try random elements
                attempts = 0
                while attempts < 100:
                    x = np.random.randint(1, n)
                    if x not in current:
                        candidate = sorted(current + [x])
                        if is_sidon(candidate):
                            current = candidate
                            break
                    attempts += 1
                else:
                    break
            if len(current) == k:
                return current
        # If not found with this k, try smaller
    return [0]

def construct():
    start_time = time.time()
    deadline = 145.0 # 150s limit, 5s margin
    
    # Find the best Sidon set
    sidon_set = find_sidon_set(N)
    
    # Create the height vector
    h = np.zeros(N)
    for idx in sidon_set:
        h[idx] = 1.0
        
    # Normalize to L1 = 1
    norm = np.sum(h)
    if norm > 0:
        h = h / norm
    else:
        h[0] = 1.0
        
    # Check time
    if time.time() > deadline:
        return h.tolist()
        
    return h.tolist()

# EVOLVE-BLOCK-START
# The construct function is defined above, but we need to keep the interface.
# The prompt asks to keep function names.
# I will wrap the logic in construct.
# EVOLVE-BLOCK-END
