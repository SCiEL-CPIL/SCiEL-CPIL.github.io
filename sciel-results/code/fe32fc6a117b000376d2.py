"""Program for integer sum-product extremal sets using vectorized SA."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0  # Safety margin for 90s limit
    n = 20
    max_val = 10000
    
    # 1. Initialization: Start with a set of smooth numbers (highly composite-like)
    # Smooth numbers have few prime factors, which tends to reduce the size of the product set A*A.
    # We generate a pool of smooth numbers and pick 20 distinct ones.
    
    def generate_smooth_numbers(limit=10000, count=100):
        # Numbers of the form 2^a * 3^b * 5^c * 7^d ...
        # We'll just generate a list of numbers with small prime factors
        candidates = []
        for i in range(1, limit + 1):
            x = i
            # Check if x is 7-smooth (primes <= 7)
            for p in [2, 3, 5, 7]:
                while x % p == 0:
                    x //= p
            if x == 1:
                candidates.append(i)
            if len(candidates) >= count:
                break
        return candidates

    smooth_pool = generate_smooth_numbers(max_val, 200)
    # Ensure we have at least 20
    if len(smooth_pool) < n:
        smooth_pool = list(range(1, max_val + 1))
        
    # Randomly sample 20 from the smooth pool to start
    current_set = np.array(random.sample(smooth_pool, n), dtype=np.int64)
    
    def evaluate(A):
        """
        Vectorized evaluation of max(|A+A|, |A*A|).
        A is a numpy array of shape (n,).
        """
        # A+A: all pairwise sums
        # We can use broadcasting: A[:, None] + A[None, :]
        # This creates an (n, n) matrix.
        sums = A[:, None] + A[None, :]
        # Unique sums
        # np.unique is efficient for 1D arrays. Flatten the matrix.
        num_sums = len(np.unique(sums))
        
        # A*A: all pairwise products
        prods = A[:, None] * A[None, :]
        num_prods = len(np.unique(prods))
        
        return max(num_sums, num_prods)

    current_score = evaluate(current_set)
    best_set = current_set.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    T = 1000.0  # Initial temperature
    alpha = 0.999  # Cooling rate
    min_T = 1.0
    
    # Precompute indices for unique pairs? No, we just mutate one element.
    # When we mutate A[i], we need to recalculate the full set sizes.
    # Vectorized evaluation is fast (~1-2ms for n=20), so we can do many iterations.
    
    iters = 0
    while time.time() < deadline and T > min_T:
        # Pick a random index to mutate
        idx = np.random.randint(0, n)
        old_val = current_set[idx]
        
        # Propose new value
        # Strategy: Mix of local perturbations and global jumps
        if np.random.random() < 0.7:
            # Local: +/- random small delta
            delta = np.random.randint(-50, 51)
            new_val = old_val + delta
        else:
            # Global: Random value from 1..10000
            new_val = np.random.randint(1, max_val + 1)
            
        # Clamp
        if new_val < 1 or new_val > max_val:
            T *= alpha
            continue
            
        # Check distinctness
        # If new_val is already in current_set (excluding idx), it's invalid
        if new_val in current_set and new_val != old_val:
            T *= alpha
            continue
            
        # Apply mutation
        current_set[idx] = new_val
        new_score = evaluate(current_set)
        
        # Acceptance criterion
        delta_score = new_score - current_score
        if delta_score < 0 or np.random.random() < np.exp(-delta_score / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_set = current_set.copy()
        else:
            # Revert
            current_set[idx] = old_val
            
        T *= alpha
        iters += 1
        
        # Optional: If we've been stuck for a while, try a restart?
        # For now, rely on SA.
        
    # Final Polish: Greedy local search on the best set found
    # Try to improve by swapping or tweaking individual elements
    current_set = best_set.copy()
    current_score = best_score
    
    # Quick greedy pass: for each element, try a few nearby values
    for _ in range(100): # Limit polish time
        if time.time() > deadline - 2.0: break
        improved = False
        for i in range(n):
            old_val = current_set[i]
            # Try a few random neighbors
            for _ in range(10):
                new_val = old_val + np.random.randint(-20, 21)
                if new_val < 1 or new_val > max_val: continue
                if new_val in current_set and new_val != old_val: continue
                
                current_set[i] = new_val
                new_score = evaluate(current_set)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    current_set[i] = old_val
            if improved:
                # Restart the scan? Or just continue
                pass
        
        if not improved:
            break
            
    if current_score < best_score:
        best_score = current_score
        best_set = current_set.copy()

    return list(best_set)
# EVOLVE-BLOCK-END
