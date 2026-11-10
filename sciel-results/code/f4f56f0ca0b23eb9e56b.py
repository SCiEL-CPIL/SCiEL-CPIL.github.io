"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def _score(A):
    """Return max(|A+A|, |A*A|) for sorted list A of length 20."""
    n = len(A)
    ss = set()
    ps = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            ss.add(ai + A[j])
            ps.add(ai * A[j])
    return max(len(ss), len(ps))

def _valid(A, n=20, maximum=10000):
    if len(A) != n:
        return False
    if len(set(A)) != n:
        return False
    if any(x <= 0 or x > maximum for x in A):
        return False
    return True

def _generate_smooth_numbers(max_val, primes=[2, 3, 5]):
    """Generate 5-smooth numbers up to max_val."""
    smooth = set()
    def gen(idx, current):
        if current > max_val:
            return
        if idx == len(primes):
            if current > 0:
                smooth.add(current)
            return
        p = primes[idx]
        val = current
        while val <= max_val:
            gen(idx + 1, val)
            if val > max_val // p:
                break
            val *= p
    
    gen(0, 1)
    return sorted(smooth)

def _generate_candidates(max_val=10000):
    """Generate a pool of candidate numbers for local search."""
    candidates = set()
    
    # 1. Smooth numbers (2, 3, 5)
    smooth = _generate_smooth_numbers(max_val)
    candidates.update(smooth)
    
    # 2. Geometric progressions
    for base in [2, 3, 4, 5]:
        val = 1
        while val <= max_val:
            candidates.add(val)
            val *= base
            
    # 3. Small integers
    for i in range(1, 100):
        candidates.add(i)
        
    # 4. Multiples of small numbers
    for i in range(1, 100):
        for j in range(1, 100):
            if i * j <= max_val:
                candidates.add(i * j)
                
    return sorted(candidates)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = 20
    maximum = 10000
    
    # Initial Candidates
    candidates_A = []
    
    # 1. Pure Geometric (Base 2, truncated) + fill
    gp2 = [2**i for i in range(14)] # 1 to 8192
    # Fill remaining with small odd numbers to keep sumset small?
    # Actually, mixing large GP with small numbers might increase sumset.
    # Let's try just the first 20 smooth numbers.
    smooth_nums = _generate_smooth_numbers(maximum)
    if len(smooth_nums) >= n:
        candidates_A.append(sorted(smooth_nums[:n]))
        
    # 2. Geometric Base 2 + some 3s
    # {2^i} U {3*2^i}
    mixed = set()
    for i in range(14):
        mixed.add(2**i)
    for i in range(10):
        if 3 * (2**i) <= maximum:
            mixed.add(3 * (2**i))
    mixed_list = sorted(mixed)
    if len(mixed_list) >= n:
        candidates_A.append(mixed_list[:n])
        
    # 3. Grid in log space: {2^i 3^j}
    grid = set()
    for i in range(14):
        for j in range(14):
            val = (2**i) * (3**j)
            if val <= maximum:
                grid.add(val)
    grid_list = sorted(grid)
    if len(grid_list) >= n:
        candidates_A.append(grid_list[:n])
        
    # 4. Random smooth sets
    rng = random.Random(42)
    for _ in range(5):
        if len(smooth_nums) >= n:
            cand = rng.sample(smooth_nums, n)
            candidates_A.append(sorted(cand))
            
    # Evaluate initial candidates
    best_A = None
    best_score = float('inf')
    for A in candidates_A:
        if _valid(A, n, maximum):
            sc = _score(A)
            if sc < best_score:
                best_score = sc
                best_A = A
                
    if best_A is None:
        best_A = list(range(1, n+1))
        best_score = _score(best_A)
        
    # Local Search
    A = list(best_A)
    A_set = set(A)
    
    # Build candidate pool for swaps
    # Include current elements, smooth numbers, and neighbors
    pool = set()
    pool.update(A)
    pool.update(smooth_nums)
    
    # Add neighbors of current elements
    for x in A:
        for d in range(-5, 6):
            if x + d > 0 and x + d <= maximum:
                pool.add(x + d)
        # Multiplicative neighbors
        if x % 2 == 0:
            pool.add(x // 2)
        if x * 2 <= maximum:
            pool.add(x * 2)
        if x % 3 == 0:
            pool.add(x // 3)
        if x * 3 <= maximum:
            pool.add(x * 3)
            
    pool = sorted(pool)
    
    # Limit pool size if it's too large (shouldn't be, but just in case)
    if len(pool) > 5000:
        # Keep a diverse subset
        pool = pool[:2000] + pool[-2000:]
        pool = sorted(set(pool))
        
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle positions to avoid bias? Or iterate all.
        # Iterating all is deterministic and thorough.
        for i in range(n):
            old_val = A[i]
            # Try replacing A[i] with values from pool
            # To speed up, we can pre-filter pool to values not in A (except old_val)
            # But A changes, so we check on the fly.
            
            # Optimization: Sort pool by "closeness" to old_val? 
            # Or just iterate.
            for new_val in pool:
                if new_val == old_val:
                    continue
                if new_val in A_set:
                    continue
                    
                A[i] = new_val
                A_set.discard(old_val)
                A_set.add(new_val)
                
                sc = _score(A)
                
                if sc < best_score:
                    best_score = sc
                    improved = True
                    # Keep the change
                    break
                else:
                    # Revert
                    A[i] = old_val
                    A_set.discard(new_val)
                    A_set.add(old_val)
                    
            if improved:
                # Restart the scan from the beginning? 
                # Or continue? 
                # Continuing is faster, but might miss global improvements in earlier indices.
                # Let's break and restart the outer loop to be safe.
                break
                
        if not improved:
            # Try random perturbations to escape local minima
            for _ in range(100):
                if time.time() > deadline:
                    break
                i = rng.randint(0, n-1)
                old_val = A[i]
                new_val = rng.choice(pool)
                if new_val == old_val or new_val in A_set:
                    continue
                    
                A[i] = new_val
                A_set.discard(old_val)
                A_set.add(new_val)
                
                sc = _score(A)
                
                if sc < best_score:
                    best_score = sc
                    improved = True
                    break
                else:
                    A[i] = old_val
                    A_set.discard(new_val)
                    A_set.add(old_val)
                    
            if not improved:
                # Try a slightly different pool or restart with a different seed?
                # For now, break.
                break
                
    # Final validation
    if not _valid(A, n, maximum):
        A = list(range(1, n+1))
        
    return A
# EVOLVE-BLOCK-END
