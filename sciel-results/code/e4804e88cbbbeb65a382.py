import time
import random
import numpy as np

def _objective(A):
    """
    Compute max(|A+A|, |A*A|) for a set A of 20 distinct positive integers.
    Uses NumPy vectorization for speed.
    """
    A = np.array(A, dtype=np.int64)
    # Sums: A[:, None] + A[None, :]
    sums = (A[:, None] + A[None, :]).ravel()
    # Products: A[:, None] * A[None, :]
    prods = (A[:, None] * A[None, :]).ravel()
    
    # Count unique values
    # Using np.unique is generally fast for ~400 elements
    n_sums = len(np.unique(sums))
    n_prods = len(np.unique(prods))
    
    return max(n_sums, n_prods)

def _is_valid(A, n=20, maximum=10000):
    """Check if A has n distinct positive integers <= maximum."""
    if len(A) != n:
        return False
    if len(set(A)) != n:
        return False
    if any(x <= 0 for x in A):
        return False
    if any(x > maximum for x in A):
        return False
    return True

def _generate_smooth_numbers(rng, n=20, max_val=10000):
    """
    Generate a set of 'smooth' numbers (numbers with small prime factors).
    These tend to have more multiplicative collisions.
    """
    # Small primes
    primes = [2, 3, 5, 7, 11, 13]
    candidates = set()
    # Generate numbers of form p1^a * p2^b * ...
    # Simple BFS/DFS on exponents
    current = [1]
    for _ in range(10): # Depth
        new_current = []
        for c in current:
            for p in primes:
                nc = c * p
                if nc <= max_val:
                    candidates.add(nc)
                    new_current.append(nc)
        current = new_current
    
    # We need 20 distinct.
    # Smooth numbers cluster at the bottom.
    # To get 20 distinct, we might need to go up to a few hundred or thousand.
    # Let's just take the first 20 smallest smooth numbers > 1? 
    # Or mix small smooth and large smooth?
    
    # Actually, a geometric progression of smooth numbers is good.
    # Let's try to construct a set that is a mix.
    
    # Strategy: Start with 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36
    # These are all <= 36 and very smooth.
    # Let's generate a list of smooth numbers up to 10000, sort them, and pick a diverse subset.
    
    smooth_list = sorted(candidates)
    if len(smooth_list) < n:
        # Fallback to random
        return rng.sample(range(1, max_val+1), n)
    
    # We want to minimize product set. 
    # A set like {2^i * 3^j} is good.
    # Let's pick numbers that are powers of 2 and 3 mostly, then fill in.
    
    # A specific good structure: 
    # A = {2^i * 3^j | i,j >= 0, 2^i * 3^j <= 10000}
    # This set is closed under multiplication (mostly).
    # Let's just take the first 20 from a grid of 2^i * 3^j.
    
    grid = []
    for i in range(14): # 2^13 = 8192
        for j in range(9): # 3^8 = 6561
            val = (2 ** i) * (3 ** j)
            if val <= max_val:
                grid.append(val)
    
    # Sort and take first 20? They will be small.
    # Small numbers -> small products? Not necessarily, but small numbers have more collisions in products if they share factors.
    # However, if all numbers are small (<= 50), the sum set is small (<= 100) and product set is small (<= 2500).
    # But we need 20 DISTINCT.
    # The first 20 smooth numbers are: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36.
    # Let's use these as a strong seed.
    
    return sorted(grid[:n])

def _generate_random_structured(rng, n=20, max_val=10000):
    """
    Generate a set with some structure: 
    Half small random, half geometric-like.
    """
    # Small numbers: 1..50
    smalls = rng.sample(range(1, 51), 10)
    # Large numbers: Geometric progression from 100 to 10000
    # 100 * 1.5^k
    larges = []
    cur = 100
    while len(larges) < 10 and cur < max_val:
        larges.append(int(cur))
        cur *= 1.5
        # Add some noise
        if rng.random() < 0.5:
            cur += rng.randint(-10, 10)
    
    # Ensure distinctness and validity
    A = smalls + larges
    # Fix duplicates
    A = list(set(A))
    while len(A) < n:
        A.append(rng.randint(1, max_val))
    
    return sorted(A[:n])

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    rng = random.Random(123)
    
    # 1. Generate initial candidates
    candidates = []
    
    # Candidate 1: Smooth numbers (2^i * 3^j)
    A1 = _generate_smooth_numbers(rng)
    if _is_valid(A1):
        candidates.append(A1)
        
    # Candidate 2: Random structured
    A2 = _generate_random_structured(rng)
    if _is_valid(A2):
        candidates.append(A2)
        
    # Candidate 3: Pure Random
    A3 = rng.sample(range(1, 10001), 20)
    candidates.append(A3)
    
    # Evaluate candidates
    best_A = None
    best_score = float('inf')
    
    for A in candidates:
        s = _objective(A)
        if s < best_score:
            best_score = s
            best_A = A[:]
            
    # 2. Collision-Directed Local Search
    # Strategy: 
    # Instead of random moves, look for pairs (i, j) and (k, l) such that A[i]*A[j] == A[k]*A[l] is close.
    # Or simply: For each element, try to replace it with a value that creates a new collision.
    # 
    # Simpler effective heuristic:
    # Try to replace A[i] with A[j] * A[k] // A[m] if integer and valid.
    # This forces a multiplicative relation.
    
    current_A = best_A[:]
    current_score = best_score
    
    no_improvement_count = 0
    max_no_improvement = 200
    
    while time.time() < deadline and no_improvement_count < max_no_improvement:
        improved = False
        rng.shuffle(current_A) # Randomize order to break symmetry
        
        for i in range(20):
            if time.time() > deadline:
                break
            
            old_val = current_A[i]
            
            # Generate potential new values based on collisions
            # Type 1: A[j] * A[k] // A[m]
            # Type 2: A[j] + A[k] - A[m] (for sum collisions)
            
            candidates_new_val = set()
            
            # Sample a few random j, k, m to generate candidates
            # We don't want to do all O(n^3) for every i, that's 8000 * 20 = 160k checks per iter.
            # Let's sample 10 random triples.
            for _ in range(10):
                j, k, m = rng.sample(range(20), 3)
                # Avoid using i in the formula if possible, or handle it
                if j == i or k == i or m == i:
                    continue
                
                # Multiplicative collision candidate
                num = current_A[j] * current_A[k]
                den = current_A[m]
                if num % den == 0:
                    val = num // den
                    if 1 <= val <= 10000:
                        candidates_new_val.add(val)
                
                # Additive collision candidate
                val2 = current_A[j] + current_A[k] - current_A[m]
                if 1 <= val2 <= 10000:
                    candidates_new_val.add(val2)
            
            # Also try small random perturbations
            for _ in range(5):
                delta = rng.randint(-50, 50)
                val3 = old_val + delta
                if 1 <= val3 <= 10000:
                    candidates_new_val.add(val3)
            
            # Filter valid candidates
            valid_candidates = [v for v in candidates_new_val if v != old_val and v not in current_A]
            
            # If no valid candidates, skip
            if not valid_candidates:
                continue
                
            # Evaluate top few candidates (or all if few)
            # Sort by some heuristic? Or just random sample if many?
            # Let's just evaluate up to 5 random valid candidates to save time
            if len(valid_candidates) > 5:
                valid_candidates = rng.sample(valid_candidates, 5)
                
            best_candidate_val = None
            best_candidate_score = current_score
            
            for v in valid_candidates:
                new_A = current_A[:]
                new_A[i] = v
                # Check distinctness again just in case (though filtered)
                if len(set(new_A)) != 20:
                    continue
                s = _objective(new_A)
                if s < best_candidate_score:
                    best_candidate_score = s
                    best_candidate_val = v
            
            if best_candidate_val is not None:
                # Accept
                current_A[i] = best_candidate_val
                current_score = best_candidate_score
                improved = True
                no_improvement_count = 0
                
                if current_score < best_score:
                    best_score = current_score
                    best_A = current_A[:]
                    
        if not improved:
            no_improvement_count += 1
            # If stuck, try a random restart of the local search neighborhood?
            # Or just break if we've tried many times.
            if no_improvement_count > 50:
                # Perturb current solution randomly to escape local minimum
                for _ in range(5):
                    i = rng.randint(0, 19)
                    current_A[i] = rng.randint(1, 10000)
                    if len(set(current_A)) != 20:
                        current_A[i] = old_val # revert if invalid
                no_improvement_count = 0 # Reset counter after perturbation
                current_score = _objective(current_A)
                
    return sorted(best_A)

# Keep the _score function name if required by strict interface, though we use _objective
def _score(A):
    return _objective(A)

def _is_valid(A, n=20, maximum=10000):
    return _is_valid(A, n, maximum)

# The construct function is the entry point.
# Ensure no global state leaks.

if __name__ == "__main__":
    res = construct()
    print(res)
    print(_score(res))
