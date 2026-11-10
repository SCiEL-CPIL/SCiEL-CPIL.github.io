import time
import random
import math
import itertools

def calc_score(A):
    """
    Calculate the max of the size of the sumset and product set of A.
    A is a list of distinct positive integers.
    """
    n = len(A)
    sumset = set()
    prodset = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sumset.add(ai + aj)
            prodset.add(ai * aj)
    return max(len(sumset), len(prodset))

def generate_candidate_pool():
    """
    Generate a pool of candidate integers likely to be in optimal sets.
    Focus on small integers (good for sumsets) and smooth numbers (good for product sets).
    """
    candidates = set()
    
    # 1. Small integers (1-300)
    # These are crucial for keeping sums small and allowing many collisions.
    candidates.update(range(1, 301))
    
    # 2. 2, 3, 5, 7-smooth numbers
    # Numbers with only small prime factors tend to have more multiplicative collisions.
    primes = [2, 3, 5, 7]
    max_val = 10000
    
    def gen_smooth(primes, max_val, current=1, index=0):
        if current > max_val:
            return
        if index == len(primes):
            candidates.add(current)
            return
        p = primes[index]
        gen_smooth(primes, max_val, current, index + 1)
        val = current
        while val * p <= max_val:
            val *= p
            gen_smooth(primes, max_val, val, index)
            
    gen_smooth(primes, max_val)
    
    return sorted(candidates)

def local_search(A, time_limit, candidate_pool, rng):
    """
    Perform local search on the set A.
    Tries to replace elements with candidates from the pool to improve the score.
    """
    A = list(A)
    A.sort()
    n = len(A)
    current_score = calc_score(A)
    best_score = current_score
    best_A = list(A)
    start_time = time.time()
    
    # Pre-compute a smaller, high-priority subset of the candidate pool
    # Small numbers and very smooth numbers are the most likely improvements.
    # We limit the pool size to keep the inner loop fast.
    # Prioritize: 1-100, then smooth numbers up to 2000.
    priority_pool = [x for x in candidate_pool if x <= 2000]
    if len(priority_pool) > 800:
        # Ensure we have a mix of small and smooth
        smalls = [x for x in priority_pool if x <= 100]
        smooths = [x for x in priority_pool if x > 100]
        # Shuffle smooths to avoid bias towards specific values in early iterations
        rng.shuffle(smooths)
        priority_pool = smalls + smooths[:800 - len(smalls)]
        
    improved = True
    # We will iterate multiple times until no improvement or time runs out
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        A_set = set(A)
        
        for i in range(n):
            if (time.time() - start_time) > time_limit:
                return best_A, best_score
            
            old_val = A[i]
            
            # Try replacing A[i] with a candidate
            # We iterate through the priority pool.
            for new_val in priority_pool:
                if new_val in A_set or new_val == old_val:
                    continue
                
                new_A = A.copy()
                new_A[i] = new_val
                # Maintain sorted order for potential future optimizations, 
                # but calc_score doesn't strictly need it.
                
                score = calc_score(new_A)
                
                if score < current_score:
                    current_score = score
                    A = new_A
                    A_set.remove(old_val)
                    A_set.add(new_val)
                    improved = True
                    
                    if score < best_score:
                        best_score = score
                        best_A = list(A)
                    
                    # If we improved, we might want to restart the inner loop 
                    # to try other candidates for this index or move to next.
                    # Breaking here ensures we re-evaluate the impact of the change 
                    # against the new set state.
                    break 
                    
    return best_A, best_score

def construct():
    start_time = time.time()
    # Safety margin: 85 seconds (limit is 90s)
    deadline = start_time + 85.0
    
    rng = random.Random(42)
    
    # Generate the candidate pool once
    candidate_pool = generate_candidate_pool()
    
    best_A = None
    best_score = float('inf')
    
    # Generate diverse initial candidates
    initial_candidates = []
    
    # 1. Arithmetic Progressions
    # A = {a, a+d, ..., a+19d}
    # Small 'd' keeps the set compact, which helps sumsets.
    # 'a' should be small to keep products small.
    for a in range(1, 11):
        for d in range(1, 11):
            last = a + 19 * d
            if last > 10000:
                continue
            A = [a + i * d for i in range(20)]
            if len(set(A)) == 20:
                initial_candidates.append(A)
                
    # 2. Hybrid Sets: S U cS
    # S = {1, ..., k}, A = S U {c, 2c, ..., kc}
    # This structure often balances sum and product sets well.
    for k in range(5, 11):
        for c in range(k + 1, k + 20):
            S = list(range(1, k + 1))
            T = [c * x for x in S]
            A = S + T
            if len(set(A)) == 20 and max(A) <= 10000:
                initial_candidates.append(sorted(A))
                
    # 3. Geometric Progressions (small ratio)
    # A = {a * r^i}
    for a in range(1, 11):
        for r in range(2, 4):
            A = []
            val = a
            for _ in range(20):
                if val > 10000:
                    break
                A.append(val)
                val *= r
            if len(A) == 20:
                initial_candidates.append(A)
                
    # 4. Random Smooth Sets
    smooth_nums = [x for x in candidate_pool if x <= 500]
    for _ in range(10):
        try:
            A = rng.sample(smooth_nums, 20)
            A = sorted(A)
            if max(A) <= 10000:
                initial_candidates.append(A)
        except ValueError:
            continue
            
    # 5. Lattice-based Construction (2,3-smooth)
    # Generate 2^a * 3^b values that are distinct and <= 10000
    # We want to pick 20 such numbers.
    # To minimize sumset, we want them small.
    # To minimize productset, we want exponents to collide.
    # Let's try a grid of (a,b) and pick 20 points.
    # We'll try a few different "shapes" of point clouds in the exponent space.
    
    # Shape 1: Compact square-ish
    pts = []
    for a in range(0, 12):
        for b in range(0, 12):
            val = (2 ** a) * (3 ** b)
            if val <= 10000:
                pts.append((a, b, val))
    
    # We need to select 20 distinct values from these.
    # Sorting by value gives the smallest sums.
    pts.sort(key=lambda x: x[2])
    
    # Try taking the first 20 smallest
    if len(pts) >= 20:
        A_lattice1 = [p[2] for p in pts[:20]]
        if len(set(A_lattice1)) == 20:
            initial_candidates.append(sorted(A_lattice1))
            
    # Try taking a more "spread out" set to potentially reduce product set size?
    # Actually, for 2^a 3^b, the product is 2^(a1+a2) 3^(b1+b2).
    # The number of distinct products is the number of distinct (a1+a2, b1+b2) pairs.
    # If we pick points on a line in exponent space, the sumset of exponents is small.
    # If we pick points on a square, the sumset of exponents is larger.
    # We want small product set, so we want the exponent sumset to be small.
    # A line in exponent space (e.g., a=b) gives distinct values 6^a.
    # But 6^0..6^19 exceeds 10000 quickly.
    # Let's try a few specific subsets of the lattice.
    
    # Subset: a in 0..9, b in 0..9, but pick a specific pattern
    # Pattern 1: a=b (powers of 6)
    A_pow6 = []
    for a in range(20):
        val = 6 ** a
        if val <= 10000:
            A_pow6.append(val)
    if len(A_pow6) == 20:
        initial_candidates.append(A_pow6)
        
    # Pattern 2: a in 0..19, b=0 (powers of 2)
    A_pow2 = [2 ** i for i in range(20)]
    if max(A_pow2) <= 10000:
        initial_candidates.append(A_pow2)
        
    # Pattern 3: a in 0..19, b=1 (2*3^i? No, 2^a * 3)
    A_pow2_3 = [2 ** i * 3 for i in range(20)]
    if max(A_pow2_3) <= 10000:
        initial_candidates.append(A_pow2_3)

    # Deduplicate initial candidates
    unique_candidates = []
    seen = set()
    for A in initial_candidates:
        key = tuple(sorted(A))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(A)
            
    # Sort initial candidates by their initial score to prioritize good starting points
    # This is a one-time cost.
    scored_candidates = []
    for A in unique_candidates:
        score = calc_score(A)
        scored_candidates.append((score, A))
    scored_candidates.sort(key=lambda x: x[0])
    
    # Allocate time for local search on the best candidates
    # We will process candidates in order of their initial score.
    # We stop if we run out of time.
    
    num_candidates = len(scored_candidates)
    # Estimate time per candidate. 
    # Local search can be fast if it converges quickly, or slow if it explores a lot.
    # We give each candidate a slice of the remaining time, but cap it to ensure we try multiple.
    # Let's aim to try at least 10-20 candidates if possible.
    
    for idx, (init_score, A) in enumerate(scored_candidates):
        if time.time() > deadline - 2.0: # Keep 2s buffer
            break
            
        # If the initial score is already worse than our best, and we've tried enough, we can skip?
        # No, local search might improve it significantly. But if init_score is very bad, it's unlikely.
        if init_score > best_score * 1.5 and idx > 5:
            # Heuristic: skip very poor candidates if we have a decent best
            continue
            
        remaining_time = deadline - time.time()
        # Allocate a fraction of the remaining time, but at least 0.5s, max 5s per candidate
        alloc_time = max(0.5, min(5.0, remaining_time / max(1, num_candidates - idx)))
        
        improved_A, score = local_search(A, alloc_time, candidate_pool, rng)
        
        if score < best_score:
            best_score = score
            best_A = improved_A
            
    # Fallback
    if best_A is None:
        best_A = list(range(1, 21))
        best_score = calc_score(best_A)
        
    return best_A

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Result: {sorted(res)}")
    print(f"Score: {calc_score(res)}")
    print(f"Time: {end - start:.2f}s")
