import time
import random
import math

def calc_score(A):
    """
    Compute the score for a set A: max(|A+A|, |A*A|).
    """
    if not A:
        return 0
    A = list(A)
    sumset = set(a + b for a in A for b in A)
    prodset = set(a * b for a in A for b in A)
    return max(len(sumset), len(prodset))

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin before 90s limit
    
    # Candidate pool: 
    # Small numbers (1-100) are dense in sums.
    # Smooth numbers (highly composite) are dense in products.
    # We want a mix.
    # Let's generate a pool of numbers that are "structurally rich".
    # 1..100
    # Plus multiples of small numbers up to 1000.
    pool = set(range(1, 101))
    for i in range(1, 20):
        for j in range(1, 20):
            if i * j <= 1000:
                pool.add(i * j)
    # Add some random numbers up to 5000 to allow for "spreading out" if needed?
    # Actually, spreading out usually increases the set size. We want compactness.
    # So stick to small/medium numbers.
    pool = list(pool)
    random.shuffle(pool)
    
    best_set = None
    best_score = float('inf')
    
    # Strategy 1: Greedy Construction
    # Start with a small seed set, then add elements one by one 
    # that cause the smallest increase in max(|A+A|, |A*A|).
    
    def greedy_construct(seed_size=5):
        # Start with a small random subset of the pool
        # Prefer very small numbers initially to keep sums small?
        # Or prefer numbers that multiply to duplicates?
        # Let's try starting with 1, 2, 4, 8, 16 (geometric) or 1..5 (arithmetic).
        # Let's try a few different seeds.
        
        seeds = [
            [1, 2, 3, 4, 5],
            [1, 2, 4, 8, 16],
            [1, 2, 3, 6, 12], # Smooth numbers
            sorted(random.sample(pool, seed_size))
        ]
        
        best_local = None
        best_local_score = float('inf')
        
        for seed in seeds:
            A = sorted(list(set(seed)))
            if len(A) < seed_size:
                continue
                
            # Greedily add elements
            for _ in range(20 - len(A)):
                best_new_elem = None
                best_new_score = float('inf')
                
                # Sample a subset of candidates to speed up greedy step
                # Evaluating all ~2000 candidates x 5 elements is 10k set ops. 
                # Each set op on 25 elements is fast. 
                # Let's evaluate a random sample of 200 candidates per step.
                candidates = random.sample(pool, min(200, len(pool)))
                
                for c in candidates:
                    if c in A:
                        continue
                    new_A = A + [c]
                    # Quick check: if c is huge, sumset/prodset will grow significantly.
                    # We can approximate or just compute.
                    s_set = set(a + b for a in new_A for b in new_A)
                    p_set = set(a * b for a in new_A for b in new_A)
                    score = max(len(s_set), len(p_set))
                    
                    if score < best_new_score:
                        best_new_score = score
                        best_new_elem = c
                        
                if best_new_elem is not None:
                    A.append(best_new_elem)
                    A.sort()
                else:
                    # Fallback: pick smallest unused
                    for c in pool:
                        if c not in A:
                            A.append(c)
                            A.sort()
                            break
                
            final_score = calc_score(A)
            if final_score < best_local_score:
                best_local_score = final_score
                best_local = A[:]
                
        return best_local, best_local_score
    
    # Run Greedy Construction
    g_set, g_score = greedy_construct()
    if g_set:
        best_set = g_set
        best_score = g_score
        
    # Strategy 2: Local Search with Incremental Delta Evaluation
    # To fit more iterations, we calculate the change in sum/prod set size
    # when swapping one element, rather than recomputing from scratch.
    # However, the "delta" for set sizes is not trivial because removing an element
    # might remove sums/products that are shared with other elements.
    # A full recompute is O(N^2). N=20, so 400 ops.
    # 1000 swaps = 400k ops. Python can do ~1M simple ops/s.
    # So ~0.4s per 1000 swaps. We have 58s. We can do ~100k swaps.
    # This is much better than the previous SA which was likely doing fewer 
    # due to overhead or bad moves.
    
    if best_set is None:
        best_set = list(range(1, 21))
        best_score = calc_score(best_set)
        
    A = best_set[:]
    current_score = best_score
    
    # Pre-compute current sets for potential optimization? 
    # No, just use the fast recompute.
    
    # Local Search Loop
    # We will perform swaps: remove one, add one.
    # To avoid getting stuck, we accept worse moves with a small probability (SA-like) 
    # but with a very fast evaluation.
    
    T = 2.0
    T_min = 0.1
    cooling = 0.999
    
    # We want to explore the pool. 
    # The pool is ~2000 items. 
    # If we pick random from pool, we rarely hit a "good" neighbor.
    # Better: Pick a random element from A to remove. 
    # Pick a random element from a "neighborhood" of the removed element 
    # or from the small end of the pool.
    
    while time.time() < deadline:
        # Pick index to remove
        idx = random.randint(0, len(A) - 1)
        old_val = A[idx]
        
        # Pick new value
        # Bias towards small numbers and numbers close to old_val?
        # Actually, just pick from the first 500 numbers in the pool (sorted)
        # as they are most likely to be useful for compactness.
        # Let's maintain a sorted pool for this.
        if not hasattr(construct, '_sorted_pool'):
            construct._sorted_pool = sorted(pool)
        
        # Sample from the smaller half of the pool
        sample_range = min(1000, len(construct._sorted_pool))
        new_val = construct._sorted_pool[random.randint(0, sample_range - 1)]
        
        if new_val == old_val or new_val in A:
            # Retry or skip
            # If new_val in A, we need to swap with another element? 
            # No, the set must remain distinct. 
            # If new_val is already in A, we can't add it. 
            # So we must pick a new_val NOT in A.
            # This rejection sampling can be slow if A is large and pool is small.
            # But pool is ~2000, A is 20. Rejection prob is 1%.
            # So this is fine.
            continue
            
        # Perform swap
        A[idx] = new_val
        A.sort()
        
        new_score = calc_score(A)
        
        delta = new_score - current_score
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_set = A[:]
        else:
            # Revert
            A[idx] = old_val
            A.sort()
            
        T *= cooling
        if T < T_min:
            T = T_min
            # Occasionally reheat?
            # If we haven't improved in a while, maybe randomize?
            # For now, just keep going.
            
    return best_set

if __name__ == "__main__":
    res = construct()
    print(res)
    print(calc_score(res))
