import time
import random
import math
import numpy as np

def calc_score(A):
    """
    Calculate the max of the size of the sumset and product set of A.
    A is a list of distinct positive integers.
    Uses NumPy for speed.
    """
    A_arr = np.array(A, dtype=np.int64)
    # Create outer sum and product matrices
    sums = A_arr[:, None] + A_arr[None, :]
    prods = A_arr[:, None] * A_arr[None, :]
    
    # Get unique counts
    # np.unique is efficient for small arrays
    n_sums = len(np.unique(sums))
    n_prods = len(np.unique(prods))
    
    return max(n_sums, n_prods)

def generate_candidates():
    """
    Generate a diverse pool of candidate integers for replacement.
    Focus on small integers and smooth numbers.
    """
    candidates = set()
    
    # Small integers provide the best sumset density
    candidates.update(range(1, 101))
    
    # 2,3,5-smooth numbers reduce product set size
    # val = 2^a * 3^b * 5^c
    max_exp = 15
    for a in range(max_exp):
        p2 = 2 ** a
        if p2 > 10000:
            break
        for b in range(max_exp):
            p23 = p2 * (3 ** b)
            if p23 > 10000:
                break
            for c in range(max_exp):
                val = p23 * (5 ** c)
                if val > 10000:
                    break
                candidates.add(val)
            if 3 ** b > 10000:
                break
        if 2 ** a > 10000:
            break
            
    return sorted(candidates)

def local_search(A, time_limit, rng, candidate_pool):
    """
    Perform local search to minimize the score.
    Includes a 'kick' mechanism to escape local minima.
    """
    A = list(A)
    A.sort()
    
    current_score = calc_score(A)
    best_score = current_score
    best_A = list(A)
    start_time = time.time()
    
    no_improve_count = 0
    max_no_improve = 50 # If stuck for 50 iterations, kick
    
    while (time.time() - start_time) < time_limit:
        improved = False
        
        # Try replacing each element
        for i in range(len(A)):
            if (time.time() - start_time) > time_limit:
                return best_A, best_score
            
            old_val = A[i]
            A_set = set(A)
            
            # Try candidates
            # Shuffle candidates to avoid bias towards smaller values in pool order
            # But sorting is deterministic. 
            # We can just iterate.
            for new_val in candidate_pool:
                if new_val in A_set or new_val == old_val:
                    continue
                
                new_A = A.copy()
                new_A[i] = new_val
                
                score = calc_score(new_A)
                
                if score < current_score:
                    current_score = score
                    A = new_A
                    improved = True
                    no_improve_count = 0
                    if score < best_score:
                        best_score = score
                        best_A = list(A)
                    # Don't break, continue to see if we can improve further in this pass?
                    # Or break to restart the pass with the new A?
                    # Breaking ensures we don't use stale A_set? A_set is local.
                    # But new_A is different from A.
                    # If we don't break, we might replace A[j] based on the NEW A[i].
                    # This is fine.
                    # However, if we change A[i], the set of valid candidates for other j changes?
                    # No, we just check `new_val in A_set`. A_set is a snapshot.
                    # If we change A, A_set is stale.
                    # So we MUST break or update A_set.
                    # Breaking is safer and allows the outer loop to re-evaluate the whole state.
                    break
            
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Kick: Randomly change one element to escape local minimum
                # Pick a random index and a random candidate
                idx = rng.randint(0, len(A)-1)
                # Pick a random candidate not in A
                valid_cands = [c for c in candidate_pool if c not in A]
                if valid_cands:
                    new_val = rng.choice(valid_cands)
                    A[idx] = new_val
                    current_score = calc_score(A)
                    no_improve_count = 0
                    # Do not update best if kick makes it worse, but we continue search from here
                else:
                    # If no valid candidates, we are stuck
                    break
        else:
            no_improve_count = 0
            
    return best_A, best_score

def construct():
    start_time = time.time()
    # Safety margin: 85 seconds
    deadline = start_time + 85.0
    
    rng = random.Random(42)
    
    # Generate the pool of candidates for replacement
    candidate_pool = generate_candidates()
    
    # Initial candidates
    candidates = []
    
    # 1. Arithmetic Progression 1..20
    candidates.append(list(range(1, 21)))
    
    # 2. Smallest 20 Smooth Numbers
    smooth_nums = sorted([x for x in candidate_pool if x <= 10000])
    candidates.append(smooth_nums[:20])
    
    # 3. Mixed: 1..15 and then next 5 smallest smooth > 15
    base_ap = list(range(1, 16))
    extra_smooth = [x for x in smooth_nums if x > 15]
    mixed = base_ap + extra_smooth[:5]
    if len(set(mixed)) == 20:
        candidates.append(mixed)
    
    # 4. Randomized Smooth-heavy start
    # Pick 20 distinct smooth numbers, biased towards small
    smooth_sample = rng.sample(smooth_nums[:50], 20)
    candidates.append(sorted(smooth_sample))
    
    # 5. Randomized Small Integers
    small_sample = rng.sample(range(1, 101), 20)
    candidates.append(sorted(small_sample))
    
    best_A = None
    best_score = float('inf')
    
    # Run local search on each candidate
    num_candidates = len(candidates)
    
    for idx, A in enumerate(candidates):
        if time.time() > deadline:
            break
        
        # Validate
        if len(set(A)) != 20:
            continue
        if any(x < 1 or x > 10000 for x in A):
            continue
        
        # Allocate time for this candidate
        remaining_time = deadline - time.time()
        remaining_count = num_candidates - idx
        alloc_time = max(1.0, remaining_time / remaining_count)
        
        improved_A, score = local_search(A, alloc_time, rng, candidate_pool)
        
        if score < best_score:
            best_score = score
            best_A = improved_A
            
    # If local search didn't find anything better (unlikely), use the best initial
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
