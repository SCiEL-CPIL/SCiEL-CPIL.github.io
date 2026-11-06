"""Model-and-optimize approach for distinct subset sums with small max element."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def _all_subset_sums_distinct(seq):
    """Check if all subset sums are distinct. Returns True if valid."""
    n = len(seq)
    bits = 1  # only sum 0
    for a in seq:
        bits |= (bits << a)
    cnt = bin(bits).count('1')
    return cnt == (1 << n)

def _greedy_superincreasing(n=12, max_val=1048576):
    """
    Greedy construction using superincreasing property: each new element is
    chosen as the smallest value that is > sum of all previous elements,
    which guarantees distinct subset sums. This minimizes the largest element
    within the superincreasing family.
    """
    seq = []
    total_sum = 0
    for i in range(n):
        # The minimum valid next element is total_sum + 1
        next_val = total_sum + 1
        if next_val > max_val:
            return None
        seq.append(next_val)
        total_sum += next_val
    return seq

def _construct_random_valid(n, max_limit, rng, time_limit):
    """
    Try to construct a valid set with max element < max_limit using random choices.
    """
    start_time = time.time()
    best_max = max_limit
    best_seq = None
    
    while time.time() - start_time < time_limit:
        seq = []
        valid = True
        for i in range(n):
            # Generate candidates
            candidates = set()
            # Random samples
            for _ in range(20):
                val = rng.randint(1, max_limit)
                if val not in seq:
                    candidates.add(val)
            # Small values
            for val in range(1, min(10, max_limit + 1)):
                if val not in seq:
                    candidates.add(val)
            
            if not candidates:
                valid = False
                break
                
            cand_list = list(candidates)
            rng.shuffle(cand_list)
            
            found = False
            for val in cand_list:
                test_seq = seq + [val]
                if _all_subset_sums_distinct(test_seq):
                    seq.append(val)
                    found = True
                    break
            
            if not found:
                valid = False
                break
        
        if valid:
            cur_max = max(seq)
            if cur_max < best_max:
                best_max = cur_max
                best_seq = list(seq)
                
    return best_seq

def _local_search_refined(seq, n=12, max_val=1048576, time_limit=10.0, rng=None):
    """Refined local search with multiple strategies."""
    if rng is None:
        rng = random.Random()
    
    start_time = time.time()
    current = list(seq)
    cur_max = max(current)
    best_seq = list(current)
    best_max = cur_max
    
    improved = True
    while improved:
        if time.time() - start_time > time_limit - 1:
            break
        improved = False
        # Try each position
        for idx in range(n):
            if time.time() - start_time > time_limit - 1:
                break
            old = current[idx]
            # Try smaller values, starting from close to old
            for new in range(1, old):
                if time.time() - start_time > time_limit - 1:
                    break
                cand = list(current)
                cand[idx] = new
                if len(set(cand)) != n:
                    continue
                if max(cand) > max_val:
                    continue
                if _all_subset_sums_distinct(cand):
                    nm = max(cand)
                    if nm < cur_max or (nm == cur_max and sum(cand) < sum(current)):
                        current = cand
                        cur_max = nm
                        if cur_max < best_max:
                            best_seq = list(current)
                            best_max = cur_max
                        improved = True
                        break
            if improved:
                break
        if not improved:
            # Try random perturbations
            if time.time() - start_time > time_limit - 1:
                break
            for _ in range(50):
                if time.time() - start_time > time_limit - 1:
                    break
                idx = rng.randint(0, n - 1)
                delta = rng.randint(-20, 20)
                new_val = current[idx] + delta
                if new_val < 1 or new_val > max_val:
                    continue
                cand = list(current)
                cand[idx] = new_val
                if len(set(cand)) != n:
                    continue
                if _all_subset_sums_distinct(cand):
                    nm = max(cand)
                    if nm < cur_max or (nm == cur_max and sum(cand) < sum(current)):
                        current = cand
                        cur_max = nm
                        if cur_max < best_max:
                            best_seq = list(current)
                            best_max = cur_max
                        improved = True
                        break
    
    return best_seq

def construct():
    """
    Construct 12 distinct positive integers <= 1048576 with all subset sums distinct.
    Minimize the largest element.
    """
    deadline = 55.0  # safety margin from 60s
    start_time = time.time()
    
    n = 12
    max_val = 1048576
    
    # Step 1: Greedy superincreasing construction for a quick valid solution
    greedy_seq = _greedy_superincreasing(n, max_val)
    if greedy_seq is None:
        # Fallback to powers of 2
        return [1 << i for i in range(n)]
    
    best_seq = greedy_seq
    best_max = max(best_seq)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return best_seq
    
    # Step 2: Random construction to find a potentially better set
    # We try to find a set with max < best_max
    remaining = deadline - elapsed - 5
    if remaining > 1:
        rng = random.Random(42)
        # Try to find a set with max < best_max
        # Use a target slightly lower than current best
        target = best_max - 1
        if target > 400: # Lower bound is ~341
            rand_seq = _construct_random_valid(n, target, rng, min(remaining, 20.0))
            if rand_seq and max(rand_seq) < best_max:
                best_seq = rand_seq
                best_max = max(best_seq)
                
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return best_seq
    
    # Step 3: Local search on the best found
    remaining = deadline - elapsed - 2
    if remaining > 1:
        rng = random.Random(99)
        improved = _local_search_refined(best_seq, n, max_val, min(remaining, 20.0), rng)
        if improved and max(improved) < best_max:
            best_seq = improved
            best_max = max(improved)
            
    return best_seq
# EVOLVE-BLOCK-END
