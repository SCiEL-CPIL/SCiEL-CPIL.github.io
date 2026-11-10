"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict
import numpy as np

def _greedy_build(candidates, counts_init=None):
    """Greedy: add candidates in given order if B2[2] constraint holds."""
    chosen = []
    counts = defaultdict(int) if counts_init is None else dict(counts_init)
    chosen_set = set()
    for x in candidates:
        if x in chosen_set:
            continue
        # Check all new sums
        ok = True
        for y in chosen:
            s = x + y
            if counts[s] + 1 > 2:
                ok = False
                break
        if ok:
            s2 = 2 * x
            if counts[s2] + 1 > 2:
                ok = False
        if ok:
            for y in chosen:
                counts[x + y] += 1
            counts[2 * x] += 1
            chosen.append(x)
            chosen_set.add(x)
    return chosen, counts

def _is_valid_b2(chosen):
    """Verify B2[2] property."""
    if not chosen:
        return True
    arr = np.array(chosen)
    # Compute all pairwise sums (i <= j)
    i_idx, j_idx = np.triu_indices(len(arr))
    sums = arr[i_idx] + arr[j_idx]
    # Count occurrences
    unique, counts_arr = np.unique(sums, return_counts=True)
    return np.all(counts_arr <= 2)

def _local_search(chosen, time_limit=10.0):
    """Try insertions and swaps to improve set size."""
    chosen = sorted(chosen)
    n = len(chosen)
    
    # Precompute counts
    counts = defaultdict(int)
    for i in range(n):
        for j in range(i, n):
            counts[chosen[i] + chosen[j]] += 1
    
    chosen_set = set(chosen)
    start_time = time.time()
    rng_ls = random.Random(12345)
    
    def _try_insert(y, chosen, counts, chosen_set):
        """Check if y can be added without violating B2[2]."""
        if y in chosen_set:
            return False
        for z in chosen:
            s = y + z
            if counts[s] + 1 > 2:
                return False
        s2 = 2 * y
        if counts[s2] + 1 > 2:
            return False
        return True
    
    def _do_insert(y, chosen, counts, chosen_set):
        for z in chosen:
            counts[y + z] += 1
        counts[2 * y] += 1
        chosen_set.add(y)
        chosen.append(y)
        chosen.sort()
    
    def _try_swap(x, y, chosen, counts, chosen_set):
        """Check if swapping x for y is valid."""
        if y in chosen_set:
            return False
        # Remove x temporarily from counts
        for z in chosen:
            if z != x:
                s = x + z
                if counts[s] < 1:
                    return False
        if counts[2 * x] < 1:
            return False
        # Now check if y can be added to (chosen - {x})
        for z in chosen:
            if z != x:
                s = y + z
                if counts[s] - 1 + 1 > 2:  # counts[s] already has x+z, we remove it then add y+z
                    return False
        s2 = 2 * y
        if counts[s2] + 1 > 2:
            return False
        return True
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time <= 0:
            break
        
        # Phase 1: Try pure insertions (grow the set)
        candidate_pool = list(range(1, 1001))
        rng_ls.shuffle(candidate_pool)
        # Try a subset based on remaining time
        num_to_try = min(len(candidate_pool), max(100, int(remaining_time * 50)))
        
        for y in candidate_pool[:num_to_try]:
            if time.time() - start_time > time_limit:
                break
            if _try_insert(y, chosen, counts, chosen_set):
                _do_insert(y, chosen, counts, chosen_set)
                improved = True
        
        # Phase 2: Try swaps (keep size same but may enable future insertions)
        if time.time() - start_time < time_limit:
            n = len(chosen)
            candidate_pool2 = list(range(1, 1001))
            rng_ls.shuffle(candidate_pool2)
            num_swaps_to_try = min(n, max(5, int((time_limit - (time.time() - start_time)) * 20)))
            
            for idx in range(min(n, num_swaps_to_try)):
                if time.time() - start_time > time_limit:
                    break
                i = rng_ls.randint(0, n - 1)
                x = chosen[i]
                
                # Try to swap x with some y
                for y in candidate_pool2[:max(50, int((time_limit - (time.time() - start_time)) * 30))]:
                    if time.time() - start_time > time_limit:
                        break
                    if y in chosen_set or y == x:
                        continue
                    
                    # Check if swap is valid
                    valid = True
                    # Remove x from counts
                    for z in chosen:
                        if z != x:
                            s = x + z
                            counts[s] -= 1
                    counts[2 * x] -= 1
                    
                    # Check if y can be added
                    for z in chosen:
                        if z != x:
                            s = y + z
                            if counts[s] + 1 > 2:
                                valid = False
                                break
                    if valid:
                        s2 = 2 * y
                        if counts[s2] + 1 > 2:
                            valid = False
                    
                    if valid:
                        # Do the swap
                        for z in chosen:
                            if z != x:
                                s = y + z
                                counts[s] += 1
                        counts[2 * y] += 1
                        chosen[i] = y
                        chosen_set.discard(x)
                        chosen_set.add(y)
                        chosen.sort()
                        improved = True
                        break
                    else:
                        # Restore x
                        for z in chosen:
                            if z != x:
                                s = x + z
                                counts[s] += 1
                        counts[2 * x] += 1
    
    return sorted(chosen)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s
    
    # Strategy 1: Standard greedy (increasing order)
    candidates = list(range(1, 1001))
    chosen1, _ = _greedy_build(candidates)
    
    best = chosen1[:]
    best_size = len(best)
    
    # Strategy 2: Try structured and random orderings
    rng = random.Random(42)
    
    # Try some structured approaches
    # Approach A: Small numbers first (already done in Strategy 1)
    # Approach B: Large numbers first
    if time.time() < deadline:
        cand_large = list(range(1000, 0, -1))
        chosen_large, _ = _greedy_build(cand_large)
        if len(chosen_large) > best_size:
            best = chosen_large[:]
            best_size = len(best)
    
    # Approach C: Mixed - alternate small and large
    if time.time() < deadline:
        small = list(range(1, 501))
        large = list(range(500, 0, -1))
        cand_mixed = []
        si, li = 0, 0
        while si < len(small) or li < len(large):
            if si < len(small):
                cand_mixed.append(small[si])
                si += 1
            if li < len(large):
                cand_mixed.append(large[li])
                li += 1
        chosen_mixed, _ = _greedy_build(cand_mixed)
        if len(chosen_mixed) > best_size:
            best = chosen_mixed[:]
            best_size = len(best)
    
    # Approach D: Prime-based or arithmetic progression based
    if time.time() < deadline:
        # Try multiples of small primes
        for p in [2, 3, 5, 7]:
            if time.time() > deadline:
                break
            cand_prime = [x for x in range(1, 1001) if x % p != 0] + [x for x in range(1, 1001) if x % p == 0]
            chosen_p, _ = _greedy_build(cand_prime)
            if len(chosen_p) > best_size:
                best = chosen_p[:]
                best_size = len(best)
    
    # Strategy 3: Try random orderings with bias
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 20.0:  # Leave time for local search
            break
        
        # Create biased candidate list: mix of structured and random
        if restart % 5 == 0:
            # Pure random
            cand = list(range(1, 1001))
            rng.shuffle(cand)
        elif restart % 5 == 1:
            # Bias towards smaller numbers
            cand = list(range(1, 1001))
            biased = [(rng.random() ** 0.5, i) for i in range(1000)]
            biased.sort(key=lambda x: x[0])
            cand = [b[1] + 1 for b in biased]
        elif restart % 5 == 2:
            # Bias towards larger numbers
            cand = list(range(1, 1001))
            biased = [(rng.random() ** 0.3, i) for i in range(1000)]
            biased.sort(key=lambda x: x[0])
            cand = [b[1] + 1 for b in biased]
        elif restart % 5 == 3:
            # Very small bias
            cand = list(range(1, 1001))
            biased = [(rng.random() ** 0.8, i) for i in range(1000)]
            biased.sort(key=lambda x: x[0])
            cand = [b[1] + 1 for b in biased]
        else:
            # Alternating blocks
            cand = list(range(1, 1001))
            # Shuffle in blocks of 10
            for i in range(0, 1000, 10):
                block = cand[i:i+10]
                rng.shuffle(block)
                cand[i:i+10] = block
        
        chosen_r, _ = _greedy_build(cand)
        if len(chosen_r) > best_size:
            best = chosen_r[:]
            best_size = len(best)
    
    # Local search on the best solution - this can now GROW the set
    remaining = deadline - time.time() - 2.0
    if remaining > 1.0:
        ls_time = min(remaining, 60.0)
        best = _local_search(best, time_limit=ls_time)
        best_size = len(best)
    
    # Second pass: try local search on other good solutions too
    if time.time() < deadline - 5.0:
        # Re-run greedy with the best as seed and try to extend
        pass
    
    # Verify validity
    if not _is_valid_b2(best):
        # Fallback to standard greedy result
        best = chosen1[:]
    
    return best
# EVOLVE-BLOCK-END
