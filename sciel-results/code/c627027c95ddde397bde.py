"""Combined B2[2] construction with vectorized local search."""
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
        ok = True
        for y in chosen:
            if counts[x + y] >= 2:
                ok = False
                break
        if ok and counts[2 * x] >= 2:
            ok = False
        if ok:
            for y in chosen:
                counts[x + y] += 1
            counts[2 * x] += 1
            chosen.append(x)
            chosen_set.add(x)
    return chosen, counts

def _is_valid_b2(chosen):
    """Verify B2[2] property using vectorized NumPy."""
    if not chosen:
        return True
    arr = np.array(chosen, dtype=np.int32)
    n = len(arr)
    i_idx, j_idx = np.triu_indices(n)
    sums = arr[i_idx] + arr[j_idx]
    unique, counts_arr = np.unique(sums, return_counts=True)
    return bool(np.all(counts_arr <= 2))

def _local_search(chosen, time_limit=10.0):
    """Optimized local search using numpy arrays for fast conflict checking."""
    chosen = sorted(chosen)
    n = len(chosen)
    arr = np.array(chosen, dtype=np.int32)
    
    # Build count array for sums (max sum = 2000)
    counts = np.zeros(2001, dtype=np.int32)
    i_idx, j_idx = np.triu_indices(n)
    sums = arr[i_idx] + arr[j_idx]
    np.add.at(counts, sums, 1)
    
    start_time = time.time()
    improved = True
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        remaining = time_limit - (time.time() - start_time)
        if remaining <= 0:
            break
        
        chosen_set = set(chosen)
        
        # Phase 1: Try pure insertions using vectorized conflict check
        # For each candidate y, check if any z in chosen has counts[y+z] >= 2
        # Vectorize: for all y in 1..1000 not in chosen, compute conflicts
        all_ys = np.arange(1, 1001, dtype=np.int32)
        mask = np.array([y not in chosen_set for y in range(1, 1001)], dtype=bool)
        candidates_y = all_ys[mask]
        
        if len(candidates_y) > 0:
            # For each candidate y, count how many z in chosen have counts[y+z] >= 2
            # Vectorized: outer product approach
            # conflicts[y_idx] = sum over z of (counts[y+z] >= 2) + (counts[2y] >= 2)
            # This is O(|candidates| * n) but vectorized per candidate
            
            # Batch process candidates in groups to avoid memory issues
            batch_size = min(len(candidates_y), 200)
            for batch_start in range(0, len(candidates_y), batch_size):
                if time.time() - start_time > time_limit:
                    break
                batch = candidates_y[batch_start:batch_start + batch_size]
                
                # For each y in batch, compute conflicts with all z in chosen
                # sums_matrix[b, k] = batch[b] + arr[k]
                sums_matrix = batch[:, None] + arr[None, :]  # shape (batch_size, n)
                # conflict_matrix[b, k] = counts[sums_matrix[b,k]] >= 2
                conflict_matrix = counts[sums_matrix] >= 2
                conflict_counts = conflict_matrix.sum(axis=1)
                
                # Also check 2*y
                two_y = 2 * batch
                conflict_counts += (counts[two_y] >= 2).astype(np.int32)
                
                # Try to insert candidates with conflict_count == 0 (guaranteed valid)
                # or conflict_count <= 1 (may still be valid depending on exact counts)
                for idx in range(len(batch)):
                    if time.time() - start_time > time_limit:
                        break
                    y = int(batch[idx])
                    if y in chosen_set:
                        continue
                    # Exact check: can we add y?
                    # Need: for all z in chosen, counts[y+z] < 2, and counts[2y] < 2
                    # i.e., no z such that counts[y+z] >= 2
                    if conflict_counts[idx] == 0:
                        # Safe to insert
                        for z in chosen:
                            counts[y + z] += 1
                        counts[2 * y] += 1
                        chosen.append(y)
                        chosen_set.add(y)
                        arr = np.array(chosen, dtype=np.int32)
                        improved = True
                        # Rebuild counts for correctness
                        counts[:] = 0
                        n_new = len(arr)
                        ii, jj = np.triu_indices(n_new)
                        s_new = arr[ii] + arr[jj]
                        np.add.at(counts, s_new, 1)
        
        # Phase 2: Try swaps (remove one, add another)
        if (time.time() - start_time) < time_limit:
            n = len(chosen)
            if n == 0:
                continue
            arr = np.array(chosen, dtype=np.int32)
            chosen_set = set(chosen)
            
            # For each position i, try removing chosen[i] and adding a better candidate
            # Sample a subset of positions to try
            num_swap_tries = min(n, max(10, int(remaining * 20)))
            rng_local = random.Random(12345)
            positions_to_try = rng_local.sample(range(n), min(num_swap_tries, n))
            
            for i in positions_to_try:
                if time.time() - start_time > time_limit:
                    break
                x = chosen[i]
                
                # Remove x from counts
                for j in range(n):
                    if j != i:
                        counts[x + chosen[j]] -= 1
                counts[2 * x] -= 1
                chosen_set.discard(x)
                
                # Find best replacement
                remaining_arr = np.array([chosen[j] for j in range(n) if j != i], dtype=np.int32)
                n_rem = len(remaining_arr)
                
                # Check candidates
                all_ys = np.arange(1, 1001, dtype=np.int32)
                mask = np.array([y not in chosen_set for y in range(1, 1001)], dtype=bool)
                candidates_y = all_ys[mask]
                
                if len(candidates_y) == 0:
                    # Restore x
                    for z in [chosen[j] for j in range(n) if j != i]:
                        counts[x + z] += 1
                    counts[2 * x] += 1
                    chosen_set.add(x)
                    continue
                
                # Vectorized conflict check
                batch_size = min(len(candidates_y), 300)
                best_y = -1
                for batch_start in range(0, len(candidates_y), batch_size):
                    if time.time() - start_time > time_limit:
                        break
                    batch = candidates_y[batch_start:batch_start + batch_size]
                    if n_rem == 0:
                        # Empty set, any y works
                        best_y = int(batch[0])
                        break
                    sums_matrix = batch[:, None] + remaining_arr[None, :]
                    conflict_matrix = counts[sums_matrix] >= 2
                    conflict_counts = conflict_matrix.sum(axis=1)
                    two_y = 2 * batch
                    conflict_counts += (counts[two_y] >= 2).astype(np.int32)
                    
                    # Find first candidate with zero conflicts
                    zero_conf = np.where(conflict_counts == 0)[0]
                    if len(zero_conf) > 0:
                        best_y = int(batch[zero_conf[0]])
                        break
                
                if best_y > 0:
                    # Do the swap
                    for z in [chosen[j] for j in range(n) if j != i]:
                        counts[best_y + z] += 1
                    counts[2 * best_y] += 1
                    chosen[i] = best_y
                    chosen_set.add(best_y)
                    improved = True
                    # Rebuild arr and counts
                    arr = np.array(chosen, dtype=np.int32)
                    counts[:] = 0
                    n_new = len(arr)
                    ii, jj = np.triu_indices(n_new)
                    s_new = arr[ii] + arr[jj]
                    np.add.at(counts, s_new, 1)
                else:
                    # Restore x
                    for z in [chosen[j] for j in range(n) if j != i]:
                        counts[x + z] += 1
                    counts[2 * x] += 1
                    chosen_set.add(x)
    
    return sorted(chosen)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    # Strategy 1: Standard greedy (increasing order)
    candidates = list(range(1, 1001))
    chosen1, _ = _greedy_build(candidates)
    
    best = chosen1[:]
    best_size = len(best)
    
    # Strategy 2: Diverse greedy restarts
    rng = random.Random(42)
    
    # Large-first
    if time.time() < deadline:
        cand_large = list(range(1000, 0, -1))
        chosen_large, _ = _greedy_build(cand_large)
        if len(chosen_large) > best_size:
            best = chosen_large[:]
            best_size = len(best)
    
    # Mixed alternating
    if time.time() < deadline:
        small = list(range(1, 501))
        large = list(range(1000, 499, -1))
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
    
    # Prime-based orderings
    for p in [2, 3, 5, 7, 11]:
        if time.time() > deadline - 30.0:
            break
        cand_p = [x for x in range(1, 1001) if x % p != 0] + [x for x in range(1, 1001) if x % p == 0]
        chosen_p, _ = _greedy_build(cand_p)
        if len(chosen_p) > best_size:
            best = chosen_p[:]
            best_size = len(best)
    
    # Random restarts with various biases
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() > deadline - 40.0:  # Leave time for local search
            break
        
        cand = list(range(1, 1001))
        
        if restart % 6 == 0:
            # Pure random
            rng.shuffle(cand)
        elif restart % 6 == 1:
            # Bias towards smaller (power 0.5)
            biased = [(rng.random() ** 0.5, i) for i in range(1000)]
            biased.sort(key=lambda x: x[0])
            cand = [b[1] + 1 for b in biased]
        elif restart % 6 == 2:
            # Bias towards larger (power 0.3)
            biased = [(rng.random() ** 0.3, i) for i in range(1000)]
            biased.sort(key=lambda x: x[0])
            cand = [b[1] + 1 for b in biased]
        elif restart % 6 == 3:
            # Very small bias (power 0.8)
            biased = [(rng.random() ** 0.8, i) for i in range(1000)]
            biased.sort(key=lambda x: x[0])
            cand = [b[1] + 1 for b in biased]
        elif restart % 6 == 4:
            # Block shuffle (blocks of 20)
            for i in range(0, 1000, 20):
                block = cand[i:i+20]
                rng.shuffle(block)
                cand[i:i+20] = block
        else:
            # Two-phase: first 100 random, rest sorted
            first_100 = cand[:100]
            rng.shuffle(first_100)
            cand = first_100 + cand[100:]
        
        chosen_r, _ = _greedy_build(cand)
        if len(chosen_r) > best_size:
            best = chosen_r[:]
            best_size = len(best)
    
    # Local search on the best solution
    remaining = deadline - time.time() - 3.0
    if remaining > 1.0:
        ls_time = min(remaining, 80.0)
        best = _local_search(best, time_limit=ls_time)
        best_size = len(best)
    
    # Second local search pass with different random seed if time allows
    if time.time() < deadline - 5.0:
        remaining2 = deadline - time.time() - 3.0
        if remaining2 > 5.0:
            best = _local_search(best, time_limit=min(remaining2, 30.0))
            best_size = len(best)
    
    # Verify validity
    if not _is_valid_b2(best):
        best = chosen1[:]
    
    return best
# EVOLVE-BLOCK-END
