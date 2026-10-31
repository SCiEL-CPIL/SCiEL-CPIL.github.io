import math
import time
import random
import numpy as np

def construct():
    DEADLINE = 140  # safety margin before 150s kill

    def fft_set_sizes(u):
        """Compute |U+U| and |U-U| using FFT in O(M log M)."""
        arr = np.array(u, dtype=np.int64)
        max_u = int(arr.max())
        min_u = 0  # since u contains 0 and nonneg
        # For sum: range [0, 2*max_u]
        # For diff: range [-max_u, max_u]
        # Use FFT on indicator vector
        M = max(2 * max_u + 1, 1)
        # Pad to next power of 2 for FFT efficiency
        N = 1
        while N < 2 * M:
            N *= 2
        
        # Indicator vector for U (positioned at indices 0..max_u)
        f = np.zeros(N, dtype=np.float64)
        for val in u:
            f[int(val)] = 1.0
        
        # Sumset: convolution of f with itself
        F = np.fft.fft(f)
        sum_conv = np.fft.ifft(F * F).real
        # Round to handle floating point
        sum_conv = np.round(sum_conv)
        # Sumset occupies indices 0 to 2*max_u
        sumset_size = int(np.sum(sum_conv[:2 * max_u + 1] > 0.5))
        
        # Difference set: use f * conj(f) -> autocorrelation
        diff_conv = np.fft.ifft(F * np.conj(F)).real
        diff_conv = np.round(diff_conv)
        # Autocorrelation: index k corresponds to shift k (for k >= 0)
        # and N-k for negative shifts
        # diff_conv[0] = |U|^2, diff_conv[k] for k in 1..max_u = count of pairs with diff k
        # diff_conv[N-k] = count of pairs with diff -k
        # Total diffset size = 1 (for 0) + 2 * (number of k in 1..max_u with diff_conv[k] > 0)
        # But we need to check both sides
        diffset_size = 1  # always includes 0
        for k in range(1, max_u + 1):
            if diff_conv[k] > 0.5 or diff_conv[N - k] > 0.5:
                diffset_size += 1  # both +k and -k present (since if a-b=k then b-a=-k)
        
        return sumset_size, diffset_size, max_u

    def compute_score(u):
        u = sorted(set(u))
        if not u or 0 not in u:
            return -1.0
        max_u = max(u)
        m = 2 * max_u + 1
        if m <= 1:
            return 1.0
        if len(u) <= 30:
            # Direct computation for small sets
            arr = np.array(u, dtype=np.int64)
            sums = arr[:, None] + arr[None, :]
            diffs = arr[:, None] - arr[None, :]
            sumset_size = len(np.unique(sums))
            diffset_size = len(np.unique(diffs))
        else:
            sumset_size, diffset_size, _ = fft_set_sizes(u)
        r = diffset_size / sumset_size
        if r <= 0:
            return -1.0
        return 1 + math.log(r) / math.log(m)

    def compute_score_fast(u):
        """Fast score for local search - uses FFT always for consistency."""
        u = sorted(set(u))
        if not u or 0 not in u:
            return -1.0
        max_u = max(u)
        m = 2 * max_u + 1
        if m <= 1:
            return 1.0
        if len(u) <= 20:
            arr = np.array(u, dtype=np.int64)
            sums = arr[:, None] + arr[None, :]
            diffs = arr[:, None] - arr[None, :]
            sumset_size = len(np.unique(sums))
            diffset_size = len(np.unique(diffs))
        else:
            sumset_size, diffset_size, _ = fft_set_sizes(u)
        r = diffset_size / sumset_size
        if r <= 0:
            return -1.0
        return 1 + math.log(r) / math.log(m)

    def is_valid(u):
        """Check that u is a valid set: nonnegative integers, contains 0."""
        if not u:
            return False
        if 0 not in u:
            return False
        if any(x < 0 for x in u):
            return False
        return True

    # Start with known good candidates
    candidates = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 20],
        [0, 1, 4, 10, 12, 17, 20, 22],
    ]

    # Local search with FFT-based scoring
    def local_search(base_u, time_limit, temperature=1.0):
        start_time = time.time()
        best_u = list(base_u)
        best_score = compute_score_fast(best_u)
        current_u = list(base_u)
        current_score = best_score
        improved = True
        no_improve_count = 0
        max_no_improve = 50
        
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
            
            current_max = max(current_u)
            n = len(current_u)
            found_improvement = False
            
            # Try adding a new element
            search_hi = current_max + max(20, n * n)
            # Sample positions rather than exhaustive for speed
            positions_to_try = []
            if search_hi - current_max <= 500:
                positions_to_try = list(range(current_max + 1, search_hi + 1))
            else:
                # Sample
                positions_to_try = list(range(current_max + 1, min(current_max + 50, search_hi + 1)))
                positions_to_try += [random.randint(current_max + 1, search_hi) for _ in range(50)]
            
            for new_mark in positions_to_try:
                if (time.time() - start_time) > time_limit:
                    break
                candidate = sorted(current_u + [new_mark])
                s = compute_score_fast(candidate)
                if s > current_score + 1e-12:
                    current_u = candidate
                    current_score = s
                    found_improvement = True
                    if s > best_score:
                        best_u = candidate
                        best_score = s
                        no_improve_count = 0
            
            # Try removing an element
            if len(current_u) > 3 and (time.time() - start_time) < time_limit:
                for i in range(1, len(current_u)):  # keep 0
                    if (time.time() - start_time) > time_limit:
                        break
                    candidate = [x for j, x in enumerate(current_u) if j != i]
                    s = compute_score_fast(candidate)
                    if s > current_score + 1e-12:
                        current_u = candidate
                        current_score = s
                        found_improvement = True
                        if s > best_score:
                            best_u = candidate
                            best_score = s
                            no_improve_count = 0
            
            # Try swapping: replace one element with another
            if (time.time() - start_time) < time_limit:
                for i in range(1, len(current_u)):  # don't swap 0
                    if (time.time() - start_time) > time_limit:
                        break
                    old_val = current_u[i]
                    # Try various replacements
                    deltas = list(range(1, max(15, len(current_u) * 2)))
                    random.shuffle(deltas)
                    for delta in deltas[:20]:
                        for new_val in [old_val - delta, old_val + delta]:
                            if (time.time() - start_time) > time_limit:
                                break
                            if new_val < 0:
                                continue
                            if new_val in current_u:
                                continue
                            candidate = sorted([x if x != old_val else new_val for x in current_u])
                            s = compute_score_fast(candidate)
                            if s > current_score + 1e-12:
                                current_u = candidate
                                current_score = s
                                found_improvement = True
                                if s > best_score:
                                    best_u = candidate
                                    best_score = s
                                    no_improve_count = 0
                        if found_improvement:
                            break
            
            # Try random perturbations for escaping local optima
            if (time.time() - start_time) < time_limit and not found_improvement:
                for _ in range(5):
                    if (time.time() - start_time) > time_limit:
                        break
                    candidate = list(current_u)
                    # Randomly add or remove
                    if random.random() < 0.5 and len(candidate) < 50:
                        new_val = random.randint(1, max(2, max(candidate) + 10))
                        if new_val not in candidate:
                            candidate.append(new_val)
                    elif len(candidate) > 4:
                        idx = random.randint(1, len(candidate) - 1)
                        candidate.pop(idx)
                    candidate = sorted(set(candidate))
                    if is_valid(candidate):
                        s = compute_score_fast(candidate)
                        # Accept with some probability (simulated annealing style)
                        if s > current_score or random.random() < 0.1:
                            current_u = candidate
                            current_score = s
                            if s > best_score:
                                best_u = candidate
                                best_score = s
                                found_improvement = True
            
            if found_improvement:
                improved = True
        
        return best_u, best_score

    # Greedy construction with FFT scoring
    def greedy_construct(n_target, time_limit):
        start_time = time.time()
        u = [0]
        for i in range(1, n_target):
            if (time.time() - start_time) > time_limit:
                break
            best_pos = None
            best_s = compute_score_fast(u)
            current_max = max(u)
            # Try positions in increasing order
            for pos in range(current_max + 1, current_max + max(50, i * i * 3)):
                if (time.time() - start_time) > time_limit:
                    break
                candidate = sorted(u + [pos])
                s = compute_score_fast(candidate)
                if s > best_s + 1e-12:
                    best_s = s
                    best_pos = pos
                if pos - current_max > 200:
                    break
            if best_pos is not None:
                u.append(best_pos)
                u.sort()
            else:
                break
        return u

    # Random restart local search
    def random_restart_local_search(time_limit):
        start_time = time.time()
        best_u = [0, 1, 3]
        best_score = compute_score_fast(best_u)
        
        while (time.time() - start_time) < time_limit:
            # Random initial set
            n = random.randint(3, 20)
            u = [0]
            current_max = 0
            for _ in range(n - 1):
                if (time.time() - start_time) > time_limit:
                    break
                new_val = random.randint(current_max + 1, current_max + random.randint(2, 20))
                u.append(new_val)
                current_max = new_val
            u = sorted(set(u))
            
            # Local search from this starting point
            time_remaining = time_limit - (time.time() - start_time)
            if time_remaining < 2:
                break
            alloc_time = min(time_remaining * 0.3, 10)
            if alloc_time < 1:
                alloc_time = 1
            ls_u, ls_score = local_search(u, alloc_time)
            if ls_score > best_score:
                best_u = ls_u
                best_score = ls_score
        
        return best_u, best_score

    # Run all strategies
    results = []
    
    # 1. Local search on known candidates
    for sp in candidates:
        remaining = DEADLINE - (time.time() - 0)  # approximate
        if remaining < 5:
            break
        time_for_this = min(remaining * 0.15, 15)
        if time_for_this < 3:
            time_for_this = 3
        ext_u, ext_score = local_search(sp, time_for_this)
        if ext_u and len(ext_u) >= 3:
            results.append((ext_u, ext_score))
    
    # 2. Greedy construction with various target sizes
    for n_target in [8, 12, 15, 20, 25, 30, 40, 50]:
        remaining = DEADLINE - (time.time() - 0)
        if remaining < 5:
            break
        time_for_this = min(remaining * 0.1, 10)
        if time_for_this < 2:
            time_for_this = 2
        greedy_u = greedy_construct(n_target, time_for_this)
        if greedy_u and len(greedy_u) >= 3:
            results.append((greedy_u, compute_score_fast(greedy_u)))
    
    # 3. Random restart local search
    remaining = DEADLINE - (time.time() - 0)
    if remaining > 10:
        time_for_rs = min(remaining * 0.3, 30)
        rs_u, rs_score = random_restart_local_search(time_for_rs)
        results.append((rs_u, rs_score))
    
    # 4. Additional local search on best found so far
    if results:
        results.sort(key=lambda x: x[1], reverse=True)
        top_u, top_score = results[0]
        remaining = DEADLINE - (time.time() - 0)
        if remaining > 5:
            time_for_polish = min(remaining * 0.4, 20)
            if time_for_polish > 3:
                polish_u, polish_score = local_search(top_u, time_for_polish)
                results.append((polish_u, polish_score))
    
    # Find overall best
    best_set = [0, 1, 3]
    best_score = 0.0
    for u, s in results:
        u = sorted(set(u))
        if 0 not in u:
            u = [0] + u
        s = compute_score(u)  # Final verification with exact computation
        if s > best_score:
            best_score = s
            best_set = u
    
    return best_set
