import math
import time
import numpy as np

def construct():
    DEADLINE = 145  # seconds, leave 5s safety margin
    _t0 = time.time()
    
    def elapsed():
        return time.time() - _t0
    
    def compute_score(u):
        u = sorted(set(u))
        if not u or 0 not in u:
            return -1.0
        max_u = max(u)
        m = 2 * max_u + 1
        if m <= 1:
            return 1.0
        arr = np.array(u, dtype=np.int64)
        sums = arr[:, None] + arr[None, :]
        diffs = arr[:, None] - arr[None, :]
        sumset_size = len(np.unique(sums))
        diffset_size = len(np.unique(diffs))
        r = diffset_size / sumset_size
        return 1 + math.log(r) / math.log(m)
    
    def is_golomb(u):
        """Fast Golomb check using sorted differences."""
        u = sorted(u)
        n = len(u)
        if n < 3:
            return True
        diffs = []
        for i in range(n):
            for j in range(i + 1, n):
                diffs.append(u[j] - u[i])
        diffs.sort()
        for k in range(1, len(diffs)):
            if diffs[k] == diffs[k-1]:
                return False
        return True
    
    def get_diffs(u):
        u = sorted(u)
        diffs = set()
        for i in range(len(u)):
            for j in range(i+1, len(u)):
                diffs.add(u[j] - u[i])
        return diffs
    
    # Diverse starting points: Golomb rulers, near-Golomb, and structured sets
    start_points = [
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 20],
        [0, 1, 4, 9, 11],
        [0, 2, 7, 15, 22, 31, 39],
        [0, 1, 4, 10, 18, 23, 31, 37],
        [0, 3, 7, 12, 20, 29, 38],
        [0, 1, 4, 10, 12, 17, 20, 22],
        [0, 2, 5, 9, 14, 20, 27, 34],
        [0, 1, 5, 10, 15, 21, 27, 34],
        [0, 2, 6, 11, 17, 24, 32, 40],
        [0, 1, 4, 11, 14, 21, 25, 33],
    ]
    
    def local_search(base_u, time_limit):
        start_time = time.time()
        best_u = sorted(list(base_u))
        best_score = compute_score(best_u)
        improved = True
        current_diffs = get_diffs(best_u)
        
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            best_candidate = None
            best_candidate_score = best_score
            best_candidate_diffs = None
            
            n = len(best_u)
            current_max = max(best_u)
            
            # Move 1: Add a new element (wider search range)
            search_hi = current_max + max(20, n * n + 10)
            for new_mark in range(current_max + 1, search_hi + 1):
                new_diffs_needed = set()
                for x in best_u:
                    new_diffs_needed.add(new_mark - x)
                if new_diffs_needed & current_diffs:
                    continue
                candidate = sorted(best_u + [new_mark])
                s = compute_score(candidate)
                if s > best_candidate_score:
                    best_candidate_score = s
                    best_candidate = candidate
                    best_candidate_diffs = current_diffs | new_diffs_needed
                if (time.time() - start_time) > time_limit:
                    break
            
            # Move 2: Remove an element
            if n > 3 and (time.time() - start_time) < time_limit:
                for i in range(1, n):
                    candidate = [x for j, x in enumerate(best_u) if j != i]
                    s = compute_score(candidate)
                    if s > best_candidate_score:
                        best_candidate_score = s
                        best_candidate = candidate
                        best_candidate_diffs = get_diffs(candidate)
                    if (time.time() - start_time) > time_limit:
                        break
            
            # Move 3: Swap one element (wider delta range)
            if (time.time() - start_time) < time_limit:
                max_delta = max(20, n * 4)
                for i in range(1, n):
                    old_val = best_u[i]
                    other_diffs = set()
                    for d in current_diffs:
                        involves_old = False
                        for x in best_u:
                            if x != old_val:
                                if old_val - x == d or x - old_val == d:
                                    involves_old = True
                                    break
                        if not involves_old:
                            other_diffs.add(d)
                    
                    for delta in range(1, max_delta + 1):
                        for new_val in [old_val - delta, old_val + delta]:
                            if new_val < 0 or new_val in best_u:
                                continue
                            new_diffs = set()
                            for x in best_u:
                                if x != old_val:
                                    new_diffs.add(abs(new_val - x))
                            if new_diffs & other_diffs:
                                continue
                            if len(new_diffs) < (n - 1):
                                continue
                            candidate = sorted([x if x != old_val else new_val for x in best_u])
                            s = compute_score(candidate)
                            if s > best_candidate_score:
                                best_candidate_score = s
                                best_candidate = candidate
                                best_candidate_diffs = other_diffs | new_diffs
                        if (time.time() - start_time) > time_limit:
                            break
                    if (time.time() - start_time) > time_limit:
                        break
            
            # Move 4: 2-swap with small perturbations
            if n >= 4 and (time.time() - start_time) < time_limit:
                for i in range(1, n):
                    for j in range(i + 1, n):
                        if (time.time() - start_time) > time_limit:
                            break
                        old_i, old_j = best_u[i], best_u[j]
                        for di in [-2, -1, 1, 2]:
                            for dj in [-2, -1, 1, 2]:
                                new_i, new_j = old_i + di, old_j + dj
                                if new_i < 0 or new_j < 0:
                                    continue
                                if new_i in best_u or new_j in best_u:
                                    continue
                                if new_i == new_j:
                                    continue
                                candidate = sorted([x if x != old_i else new_i for x in best_u])
                                candidate = sorted([x if x != old_j else new_j for x in candidate])
                                if len(set(candidate)) != len(candidate):
                                    continue
                                s = compute_score(candidate)
                                if s > best_candidate_score:
                                    best_candidate_score = s
                                    best_candidate = candidate
                                    best_candidate_diffs = get_diffs(candidate)
                        if (time.time() - start_time) > time_limit:
                            break
                    if (time.time() - start_time) > time_limit:
                        break
            
            # Move 5: Insert a new element in the middle (not just at the end)
            if (time.time() - start_time) < time_limit:
                for new_mark in range(1, current_max + 1):
                    if new_mark in best_u:
                        continue
                    new_diffs_needed = set()
                    for x in best_u:
                        new_diffs_needed.add(abs(new_mark - x))
                    if new_diffs_needed & current_diffs:
                        continue
                    candidate = sorted(best_u + [new_mark])
                    s = compute_score(candidate)
                    if s > best_candidate_score:
                        best_candidate_score = s
                        best_candidate = candidate
                        best_candidate_diffs = current_diffs | new_diffs_needed
                    if (time.time() - start_time) > time_limit:
                        break
            
            if best_candidate is not None:
                best_u = best_candidate
                best_score = best_candidate_score
                if best_candidate_diffs is not None:
                    current_diffs = best_candidate_diffs
                else:
                    current_diffs = get_diffs(best_u)
                improved = True
        
        return best_u, best_score
    
    # Score and sort starting points
    scored_starts = []
    for sp in start_points:
        s = compute_score(sp)
        scored_starts.append((s, sp))
    scored_starts.sort(key=lambda x: -x[0])
    
    # Run local search on each starting point with adaptive time allocation
    results = []
    for idx, (s, sp) in enumerate(scored_starts):
        remaining = DEADLINE - elapsed()
        if remaining < 8:
            break
        # Allocate more time to earlier (higher-scoring) starts
        frac = max(0.15, 0.35 - 0.03 * idx)
        time_for_this = min(remaining * frac, 25)
        if time_for_this < 4:
            time_for_this = max(4, remaining * 0.2)
        ext_u, ext_score = local_search(sp, time_for_this)
        if ext_u and len(ext_u) >= 3:
            results.append((ext_score, ext_u))
    
    # Greedy Golomb construction for additional diversity
    def greedy_golomb(n, time_limit):
        start_time = time.time()
        u = [0]
        for i in range(1, n):
            if (time.time() - start_time) > time_limit:
                break
            pos = 0
            found = False
            while not found:
                pos += 1
                candidate = sorted(u + [pos])
                if is_golomb(candidate):
                    u = candidate
                    found = True
                if pos > n * n * 30:
                    break
        return u
    
    if DEADLINE - elapsed() > 6:
        greedy_u = greedy_golomb(10, time_limit=min(6, DEADLINE - elapsed() - 2))
        if greedy_u and len(greedy_u) >= 5:
            results.append((compute_score(greedy_u), greedy_u))
    
    # Find best among all results
    best_set = [0, 1, 3]
    best_score = 0.0
    for s, u in results:
        u = sorted(set(u))
        if 0 not in u:
            u = [0] + u
        s = compute_score(u)
        if s > best_score:
            best_score = s
            best_set = u
    
    # Final polish: focused local search with wider moves on the best
    remaining = DEADLINE - elapsed()
    if remaining > 12 and best_score > 0:
        polish_u, polish_score = local_search(best_set, remaining - 5)
        if polish_score > best_score:
            best_set = polish_u
            best_score = polish_score
    
    return best_set
