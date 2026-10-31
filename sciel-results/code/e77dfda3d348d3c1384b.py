import math
import time
import numpy as np

def construct():
    DEADLINE = 145  # seconds, leave 5s safety margin
    T0 = time.time()
    
    def elapsed():
        return time.time() - T0
    
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
    
    def is_golomb_fast(u):
        """Fast Golomb check using difference set."""
        n = len(u)
        if n < 2:
            return True
        diffs = set()
        for i in range(n):
            ui = u[i]
            for j in range(i + 1, n):
                d = u[j] - ui
                if d in diffs:
                    return False
                diffs.add(d)
        return True
    
    # Known verified Golomb rulers
    candidates = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 20],
        [0, 1, 4, 10, 12, 17, 20, 22],
        [0, 1, 4, 10, 18, 23, 29, 34],
        [0, 1, 4, 10, 18, 23, 29, 34, 39],
        [0, 1, 4, 10, 18, 23, 29, 34, 39, 44],
        [0, 1, 4, 10, 18, 23, 29, 34, 39, 44, 48],
        [0, 1, 4, 10, 18, 23, 29, 34, 39, 44, 48, 51],
    ]
    
    def local_search(base_u, time_limit):
        start_time = time.time()
        best_u = list(base_u)
        best_score = compute_score(best_u)
        improved = True
        
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            best_candidate = None
            best_candidate_score = best_score
            
            n = len(best_u)
            current_max = max(best_u) if best_u else 0
            
            # Try adding a new element - wider range
            search_hi = current_max + max(3 * n * n, 10 * n, 50)
            for new_mark in range(current_max + 1, search_hi + 1):
                candidate = sorted(best_u + [new_mark])
                if not is_golomb_fast(candidate):
                    continue
                s = compute_score(candidate)
                if s > best_candidate_score:
                    best_candidate_score = s
                    best_candidate = candidate
                if (time.time() - start_time) > time_limit:
                    break
            
            # Try removing an element (if size > 3)
            if best_u and len(best_u) > 3 and (time.time() - start_time) < time_limit:
                for i in range(1, len(best_u)):  # keep 0
                    candidate = [x for j, x in enumerate(best_u) if j != i]
                    if not is_golomb_fast(candidate):
                        continue
                    s = compute_score(candidate)
                    if s > best_candidate_score:
                        best_candidate_score = s
                        best_candidate = candidate
                    if (time.time() - start_time) > time_limit:
                        break
            
            # Try swapping: replace one element with another - wider delta
            if best_u and (time.time() - start_time) < time_limit:
                for i in range(1, len(best_u)):  # don't swap 0
                    old_val = best_u[i]
                    max_delta = max(50, len(best_u) * 15)
                    for delta in range(1, max_delta + 1):
                        for new_val in [old_val - delta, old_val + delta]:
                            if new_val < 0:
                                continue
                            if new_val in best_u:
                                continue
                            candidate = sorted([x if x != old_val else new_val for x in best_u])
                            if not is_golomb_fast(candidate):
                                continue
                            s = compute_score(candidate)
                            if s > best_candidate_score:
                                best_candidate_score = s
                                best_candidate = candidate
                        if (time.time() - start_time) > time_limit:
                            break
            
            # Try "shift" move: shift all elements except 0 by a small amount
            if best_u and len(best_u) > 3 and (time.time() - start_time) < time_limit:
                for shift in range(1, 10):
                    for direction in [1, -1]:
                        candidate = [0] + [x + shift * direction for x in best_u[1:]]
                        if any(x < 0 for x in candidate):
                            continue
                        if len(set(candidate)) != len(candidate):
                            continue
                        candidate = sorted(candidate)
                        if not is_golomb_fast(candidate):
                            continue
                        s = compute_score(candidate)
                        if s > best_candidate_score:
                            best_candidate_score = s
                            best_candidate = candidate
                if (time.time() - start_time) > time_limit:
                    break
            
            if best_candidate is not None:
                best_u = best_candidate
                best_score = best_candidate_score
                improved = True
        
        return best_u, best_score
    
    # Run local search on multiple starting points
    start_points = [
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 20],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 18, 23, 29, 34],
    ]
    
    # Allocate time: first pass with shorter limits, then polish
    for idx, sp in enumerate(start_points):
        remaining = DEADLINE - elapsed()
        if remaining < 10:
            break
        # First few starts get less time, later ones get more
        if idx < len(start_points) - 1:
            time_for_this = min(remaining * 0.2, 30)
        else:
            time_for_this = min(remaining * 0.4, 50)
        if time_for_this < 5:
            time_for_this = max(5, remaining * 0.3)
        ext_u, ext_score = local_search(sp, time_for_this)
        if ext_u and len(ext_u) >= 3:
            candidates.append(ext_u)
    
    # Greedy Golomb ruler construction
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
                if is_golomb_fast(candidate):
                    u = candidate
                    found = True
                if pos > n * n * 30:
                    break
        return u
    
    if elapsed() < DEADLINE - 15:
        greedy_u = greedy_golomb(12, time_limit=10)
        if greedy_u and len(greedy_u) >= 6:
            candidates.append(greedy_u)
    
    # Find best among all candidates
    best_set = [0, 1, 3]
    best_score = 0.0
    for u in candidates:
        u = sorted(set(u))
        if 0 not in u:
            u = [0] + u
        s = compute_score(u)
        if s > best_score:
            best_score = s
            best_set = u
    
    # Final polish: run local search on the best candidate with remaining time
    remaining = DEADLINE - elapsed()
    if remaining > 5:
        best_u, best_score = local_search(best_set, remaining - 3)
        best_set = best_u
    
    return best_set
