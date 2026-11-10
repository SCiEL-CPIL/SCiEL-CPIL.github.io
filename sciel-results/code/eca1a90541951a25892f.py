"""Rewritten program for B2[2] set construction in [1, 1000]."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def _greedy_build(candidates, counts_init=None):
    """Greedy: add candidates in given order if B2[2] constraint holds."""
    chosen = []
    counts = defaultdict(int) if counts_init is None else dict(counts_init)
    chosen_set = set()
    for x in candidates:
        if x in chosen_set:
            continue
        ok = True
        new_sums = []
        for y in chosen:
            s = x + y
            new_sums.append(s)
            if counts[s] + 1 > 2:
                ok = False
                break
        if ok:
            s2 = 2 * x
            new_sums.append(s2)
            if counts[s2] + 1 > 2:
                ok = False
        if ok:
            for s in new_sums:
                counts[s] += 1
            chosen.append(x)
            chosen_set.add(x)
    return chosen, counts

def _is_valid_b2(chosen):
    """Verify B2[2] property."""
    counts = defaultdict(int)
    n = len(chosen)
    for i in range(n):
        for j in range(i, n):
            counts[chosen[i] + chosen[j]] += 1
    return all(v <= 2 for v in counts.values())

def _pressure_greedy(seed_val, N=1000):
    """Greedy with dynamic pressure-based selection.
    
    At each step, select the smallest unused value that minimizes the
    'pressure' it would add to sums already at capacity. This avoids
    clustering and enables better gap-filling.
    """
    rng = random.Random(seed_val)
    max_sum = 2 * N
    cnt = [0] * (max_sum + 1)
    S = set()
    S_list = []
    
    # Precompute candidate list
    all_vals = list(range(1, N + 1))
    
    for _ in range(N):
        if not all_vals:
            break
        
        # Score each candidate by how much pressure it would add
        best_val = None
        best_score = float('inf')
        best_conflicts = 0
        
        # Sample candidates for efficiency when set is large
        if len(S_list) > 50:
            # Only consider a subset of candidates
            sample_size = min(100, len(all_vals))
            candidates_to_try = rng.sample(all_vals, sample_size)
        else:
            candidates_to_try = all_vals
        
        for v in candidates_to_try:
            if v in S:
                continue
            
            # Quick check: if 2*v is already at capacity, skip
            if cnt[2 * v] >= 2:
                continue
            
            # Count conflicts: how many sums would exceed capacity
            conflicts = 0
            pressure = 0
            for y in S_list:
                s = v + y
                if cnt[s] >= 2:
                    conflicts += 1
                    if conflicts > 5:  # Early exit
                        break
                elif cnt[s] == 1:
                    pressure += 1  # This sum would go to capacity
            
            if conflicts == 0:
                # Score: prefer values that create less pressure
                score = pressure * 10 + v  # Tie-break by smaller value
            else:
                score = float('inf')
            
            if score < best_score:
                best_score = score
                best_val = v
                best_conflicts = conflicts
        
        if best_val is None:
            # Fallback: try all remaining values
            for v in all_vals:
                if v in S:
                    continue
                if cnt[2 * v] >= 2:
                    continue
                conflicts = 0
                for y in S_list:
                    if cnt[v + y] >= 2:
                        conflicts += 1
                        break
                if conflicts == 0:
                    best_val = v
                    break
        
        if best_val is None:
            break
        
        # Add best_val
        S.add(best_val)
        S_list.append(best_val)
        all_vals.remove(best_val)
        cnt[2 * best_val] += 1
        for y in S_list:
            if y != best_val:
                cnt[best_val + y] += 1
    
    return S, cnt

def _local_search_array(S, cnt, start_time, time_limit, N=1000, max_sum=2000):
    """Array-based local search with insertion-first strategy."""
    S_list = list(S)
    best_size = len(S_list)
    best_S = set(S_list)
    
    no_improve = 0
    max_no_improve = 500
    rng = random.Random(99999)
    
    while time.time() - start_time < time_limit:
        if time.time() - start_time > time_limit - 2:
            break
        
        # Try insertions first
        added = False
        
        # Smart candidate ordering: try values with minimal conflict
        potential = []
        S_set = S
        for v in range(1, N + 1):
            if v in S_set:
                continue
            if cnt[2 * v] >= 2:
                continue
            conflict_count = 0
            for y in S:
                if cnt[v + y] >= 2:
                    conflict_count += 1
                    if conflict_count > 3:
                        break
            if conflict_count == 0:
                potential.append((0, v))
        
        if potential:
            # Try zero-conflict candidates in random order
            rng.shuffle(potential)
            for _, v in potential:
                if time.time() - start_time > time_limit - 2:
                    break
                if v in S:
                    continue
                if cnt[2 * v] >= 2:
                    continue
                ok = True
                for y in S:
                    if cnt[v + y] >= 2:
                        ok = False
                        break
                if ok:
                    S.add(v)
                    cnt[2 * v] += 1
                    for y in S:
                        if y != v:
                            cnt[v + y] += 1
                    added = True
                    break
        
        # If no smart insertion worked, try random
        if not added:
            for _ in range(50):
                v = rng.randint(1, N)
                if v in S:
                    continue
                if cnt[2 * v] >= 2:
                    continue
                ok = True
                for y in S:
                    if cnt[v + y] >= 2:
                        ok = False
                        break
                if not ok:
                    continue
                S.add(v)
                cnt[2 * v] += 1
                for y in S:
                    if y != v:
                        cnt[v + y] += 1
                added = True
                break
        
        if added:
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)
            no_improve = 0
        else:
            # Try swaps
            swapped = False
            
            # Compute which elements are most "expensive" to keep
            # An element x is expensive if removing it frees up many capacity slots
            element_value = {}
            for x in S:
                freed_capacity = 0
                for y in S:
                    if y >= x:
                        s = x + y
                        if cnt[s] == 2:
                            freed_capacity += 1
                element_value[x] = freed_capacity
            
            # Sort by value (highest first) - these are candidates for removal
            sorted_elements = sorted(S, key=lambda x: element_value.get(x, 0), reverse=True)
            
            for u in sorted_elements[:15]:
                if time.time() - start_time > time_limit - 2:
                    break
                
                # Remove u
                S.remove(u)
                cnt[2 * u] -= 1
                for y in S:
                    cnt[u + y] -= 1
                
                # Find best replacement
                best_v = None
                best_conflicts = float('inf')
                
                S_set = S
                for v in range(1, N + 1):
                    if v in S_set or v == u:
                        continue
                    if cnt[2 * v] >= 2:
                        continue
                    conflict_count = 0
                    for y in S:
                        if cnt[v + y] >= 2:
                            conflict_count += 1
                            if conflict_count > best_conflicts:
                                break
                    if conflict_count < best_conflicts:
                        best_conflicts = conflict_count
                        best_v = v
                        if conflict_count == 0:
                            break
                
                if best_v is not None and best_conflicts == 0:
                    S.add(best_v)
                    cnt[2 * best_v] += 1
                    for y in S:
                        if y != best_v:
                            cnt[best_v + y] += 1
                    swapped = True
                    break
                else:
                    # Revert
                    S.add(u)
                    cnt[2 * u] += 1
                    for y in S:
                        if y != u:
                            cnt[u + y] += 1
            
            if not swapped:
                # Random swap fallback
                for _ in range(50):
                    if not S:
                        break
                    u = rng.choice(list(S))
                    v = rng.randint(1, N)
                    if v in S or v == u:
                        continue
                    
                    S.remove(u)
                    cnt[2 * u] -= 1
                    for y in S:
                        cnt[u + y] -= 1
                    
                    if cnt[2 * v] >= 2:
                        S.add(u)
                        cnt[2 * u] += 1
                        for y in S:
                            if y != u:
                                cnt[u + y] += 1
                        continue
                    ok = True
                    for y in S:
                        if cnt[v + y] >= 2:
                            ok = False
                            break
                    if not ok:
                        S.add(u)
                        cnt[2 * u] += 1
                        for y in S:
                            if y != u:
                                cnt[u + y] += 1
                        continue
                    
                    S.add(v)
                    cnt[2 * v] += 1
                    for y in S:
                        if y != v:
                            cnt[v + y] += 1
                    swapped = True
                    break
            
            if swapped:
                no_improve += 1
            else:
                no_improve += 1
            
            if no_improve > max_no_improve:
                # Shake: remove 3 most expensive elements
                if len(S) > 3:
                    element_value = {}
                    for x in S:
                        freed_capacity = 0
                        for y in S:
                            if y >= x:
                                s = x + y
                                if cnt[s] == 2:
                                    freed_capacity += 1
                        element_value[x] = freed_capacity
                    sorted_elements = sorted(S, key=lambda x: element_value.get(x, 0), reverse=True)
                    to_remove = sorted_elements[:3]
                    for u in to_remove:
                        if u in S:
                            S.remove(u)
                            cnt[2 * u] -= 1
                            for y in S:
                                cnt[u + y] -= 1
                no_improve = 0
    
    return best_S

def _final_polish(S, cnt, start_time, time_limit, N=1000, max_sum=2000):
    """Final high-precision polish: try all remaining values in multiple orders."""
    S_set = S
    improved = True
    orders = [
        range(1, N + 1),
        range(N, 0, -1),
        range(1, N + 1, 2),
        range(2, N + 1, 2),
    ]
    
    while improved and time.time() - start_time < time_limit - 2:
        improved = False
        for order in orders:
            for v in order:
                if time.time() - start_time > time_limit - 2:
                    break
                if v in S_set:
                    continue
                if cnt[2 * v] >= 2:
                    continue
                ok = True
                for y in S:
                    if cnt[v + y] >= 2:
                        ok = False
                        break
                if ok:
                    S.add(v)
                    cnt[2 * v] += 1
                    for y in S:
                        if y != v:
                            cnt[v + y] += 1
                    improved = True
                    break
            if improved:
                break
    
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s
    
    N = 1000
    max_sum = 2 * N
    
    best = []
    best_size = 0
    best_S = set()
    best_cnt = [0] * (max_sum + 1)
    
    # Strategy 1: Standard greedy (increasing order)
    candidates = list(range(1, N + 1))
    chosen1, _ = _greedy_build(candidates)
    best = chosen1[:]
    best_size = len(best)
    best_S = set(best)
    for i in range(len(best)):
        for j in range(i, len(best)):
            best_cnt[best[i] + best[j]] += 1
    
    # Strategy 2: Pressure-based greedy with multiple seeds
    for seed in range(20):
        if time.time() > deadline - 30.0:
            break
        S, cnt = _pressure_greedy(seed, N)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
            best_cnt = cnt
            best = list(S)
    
    # Strategy 3: Randomized greedy with diverse orderings
    rng = random.Random(42)
    for restart in range(15):
        if time.time() > deadline - 40.0:
            break
        cand = list(range(1, N + 1))
        if restart % 3 == 0:
            rng.shuffle(cand)
        elif restart % 3 == 1:
            # Bias towards smaller
            weights = [1.0 / (1 + i * 0.02) for i in range(N)]
            indexed = list(zip(weights, range(1, N + 1)))
            rng.shuffle(indexed)
            cand = [x[1] for x in indexed]
        else:
            # Interleaved
            half = N // 2
            first_half = list(range(1, half + 1))
            second_half = list(range(half + 1, N + 1))
            rng.shuffle(first_half)
            rng.shuffle(second_half)
            cand = first_half + second_half
        
        chosen_r, _ = _greedy_build(cand)
        if len(chosen_r) > best_size:
            best = chosen_r[:]
            best_size = len(best)
            best_S = set(best)
            best_cnt = [0] * (max_sum + 1)
            for i in range(len(best)):
                for j in range(i, len(best)):
                    best_cnt[best[i] + best[j]] += 1
    
    # Local search on the best solution
    if time.time() < deadline - 10.0:
        ls_time = min(60.0, deadline - time.time() - 5.0)
        if ls_time > 1.0:
            final_S = _local_search_array(best_S, best_cnt, start_time, ls_time, N, max_sum)
            if len(final_S) > best_size:
                best_S = final_S
                best_size = len(best_S)
                best = list(best_S)
                best_cnt = [0] * (max_sum + 1)
                for i in range(len(best)):
                    for j in range(i, len(best)):
                        best_cnt[best[i] + best[j]] += 1
    
    # Final polish pass
    if time.time() < deadline - 5.0:
        polish_time = min(15.0, deadline - time.time() - 2.0)
        if polish_time > 1.0:
            polished_S = _final_polish(best_S, best_cnt, start_time, polish_time, N, max_sum)
            if len(polished_S) > best_size:
                best_S = polished_S
                best_size = len(best_S)
                best = list(best_S)
    
    # Final verification
    if not _is_valid_b2(best):
        best = list(range(1, 1001))
        chosen1, _ = _greedy_build(best)
        best = chosen1[:]
    
    return best
# EVOLVE-BLOCK-END
