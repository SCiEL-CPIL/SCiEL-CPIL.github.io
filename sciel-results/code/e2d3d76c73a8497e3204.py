import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    T_min = 128
    T_max = 256
    N_max = 129
    R = T_max - T_min + 1  # 129

    best_A = None
    best_max = float('inf')
    best_sum_sq = float('inf')

    def compute_counts_full(ind):
        conv = np.convolve(ind, ind, mode='full')
        counts = np.empty(R, dtype=np.int32)
        for i, s in enumerate(range(T_min, T_max + 1)):
            c_ord = conv[s]
            if s % 2 == 0:
                half = s // 2
                n_eq = 1 if (half < N_max and ind[half] == 1) else 0
            else:
                n_eq = 0
            counts[i] = (c_ord + n_eq) // 2
        return counts

    def is_valid(counts):
        return bool(np.all(counts > 0))

    def get_metrics(counts):
        return int(np.max(counts)), float(np.sum(counts * counts))

    def remove_element(x, A_cur, ind_cur, counts_cur):
        """Remove element x from A, return new counts"""
        new_counts = counts_cur.copy()
        s_2x = 2 * x
        if T_min <= s_2x <= T_max:
            new_counts[s_2x - T_min] -= 1
        for a in A_cur:
            if a == x: continue
            s_xa = x + a
            if T_min <= s_xa <= T_max:
                new_counts[s_xa - T_min] -= 1
        A_cur.remove(x)
        ind_cur[x] = 0
        return new_counts

    def add_element(y, A_cur, ind_cur, counts_cur):
        """Add element y to A, return new counts"""
        new_counts = counts_cur.copy()
        s_2y = 2 * y
        if T_min <= s_2y <= T_max:
            new_counts[s_2y - T_min] += 1
        for a in A_cur:
            s_ya = y + a
            if T_min <= s_ya <= T_max:
                new_counts[s_ya - T_min] += 1
        A_cur.add(y)
        ind_cur[y] = 1
        return new_counts

    def try_improve(A_initial, time_budget):
        nonlocal best_A, best_max, best_sum_sq
        
        A_set = set(A_initial)
        ind = np.zeros(N_max, dtype=np.int32)
        for x in A_set:
            ind[x] = 1
            
        # Initial counts
        counts = compute_counts_full(ind)
        
        # Repair if invalid
        if not is_valid(counts):
            # Simple greedy repair
            for _ in range(100):
                if is_valid(counts): break
                gaps = np.where(counts == 0)[0]
                if len(gaps) == 0: break
                best_y = None
                best_gain = -1
                for y in range(N_max):
                    if ind[y] == 1: continue
                    gain = 0
                    for gi in gaps:
                        s = T_min + gi
                        a = s - y
                        if 0 <= a < N_max and (ind[a] == 1 or a == y):
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_y = y
                if best_y is not None:
                    A_set.add(best_y)
                    ind[best_y] = 1
                    counts = compute_counts_full(ind)
                else:
                    break
            if not is_valid(counts):
                return

        cur_max, cur_sum_sq = get_metrics(counts)
        if cur_max < best_max or (cur_max == best_max and cur_sum_sq < best_sum_sq):
            best_max = cur_max
            best_sum_sq = cur_sum_sq
            best_A = set(A_set)

        A_cur = set(A_set)
        ind_cur = ind.copy()
        counts_cur = counts.copy()
        max_cur = cur_max
        sum_sq_cur = cur_sum_sq
        
        # Local Search with peak reduction phase
        no_improve = 0
        T = 1.0
        
        # Phase 1: Standard local search
        while time.time() < start_time + time_budget - 10.0:
            if len(A_cur) == 0: break
            
            # 1. Try to remove an element (prune)
            if len(A_cur) > 20 and random.random() < 0.3:
                removable = [x for x in A_cur if random.random() < 0.1]
                if removable:
                    x = random.choice(removable)
                    new_counts = remove_element(x, A_cur.copy(), ind_cur.copy(), counts_cur)
                    
                    if is_valid(new_counts):
                        new_max = int(np.max(new_counts))
                        new_sum_sq = float(np.sum(new_counts * new_counts))
                        if new_max < max_cur or (new_max == max_cur and new_sum_sq < sum_sq_cur):
                            A_cur.remove(x)
                            ind_cur[x] = 0
                            counts_cur = new_counts
                            max_cur = new_max
                            sum_sq_cur = new_sum_sq
                            no_improve = 0
                            if max_cur < best_max or (max_cur == best_max and sum_sq_cur < best_sum_sq):
                                best_max = max_cur
                                best_sum_sq = sum_sq_cur
                                best_A = set(A_cur)
                            continue
                        else:
                            no_improve += 1
                            if no_improve > 100:
                                no_improve = 0
                            continue

            # 2. Swap with bias toward high-multiplicity reduction
            x = random.choice(list(A_cur))
            
            # Find sums affected by removing x
            affected_sums_x = []
            s_2x = 2 * x
            if T_min <= s_2x <= T_max:
                affected_sums_x.append(s_2x)
            for a in A_cur:
                if a == x: continue
                s_xa = x + a
                if T_min <= s_xa <= T_max:
                    affected_sums_x.append(s_xa)
            
            # Choose y that would reduce high-multiplicity sums
            candidates = [y for y in range(N_max) if ind_cur[y] == 0]
            if not candidates:
                continue
            
            # Score candidates by how much they reduce peak
            best_y = None
            best_score = float('inf')
            for y in random.sample(candidates, min(20, len(candidates))):
                # Estimate impact
                s_2y = 2 * y
                if T_min <= s_2y <= T_max:
                    idx = s_2y - T_min
                    if counts_cur[idx] > max_cur:
                        continue
                
                score = 0
                for a in A_cur:
                    if a == x: continue
                    s_ya = y + a
                    if T_min <= s_ya <= T_max:
                        idx = s_ya - T_min
                        score += counts_cur[idx]
                
                if score < best_score:
                    best_score = score
                    best_y = y
            
            if best_y is None:
                continue
            y = best_y
            
            new_counts = counts_cur.copy()
            
            # Remove x
            s_2x = 2 * x
            if T_min <= s_2x <= T_max:
                new_counts[s_2x - T_min] -= 1
            for a in A_cur:
                if a == x: continue
                s_xa = x + a
                if T_min <= s_xa <= T_max:
                    new_counts[s_xa - T_min] -= 1
            
            # Add y
            s_2y = 2 * y
            if T_min <= s_2y <= T_max:
                new_counts[s_2y - T_min] += 1
            for a in A_cur:
                if a == x: continue
                s_ya = y + a
                if T_min <= s_ya <= T_max:
                    new_counts[s_ya - T_min] += 1
            
            if not is_valid(new_counts):
                no_improve += 1
                if no_improve > 500:
                    break
                continue

            new_max = int(np.max(new_counts))
            new_sum_sq = float(np.sum(new_counts * new_counts))
            
            accepted = False
            if new_max < max_cur:
                accepted = True
            elif new_max == max_cur:
                if new_sum_sq < sum_sq_cur:
                    accepted = True
                elif random.random() < 0.05:
                    accepted = True
            else:
                if T > 0.01:
                    prob = np.exp(-(new_max - max_cur) / T)
                    if random.random() < prob:
                        accepted = True
                    T *= 0.995
                
            if accepted:
                A_cur.remove(x)
                A_cur.add(y)
                ind_cur[x] = 0
                ind_cur[y] = 1
                counts_cur = new_counts
                max_cur = new_max
                sum_sq_cur = new_sum_sq
                no_improve = 0
                
                if max_cur < best_max or (max_cur == best_max and sum_sq_cur < best_sum_sq):
                    best_max = max_cur
                    best_sum_sq = sum_sq_cur
                    best_A = set(A_cur)
            else:
                no_improve += 1
                if no_improve > 500:
                    break
        
        # Phase 2: Targeted peak reduction - focus on high-multiplicity sums
        if time.time() < start_time + time_budget - 2.0:
            peak_threshold = max_cur
            for _ in range(50):
                if time.time() > start_time + time_budget - 2.0:
                    break
                
                # Find sums with peak multiplicity
                peak_indices = np.where(counts_cur >= peak_threshold)[0]
                if len(peak_indices) == 0:
                    break
                
                # Try to reduce peak by swapping elements that contribute to peak sums
                improved = False
                for peak_idx in peak_indices:
                    if time.time() > start_time + time_budget - 2.0:
                        break
                    
                    s_peak = T_min + peak_idx
                    # Find pairs in A that sum to s_peak
                    contributing_pairs = []
                    for a in A_cur:
                        b = s_peak - a
                        if b in A_cur and a <= b:
                            contributing_pairs.append((a, b))
                    
                    if len(contributing_pairs) < 2:
                        continue
                    
                    # Try to remove one element from a contributing pair and replace it
                    for (a, b) in contributing_pairs:
                        # Try removing a and adding a new element that doesn't contribute to s_peak
                        for y in range(N_max):
                            if ind_cur[y] == 1:
                                continue
                            if y + a == s_peak or y + b == s_peak:
                                continue
                            
                            new_counts = counts_cur.copy()
                            # Remove a
                            s_2a = 2 * a
                            if T_min <= s_2a <= T_max:
                                new_counts[s_2a - T_min] -= 1
                            for c in A_cur:
                                if c == a: continue
                                s_ac = a + c
                                if T_min <= s_ac <= T_max:
                                    new_counts[s_ac - T_min] -= 1
                            
                            # Add y
                            s_2y = 2 * y
                            if T_min <= s_2y <= T_max:
                                new_counts[s_2y - T_min] += 1
                            for c in A_cur:
                                if c == a: continue
                                s_yc = y + c
                                if T_min <= s_yc <= T_max:
                                    new_counts[s_yc - T_min] += 1
                            
                            if not is_valid(new_counts):
                                continue
                            
                            new_max = int(np.max(new_counts))
                            new_sum_sq = float(np.sum(new_counts * new_counts))
                            
                            if new_max < max_cur or (new_max == max_cur and new_sum_sq < sum_sq_cur):
                                A_cur.remove(a)
                                A_cur.add(y)
                                ind_cur[a] = 0
                                ind_cur[y] = 1
                                counts_cur = new_counts
                                max_cur = new_max
                                sum_sq_cur = new_sum_sq
                                improved = True
                                if max_cur < best_max or (max_cur == best_max and sum_sq_cur < best_sum_sq):
                                    best_max = max_cur
                                    best_sum_sq = sum_sq_cur
                                    best_A = set(A_cur)
                                break
                        
                        if improved:
                            break
                
                if not improved:
                    break

    # Initializations
    starts = []
    # 1. Greedy construction to ensure coverage
    # Start with empty, add elements that cover most gaps
    ind_greedy = np.zeros(N_max, dtype=np.int32)
    counts_greedy = np.zeros(R, dtype=np.int32)
    A_greedy = set()
    for _ in range(50):
        if is_valid(counts_greedy): break
        best_y = None
        best_gain = -1
        for y in range(N_max):
            if ind_greedy[y] == 1: continue
            gain = 0
            for i in range(R):
                s = T_min + i
                if counts_greedy[i] == 0:
                    a = s - y
                    if 0 <= a < N_max and (ind_greedy[a] == 1 or a == y):
                        gain += 1
            if gain > best_gain:
                best_gain = gain
                best_y = y
        if best_y is not None:
            A_greedy.add(best_y)
            ind_greedy[best_y] = 1
            counts_greedy = compute_counts_full(ind_greedy)
        else:
            break
    if is_valid(counts_greedy):
        starts.append(list(A_greedy))
    
    # 2. Random sets with varying densities
    for sz in [30, 40, 50, 60, 70]:
        for _ in range(3):
            starts.append(random.sample(range(0, N_max), sz))
            
    # 3. Structured sets
    starts.append(list(range(64, 129)))
    starts.append(list(range(0, 129, 2)))
    starts.append(list(range(1, 129, 2)))
    
    # Time allocation
    total_time = 85.0
    time_per_start = total_time / len(starts)
    
    for i, s in enumerate(starts):
        if time.time() > start_time + total_time - 2.0:
            break
        remaining = (start_time + total_time) - time.time()
        budget = min(time_per_start, remaining - 1.0)
        if budget > 0.5:
            try_improve(s, budget)

    if best_A is None:
        return list(range(64, 129))
    return list(best_A)
