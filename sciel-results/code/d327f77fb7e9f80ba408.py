import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 88.0  # Safety margin before 90s kill

    N = 128
    min_sum = 128
    max_sum = 256
    target_sums = list(range(min_sum, max_sum + 1))
    num_targets = len(target_sums)  # 129

    # Precompute index mapping for sums in target range
    sum_to_idx = np.full(max_sum + 1, -1, dtype=np.int32)
    for i, s in enumerate(target_sums):
        sum_to_idx[s] = i

    def get_counts_fast(A):
        """Returns array of counts for sums in [0, 256]"""
        A_arr = np.array(sorted(A), dtype=np.int64)
        k = len(A_arr)
        if k == 0:
            return np.zeros(max_sum + 1, dtype=np.int32)
        
        # Vectorized approach for small sets, loop for larger to avoid huge memory
        if k <= 60:
            A_i = A_arr[:, None]
            A_j = A_arr[None, :]
            sums = A_i + A_j
            # Only upper triangle including diagonal for unordered pairs with self
            mask = np.triu(np.ones((k, k), dtype=bool), k=0)
            valid_sums = sums[mask]
            valid_sums = valid_sums[valid_sums <= max_sum]
            if len(valid_sums) > 0:
                counts = np.bincount(valid_sums, minlength=max_sum + 1).astype(np.int32)
            else:
                counts = np.zeros(max_sum + 1, dtype=np.int32)
            return counts
        else:
            counts = np.zeros(max_sum + 1, dtype=np.int32)
            for i in range(k):
                ai = A_arr[i]
                s = ai + ai
                if s <= max_sum:
                    counts[s] += 1
                for j in range(i + 1, k):
                    aj = A_arr[j]
                    s = ai + aj
                    if s <= max_sum:
                        counts[s] += 1
            return counts

    def get_cost_detailed(A):
        """Returns (max_mult, uncovered_count, counts_array)"""
        counts = get_counts_fast(A)
        target_counts = counts[min_sum:max_sum + 1]
        uncovered = np.sum(target_counts == 0)
        if uncovered > 0:
            return (1000 + uncovered, uncovered, counts)
        return (int(np.max(target_counts)), 0, counts)

    def incremental_swap_cost(A, x_in, x_out, current_counts):
        """
        Compute new counts after removing x_in and adding x_out.
        Returns (new_counts, new_cost, new_uncovered)
        """
        new_counts = current_counts.copy()

        # Subtract contributions involving x_in
        s = 2 * x_in
        if s <= max_sum:
            new_counts[s] -= 1
        for y in A:
            if y == x_in:
                continue
            s = x_in + y
            if s <= max_sum:
                new_counts[s] -= 1

        # Add contributions involving x_out
        s = 2 * x_out
        if s <= max_sum:
            new_counts[s] += 1
        for y in A:
            if y == x_in:
                continue
            s = x_out + y
            if s <= max_sum:
                new_counts[s] += 1

        target_counts = new_counts[min_sum:max_sum + 1]
        uncovered = np.sum(target_counts == 0)
        if uncovered > 0:
            new_cost = 1000 + uncovered
        else:
            new_cost = int(np.max(target_counts))

        return new_counts, new_cost, uncovered

    def incremental_add_cost(A, x_add, current_counts):
        """
        Compute new counts after adding x_add to A.
        Returns (new_counts, new_cost, new_uncovered)
        """
        new_counts = current_counts.copy()

        s = 2 * x_add
        if s <= max_sum:
            new_counts[s] += 1
        for y in A:
            s = x_add + y
            if s <= max_sum:
                new_counts[s] += 1

        target_counts = new_counts[min_sum:max_sum + 1]
        uncovered = np.sum(target_counts == 0)
        if uncovered > 0:
            new_cost = 1000 + uncovered
        else:
            new_cost = int(np.max(target_counts))

        return new_counts, new_cost, uncovered

    def incremental_remove_cost(A, x_rem, current_counts):
        """
        Compute new counts after removing x_rem from A.
        Returns (new_counts, new_cost, new_uncovered)
        """
        new_counts = current_counts.copy()

        s = 2 * x_rem
        if s <= max_sum:
            new_counts[s] -= 1
        for y in A:
            if y == x_rem:
                continue
            s = x_rem + y
            if s <= max_sum:
                new_counts[s] -= 1

        target_counts = new_counts[min_sum:max_sum + 1]
        uncovered = np.sum(target_counts == 0)
        if uncovered > 0:
            new_cost = 1000 + uncovered
        else:
            new_cost = int(np.max(target_counts))

        return new_counts, new_cost, uncovered

    # Direct construction: try several structured initial sets
    best_A = None
    best_cost = float('inf')

    candidates = []

    # Candidate 1: k=16-20, spread out evenly
    for k in [16, 17, 18, 19, 20]:
        base_points = np.linspace(0, 128, k, dtype=int)
        base_set = set(int(x) for x in base_points)
        if len(base_set) < k:
            base_set = set()
            used = set()
            for i in range(k):
                x = round(i * 128 / (k - 1))
                while x in used:
                    x += 1
                    if x > 128:
                        x = 0
                base_set.add(x)
                used.add(x)
            base_set = set(x for x in base_set if 0 <= x <= 128)
        candidates.append(set(base_set))

        for offset in range(-3, 4):
            perturbed = set()
            for i in range(k):
                x = int(round(i * 128 / (k - 1))) + offset
                if 0 <= x <= 128:
                    perturbed.add(x)
            if len(perturbed) >= k - 2:
                candidates.append(perturbed)

    # Candidate: Two clusters approach
    for a in range(5, 15):
        for b in range(113, 128):
            if b > a + 1:
                clus = set(range(0, a + 1)) | set(range(b, 129))
                candidates.append(clus)

    # Candidate: Mix of small and large
    for m in range(6, 12):
        small = set(range(0, m))
        large = set(range(128 - m + 1, 129))
        mid = set(range(60 - m // 2, 60 + m // 2 + 1))
        mixed = small | large | mid
        if mixed <= set(range(129)):
            candidates.append(mixed)

    # Evaluate all candidates
    for cand in candidates:
        cand = set(x for x in cand if 0 <= x <= 128)
        if len(cand) == 0:
            continue
        cost, uncovered, _ = get_cost_detailed(cand)
        if cost < best_cost:
            best_cost = cost
            best_A = set(cand)

    # Now optimize the best candidate with local search using incremental updates
    if best_A is not None:
        A = set(best_A)
        current_counts = get_counts_fast(A)
        current_cost, _, _ = get_cost_detailed(A)

        # Local search: try swaps and additions/removals with incremental cost
        # Use multiple random seeds/restarts if time permits
        rng = random.Random(12345)
        outside = list(set(range(129)) - A)
        A_list = list(A)
        
        # Sort lists for faster removal? No, sets are O(1) remove. Lists are O(N).
        # But N is small (~20-30). List removal is fine.
        
        # Simulated Annealing parameters
        T_start = 5.0
        T_end = 0.01
        alpha = 0.9998
        
        T = T_start
        steps = 0
        best_peak = current_cost
        best_A_local = set(A)

        # Main search loop
        while time.time() < deadline - 2.0:
            steps += 1
            
            # Update temperature
            if steps % 50 == 0:
                T *= alpha
                if T < T_end:
                    break

            move_type = rng.random()
            new_A = set(A)
            new_counts = None
            new_cost = None
            accepted = False

            if move_type < 0.6 and len(outside) > 0:
                # Swap
                x_in = rng.choice(A_list)
                x_out = rng.choice(outside)

                new_counts, new_cost, _ = incremental_swap_cost(A, x_in, x_out, current_counts)

                if new_cost < current_cost:
                    # Improvement
                    accepted = True
                    A.remove(x_in)
                    A.add(x_out)
                    outside.remove(x_out)
                    outside.append(x_in)
                    A_list.remove(x_in)
                    A_list.append(x_out)
                    current_counts = new_counts
                    current_cost = new_cost
                    if current_cost < best_peak:
                        best_peak = current_cost
                        best_A_local = set(A)
                else:
                    # Accept with probability
                    delta_cost = new_cost - current_cost
                    if delta_cost > 0:
                        prob = np.exp(-delta_cost / T)
                        if rng.random() < prob:
                            accepted = True
                            A.remove(x_in)
                            A.add(x_out)
                            outside.remove(x_out)
                            outside.append(x_in)
                            A_list.remove(x_in)
                            A_list.append(x_out)
                            current_counts = new_counts
                            current_cost = new_cost
                            if current_cost < best_peak:
                                best_peak = current_cost
                                best_A_local = set(A)

            elif move_type < 0.8 and len(A) > 14:
                # Removal
                x = rng.choice(A_list)

                new_counts, new_cost, _ = incremental_remove_cost(A, x, current_counts)

                if new_cost < current_cost:
                    accepted = True
                    A.remove(x)
                    outside.append(x)
                    A_list.remove(x)
                    current_counts = new_counts
                    current_cost = new_cost
                    if current_cost < best_peak:
                        best_peak = current_cost
                        best_A_local = set(A)
                else:
                    delta_cost = new_cost - current_cost
                    if delta_cost > 0:
                        prob = np.exp(-delta_cost / T)
                        if rng.random() < prob:
                            accepted = True
                            A.remove(x)
                            outside.append(x)
                            A_list.remove(x)
                            current_counts = new_counts
                            current_cost = new_cost
                            if current_cost < best_peak:
                                best_peak = current_cost
                                best_A_local = set(A)

            elif move_type < 1.0 and len(A) < 30:
                # Addition
                x = rng.choice(outside)

                new_counts, new_cost, _ = incremental_add_cost(A, x, current_counts)

                if new_cost < current_cost:
                    accepted = True
                    A.add(x)
                    outside.remove(x)
                    A_list.append(x)
                    current_counts = new_counts
                    current_cost = new_cost
                    if current_cost < best_peak:
                        best_peak = current_cost
                        best_A_local = set(A)
                else:
                    delta_cost = new_cost - current_cost
                    if delta_cost > 0:
                        prob = np.exp(-delta_cost / T)
                        if rng.random() < prob:
                            accepted = True
                            A.add(x)
                            outside.remove(x)
                            A_list.append(x)
                            current_counts = new_counts
                            current_cost = new_cost
                            if current_cost < best_peak:
                                best_peak = current_cost
                                best_A_local = set(A)

            if best_peak <= 2:
                break

        # Update global best
        if best_peak < best_cost:
            best_cost = best_peak
            best_A = best_A_local

    # Final verification and cleanup
    if best_A is None:
        # Fallback: return a valid set
        best_A = set(range(0, 129, 8))

    return list(best_A)
