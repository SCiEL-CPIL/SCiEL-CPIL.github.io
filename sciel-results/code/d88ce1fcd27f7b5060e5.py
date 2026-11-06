import time
import random
import math
import numpy as np

def compute_metrics(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers."""
    n = len(A)
    # Sumset: A[i] + A[j] for i <= j
    sums = set()
    prods = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            sums.add(ai + A[j])
            prods.add(ai * A[j])
    return max(len(sums), len(prods))

def compute_metrics_fast(A):
    """Faster metric using numpy for the inner loop."""
    n = len(A)
    arr = np.array(A, dtype=np.int64)
    # Upper triangle indices
    ii, jj = np.triu_indices(n)
    sums = arr[ii] + arr[jj]
    prods = arr[ii] * arr[jj]
    return max(len(np.unique(sums)), len(np.unique(prods)))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # Generate 2,3-smooth numbers <= 10000
    smooth_23 = set()
    for i in range(0, 14):
        for j in range(0, 10):
            val = (1 << i) * (3 ** j)
            if val <= 10000:
                smooth_23.add(val)
    smooth_23 = sorted(smooth_23)

    # Generate 2,3,5-smooth numbers <= 10000
    smooth_235 = set()
    for i in range(0, 14):
        for j in range(0, 10):
            for k in range(0, 7):
                val = (1 << i) * (3 ** j) * (5 ** k)
                if val <= 10000:
                    smooth_235.add(val)
    smooth_235 = sorted(smooth_235)

    # Also generate 2,3,5,7-smooth for broader pool
    smooth_2357 = set(smooth_235)
    for i in range(0, 14):
        for j in range(0, 10):
            for k in range(0, 7):
                for m in range(0, 5):
                    val = (1 << i) * (3 ** j) * (5 ** k) * (7 ** m)
                    if val <= 10000:
                        smooth_2357.add(val)
    smooth_2357 = sorted(smooth_2357)

    initial_candidates = []

    # Family 1: Subsets of 2,3-smooth numbers
    if len(smooth_23) >= 20:
        initial_candidates.append(smooth_23[:20])
    for offset in range(0, min(60, len(smooth_23) - 20 + 1)):
        cand = smooth_23[offset:offset + 20]
        if len(set(cand)) == 20 and max(cand) <= 10000:
            initial_candidates.append(cand)

    # Family 2: Subsets of 2,3,5-smooth numbers
    if len(smooth_235) >= 20:
        initial_candidates.append(smooth_235[:20])
    for offset in range(0, min(40, len(smooth_235) - 20 + 1)):
        cand = smooth_235[offset:offset + 20]
        if len(set(cand)) == 20 and max(cand) <= 10000:
            initial_candidates.append(cand)

    # Family 3: Powers of 2
    A_powers2 = [1 << i for i in range(20)]
    if max(A_powers2) <= 10000:
        initial_candidates.append(A_powers2)

    # Family 4: Powers of 3
    A_powers3 = [3 ** i for i in range(1, 21)]
    if max(A_powers3) <= 10000:
        initial_candidates.append(A_powers3)

    # Family 5: Triangular smooth
    tri_candidates = []
    for i in range(0, 8):
        for j in range(0, 8 - i):
            val = (1 << i) * (3 ** j)
            if val <= 10000:
                tri_candidates.append(val)
    tri_candidates = sorted(set(tri_candidates))
    if len(tri_candidates) >= 20:
        initial_candidates.append(tri_candidates[:20])

    # Family 6: Arithmetic progressions
    for start in [1, 2, 3, 5, 10]:
        for step in [1, 2, 3, 5]:
            cand = [start + i * step for i in range(20)]
            if all(x <= 10000 for x in cand) and len(set(cand)) == 20:
                initial_candidates.append(sorted(cand))

    # Evaluate initial candidates
    best_set = None
    best_score = float('inf')
    for cand in initial_candidates:
        s = compute_metrics_fast(cand)
        if s < best_score:
            best_score = s
            best_set = cand[:]

    if not best_set:
        best_set = list(range(1, 21))
        best_score = compute_metrics_fast(best_set)

    current_set = best_set[:]
    current_score = best_score

    # Create pool for replacement
    pool = list(smooth_2357[:300])  # Smallest 300 2,3,5,7-smooth numbers
    pool_set = set(pool)
    pool_set.update(range(1, 150))
    pool = sorted(pool_set)

    # Simulated Annealing with restarts
    T_start = 15.0
    T_end = 0.005
    num_steps = 300000
    max_restarts = 3
    restart_threshold = 30000  # steps without improvement before considering restart

    step = 0
    steps_since_improve = 0
    restart_count = 0

    while step < num_steps:
        if time.time() > deadline - 2.0:
            break

        step += 1

        # Cooling schedule
        T = T_start * (T_end / T_start) ** (step / num_steps)

        # Choose a random index to replace
        idx = random.randint(0, 19)
        old_val = current_set[idx]

        # Choose a new value - bias towards smooth numbers
        r = random.random()
        if r < 0.5:
            new_val = random.choice(pool[:80])
        elif r < 0.8:
            new_val = random.choice(pool[:200])
        else:
            new_val = random.choice(pool)

        # Ensure distinctness
        if new_val in current_set:
            continue

        # Compute new score
        current_set[idx] = new_val
        new_score = compute_metrics_fast(current_set)

        delta = new_score - current_score

        if delta <= 0 or random.random() < math.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_set = current_set[:]
                steps_since_improve = 0
            else:
                steps_since_improve += 1
        else:
            current_set[idx] = old_val
            steps_since_improve += 1

        # Occasionally do a double swap
        if step % 7 == 0:
            idx1 = random.randint(0, 19)
            idx2 = random.randint(0, 19)
            if idx1 != idx2:
                v1 = current_set[idx1]
                v2 = current_set[idx2]
                new_v1 = random.choice(pool[:150])
                new_v2 = random.choice(pool[:150])

                if new_v1 not in current_set and new_v2 not in current_set and new_v1 != new_v2:
                    current_set[idx1] = new_v1
                    current_set[idx2] = new_v2
                    new_score = compute_metrics_fast(current_set)
                    delta = new_score - current_score
                    if delta <= 0 or random.random() < math.exp(-delta / T):
                        current_score = new_score
                        if new_score < best_score:
                            best_score = new_score
                            best_set = current_set[:]
                            steps_since_improve = 0
                        else:
                            steps_since_improve += 1
                    else:
                        current_set[idx1] = v1
                        current_set[idx2] = v2
                        steps_since_improve += 1

        # Triple swap occasionally
        if step % 13 == 0:
            indices = random.sample(range(20), 3)
            old_vals = [current_set[i] for i in indices]
            new_vals = [random.choice(pool[:120]) for _ in range(3)]
            # Check distinctness
            if len(set(new_vals)) == 3:
                for nv in new_vals:
                    if nv in current_set:
                        break
                else:
                    for i, nv in zip(indices, new_vals):
                        current_set[i] = nv
                    new_score = compute_metrics_fast(current_set)
                    delta = new_score - current_score
                    if delta <= 0 or random.random() < math.exp(-delta / T):
                        current_score = new_score
                        if new_score < best_score:
                            best_score = new_score
                            best_set = current_set[:]
                            steps_since_improve = 0
                        else:
                            steps_since_improve += 1
                    else:
                        for i, ov in zip(indices, old_vals):
                            current_set[i] = ov
                        steps_since_improve += 1

        # Restart mechanism: if stuck, perturb best solution
        if steps_since_improve > restart_threshold and restart_count < max_restarts:
            restart_count += 1
            current_set = best_set[:]
            # Perturb: replace 3-5 random elements
            n_replace = random.randint(3, 5)
            replace_idx = random.sample(range(20), n_replace)
            for ri in replace_idx:
                current_set[ri] = random.choice(pool[:100])
                while current_set[ri] in current_set[:ri] or current_set[ri] in current_set[ri+1:]:
                    current_set[ri] = random.choice(pool[:100])
            current_score = compute_metrics_fast(current_set)
            steps_since_improve = 0
            T = T_start * 0.3  # Reset temperature

    # Final local search polish
    if time.time() < deadline - 3.0:
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for i in range(20):
                old_val = best_set[i]
                # Try smooth pool
                for new_val in pool[:200]:
                    if new_val in best_set:
                        continue
                    best_set[i] = new_val
                    s = compute_metrics_fast(best_set)
                    if s < best_score:
                        best_score = s
                        improved = True
                    else:
                        best_set[i] = old_val
                best_set[i] = old_val

    # Second polish pass with broader pool
    if time.time() < deadline - 1.0:
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            for i in range(20):
                old_val = best_set[i]
                for new_val in pool:
                    if new_val in best_set:
                        continue
                    best_set[i] = new_val
                    s = compute_metrics_fast(best_set)
                    if s < best_score:
                        best_score = s
                        improved = True
                    else:
                        best_set[i] = old_val
                best_set[i] = old_val

    # Final 2-swap local search
    if time.time() < deadline - 0.5:
        improved = True
        while improved and time.time() < deadline - 0.3:
            improved = False
            for i in range(20):
                for j in range(i + 1, 20):
                    old_i = best_set[i]
                    old_j = best_set[j]
                    for new_i in pool[:100]:
                        if new_i in best_set:
                            continue
                        for new_j in pool[:100]:
                            if new_j in best_set or new_j == new_i:
                                continue
                            best_set[i] = new_i
                            best_set[j] = new_j
                            s = compute_metrics_fast(best_set)
                            if s < best_score:
                                best_score = s
                                improved = True
                                break
                            else:
                                best_set[i] = old_i
                                best_set[j] = old_j
                        if improved:
                            break
                    best_set[i] = old_i
                    best_set[j] = old_j

    return sorted(best_set)
