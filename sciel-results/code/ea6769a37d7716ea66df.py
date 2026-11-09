import time
import random
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit

    N = 256

    # Precompute 4-APs
    four_ap_by_elem = [[] for _ in range(N)]
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            a0 = a - 1
            b0 = a + d - 1
            c0 = a + 2 * d - 1
            d0 = a + 3 * d - 1
            mask = (1 << a0) | (1 << b0) | (1 << c0) | (1 << d0)
            four_ap_by_elem[a0].append(mask)
            four_ap_by_elem[b0].append(mask)
            four_ap_by_elem[c0].append(mask)
            four_ap_by_elem[d0].append(mask)

    # Precompute 3-AP neighbors
    three_ap_by_elem = [[] for _ in range(N)]
    for d in range(1, (N - 1) // 2 + 1):
        max_a = N - 2 * d
        for a in range(1, max_a + 1):
            a0 = a - 1
            b0 = a + d - 1
            c0 = a + 2 * d - 1
            three_ap_by_elem[a0].append((b0, c0))
            three_ap_by_elem[b0].append((a0, c0))
            three_ap_by_elem[c0].append((a0, b0))

    def can_add(state, x):
        for mask in four_ap_by_elem[x]:
            if (state & mask) == (mask ^ (1 << x)):
                return False
        return True

    def calc_delta_add(state, x):
        delta = 0
        for u, v in three_ap_by_elem[x]:
            if ((state >> u) & 1) and ((state >> v) & 1):
                delta += 1
        return delta

    def count_3aps(state):
        count = 0
        for x in range(N):
            if (state >> x) & 1:
                for u, v in three_ap_by_elem[x]:
                    if ((state >> u) & 1) and ((state >> v) & 1):
                        count += 1
        return count // 3

    # --- Phase 1: Density-Heuristic Greedy Construction ---
    best_state = 0
    best_score = 0

    num_restarts = 3

    for r in range(num_restarts):
        if time.time() > deadline - 15.0:
            break

        state = 0
        current_score = 0
        available = list(range(N))
        random.shuffle(available)

        # Use a priority-based greedy: at each step, pick the element with highest
        # 3-AP gain, breaking ties by random order for diversity.
        while available and time.time() < deadline - 10.0:
            best_x = -1
            best_gain = -1
            best_idx = -1

            # To add diversity, iterate in shuffled order but pick the best gain
            # among the first K candidates, or scan all if N is small enough.
            # N=256, so scanning all is fine.
            for idx, x in enumerate(available):
                if not can_add(state, x):
                    continue
                gain = calc_delta_add(state, x)
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    best_idx = idx

            if best_x == -1:
                break

            state |= (1 << best_x)
            current_score += best_gain
            available.pop(best_idx)

        if current_score > best_score:
            best_score = current_score
            best_state = state

    # --- Phase 2: Simulated Annealing with Fast Bitset Validation ---
    sa_state = best_state
    sa_score = best_score
    best_sa_state = best_state
    best_sa_score = best_score

    T_init = 5.0
    T_final = 0.1
    cooling = 0.9995
    T = T_init

    max_sa_iters = 50000

    for it in range(max_sa_iters):
        if time.time() > deadline - 3.0:
            break

        # Random swap: pick x in state, y not in state
        state_bits = [i for i in range(N) if (sa_state >> i) & 1]
        non_state_bits = [i for i in range(N) if not (sa_state >> i) & 1]

        if not state_bits or not non_state_bits:
            break

        x = random.choice(state_bits)
        y = random.choice(non_state_bits)

        # Check validity of swap: remove x, add y
        valid = True
        for mask in four_ap_by_elem[y]:
            other_mask = mask ^ (1 << y)
            if (sa_state & other_mask) == other_mask:
                if not (other_mask & (1 << x)):
                    valid = False
                    break

        if not valid:
            continue

        # Calculate delta
        loss = 0
        for u, v in three_ap_by_elem[x]:
            if ((sa_state >> u) & 1) and ((sa_state >> v) & 1):
                loss += 1

        gain = 0
        state_minus_x = sa_state & ~(1 << x)
        for u, v in three_ap_by_elem[y]:
            if ((state_minus_x >> u) & 1) and ((state_minus_x >> v) & 1):
                gain += 1

        delta = gain - loss

        # Acceptance
        if delta > 0 or random.random() < math.exp(delta / T):
            sa_state = (sa_state & ~(1 << x)) | (1 << y)
            sa_score += delta

            if sa_score > best_sa_score:
                best_sa_score = sa_score
                best_sa_state = sa_state

        T *= cooling

    # Use best SA result
    if best_sa_score > best_score:
        best_score = best_sa_score
        best_state = best_sa_state

    # --- Phase 3: Final Hill Climbing Polish ---
    current_state = best_state
    current_score = best_score

    improved = True
    max_local_iters = 5000
    iter_count = 0

    while improved and iter_count < max_local_iters and time.time() < deadline - 1.0:
        improved = False
        iter_count += 1

        # Try Add
        for x in range(N):
            if (current_state >> x) & 1:
                continue
            if not can_add(current_state, x):
                continue
            delta = calc_delta_add(current_state, x)
            if delta > 0:
                current_state |= (1 << x)
                current_score += delta
                improved = True
                break

        if improved:
            continue

        # Try Swaps (full sweep, random order)
        state_indices = [i for i in range(N) if (current_state >> i) & 1]
        non_state_indices = [i for i in range(N) if not (current_state >> i) & 1]
        random.shuffle(state_indices)
        random.shuffle(non_state_indices)

        for x in state_indices:
            if time.time() > deadline - 1.0:
                break
            for y in non_state_indices:
                if time.time() > deadline - 1.0:
                    break

                valid = True
                for mask in four_ap_by_elem[y]:
                    other_mask = mask ^ (1 << y)
                    if (current_state & other_mask) == other_mask:
                        if not (other_mask & (1 << x)):
                            valid = False
                            break

                if not valid:
                    continue

                loss = 0
                for u, v in three_ap_by_elem[x]:
                    if ((current_state >> u) & 1) and ((current_state >> v) & 1):
                        loss += 1

                gain = 0
                state_minus_x = current_state & ~(1 << x)
                for u, v in three_ap_by_elem[y]:
                    if ((state_minus_x >> u) & 1) and ((state_minus_x >> v) & 1):
                        gain += 1

                delta = gain - loss
                if delta > 0:
                    current_state = (current_state & ~(1 << x)) | (1 << y)
                    current_score += delta
                    improved = True
                    break

            if improved:
                break

    # Final polish: Add any remaining beneficial elements
    changed = True
    while changed and time.time() < deadline - 0.5:
        changed = False
        for x in range(N):
            if (current_state >> x) & 1:
                continue
            if not can_add(current_state, x):
                continue
            delta = calc_delta_add(current_state, x)
            if delta > 0:
                current_state |= (1 << x)
                current_score += delta
                changed = True

    # Convert to list of 1-based integers
    result = []
    for i in range(N):
        if (current_state >> i) & 1:
            result.append(i + 1)

    return sorted(result)
