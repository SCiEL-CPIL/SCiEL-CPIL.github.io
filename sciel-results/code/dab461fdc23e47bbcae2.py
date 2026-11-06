import time
import random
import math

def construct():
    t0 = time.time()
    DEADLINE = t0 + 85.0

    N = 256
    all_nums = list(range(1, N + 1))

    # Precompute conflict_mask
    conflict_mask = [0] * N
    for i in range(N):
        a = i + 1
        for j in range(N):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) == 0:
                conflict_mask[i] |= (1 << j)

    # Fast popcount with 16-bit lookup
    popcount_table = [0] * (1 << 16)
    for i in range(1, 1 << 16):
        popcount_table[i] = popcount_table[i >> 1] + (i & 1)

    def fast_popcount(x):
        count = 0
        while x:
            count += popcount_table[x & 0xFFFF]
            x >>= 16
        return count

    odds_indices = [i for i in range(N) if (i + 1) % 2 == 1]
    evens_indices = [i for i in range(N) if (i + 1) % 2 == 0]

    # Start with all odds
    current_mask = 0
    for i in odds_indices:
        current_mask |= (1 << i)

    best_mask = current_mask
    best_size = fast_popcount(current_mask)

    # Phase 1: Add evens with 0 conflicts
    changed = True
    while changed and time.time() < DEADLINE - 50:
        changed = False
        for e in evens_indices:
            if time.time() > DEADLINE - 50:
                break
            if (current_mask >> e) & 1:
                continue
            if (conflict_mask[e] & current_mask) == 0:
                current_mask |= (1 << e)
                best_mask = current_mask
                best_size = fast_popcount(current_mask)
                changed = True

    # Phase 1b: Try adding evens with minimal conflicts (net gain)
    changed = True
    while changed and time.time() < DEADLINE - 45:
        changed = False
        for e in evens_indices:
            if time.time() > DEADLINE - 45:
                break
            if (current_mask >> e) & 1:
                continue
            conflicts = conflict_mask[e] & current_mask
            num_conflicts = fast_popcount(conflicts)
            if num_conflicts <= 1:
                new_mask = (current_mask & ~conflicts) | (1 << e)
                new_size = best_size - num_conflicts + 1
                if new_size > best_size:
                    current_mask = new_mask
                    best_mask = current_mask
                    best_size = new_size
                    changed = True

    # Phase 2: Simulated Annealing
    T = 2.0
    cooling = 0.9995
    max_iters = 200000

    current_size = best_size
    no_improve = 0
    NO_IMPROVE_LIMIT = 5000

    for i_iter in range(max_iters):
        if time.time() > DEADLINE - 5:
            break

        r = random.random()
        if r < 0.55:
            # Single add move
            if random.random() < 0.7:
                e_candidates = [idx for idx in evens_indices if not ((current_mask >> idx) & 1)]
                if not e_candidates:
                    continue
                e = random.choice(e_candidates)
            else:
                o_candidates = [idx for idx in odds_indices if not ((current_mask >> idx) & 1)]
                if not o_candidates:
                    continue
                e = random.choice(o_candidates)

            conflicts = conflict_mask[e] & current_mask
            num_conflicts = fast_popcount(conflicts)
            new_size = current_size - num_conflicts + 1

            if new_size > current_size:
                current_mask = (current_mask & ~conflicts) | (1 << e)
                current_size = new_size
                if current_size > best_size:
                    best_mask = current_mask
                    best_size = current_size
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                delta = new_size - current_size
                if delta == 0:
                    if random.random() < 0.4:
                        current_mask = (current_mask & ~conflicts) | (1 << e)
                        no_improve += 1
                else:
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        current_mask = (current_mask & ~conflicts) | (1 << e)
                        current_size = new_size
                        no_improve += 1
        elif r < 0.75:
            # 1-swap: remove 1, add 1
            bits = [i for i in range(N) if (current_mask >> i) & 1]
            if not bits:
                continue
            rm_idx = random.choice(bits)
            temp_mask = current_mask & ~(1 << rm_idx)

            not_in_temp = [idx for idx in range(N) if not ((temp_mask >> idx) & 1)]
            if not not_in_temp:
                continue

            # Prefer zero-conflict candidates
            zero_conflict = [idx for idx in not_in_temp if (conflict_mask[idx] & temp_mask) == 0]
            if zero_conflict:
                e = random.choice(zero_conflict)
            else:
                e = random.choice(not_in_temp)

            conflicts_with_temp = conflict_mask[e] & temp_mask
            if conflicts_with_temp != 0:
                continue

            # Neutral move: size unchanged
            if random.random() < 0.3:
                current_mask = temp_mask | (1 << e)
                no_improve += 1
        elif r < 0.90:
            # Multi-exchange: remove k, add 1
            k = random.choice([2, 3])
            bits = [i for i in range(N) if (current_mask >> i) & 1]
            if len(bits) < k:
                continue
            removed = random.sample(bits, k)

            temp_mask = current_mask
            for rm in removed:
                temp_mask &= ~(1 << rm)

            not_in_temp = [idx for idx in range(N) if not ((temp_mask >> idx) & 1)]
            if not not_in_temp:
                continue

            # Prefer zero-conflict candidates
            zero_conflict = [idx for idx in not_in_temp if (conflict_mask[idx] & temp_mask) == 0]
            if zero_conflict:
                e = random.choice(zero_conflict)
            else:
                e = random.choice(not_in_temp)

            conflicts_with_temp = conflict_mask[e] & temp_mask
            num_conf = fast_popcount(conflicts_with_temp)
            new_size = current_size - k - num_conf + 1

            if new_size > current_size:
                current_mask = temp_mask | (1 << e)
                current_size = new_size
                if current_size > best_size:
                    best_mask = current_mask
                    best_size = current_size
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                delta = new_size - current_size
                if delta == 0:
                    if random.random() < 0.3:
                        current_mask = temp_mask | (1 << e)
                        no_improve += 1
                else:
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        current_mask = temp_mask | (1 << e)
                        current_size = new_size
                        no_improve += 1
        else:
            # Random removal
            bits = [i for i in range(N) if (current_mask >> i) & 1]
            if bits:
                r_idx = random.choice(bits)
                current_mask &= ~(1 << r_idx)
                current_size -= 1
                no_improve += 1

        T *= cooling
        if T < 0.05:
            T = 0.05

        if no_improve > NO_IMPROVE_LIMIT:
            T = max(T, 1.0)
            no_improve = 0

    # Final polish
    for e in evens_indices + odds_indices:
        if time.time() > DEADLINE - 1:
            break
        if (best_mask >> e) & 1:
            continue
        if (conflict_mask[e] & best_mask) == 0:
            best_mask |= (1 << e)
            best_size = fast_popcount(best_mask)

    result = sorted([i + 1 for i in range(N) if (best_mask >> i) & 1])
    return result
