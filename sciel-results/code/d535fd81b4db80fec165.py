import time
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin for 90s limit

    N = 256
    values = list(range(2, N + 1))
    size = len(values)  # 255

    # Precompute pairwise conflict: i and j are incompatible if lcm(values[i], values[j]) <= 256
    conflict_mask = [0] * size
    for i in range(size):
        vi = values[i]
        for j in range(i + 1, size):
            vj = values[j]
            g = gcd(vi, vj)
            if vi * vj // g <= 256:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)

    # Precompute reciprocal weights as floats for speed
    weights = [1.0 / values[i] for i in range(size)]

    def score_mask_float(mask):
        total = 0.0
        i = 0
        m = mask
        while m:
            if m & 1:
                total += weights[i]
            i += 1
            m >>= 1
        return total

    def is_valid_mask(mask):
        i = 0
        m = mask
        while m:
            if m & 1:
                if conflict_mask[i] & mask:
                    return False
            i += 1
            m >>= 1
        return True

    def greedy_construct(order):
        S_mask = 0
        for i in order:
            if not (conflict_mask[i] & S_mask):
                S_mask |= (1 << i)
        return S_mask

    # Conflict counts for diversity
    conflict_counts = [bin(cm).count('1') for cm in conflict_mask]

    # Pre-sorted orders
    order_asc = list(range(size))
    order_desc = list(range(size - 1, -1, -1))
    order_by_conflict = sorted(range(size), key=lambda x: (conflict_counts[x], x))
    order_by_ratio = sorted(range(size), key=lambda x: (weights[x] / (conflict_counts[x] + 1)), reverse=True)

    # Weight-sorted indices for fast best-v lookup in swaps
    weight_sorted = sorted(range(size), key=lambda x: weights[x], reverse=True)

    def local_search(S_mask, deadline):
        current_score = score_mask_float(S_mask)
        in_set = [False] * size
        for i in range(size):
            if S_mask & (1 << i):
                in_set[i] = True

        improved = True
        while improved:
            improved = False
            elapsed = time.time() - start_time
            if elapsed > deadline:
                break

            # Try additions first (cheap)
            for v in range(size):
                if not in_set[v] and not (conflict_mask[v] & S_mask):
                    S_mask |= (1 << v)
                    in_set[v] = True
                    current_score += weights[v]
                    improved = True
                    if time.time() - start_time > deadline:
                        return S_mask, current_score

            # Try swaps: remove u, add v
            # Iterate u in set
            for u in range(size):
                if not in_set[u]:
                    continue
                
                S_without_u = S_mask & ~(1 << u)
                best_v = -1
                best_gain = 0.0
                
                # We want v not in set, compatible with S_without_u, and weights[v] > weights[u]
                # Iterate v in weight_sorted (descending)
                # If weights[v] <= weights[u], no further v can improve (since sorted desc)
                for v in weight_sorted:
                    if weights[v] <= weights[u]:
                        break
                    if in_set[v]:
                        continue
                    if conflict_mask[v] & S_without_u:
                        continue
                    # Found a valid v with higher weight
                    # Since weight_sorted is descending, the first valid v is the best
                    best_v = v
                    best_gain = weights[v] - weights[u]
                    break

                if best_v >= 0 and best_gain > 0:
                    S_mask = S_without_u | (1 << best_v)
                    in_set[u] = False
                    in_set[best_v] = True
                    current_score += best_gain
                    improved = True
                    if time.time() - start_time > deadline:
                        return S_mask, current_score

        return S_mask, current_score

    # Generate diverse initial solutions
    initial_solutions = [
        greedy_construct(order_asc),
        greedy_construct(order_desc),
        greedy_construct(order_by_conflict),
        greedy_construct(order_by_ratio),
    ]

    # Add randomized starts
    import random
    random.seed(42)
    
    # We can afford more restarts because local search is faster (float, no 2-swap)
    num_random = 50
    for _ in range(num_random):
        shuffled = list(range(size))
        random.shuffle(shuffled)
        initial_solutions.append(greedy_construct(shuffled))

    # Local search on each
    best_mask = initial_solutions[0]
    best_score = score_mask_float(best_mask)

    for S in initial_solutions:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            break
        S_opt, val = local_search(S, timeout - 2)
        if val > best_score:
            best_score = val
            best_mask = S_opt

    # Final check and fallback
    if not is_valid_mask(best_mask) or best_mask == 0:
        fallback = greedy_construct(order_asc)
        fallback, val = local_search(fallback, timeout - 2)
        if val > best_score:
            best_score = val
            best_mask = fallback

    result = [values[i] for i in range(size) if best_mask & (1 << i)]
    return result if result else [2]

# EVOLVE-BLOCK-END
