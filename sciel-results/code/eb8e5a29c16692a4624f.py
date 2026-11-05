"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    n = 128
    target_low = n
    target_high = 2 * n

    # We need A subset of [0, n] such that every t in [n, 2n] is a sum a+b with a,b in A, a<=b.
    # Minimize max over t of (# of unordered pairs {a,b} in A x A with a<=b and a+b=t).

    def count_reps(A_set):
        """Count representations for each t in [n, 2n]."""
        A = sorted(A_set)
        counts = np.zeros(2 * n + 1, dtype=np.int32)
        for i, a in enumerate(A):
            for b in A[i:]:  # b >= a to ensure unordered
                s = a + b
                if target_low <= s <= target_high:
                    counts[s] += 1
        return counts

    def peak_mult(A_set):
        counts = count_reps(A_set)
        return int(np.max(counts[target_low:target_high + 1]))

    def is_valid(A_set):
        counts = count_reps(A_set)
        return np.all(counts[target_low:target_high + 1] > 0)

    # Initial candidate: the seed program's answer
    A_seed = set(range(64, 129))
    best_A = A_seed
    best_peak = peak_mult(A_seed)

    # Try some structured candidates
    # 1. Full interval [k, n] for various k
    for k in range(0, n + 1):
        if time.time() > deadline:
            break
        A_cand = set(range(k, n + 1))
        if is_valid(A_cand):
            p = peak_mult(A_cand)
            if p < best_peak:
                best_peak = p
                best_A = A_cand

    # 2. Symmetric around n/2: {0..m} union {n-m..n}
    for m in range(0, n // 2 + 1):
        if time.time() > deadline:
            break
        A_cand = set(range(0, m + 1)) | set(range(n - m, n + 1))
        if is_valid(A_cand):
            p = peak_mult(A_cand)
            if p < best_peak:
                best_peak = p
                best_A = A_cand

    # 3. Random search with local improvement
    def random_subset():
        # Random subset of [0, n]
        size = random.randint(n // 2, n)
        return set(random.sample(range(n + 1), size))

    def local_search(A_set, max_iters=2000):
        """Try adding/removing elements to reduce peak multiplicity."""
        A = set(A_set)
        current_peak = peak_mult(A)
        if not is_valid(A):
            return A, current_peak

        for _ in range(max_iters):
            if time.time() > deadline:
                break

            improved = False

            # Try removing an element
            for a in list(A):
                if time.time() > deadline:
                    break
                A_new = A - {a}
                if len(A_new) == 0:
                    continue
                if is_valid(A_new):
                    p = peak_mult(A_new)
                    if p < current_peak:
                        A = A_new
                        current_peak = p
                        improved = True
                        break

            if improved:
                continue

            # Try adding an element
            for a in range(n + 1):
                if time.time() > deadline:
                    break
                if a in A:
                    continue
                A_new = A | {a}
                p = peak_mult(A_new)
                if p < current_peak:
                    A = A_new
                    current_peak = p
                    improved = True
                    break

            if not improved:
                break

        return A, current_peak

    # Random restarts
    num_restarts = 0
    while time.time() < deadline - 5.0:
        num_restarts += 1
        A_rand = random_subset()
        if not is_valid(A_rand):
            # Try to repair: add elements until valid
            A_repair = set(A_rand)
            for _ in range(50):
                if time.time() > deadline - 5.0:
                    break
                if is_valid(A_repair):
                    break
                # Find which targets are uncovered
                counts = count_reps(A_repair)
                uncovered = [t for t in range(target_low, target_high + 1) if counts[t] == 0]
                if not uncovered:
                    break
                t = random.choice(uncovered)
                # Find pairs that sum to t and add one element
                found = False
                for a in range(0, t + 1):
                    b = t - a
                    if b > n:
                        continue
                    if a <= b:
                        if a not in A_repair and b not in A_repair:
                            # Add both
                            A_repair.add(a)
                            A_repair.add(b)
                            found = True
                            break
                        elif a not in A_repair:
                            A_repair.add(a)
                            found = True
                            break
                        elif b not in A_repair:
                            A_repair.add(b)
                            found = True
                            break
                if not found:
                    break

        if is_valid(A_rand):
            A_ls, p_ls = local_search(A_rand)
            if p_ls < best_peak:
                best_peak = p_ls
                best_A = A_ls

        # Also try local search on the current best
        if num_restarts % 5 == 0:
            A_ls, p_ls = local_search(best_A)
            if p_ls < best_peak:
                best_peak = p_ls
                best_A = A_ls

    # Final: try to polish the best solution
    A_final, p_final = local_search(best_A, max_iters=5000)
    if p_final < best_peak:
        best_peak = p_final
        best_A = A_final

    return sorted(best_A)

# EVOLVE-BLOCK-END
