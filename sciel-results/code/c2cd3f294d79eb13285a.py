import time
import random
import sys
import numpy as np
from itertools import product

Q = 3
N = 7
RADIUS = 1
TOTAL_WORDS = 3 ** N  # 2187


def decode_word(index):
    digits = []
    for _ in range(N):
        digits.append(index % Q)
        index //= Q
    return digits


def encode_word(digits):
    val = 0
    for i in range(N - 1, -1, -1):
        val = val * Q + digits[i]
    return val


# Precompute all words and balls as bitsets
words = [decode_word(i) for i in range(TOTAL_WORDS)]

balls = [0] * TOTAL_WORDS
# Also store neighbor lists for count updates
neighbors = [[] for _ in range(TOTAL_WORDS)]

for i in range(TOTAL_WORDS):
    w = words[i]
    mask = 1 << i
    nbrs = [i]
    for pos in range(N):
        orig_val = w[pos]
        for new_val in range(Q):
            if new_val == orig_val:
                continue
            new_w = w[:]
            new_w[pos] = new_val
            j = encode_word(new_w)
            mask |= (1 << j)
            nbrs.append(j)
    balls[i] = mask
    neighbors[i] = nbrs

FULL_MASK = (1 << TOTAL_WORDS) - 1


def greedy_construct(random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)

    uncovered = FULL_MASK
    code = []

    while uncovered:
        best_idx = -1
        best_count = -1
        best_candidates = []

        for i in range(TOTAL_WORDS):
            cov = (balls[i] & uncovered).bit_count()
            if cov > best_count:
                best_count = cov
                best_idx = i
                best_candidates = [i]
            elif cov == best_count and cov > 0:
                best_candidates.append(i)

        if best_count == 0:
            break

        selected = random.choice(best_candidates)
        code.append(selected)
        uncovered &= ~balls[selected]

    return code


def local_search(code, time_limit=5.0):
    start = time.time()
    code = list(code)
    code_set = set(code)

    # Coverage counts
    counts = [0] * TOTAL_WORDS
    for i in code:
        for idx in neighbors[i]:
            counts[idx] += 1

    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False

        # Phase 1: Remove redundant codewords
        changed = True
        while changed:
            changed = False
            new_code = []
            for i in code:
                redundant = True
                for idx in neighbors[i]:
                    if counts[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    for idx in neighbors[i]:
                        counts[idx] -= 1
                    code_set.discard(i)
                    changed = True
                else:
                    new_code.append(i)
            code = new_code
            if changed:
                improved = True

        if (time.time() - start) >= time_limit:
            break

        # Phase 2: Try swap moves - replace one codeword with another to enable further deletions
        for idx in range(len(code)):
            if (time.time() - start) >= time_limit:
                break

            i = code[idx]

            # Check if i is removable
            removable = True
            for j in neighbors[i]:
                if counts[j] == 1:
                    removable = False
                    break

            if not removable:
                continue

            # Temporarily remove i
            for j in neighbors[i]:
                counts[j] -= 1

            # Find best replacement: maximize new coverage
            complement_i = FULL_MASK ^ balls[i]

            best_j = -1
            best_gain = -1
            for j in range(TOTAL_WORDS):
                if j in code_set:
                    continue
                gain = (balls[j] & complement_i).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_j = j

            if best_j >= 0 and best_gain > 0:
                # Apply swap
                code[idx] = best_j
                code_set.discard(i)
                code_set.add(best_j)
                for j in neighbors[best_j]:
                    counts[j] += 1
                improved = True
            else:
                # Revert
                for j in neighbors[i]:
                    counts[j] += 1

    return code


def perturb(code, rng):
    code = list(code)
    n = len(code)
    k = max(1, n // 10)
    removed = set(rng.sample(code, k))
    remaining = [c for c in code if c not in removed]
    uncovered = FULL_MASK
    for c in remaining:
        uncovered &= ~balls[c]
    while uncovered:
        best_idx = -1
        best_count = -1
        for i in range(TOTAL_WORDS):
            if i in set(remaining):
                continue
            cov = (balls[i] & uncovered).bit_count()
            if cov > best_count:
                best_count = cov
                best_idx = i
        if best_count == 0:
            break
        remaining.append(best_idx)
        uncovered &= ~balls[best_idx]
    return remaining


def iterated_local_search(random_seed=0):
    rng = random.Random(random_seed)
    code = greedy_construct(random_seed)
    code = local_search(code, time_limit=3.0)
    best_code = list(code)
    best_size = len(code)
    while True:
        perturbed = perturb(code, rng)
        improved = local_search(perturbed, time_limit=3.0)
        if len(improved) < best_size:
            best_size = len(improved)
            best_code = list(improved)
        if len(improved) <= best_size or rng.random() < 0.3:
            code = improved
        else:
            code = best_code
    return best_code


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0

    best_code = None
    best_size = float('inf')

    # Initial run
    code = greedy_construct(0)
    code = local_search(code, time_limit=3.0)
    if len(code) < best_size:
        best_size = len(code)
        best_code = code

    # ILS with multiple seeds
    seed = 1
    while time.time() < deadline:
        remaining_time = deadline - time.time()
        if remaining_time < 2.0:
            break
        code = iterated_local_search(seed)
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            if best_size <= 186:
                break
        seed += 1

    # Final local search on best with tighter time budget for more iterations
    if best_code is not None and time.time() < deadline - 1.0:
        remaining = deadline - time.time() - 1.0
        if remaining > 0.5:
            best_code = local_search(best_code, time_limit=remaining)

    return [words[i] for i in best_code]


if __name__ == "__main__":
    res = construct(3, 7, 1)
    print(f"Size: {len(res)}")
    covered = 0
    for i in range(len(res)):
        covered |= balls[encode_word(res[i])]
    if covered == FULL_MASK:
        print("Valid")
    else:
        print("Invalid")
