"""Combined covering code solver for K_3(6,1).

Strategy:
  * Start from the strong tetracode-product construction (81 words).
  * Perform a fast pruning pass (remove redundant codewords).
  * Run several randomized greedy set-cover restarts using bitset coverage.
  * Prune each greedy result and keep the best code found.
  * Use an explicit wall-clock deadline well before the 150 s limit.
"""
import time
import itertools
import random


def construct(q, n, radius):
    start_time = time.time()
    # Safety margin: return before the 150 s hard kill.
    deadline = start_time + 135.0

    # Fallback / general case: tetracode product for q=3, n>=4, radius=1.
    if not (q == 3 and n == 6 and radius == 1):
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3]
                         for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode
                    for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(c) for c in itertools.product(range(q), repeat=n)]

    # Specific solver for K_3(6, 1)
    N = q ** n  # 729
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Precompute spheres as Python ints (bitsets).
    # spheres[i] has bit j set iff Hamming distance <= radius between all_words[i] and all_words[j].
    spheres = [0] * N
    for i, w in enumerate(all_words):
        mask = 1 << i
        for pos in range(n):
            old_val = w[pos]
            for new_val in range(q):
                if new_val == old_val:
                    continue
                lst = list(w)
                lst[pos] = new_val
                j = word_to_idx[tuple(lst)]
                mask |= (1 << j)
        spheres[i] = mask

    full_mask = (1 << N) - 1

    def prune(code_indices):
        """Remove redundant codewords from code_indices (list of word indices)."""
        current = list(code_indices)
        changed = True
        while changed:
            changed = False
            for i in range(len(current)):
                union_others = 0
                for j in range(len(current)):
                    if i == j:
                        continue
                    union_others |= spheres[current[j]]
                    if union_others == full_mask:
                        break
                if union_others == full_mask:
                    current.pop(i)
                    changed = True
                    break
        return current

    best_code = None
    best_size = N

    # 1) Start from tetracode product and prune.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3]
                 for a in range(3) for b in range(3)]
    init_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            w = tuple(h + list(tail))
            init_code.append(word_to_idx[w])

    pruned_init = prune(init_code)
    if len(pruned_init) < best_size:
        best_size = len(pruned_init)
        best_code = [all_words[i] for i in pruned_init]

    # 2) Randomized greedy restarts with pruning.
    # Each restart: shuffle candidate order, run greedy, prune, keep best.
    num_restarts = 2000  # cap; will be limited by deadline
    for r in range(num_restarts):
        if time.time() > deadline:
            break

        rng = random.Random(r)
        candidates = list(range(N))
        rng.shuffle(candidates)

        covered = 0
        chosen = []

        while covered != full_mask:
            best_gain = -1
            best_idx = -1
            for idx in candidates:
                new_covered = spheres[idx] & ~covered
                gain = new_covered.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            if best_gain == 0:
                break
            chosen.append(best_idx)
            covered |= spheres[best_idx]

        pruned_greedy = prune(chosen)
        if len(pruned_greedy) < best_size:
            best_size = len(pruned_greedy)
            best_code = [all_words[i] for i in pruned_greedy]

        # Early exit if we hit the known optimum of 73
        if best_size <= 73:
            break

    if best_code is None:
        # Fallback: tetracode product
        return [list(h) + list(tail) for h in tetracode
                for tail in itertools.product(range(3), repeat=n - 4)]

    return [list(c) for c in best_code]
