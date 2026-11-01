import time
import numpy as np
import random
from collections import defaultdict
import itertools

_TETRA = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]


def _product_code():
    return [h + t for h in _TETRA for t in _TETRA]


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0

    if q == 3 and n == 8 and radius == 1:
        return _ternary_8_1(deadline)

    # Fallbacks (best-effort) for other parameters
    if q == 3 and n >= 4:
        return [h + list(tail) for h in _TETRA
                for tail in itertools.product(range(3), repeat=n - 4)]
    return [list(c) for c in itertools.product(range(q), repeat=n)]


def _ternary_8_1(deadline):
    n = 8
    q = 3
    total = 3 ** n
    ball_size = 1 + 2 * n  # 17

    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int64)

    # Precompute balls: balls[w] = all words within distance 1 of w
    balls = np.empty((total, ball_size), dtype=np.int64)
    balls[:, 0] = np.arange(total)
    for pos in range(n):
        w = weights[pos]
        digs = (np.arange(total) // w) % 3
        balls[:, 1 + 2 * pos] = np.arange(total) - digs * w + w
        balls[:, 2 + 2 * pos] = np.arange(total) - digs * w + 2 * w

    # Product-code seed (valid, 486 words)
    seed_words = _product_code()
    seed = np.array([np.sum(np.array(w) * weights) for w in seed_words], dtype=np.int64)

    # Verify seed is valid
    cc_seed = np.zeros(total, dtype=np.int64)
    for c in seed:
        cc_seed[balls[c]] += 1
    assert (cc_seed >= 1).all(), "Product code seed is not a valid covering"

    best = seed.copy()
    best_size = len(best)

    # Coverage count for current best
    cc = np.zeros(total, dtype=np.int64)
    for c in best:
        cc[balls[c]] += 1

    # Build coverage structure: for each word, which codewords cover it
    # cov[word] = list of indices into best array that cover word
    # This is expensive to maintain dynamically, so we'll use a different approach

    # --- Phase 1: Aggressive removal sweep ---
    changed = True
    while changed and time.time() < deadline - 20.0:
        changed = False
        # Randomize order for better removal
        order = np.random.permutation(len(best))
        for idx in order:
            c = best[idx]
            if (cc[balls[c]] > 1).all():
                cc[balls[c]] -= 1
                best = np.delete(best, idx)
                changed = True
                break  # restart sweep after modification

    best_size = len(best)
    if best_size < 486:
        print(f"After removal: {best_size}")

    # --- Phase 2: Local search with swaps ---
    # For each codeword, find words it uniquely covers
    # Then try to swap to a better position

    def compute_unique_coverages(arr):
        """For each position in arr, return the set of words uniquely covered."""
        n_arr = len(arr)
        uniq = [set() for _ in range(n_arr)]
        # This is O(n_arr * 17) per word, but we need to check all words
        # Alternative: for each word, count how many codewords cover it
        # We already have cc, so words with cc==1 are uniquely covered
        # But we need to know WHICH codeword covers them
        # Let's compute it differently
        for i, c in enumerate(arr):
            for word in balls[c]:
                if cc[word] == 1:
                    uniq[i].add(int(word))
        return uniq

    def try_swap_remove(cc_arr, arr, i, c_new):
        """Try to replace arr[i] with c_new or just remove arr[i]."""
        b_old = balls[arr[i]]
        # Check if removing arr[i] is safe
        if (cc_arr[b_old] > 1).all():
            cc_arr[b_old] -= 1
            arr = np.delete(arr, i)
            return arr, True
        # Check if swapping is possible
        if c_new is not None:
            b_new = balls[c_new]
            # The words only covered by arr[i] must also be covered by c_new
            only_old = b_old[cc_arr[b_old] == 1]
            if len(only_old) == 0 or (cc_arr[b_new][only_old] >= 1).all():
                cc_arr[b_old] -= 1
                cc_arr[b_new] += 1
                arr = np.delete(arr, i)
                arr = np.append(arr, c_new)
                return arr, True
        return arr, False

    # Main local search loop
    restart = 0
    while time.time() < deadline - 5.0:
        restart += 1

        # Occasionally do a fresh restart from the product code
        if restart % 10 == 0:
            arr = seed.copy()
            cc_ls = np.zeros(total, dtype=np.int64)
            for c in arr:
                cc_ls[balls[c]] += 1
        else:
            arr = best.copy()
            cc_ls = np.zeros(total, dtype=np.int64)
            for c in arr:
                cc_ls[balls[c]] += 1

        # Iterative improvement
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False

            # Try removals
            order = np.random.permutation(len(arr))
            for i in order:
                if time.time() > deadline - 5.0:
                    break
                b_i = balls[arr[i]]
                if (cc_ls[b_i] > 1).all():
                    cc_ls[b_i] -= 1
                    arr = np.delete(arr, i)
                    improved = True

            # Try swaps: for each codeword, find a better alternative
            if time.time() > deadline - 5.0:
                break
            n_arr = len(arr)
            for i in range(n_arr):
                if time.time() > deadline - 5.0:
                    break
                c_i = arr[i]
                b_i = balls[c_i]
                # Words uniquely covered by c_i
                uniq_words = b_i[cc_ls[b_i] == 1]
                if len(uniq_words) == 0:
                    # Can just remove
                    cc_ls[b_i] -= 1
                    arr = np.delete(arr, i)
                    improved = True
                    continue

                # Find a candidate that covers all unique words
                # Sample candidates
                n_candidates = 50
                found_better = False
                for _ in range(n_candidates):
                    # Generate a random candidate near c_i
                    # Mutate c_i in 1-2 positions
                    c_new_digits = (c_i // weights) % 3
                    c_new_digits = c_new_digits.copy()
                    # Randomly change 1 or 2 digits
                    n_changes = random.choice([1, 1, 1, 2])
                    pos_changes = random.sample(range(n), n_changes)
                    for pos in pos_changes:
                        c_new_digits[pos] = (c_new_digits[pos] + random.choice([1, 2])) % 3
                    c_new = int(np.sum(c_new_digits * weights))

                    if c_new == c_i:
                        continue

                    # Check if c_new covers all unique words
                    b_new = balls[c_new]
                    if (cc_ls[b_new][uniq_words] >= 1).all():
                        # Check if this swap is beneficial
                        # Count how many words c_new covers that are also covered by c_i
                        overlap = np.intersect1d(b_i, b_new)
                        # Words c_new covers that c_i doesn't
                        new_words = np.setdiff1d(b_new, b_i)
                        # Words c_i covers that c_new doesn't
                        old_words = np.setdiff1d(b_i, b_new)

                        # After swap, words in old_words lose coverage of c_i
                        # If any of those words had cc==1, the swap is bad
                        if (cc_ls[old_words] > 1).all():
                            cc_ls[b_i] -= 1
                            cc_ls[b_new] += 1
                            arr[i] = c_new
                            improved = True
                            found_better = True
                            break

                if found_better:
                    break

        # Update best if improved
        if len(arr) < best_size:
            best_size = len(arr)
            best = arr.copy()
            print(f"New best: {best_size}")

        # Also try a targeted approach: find all uniquely covered words,
        # and for each, try to find a codeword that covers it and has maximum overlap
        if time.time() < deadline - 5.0:
            cc_t = np.zeros(total, dtype=np.int64)
            for c in best:
                cc_t[balls[c]] += 1

            for i in range(len(best)):
                if time.time() > deadline - 5.0:
                    break
                c_i = best[i]
                b_i = balls[c_i]
                uniq = b_i[cc_t[b_i] == 1]
                if len(uniq) == 0:
                    cc_t[b_i] -= 1
                    best = np.delete(best, i)
                    best_size = len(best)
                    continue

                # For the first unique word, find all codewords that cover it
                target_word = uniq[0]
                # Find codewords that cover target_word
                # A codeword c covers target_word if dist(c, target_word) <= 1
                # This means c is in the ball of target_word
                candidates = balls[target_word]
                for c_new in candidates:
                    if int(c_new) == int(c_i):
                        continue
                    b_new = balls[c_new]
                    if (cc_t[b_new][uniq] >= 1).all():
                        # Check if removing c_i's unique coverage is safe
                        old_only = b_i[cc_t[b_i] == 1]
                        if (cc_t[b_new][old_only] >= 1).all():
                            cc_t[b_i] -= 1
                            cc_t[b_new] += 1
                            best[i] = c_new
                            break

    # Final cleanup
    cc_final = np.zeros(total, dtype=np.int64)
    for c in best:
        cc_final[balls[c]] += 1

    changed = True
    while changed and time.time() < deadline - 1.0:
        changed = False
        for i in range(len(best) - 1, -1, -1):
            if (cc_final[balls[best[i]]] > 1).all():
                cc_final[balls[best[i]]] -= 1
                best = np.delete(best, i)
                changed = True

    # Verify validity
    assert (cc_final >= 1).all(), "Final code is not a valid covering"

    result = []
    for c in best:
        digs = (c // weights) % 3
        result.append([int(x) for x in digs])
    return result
