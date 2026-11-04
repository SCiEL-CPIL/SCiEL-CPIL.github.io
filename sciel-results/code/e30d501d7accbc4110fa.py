"""Improved program for K_3(6,1) covering code problem."""
import itertools
import random
import time
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    if q == 3 and n == 6 and radius == 1:
        return _construct_3_6_1(start_time, deadline)

    # General fallback: direct sum construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]


def _construct_3_6_1(start_time, deadline):
    q, n, radius = 3, 6, 1
    total_words = 3 ** n  # 729
    full_mask = (1 << total_words) - 1

    # Precompute coverage masks
    masks = [0] * total_words
    for i in range(total_words):
        w = [0] * n
        temp = i
        for k in range(n - 1, -1, -1):
            w[k] = temp % 3
            temp //= 3

        mask = 1 << i
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                w[pos] = new_val
                j = 0
                for k in range(n):
                    j = j * 3 + w[k]
                mask |= (1 << j)
            w[pos] = orig
        masks[i] = mask

    def int_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % 3
            idx //= 3
        return w

    # Greedy initialization
    uncovered = full_mask
    codewords = []

    while uncovered and time.time() < deadline - 40:
        best_idx = -1
        best_gain = 0
        for i in range(total_words):
            gain = bin(masks[i] & uncovered).count('1')
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx == -1:
            break
        codewords.append(best_idx)
        uncovered &= ~masks[best_idx]

    # Ensure full coverage
    while uncovered and time.time() < deadline - 20:
        for i in range(total_words):
            if masks[i] & uncovered:
                codewords.append(i)
                uncovered &= ~masks[i]
                break
        else:
            break

    # Local search: minimize size
    current_set = set(codewords)
    current_mask = 0
    for i in current_set:
        current_mask |= masks[i]

    if current_mask != full_mask:
        while current_mask != full_mask and time.time() < deadline - 10:
            for i in range(total_words):
                if masks[i] & ~current_mask:
                    current_set.add(i)
                    current_mask |= masks[i]
                    break
            else:
                break

    best_size = len(current_set)
    best_set = set(current_set)

    # Iterative local search with multiple strategies
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False

        # Phase 1: Removal - try to remove each codeword
        for w in list(current_set):
            if time.time() > deadline - 10:
                break
            new_mask = current_mask & ~masks[w]
            # Check if new_mask still covers everything
            if new_mask == full_mask:
                current_set.remove(w)
                current_mask = new_mask
                improved = True
                if len(current_set) < best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)

        # Phase 2: Swapping - replace one word with another
        if time.time() < deadline - 10:
            for w in list(current_set):
                if time.time() > deadline - 10:
                    break
                base_mask = current_mask & ~masks[w]
                needed = full_mask & ~base_mask
                if needed == 0:
                    continue
                for v in range(total_words):
                    if v in current_set:
                        continue
                    if (masks[v] & needed) == needed:
                        current_set.remove(w)
                        current_set.add(v)
                        current_mask = base_mask | masks[v]
                        improved = True
                        if len(current_set) < best_size:
                            best_size = len(current_set)
                            best_set = set(current_set)
                        break

        # Phase 3: Add-then-remove heuristic
        # Add a random word, then try to remove multiple words
        if time.time() < deadline - 10 and random.random() < 0.3:
            # Add a random word that helps most
            best_add = -1
            best_add_gain = 0
            for _ in range(50):
                v = random.randint(0, total_words - 1)
                if v not in current_set:
                    gain = bin(masks[v] & ~current_mask).count('1')
                    if gain > best_add_gain:
                        best_add_gain = gain
                        best_add = v
            if best_add != -1 and best_add_gain > 0:
                current_set.add(best_add)
                current_mask |= masks[best_add]
                # Now try aggressive removal
                for w in list(current_set):
                    if time.time() > deadline - 10:
                        break
                    new_mask = current_mask & ~masks[w]
                    if new_mask == full_mask:
                        current_set.remove(w)
                        current_mask = new_mask
                        improved = True
                        if len(current_set) < best_size:
                            best_size = len(current_set)
                            best_set = set(current_set)

    # Convert best_set to list of lists
    result = [int_to_word(idx) for idx in best_set]

    # Ensure distinct
    unique_result = []
    seen = set()
    for r in result:
        key = tuple(r)
        if key not in seen:
            seen.add(key)
            unique_result.append(r)

    return unique_result
