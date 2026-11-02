import time
import sys
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    total_space = q ** n

    # Precompute neighbor masks using numpy for speed
    # neighbors[i] is a set of indices within Hamming distance 'radius' of i
    # For q=3, n=8, radius=1: neighborhood size = 1 + 8*2 = 17

    # We'll use a list of sets or bitmasks
    # Using Python ints as bitmasks since total_space = 6561, needs ~6561 bits
    # Python ints handle arbitrary size

    def int_to_digits(num):
        digits = []
        for _ in range(n):
            digits.append(num % q)
            num //= q
        return digits

    # Precompute powers of q
    q_powers = [1]
    for i in range(1, n):
        q_powers.append(q_powers[-1] * q)

    # Precompute neighbor masks
    neighbor_masks = [0] * total_space
    for i in range(total_space):
        digits = []
        temp = i
        for _ in range(n):
            digits.append(temp % q)
            temp //= q

        mask = 1 << i
        if radius >= 1:
            for pos in range(n):
                orig_val = digits[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    delta = (val - orig_val) * q_powers[pos]
                    new_index = i + delta
                    if 0 <= new_index < total_space:
                        mask |= (1 << new_index)

        neighbor_masks[i] = mask

    # Start with a greedy set cover
    uncovered = (1 << total_space) - 1
    code_indices = []

    while uncovered != 0:
        best_word = -1
        max_cover = 0

        for w in range(total_space):
            cover_count = (uncovered & neighbor_masks[w]).bit_count()

            if cover_count > max_cover:
                max_cover = cover_count
                best_word = w

        if best_word == -1:
            break

        code_indices.append(best_word)
        uncovered &= ~neighbor_masks[best_word]

        if time.time() > deadline - 5:
            break

    # Local search: iteratively remove redundant codewords
    # Multiple passes to ensure convergence
    for _pass in range(10):
        if time.time() > deadline - 5:
            break

        if not code_indices:
            break

        # Build coverage count
        coverage_count = [0] * total_space
        for idx in code_indices:
            mask = neighbor_masks[idx]
            m = mask
            while m:
                lb = m & (-m)
                pos = lb.bit_length() - 1
                coverage_count[pos] += 1
                m ^= lb

        # Try to remove each codeword
        removed = []
        for i, idx in enumerate(code_indices):
            if time.time() > deadline - 5:
                break
            mask = neighbor_masks[idx]
            m = mask
            redundant = True
            while m:
                lb = m & (-m)
                pos = lb.bit_length() - 1
                if coverage_count[pos] <= 1:
                    redundant = False
                    break
                m ^= lb

            if redundant:
                removed.append(i)
                m = mask
                while m:
                    lb = m & (-m)
                    pos = lb.bit_length() - 1
                    coverage_count[pos] -= 1
                    m ^= lb

        if removed:
            remove_set = set(removed)
            code_indices = [idx for i, idx in enumerate(code_indices) if i not in remove_set]
        else:
            break  # No more removals possible

    # Second phase: try swapping - replace a codeword with another that covers more uncovered
    # But first, let's try a more aggressive local search: 
    # For each codeword, try replacing it with a different codeword that still maintains coverage
    # and potentially allows further removals

    # Actually, let's try a different approach: 
    # After greedy + removal, try to add back coverage more efficiently
    # by checking if we can replace pairs of codewords with one

    # Let's do a final verification and cleanup
    # Verify coverage
    covered = 0
    for idx in code_indices:
        covered |= neighbor_masks[idx]

    if covered != (1 << total_space) - 1:
        # Coverage lost, need to fix
        uncovered = ((1 << total_space) - 1) & ~covered
        while uncovered != 0 and time.time() < deadline - 2:
            best_word = -1
            max_cover = 0
            for w in range(total_space):
                cover_count = (uncovered & neighbor_masks[w]).bit_count()
                if cover_count > max_cover:
                    max_cover = cover_count
                    best_word = w
            if best_word == -1:
                break
            code_indices.append(best_word)
            uncovered &= ~neighbor_masks[best_word]

    # Convert indices to lists of integers
    result = []
    for idx in code_indices:
        digits = []
        temp = idx
        for _ in range(n):
            digits.append(temp % q)
            temp //= q
        result.append(digits)

    return result
