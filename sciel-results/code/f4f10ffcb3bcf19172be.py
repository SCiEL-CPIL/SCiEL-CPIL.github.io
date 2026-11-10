import time
from math import gcd
import numpy as np

def construct():
    start = time.time()
    DEADLINE = start + 85.0

    LIMIT = 512
    N = 512

    # Precompute compatibility as numpy boolean matrix
    # compat[i][j] = True iff lcm(i,j) <= LIMIT
    compat = np.zeros((N + 1, N + 1), dtype=bool)
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            l = a * b // gcd(a, b)
            if l <= LIMIT:
                compat[a, b] = True
                if a != b:
                    compat[b, a] = True

    # Precompute for each element, the bitmask of compatible elements
    # Use Python ints for bit operations
    compat_bits = [0] * (N + 1)
    for a in range(1, N + 1):
        bits = 0
        for b in range(1, N + 1):
            if compat[a, b]:
                bits |= (1 << b)
        compat_bits[a] = bits

    def can_add(x, current_bits):
        """Check if x can be added to the set represented by current_bits."""
        return (current_bits & compat_bits[x]) == current_bits

    def bits_to_list(bits):
        """Convert bits to sorted list."""
        return [a for a in range(1, N + 1) if (bits >> a) & 1]

    def set_to_bits(s):
        b = 0
        for a in s:
            b |= (1 << a)
        return b

    # Strategy 1: Greedy with multiple orderings
    # Compute degree (number of compatible elements)
    degree = [0] * (N + 1)
    for a in range(1, N + 1):
        degree[a] = int(np.sum(compat[a, 1:N + 1]))

    # Multiple ordering strategies
    orderings = [
        sorted(range(1, N + 1), key=lambda x: degree[x]),
        sorted(range(1, N + 1), key=lambda x: -degree[x]),
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: (x.bit_length(), x)),
        sorted(range(1, N + 1), key=lambda x: (-x.bit_length(), x)),
        sorted(range(1, N + 1), key=lambda x: (degree[x], -x)),
        sorted(range(1, N + 1), key=lambda x: (-degree[x], x)),
    ]

    best_count = 0
    best_bits = 0

    for order in orderings:
        if time.time() > DEADLINE:
            break
        bits = 0
        count = 0
        for x in order:
            if time.time() > DEADLINE:
                break
            if (bits >> x) & 1:
                continue
            if can_add(x, bits):
                bits |= (1 << x)
                count += 1
        if count > best_count:
            best_count = count
            best_bits = bits

    # Strategy 2: Numpy-vectorized iterative removal
    # Start with all elements, repeatedly remove the element with most conflicts
    if time.time() < DEADLINE:
        active = set(range(1, N + 1))
        active_bits = 0
        for a in active:
            active_bits |= (1 << a)

        # Iteratively remove the element that conflicts with the most other active elements
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            if len(active) == 0:
                break

            # Find the element with the most conflicts among active elements
            # For each active element, count how many other active elements it's incompatible with
            best_remove = None
            max_conflicts = 0
            for a in active:
                # Count active elements incompatible with a
                conflicts = 0
                for b in active:
                    if a != b and not compat[a, b]:
                        conflicts += 1
                if conflicts > max_conflicts:
                    max_conflicts = conflicts
                    best_remove = a
                elif conflicts == max_conflicts and best_remove is not None and a < best_remove:
                    best_remove = a

            if best_remove is not None and max_conflicts > 0:
                active.discard(best_remove)
                active_bits &= ~(1 << best_remove)
                improved = True

        if len(active) > best_count:
            best_count = len(active)
            best_bits = active_bits

    # Strategy 3: Local search on the best solution found so far
    current_set = set(bits_to_list(best_bits))
    current_bits = best_bits
    improved = True
    max_rounds = 50
    round_num = 0

    while improved and time.time() < DEADLINE and round_num < max_rounds:
        improved = False
        round_num += 1
        current_list = sorted(current_set)
        n_cur = len(current_list)

        # Move type 1: 1-for-2 (remove one, add two)
        for remove in current_list:
            if time.time() > DEADLINE:
                break
            remaining = set(current_list)
            remaining.discard(remove)
            remaining_bits = set_to_bits(remaining)

            candidates = []
            for x in range(1, N + 1):
                if (current_bits >> x) & 1:
                    continue
                if x == remove:
                    continue
                if can_add(x, remaining_bits):
                    candidates.append(x)

            if len(candidates) >= 2:
                found = False
                for ci in range(len(candidates)):
                    if found or time.time() > DEADLINE:
                        break
                    c1 = candidates[ci]
                    bits_with_c1 = remaining_bits | (1 << c1)
                    for cj in range(ci + 1, len(candidates)):
                        if time.time() > DEADLINE:
                            break
                        c2 = candidates[cj]
                        if can_add(c2, bits_with_c1):
                            new_set = remaining | {c1, c2}
                            if len(new_set) > len(current_set):
                                current_set = new_set
                                current_bits = set_to_bits(current_set)
                                best_count = len(current_set)
                                best_bits = current_bits
                                improved = True
                                found = True
                                break

        # Move type 2: 2-for-3 (remove two, add three)
        if time.time() < DEADLINE and n_cur >= 2:
            for i in range(n_cur):
                if time.time() > DEADLINE:
                    break
                for j in range(i + 1, n_cur):
                    if time.time() > DEADLINE:
                        break
                    r1, r2 = current_list[i], current_list[j]
                    remaining = set(current_list)
                    remaining.discard(r1)
                    remaining.discard(r2)
                    remaining_bits = set_to_bits(remaining)

                    candidates = []
                    for x in range(1, N + 1):
                        if (current_bits >> x) & 1:
                            continue
                        if x == r1 or x == r2:
                            continue
                        if can_add(x, remaining_bits):
                            candidates.append(x)

                    if len(candidates) >= 3:
                        found = False
                        for ci in range(len(candidates)):
                            if found or time.time() > DEADLINE:
                                break
                            c1 = candidates[ci]
                            b1 = remaining_bits | (1 << c1)
                            for cj in range(ci + 1, len(candidates)):
                                if found or time.time() > DEADLINE:
                                    break
                                c2 = candidates[cj]
                                if not can_add(c2, b1):
                                    continue
                                b2 = b1 | (1 << c2)
                                for ck in range(cj + 1, len(candidates)):
                                    if time.time() > DEADLINE:
                                        break
                                    c3 = candidates[ck]
                                    if can_add(c3, b2):
                                        new_set = remaining | {c1, c2, c3}
                                        if len(new_set) > len(current_set):
                                            current_set = new_set
                                            current_bits = set_to_bits(current_set)
                                            best_count = len(current_set)
                                            best_bits = current_bits
                                            improved = True
                                            found = True
                                            break

        # Move type 3: 1-for-1 swap + greedy fill
        if time.time() < DEADLINE:
            for elem in current_list:
                if time.time() > DEADLINE:
                    break
                remaining = set(current_list)
                remaining.discard(elem)
                remaining_bits = set_to_bits(remaining)

                for x in range(1, N + 1):
                    if time.time() > DEADLINE:
                        break
                    if (current_bits >> x) & 1:
                        continue
                    if can_add(x, remaining_bits):
                        new_bits = remaining_bits | (1 << x)
                        # Greedily add more
                        for y in range(1, N + 1):
                            if time.time() > DEADLINE:
                                break
                            if (new_bits >> y) & 1:
                                continue
                            if can_add(y, new_bits):
                                new_bits |= (1 << y)
                        new_list = bits_to_list(new_bits)
                        if len(new_list) > len(current_set):
                            current_set = set(new_list)
                            current_bits = new_bits
                            best_count = len(current_set)
                            best_bits = new_bits
                            improved = True
                        break

    # Final: try adding any remaining elements
    if time.time() < DEADLINE:
        for x in range(1, N + 1):
            if time.time() > DEADLINE:
                break
            if (current_bits >> x) & 1:
                continue
            if can_add(x, current_bits):
                current_bits |= (1 << x)
                current_set.add(x)
                best_count += 1
        best_bits = current_bits

    # Convert to sorted list
    result = bits_to_list(best_bits)

    if not result:
        result = [1]

    return result
