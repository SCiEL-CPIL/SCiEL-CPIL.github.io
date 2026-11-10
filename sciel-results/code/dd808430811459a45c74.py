import time
from math import gcd

def construct():
    start = time.time()
    DEADLINE = start + 85.0

    LIMIT = 512
    N = 512

    # Precompute compatibility bitsets
    compat = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            l = a * b // gcd(a, b)
            if l <= LIMIT:
                compat[a] |= (1 << b)
                if a != b:
                    compat[b] |= (1 << a)

    def can_add(x, current_bits):
        return (current_bits & compat[x]) == current_bits

    def bits_to_list(bits):
        result = []
        for a in range(1, N + 1):
            if (bits >> a) & 1:
                result.append(a)
        return result

    def set_to_bits(s):
        b = 0
        for a in s:
            b |= (1 << a)
        return b

    # Find max contiguous prefix [1..k]
    k = 1
    while True:
        nk = k + 1
        ok = True
        for a in range(1, k + 1):
            if nk * a // gcd(nk, a) > LIMIT:
                ok = False
                break
        if not ok:
            break
        k = nk
        if time.time() > DEADLINE:
            break

    # Compute degree for ordering
    degree = [0] * (N + 1)
    for a in range(1, N + 1):
        degree[a] = bin(compat[a]).count('1')

    def lpf(n):
        if n <= 1:
            return 1
        f = 2
        m = n
        while f * f <= m:
            while m % f == 0:
                m //= f
            f += 1
        return m if m > 1 else f

    # Build multiple orderings
    orderings = [
        sorted(range(1, N + 1), key=lambda x: degree[x]),
        sorted(range(1, N + 1), key=lambda x: -degree[x]),
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: (lpf(x), x)),
        sorted(range(1, N + 1), key=lambda x: (-degree[x], x)),
        sorted(range(1, N + 1), key=lambda x: (degree[x], -x)),
        # Prefix-first orderings: put [1..k] first, then rest by degree
        list(range(1, k + 1)) + sorted(range(k + 1, N + 1), key=lambda x: degree[x]),
        list(range(1, k + 1)) + sorted(range(k + 1, N + 1), key=lambda x: -degree[x]),
        list(range(1, k + 1)) + sorted(range(k + 1, N + 1), key=lambda x: (lpf(x), x)),
    ]

    # Multi-start greedy
    best_count = 0
    best_set_bits = 0

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
            best_set_bits = bits

    # Local search with multiple move types
    current_bits = best_set_bits
    current_list = bits_to_list(current_bits)
    improved = True
    max_rounds = 100
    round_num = 0

    while improved and time.time() < DEADLINE and round_num < max_rounds:
        improved = False
        round_num += 1
        current_list = bits_to_list(current_bits)
        n_cur = len(current_list)

        # Move type 1: 1-for-2 (remove one, add two)
        for i, remove in enumerate(current_list):
            if time.time() > DEADLINE:
                break
            remaining_bits = current_bits & ~(1 << remove)

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
                            new_bits = remaining_bits | (1 << c1) | (1 << c2)
                            if bin(new_bits).count('1') > bin(current_bits).count('1'):
                                current_bits = new_bits
                                best_set_bits = new_bits
                                best_count = bin(new_bits).count('1')
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
                    remaining_bits = current_bits & ~(1 << r1) & ~(1 << r2)

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
                                        new_bits = b2 | (1 << c3)
                                        if bin(new_bits).count('1') > bin(current_bits).count('1'):
                                            current_bits = new_bits
                                            best_set_bits = new_bits
                                            best_count = bin(new_bits).count('1')
                                            improved = True
                                            found = True
                                            break

        # Move type 3: 1-for-1 swap + subsequent greedy additions
        if time.time() < DEADLINE and round_num % 3 == 0:
            for i, elem in enumerate(current_list):
                if time.time() > DEADLINE:
                    break
                remaining_bits = current_bits & ~(1 << elem)

                for x in range(1, N + 1):
                    if time.time() > DEADLINE:
                        break
                    if (current_bits >> x) & 1:
                        continue
                    if can_add(x, remaining_bits):
                        new_bits = remaining_bits | (1 << x)
                        # Greedily add more elements
                        for y in range(1, N + 1):
                            if time.time() > DEADLINE:
                                break
                            if (new_bits >> y) & 1:
                                continue
                            if can_add(y, new_bits):
                                new_bits |= (1 << y)
                        if bin(new_bits).count('1') > bin(current_bits).count('1'):
                            current_bits = new_bits
                            best_set_bits = new_bits
                            best_count = bin(new_bits).count('1')
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
                best_count += 1
        best_set_bits = current_bits

    # Convert to sorted list
    result = []
    for a in range(1, N + 1):
        if (best_set_bits >> a) & 1:
            result.append(a)

    if not result:
        result = [1]

    return result
