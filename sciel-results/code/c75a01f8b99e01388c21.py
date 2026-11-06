import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 80.0  # safety margin from 90s limit

    mask = (1 << 201) - 1
    TARGET = 200

    def can_add(b, x):
        return ((b | (b << x)) >> TARGET) & 1 == 0

    def recompute_bits(elements):
        b = 1
        for e in elements:
            b |= (b << e)
            b &= mask
        return b

    def build_prefix_suffix(elements):
        """Build prefix and suffix bitset arrays for fast removal."""
        n = len(elements)
        if n == 0:
            return [], []
        prefix = [0] * n
        suffix = [0] * n
        b = 1
        for i in range(n):
            b |= (b << elements[i])
            b &= mask
            prefix[i] = b
        b = 1
        for i in range(n - 1, -1, -1):
            b |= (b << elements[i])
            b &= mask
            suffix[i] = b
        return prefix, suffix

    def bits_without_idx(sorted_elements, idx):
        """O(1) removal using precomputed prefix/suffix."""
        n = len(sorted_elements)
        if n == 1:
            return 1
        # We need to combine prefix[idx-1] and suffix[idx+1]
        # But we need to "merge" two bitsets: the set of sums from A and B
        # is not simply OR; we need to compute all sums a+b where a in A, b in B
        # Actually for subset sums, if we have prefix sums P (from elements before idx)
        # and suffix sums S (from elements after idx), the combined set is:
        # all sums p + s where p in P and s in S
        # This is a convolution-like operation on bitsets
        # For bitsets, this is: result = P; for each set bit in S, result |= P << bit
        # But that's O(n) per removal. Let's just recompute for now but smarter.
        
        # Actually, let's just do a linear scan which is O(n) but with small constant
        if idx == 0:
            b = 1
            for i in range(1, n):
                b |= (b << sorted_elements[i])
                b &= mask
            return b
        elif idx == n - 1:
            b = 1
            for i in range(n - 1):
                b |= (b << sorted_elements[i])
                b &= mask
            return b
        else:
            b = 1
            for i in range(idx):
                b |= (b << sorted_elements[i])
                b &= mask
            for i in range(idx + 1, n):
                b |= (b << sorted_elements[i])
                b &= mask
            return b

    def bits_without_two(sorted_elements, idx1, idx2):
        """Remove two elements from sorted list."""
        n = len(sorted_elements)
        b = 1
        for i in range(n):
            if i != idx1 and i != idx2:
                b |= (b << sorted_elements[i])
                b &= mask
        return b

    # ---- Phase 1: Greedy initialization with multiple orders ----
    best_set = set()
    best_bits = 1
    best_size = 0

    orders = [
        list(range(100, 0, -1)),
        list(range(1, 101)),
        list(range(50, 0, -1)) + list(range(100, 50, -1)),
        list(range(100, 49, -1)) + list(range(50, 0, -1)),
        list(range(1, 51)) + list(range(100, 49, -1)),
    ]
    for _ in range(5):
        rand_order = list(range(1, 101))
        random.shuffle(rand_order)
        orders.append(rand_order)

    for order in orders:
        s = set()
        b = 1
        for x in order:
            if can_add(b, x):
                s.add(x)
                b = (b | (b << x)) & mask
        if len(s) > best_size:
            best_set = s
            best_bits = b
            best_size = len(s)

    # ---- Phase 2: Local search ----
    s = set(best_set)
    b = best_bits
    size = best_size

    global_best_set = set(s)
    global_best_bits = b
    global_best_size = size

    def try_multi_add(s, b, size):
        ns = set(s)
        nb = b
        nsize = size
        changed = True
        while changed:
            changed = False
            for x in range(1, 101):
                if x not in ns and can_add(nb, x):
                    ns.add(x)
                    nb = (nb | (nb << x)) & mask
                    nsize += 1
                    changed = True
        if nsize > size:
            return ns, nb, nsize
        return None

    def try_swap_add(s, b, size):
        """Try 1-remove + 1-add + 1-add. Net gain +1."""
        elements = sorted(s)
        candidates = [x for x in range(1, 101) if x not in s]
        random.shuffle(candidates)
        candidates = candidates[:30]
        
        for idx, y in enumerate(elements):
            if time.time() - start_time > DEADLINE:
                return None
            b_minus = bits_without_idx(elements, idx)
            s_minus = s - {y}

            for x in candidates:
                if can_add(b_minus, x):
                    nb = (b_minus | (b_minus << x)) & mask
                    ns = s_minus | {x}
                    for z in range(1, 101):
                        if z not in ns and can_add(nb, z):
                            return ns | {z}, (nb | (nb << z)) & mask, size + 1
        return None

    def try_double_swap(s, b, size):
        """Try 2-remove + 2-add + 1-add. Net gain +1."""
        elements = sorted(s)
        candidates = [x for x in range(1, 101) if x not in s]
        random.shuffle(candidates)
        candidates = candidates[:20]
        
        n = len(elements)
        for i in range(n):
            for j in range(i + 1, n):
                if time.time() - start_time > DEADLINE:
                    return None
                y1, y2 = elements[i], elements[j]
                b_minus = bits_without_two(elements, i, j)
                s_minus = s - {y1, y2}

                for x in candidates:
                    if can_add(b_minus, x):
                        b1 = (b_minus | (b_minus << x)) & mask
                        s1 = s_minus | {x}
                        for x2 in candidates:
                            if x2 == x:
                                continue
                            if can_add(b1, x2):
                                b2 = (b1 | (b1 << x2)) & mask
                                s2 = s1 | {x2}
                                for z in range(1, 101):
                                    if z not in s2 and can_add(b2, z):
                                        return s2 | {z}, (b2 | (b2 << z)) & mask, size + 1
        return None

    # Main search loop
    improved = True
    stagnation = 0
    max_stagnation = 20

    while improved and (time.time() - start_time < DEADLINE):
        improved = False

        result = try_multi_add(s, b, size)
        if result is not None:
            s, b, size = result
            improved = True
            stagnation = 0
            if size > global_best_size:
                global_best_set = set(s)
                global_best_bits = b
                global_best_size = size
            continue

        result = try_swap_add(s, b, size)
        if result is not None:
            s, b, size = result
            improved = True
            stagnation = 0
            if size > global_best_size:
                global_best_set = set(s)
                global_best_bits = b
                global_best_size = size
            continue

        result = try_double_swap(s, b, size)
        if result is not None:
            s, b, size = result
            improved = True
            stagnation = 0
            if size > global_best_size:
                global_best_set = set(s)
                global_best_bits = b
                global_best_size = size
            continue

        stagnation += 1
        if stagnation >= max_stagnation:
            break

        if time.time() - start_time > DEADLINE:
            break

        elements = list(s)
        if len(elements) > 3:
            if stagnation < 5:
                n_remove = random.choice([2, 3])
            else:
                n_remove = random.choice([3, 4, 5, 6])
            n_remove = min(n_remove, len(elements) - 1)
            to_remove = random.sample(elements, n_remove)
            s_pert = s - set(to_remove)
            b_pert = recompute_bits(list(s_pert))
            s = s_pert
            b = b_pert
            size = len(s)

            result = try_multi_add(s, b, size)
            if result is not None:
                s, b, size = result
            if size > global_best_size:
                global_best_set = set(s)
                global_best_bits = b
                global_best_size = size

    if time.time() - start_time < DEADLINE:
        result = try_multi_add(global_best_set, global_best_bits, global_best_size)
        if result is not None:
            global_best_set, global_best_bits, global_best_size = result

    if len(global_best_set) > 0:
        return list(global_best_set)
    return list(s)
