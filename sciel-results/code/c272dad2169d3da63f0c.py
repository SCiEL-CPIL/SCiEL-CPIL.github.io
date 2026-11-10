import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit

    N = 512
    nodes = list(range(1, N + 1))

    # Build bitset-based compatibility: compat[i] is a bitset where bit j is set iff lcm(i,j) <= N
    compat = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = gcd(i, j)
            if (i // g) * j <= N:
                compat[i] |= (1 << j)
                compat[j] |= (1 << i)
    # Self-compatibility
    for i in range(1, N + 1):
        compat[i] |= (1 << i)

    def get_divisors(m):
        divs = []
        i = 1
        while i * i <= m:
            if m % i == 0:
                divs.append(i)
                if i != m // i:
                    divs.append(m // i)
            i += 1
        return divs

    # Find best single-m divisor set as initial seed
    best_m = 1
    max_divs = 0
    for m in range(1, N + 1):
        d = get_divisors(m)
        if len(d) > max_divs:
            max_divs = len(d)
            best_m = m

    best_clique = set(get_divisors(best_m))
    best_clique_list = sorted(best_clique)

    # Helper: check if x is compatible with all elements in set S using bitset
    def is_compatible_with_set(x, S):
        for y in S:
            if not (compat[x] >> y) & 1:
                return False
        return True

    # Helper: count conflicts of x with set S
    def count_conflicts(x, S):
        count = 0
        for y in S:
            if not (compat[x] >> y) & 1:
                count += 1
        return count

    # Step 2: Greedy improvement pass using bitsets
    candidates = [x for x in range(1, N + 1) if x not in best_clique]

    changed = True
    while changed and time.time() < deadline:
        changed = False
        candidates.sort(key=lambda x: count_conflicts(x, best_clique))
        for x in list(candidates):
            if x in best_clique:
                continue
            if is_compatible_with_set(x, best_clique):
                best_clique.add(x)
                best_clique_list.append(x)
                candidates.remove(x)
                changed = True

    # Step 3: Local search with swaps (1-for-1, 1-for-2, 2-for-3)
    all_elements = set(range(1, N + 1))

    swap_changed = True
    max_swaps = 200
    swaps_done = 0
    while swap_changed and swaps_done < max_swaps and time.time() < deadline:
        swap_changed = False
        swaps_done += 1

        clique_list = list(best_clique)
        n_c = len(clique_list)

        best_gain = 0
        best_removal = None
        best_addition = None

        # Try removing 1 element
        for ri in range(n_c):
            r1 = clique_list[ri]
            remaining = best_clique - {r1}
            addable = []
            for x in all_elements:
                if x not in remaining and is_compatible_with_set(x, remaining):
                    addable.append(x)
            gain = len(addable) - 1
            if gain > best_gain:
                best_gain = gain
                best_removal = [r1]
                best_addition = addable

        # Try removing 2 elements
        for ri in range(n_c):
            for rj in range(ri + 1, n_c):
                r1, r2 = clique_list[ri], clique_list[rj]
                remaining = best_clique - {r1, r2}
                addable = []
                for x in all_elements:
                    if x not in remaining and is_compatible_with_set(x, remaining):
                        addable.append(x)
                gain = len(addable) - 2
                if gain > best_gain:
                    best_gain = gain
                    best_removal = [r1, r2]
                    best_addition = addable

        if best_gain > 0 and best_removal is not None and best_addition:
            for r in best_removal:
                best_clique.discard(r)
                if r in best_clique_list:
                    best_clique_list.remove(r)
            for a in best_addition:
                best_clique.add(a)
                best_clique_list.append(a)
            swap_changed = True

    # Step 4: Optimized Bron-Kerbosch with bitsets
    # Use bit_count() instead of bin().count('1') for speed

    best_size = len(best_clique_list)

    def bron_kerbosch_bitset(R_bits, P_bits, X_bits):
        nonlocal best_clique_list, best_size

        if time.time() > deadline:
            return

        r_count = R_bits.bit_count()
        p_count = P_bits.bit_count()

        # Update best
        if r_count > best_size:
            best_size = r_count
            best_clique_list = [i for i in range(1, N + 1) if (R_bits >> i) & 1]

        if P_bits == 0:
            return

        # Prune
        if r_count + p_count <= best_size:
            return

        # Pivot: choose u in P that maximizes |P ∩ N(u)|
        p_list = [i for i in range(1, N + 1) if (P_bits >> i) & 1]

        if not p_list:
            return

        pivot = max(p_list, key=lambda u: (P_bits & compat[u]).bit_count())
        candidates_bits = P_bits & ~compat[pivot]

        c_list = [i for i in range(1, N + 1) if (candidates_bits >> i) & 1]

        for v in c_list:
            new_R = R_bits | (1 << v)
            new_P = P_bits & compat[v]
            new_X = X_bits & compat[v]
            bron_kerbosch_bitset(new_R, new_P, new_X)
            P_bits &= ~(1 << v)
            X_bits |= (1 << v)

            if time.time() > deadline:
                return

    # Run BK starting from empty set
    R_bits = 0
    P_bits = sum(1 << i for i in range(1, N + 1))
    X_bits = 0
    bron_kerbosch_bitset(R_bits, P_bits, X_bits)

    # Final validation
    valid = True
    for i in range(len(best_clique_list)):
        for j in range(i + 1, len(best_clique_list)):
            a, b = best_clique_list[i], best_clique_list[j]
            if (a * b) // gcd(a, b) > N:
                valid = False
                break
        if not valid:
            break

    if not valid:
        return get_divisors(best_m)

    return best_clique_list
