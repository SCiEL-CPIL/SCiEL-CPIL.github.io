import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    N = 512

    # Build bitset-based compatibility: compat[i] is a bitset where bit j is set iff lcm(i,j) <= N
    compat = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = gcd(i, j)
            if (i // g) * j <= N:
                compat[i] |= (1 << j)
                compat[j] |= (1 << i)
    for i in range(1, N + 1):
        compat[i] |= (1 << i)

    # Precompute degree for each vertex
    degree = [bin(compat[i]).count('1') for i in range(N + 1)]

    # Greedy clique construction: repeatedly add the vertex with highest degree among compatible vertices
    best_clique = set()
    best_clique_list = []
    remaining = set(range(1, N + 1))
    
    # Start with the vertex that has the most compatible partners
    while remaining:
        # Pick vertex with max degree in remaining
        v = max(remaining, key=lambda x: bin(compat[x] & sum(1 << i for i in remaining)).count('1'))
        if not best_clique:
            best_clique.add(v)
            best_clique_list.append(v)
        elif all((compat[v] >> y) & 1 for y in best_clique):
            best_clique.add(v)
            best_clique_list.append(v)
        remaining.discard(v)
        if time.time() > deadline - 50:
            break

    # Local search: try removing 1-2 elements and adding more
    all_elements = set(range(1, N + 1))
    for iteration in range(200):
        if time.time() > deadline - 15:
            break
        clique_list = list(best_clique)
        n_c = len(clique_list)
        best_gain = 0
        best_removal = None
        best_addition = None

        # Try removing 1 element
        for r1 in clique_list:
            remaining_set = best_clique - {r1}
            addable = [x for x in all_elements if x not in remaining_set and all((compat[x] >> y) & 1 for y in remaining_set)]
            gain = len(addable) - 1
            if gain > best_gain:
                best_gain = gain
                best_removal = [r1]
                best_addition = addable

        # Try removing 2 elements
        for i in range(n_c):
            for j in range(i + 1, n_c):
                r1, r2 = clique_list[i], clique_list[j]
                remaining_set = best_clique - {r1, r2}
                addable = [x for x in all_elements if x not in remaining_set and all((compat[x] >> y) & 1 for y in remaining_set)]
                gain = len(addable) - 2
                if gain > best_gain:
                    best_gain = gain
                    best_removal = [r1, r2]
                    best_addition = addable

        if best_gain > 0 and best_removal is not None:
            for r in best_removal:
                best_clique.discard(r)
                if r in best_clique_list:
                    best_clique_list.remove(r)
            for a in best_addition:
                best_clique.add(a)
                best_clique_list.append(a)
        else:
            break

    # Bron-Kerbosch with bitsets for maximum clique - optimized version
    best_size = len(best_clique_list)
    best_result = list(best_clique_list)
    
    def bron_kerbosch(R_bits, P_bits, X_bits, depth=0):
        nonlocal best_size, best_result

        if time.time() > deadline:
            return

        r_count = bin(R_bits).count('1')
        p_count = bin(P_bits).count('1')

        if r_count > best_size:
            best_size = r_count
            best_result = [i for i in range(1, N + 1) if (R_bits >> i) & 1]

        if P_bits == 0:
            return

        if r_count + p_count <= best_size:
            return

        # Choose pivot with maximum neighbors in P
        p_list = [i for i in range(1, N + 1) if (P_bits >> i) & 1]
        if not p_list:
            return

        # Pick pivot that maximizes |P ∩ N(pivot)|
        pivot = max(p_list, key=lambda u: bin(P_bits & compat[u]).count('1'))
        candidates_bits = P_bits & ~compat[pivot]
        c_list = [i for i in range(1, N + 1) if (candidates_bits >> i) & 1]

        for v in c_list:
            bron_kerbosch(R_bits | (1 << v), P_bits & compat[v], X_bits & compat[v], depth + 1)
            P_bits &= ~(1 << v)
            X_bits |= (1 << v)
            if time.time() > deadline:
                return

    if time.time() < deadline - 2:
        R_bits = 0
        P_bits = sum(1 << i for i in range(1, N + 1))
        X_bits = 0
        bron_kerbosch(R_bits, P_bits, X_bits)

    # Use the best found
    best_clique_list = best_result

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
        # Fallback: return divisors of 288 (which has many divisors)
        divs = []
        m = 288
        i = 1
        while i * i <= m:
            if m % i == 0:
                divs.append(i)
                if i != m // i:
                    divs.append(m // i)
            i += 1
        return sorted(divs)

    return best_clique_list
