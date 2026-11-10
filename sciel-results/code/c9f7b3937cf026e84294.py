import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    N = 512

    # Step 1: Find M with maximum divisors -> initial clique
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

    best_m = 1
    max_divs = 0
    for m in range(1, N + 1):
        d = get_divisors(m)
        if len(d) > max_divs:
            max_divs = len(d)
            best_m = m

    best_clique = set(get_divisors(best_m))

    # Step 2: Greedy improvement - try adding elements that don't conflict
    def conflicts_with_set(x, S):
        for y in S:
            if (x * y) // gcd(x, y) > N:
                return True
        return False

    candidates = [x for x in range(1, N + 1) if x not in best_clique]
    changed = True
    while changed and time.time() < deadline:
        changed = False
        candidates.sort(key=lambda x: sum(1 for y in best_clique if (x * y) // gcd(x, y) > N))
        for x in list(candidates):
            if x in best_clique:
                continue
            if not conflicts_with_set(x, best_clique):
                best_clique.add(x)
                candidates.remove(x)
                changed = True

    # Step 3: Build compatibility using precomputed LCM table
    # lcm[i][j] = lcm(i,j) <= N check
    # Use a set of valid pairs for O(1) lookup
    valid_pairs = set()
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            l = (i * j) // gcd(i, j)
            if l <= N:
                valid_pairs.add((i, j))
    if time.time() > deadline:
        return list(best_clique)

    def is_compatible(a, b):
        if a == b:
            return True
        if a > b:
            a, b = b, a
        return (a, b) in valid_pairs

    # Verify best_clique is a clique
    for a in best_clique:
        for b in best_clique:
            if a != b and not is_compatible(a, b):
                return get_divisors(best_m)

    # Step 4: Local search with swaps
    # Try to improve by removing one element and adding one or more
    def try_improve(clique):
        """Try to improve clique by single swaps and double swaps."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            clique_list = list(clique)
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    # Try removing clique_list[j] and adding a new element
                    removed = clique_list[j]
                    new_clique = clique - {removed}
                    # Find best element to add
                    best_add = None
                    for x in range(1, N + 1):
                        if x in new_clique:
                            continue
                        if all(is_compatible(x, y) for y in new_clique):
                            # Check if this improves (it does if x wasn't in clique)
                            best_add = x
                            break
                    if best_add is not None and best_add != removed:
                        clique.add(best_add)
                        improved = True
                        break
                if improved:
                    break
        return clique

    # Step 5: Bron-Kerbosch with pivot using bitsets (optimized)
    # Build adjacency as sets for faster intersection
    adj = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if is_compatible(i, j):
                adj[i].add(j)
                adj[j].add(i)
    if time.time() > deadline:
        return list(best_clique)

    best_size = len(best_clique)
    best_set = set(best_clique)

    def bron_kerbosch(R, P, X):
        nonlocal best_size, best_set

        if time.time() > deadline:
            return

        r_size = len(R)
        p_size = len(P)

        if r_size > best_size:
            best_size = r_size
            best_set = set(R)

        if p_size == 0:
            return

        if r_size + p_size <= best_size:
            return

        # Pivot selection: maximize |P ∩ N(u)| over u in P ∪ X
        pivot_candidates = P | X
        if not pivot_candidates:
            return

        pivot = None
        max_count = -1
        for u in pivot_candidates:
            count = len(P & adj[u])
            if count > max_count:
                max_count = count
                pivot = u

        if pivot is None:
            return

        candidates = P - adj[pivot]

        for v in list(candidates):
            R.add(v)
            bron_kerbosch(R, P & adj[v], X & adj[v])
            R.remove(v)
            P.discard(v)
            X.add(v)

            if time.time() > deadline:
                return

    # Run BK starting from full set
    P = set(range(1, N + 1))
    X = set()
    R = set()
    bron_kerbosch(R, P, X)

    # Step 6: Final local search polish
    if time.time() < deadline:
        final_clique = try_improve(set(best_set))
        if len(final_clique) > len(best_set):
            best_set = final_clique

    # Final validation
    result = list(best_set)
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            if (a * b) // gcd(a, b) > N:
                valid = False
                break
        if not valid:
            break

    if not valid:
        return get_divisors(best_m)

    return result
