import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s kill

    n = 512
    nodes = list(range(1, n + 1))

    # Precompute compatibility: adj[i] is a bitmask where bit j is set if lcm(i,j) <= n
    adj = [0] * (n + 1)

    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            g = gcd(i, j)
            l = (i // g) * j
            if l <= n:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    if time.time() > deadline:
        return list(range(1, 10))

    def verify_solution(lst):
        """Verify all pairs have lcm <= 512"""
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                g = gcd(lst[i], lst[j])
                l = (lst[i] // g) * lst[j]
                if l > n:
                    return False
        return True

    # Tomita's algorithm for maximum clique using greedy coloring with bitmasks
    def max_clique_tomita(adj, n, deadline):
        best_mask = 0
        best_size = 0

        def expand(R, P, X):
            nonlocal best_mask, best_size

            if time.time() > deadline:
                return

            if len(R) > best_size:
                best_size = len(R)
                best_mask = 0
                for v in R:
                    best_mask |= (1 << v)

            if P == 0:
                return

            # Extract vertices from P
            P_list = []
            temp = P
            while temp:
                v = (temp & -temp).bit_length() - 1
                P_list.append(v)
                temp &= temp - 1

            if not P_list:
                return

            # Color P_list using greedy coloring
            # Order by degree in P (descending) for better coloring
            deg_in_P = {}
            for v in P_list:
                deg_in_P[v] = bin(adj[v] & P).count('1')

            P_sorted = sorted(P_list, key=lambda v: -deg_in_P[v])

            # Greedy coloring: assign smallest available color
            color = {}
            color_classes = []  # list of sets of vertices with same color
            for v in P_sorted:
                placed = False
                for ci, cls in enumerate(color_classes):
                    # v can go in class ci if not adjacent to any vertex in cls
                    conflict = False
                    for u in cls:
                        if adj[v] & (1 << u):
                            conflict = True
                            break
                    if not conflict:
                        cls.add(v)
                        color[v] = ci + 1
                        placed = True
                        break
                if not placed:
                    color_classes.add = None  # prevent accidental use
                    color_classes.append(set([v]))
                    color[v] = len(color_classes)

            max_color = max(color.values()) if color else 0

            # Upper bound: len(R) + max_color
            if len(R) + max_color <= best_size:
                return

            # Process vertices in reverse color order (highest color first)
            # Build ordered list by color
            ordered = sorted(P_sorted, key=lambda v: -color[v])

            for v in ordered:
                c = color[v]

                # Bound: if len(R) + c <= best_size, we can stop
                if len(R) + c <= best_size:
                    break

                # v must still be in P
                if not (P & (1 << v)):
                    continue

                # Recurse
                R.add(v)
                new_P = P & adj[v]
                new_X = X & adj[v]
                expand(R, new_P, new_X)
                R.remove(v)

                P &= ~(1 << v)
                X |= (1 << v)

                if time.time() > deadline:
                    return

        # Initialize: all vertices 1..n
        P = 0
        for i in range(1, n + 1):
            P |= (1 << i)

        X = 0
        R = set()

        expand(R, P, X)

        # Extract best clique from best_mask
        result = []
        temp = best_mask
        while temp:
            v = (temp & -temp).bit_length() - 1
            result.append(v)
            temp &= temp - 1
        return result

    # Run Tomita
    try:
        best_clique = max_clique_tomita(adj, n, deadline)
    except Exception as e:
        best_clique = [1, 2]

    # Verify and fix if needed
    if not verify_solution(best_clique):
        # Fall back to a known valid solution
        best_clique = [2**i for i in range(10) if 2**i <= 512]

    # Try to improve with local search: try adding vertices
    improved = list(best_clique)
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for v in range(1, n + 1):
            if v in improved:
                continue
            # Check if v is compatible with all in improved
            ok = True
            for u in improved:
                if not (adj[v] & (1 << u)):
                    ok = False
                    break
            if ok:
                improved.append(v)
                changed = True

    # Try 2-opt: remove one vertex, try to add two
    for _ in range(3):
        if time.time() > deadline - 5:
            break
        for i in range(len(improved)):
            if time.time() > deadline - 5:
                break
            saved = improved.pop(i)
            added = []
            for v in range(1, n + 1):
                if v in improved or v == saved:
                    continue
                ok = True
                for u in improved:
                    if not (adj[v] & (1 << u)):
                        ok = False
                        break
                if ok and all(adj[v] & (1 << a) for a in added):
                    added.append(v)
                if len(added) >= 2:
                    break
            if len(added) > 1:
                improved.extend(added)
            else:
                improved.insert(i, saved)

    # Final verification
    if not verify_solution(improved):
        if verify_solution(best_clique):
            return best_clique
        else:
            return [1, 2]

    return improved
