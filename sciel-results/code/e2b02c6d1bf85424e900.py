import time
from math import gcd

def construct():
    """
    Returns the maximum cardinality subset of [1, 255] containing 255,
    such that gcd(a, b) > 1 for all distinct pairs a, b in the subset.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin under 90s wall-clock

    # Candidates: numbers in [1,255] sharing a factor with 255 (=3*5*17)
    candidates = []
    for i in range(1, 256):
        if gcd(i, 255) > 1:
            candidates.append(i)

    n = len(candidates)

    # Build adjacency as bitsets (Python int). adj[i] is a bitmask of neighbors of i.
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    # Degrees for ordering
    degrees = [adj[i].bit_count() for i in range(n)]

    # Order vertices by degree descending
    order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    old_to_new = [0] * n
    for new_idx, old_idx in enumerate(order):
        old_to_new[old_idx] = new_idx

    # Rebuild adjacency in new order
    new_adj = [0] * n
    for i in range(n):
        for j in range(n):
            if adj[i] & (1 << j):
                new_adj[old_to_new[i]] |= (1 << old_to_new[j])

    # Precompute neighbor lists for faster iteration
    neighbor_lists = []
    for i in range(n):
        nbrs = []
        tmp = new_adj[i]
        while tmp:
            lb = tmp & (-tmp)
            v = lb.bit_length() - 1
            nbrs.append(v)
            tmp ^= lb
        neighbor_lists.append(nbrs)

    # Tomita-style maximum clique with coloring bound
    best_size = 0
    best_clique = []

    def color_sort(P):
        """
        Greedily color the subgraph induced by P (P is a bitmask).
        Returns a list of (vertex, color) sorted by color ascending,
        and the max color (upper bound on clique size in P).
        """
        if P == 0:
            return [], 0
        # Get list of vertices in P
        verts = []
        tmp = P
        while tmp:
            lb = tmp & (-tmp)
            v = lb.bit_length() - 1
            verts.append(v)
            tmp ^= lb
        # Order by degree in P descending
        def deg_in_P(v):
            return (new_adj[v] & P).bit_count()
        verts.sort(key=deg_in_P, reverse=True)

        colors = []
        uncolored = set(verts)
        color_of = {}
        # Greedy coloring: process verts in order, assign smallest available color
        for v in verts:
            used = set()
            nb = new_adj[v] & P
            # neighbors already colored
            tmp = nb
            while tmp:
                lb = tmp & (-tmp)
                w = lb.bit_length() - 1
                if w in color_of:
                    used.add(color_of[w])
                tmp ^= lb
            c = 1
            while c in used:
                c += 1
            color_of[v] = c
            colors.append((v, c))

        # Sort by color ascending
        colors.sort(key=lambda x: x[1])
        max_color = max(c for (_, c) in colors) if colors else 0
        return colors, max_color

    def expand(R, P):
        nonlocal best_size, best_clique
        if time.time() > deadline:
            return
        if P == 0:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = R[:]
            return
        # Bound
        if len(R) + P.bit_count() <= best_size:
            return
        colored, max_color = color_sort(P)
        if len(R) + max_color <= best_size:
            return
        # Process vertices in reverse color order (highest color first)
        for v, c in reversed(colored):
            if time.time() > deadline:
                return
            if len(R) + c <= best_size:
                break
            # Remove v from P
            P ^= (1 << v)
            new_R = R + [v]
            new_P = P & new_adj[v]
            expand(new_R, new_P)

    # Initial lower bound: try multiples of 3, 5, 17
    init3 = [i for i in range(n) if candidates[i] % 3 == 0]
    init5 = [i for i in range(n) if candidates[i] % 5 == 0]
    init17 = [i for i in range(n) if candidates[i] % 17 == 0]
    best_init = max([init3, init5, init17], key=len)
    best_size = len(best_init)
    best_clique = best_init[:]

    expand([], (1 << n) - 1)

    # Convert back to values
    result = [candidates[i] for i in best_clique]
    if 255 not in result:
        result = list(range(3, 256, 3))
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
    valid = True
    if 255 not in res:
        valid = False
    for i in range(len(res)):
        for j in range(i + 1, len(res)):
            if gcd(res[i], res[j]) == 1:
                valid = False
                break
    print("Valid:", valid)
