"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations

def _moore_bound(k, g):
    """Compute the Moore bound (lower bound) for a k-regular graph of girth g."""
    if k <= 2:
        return g if g % 2 == 0 else 2 * g - 1
    if g % 2 == 0:
        half = g // 2
        s = 0
        term = 1
        for i in range(half):
            s += term
            term *= (k - 1)
        return 2 * k * s
    else:
        half = (g - 1) // 2
        s = 0
        term = 1
        for i in range(half + 1):
            s += term
            term *= (k - 1)
        return 1 + k * s

def _girth_circulant(n, gens):
    """Compute girth of a circulant graph Z_n with given generators (undirected).
    Uses BFS from vertex 0 only, exploiting circulant symmetry."""
    # For circulant graphs, girth can be found by BFS from one vertex
    # since all vertices are equivalent under translation
    dist = [-1] * n
    parent = [-1] * n
    dist[0] = 0
    queue = [0]
    head = 0
    min_girth = float('inf')
    
    while head < len(queue):
        u = queue[head]
        head += 1
        for g in gens:
            v = (u + g) % n
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                parent[v] = u
                queue.append(v)
            elif v != parent[u] and u != 0:
                cycle_len = dist[u] + dist[v] + 1
                if cycle_len < min_girth:
                    min_girth = cycle_len
            # Also check the reverse direction
            v2 = (u - g) % n
            if dist[v2] == -1:
                dist[v2] = dist[u] + 1
                parent[v2] = u
                queue.append(v2)
            elif v2 != parent[u] and u != 0:
                cycle_len = dist[u] + dist[v2] + 1
                if cycle_len < min_girth:
                    min_girth = cycle_len
    return min_girth

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Uses circulant graph search for k=5, g=9.
    Falls back to LUW construction for other parameters.
    """
    t_start = time.time()
    deadline = t_start + 140.0

    if k == 5 and g == 9:
        # Search for circulant graph Z_n with 5 generators
        # For 5-regular: generators {a, -a, b, -b, c} where c = n/2
        # n must be even
        best_n = None
        best_edges = None
        
        mb = _moore_bound(k, g)
        # Search from a reasonable lower bound
        for n in range(428, 1200, 2):  # even n only
            if time.time() > deadline:
                break
            if n < mb:
                continue
            c = n // 2
            # Try different (a, b) pairs
            # a, b must be distinct from 0, c, and from each other (mod n)
            # Also -a, -b must be distinct from a, b, c
            found = False
            for a in range(1, n // 2):
                if time.time() > deadline:
                    break
                na = (-a) % n
                if na == a:  # a = n/2, skip
                    continue
                for b in range(1, n // 2):
                    if b == a or b == na:
                        continue
                    nb = (-b) % n
                    if nb == b:  # b = n/2, skip
                        continue
                    if nb == a or nb == na:
                        continue
                    # Check all 5 generators are distinct
                    gens_set = {a, na, b, nb, c}
                    if len(gens_set) != 5:
                        continue
                    # Check no generator is 0
                    if 0 in gens_set:
                        continue
                    # Compute girth
                    girth = _girth_circulant(n, list(gens_set))
                    if girth >= g:
                        # Build edge list
                        edge_list = []
                        for i in range(n):
                            for gen in sorted(gens_set):
                                j = (i + gen) % n
                                if i < j:
                                    edge_list.append((i, j))
                        if best_n is None or n < best_n:
                            best_n = n
                            best_edges = edge_list
                        found = True
                        break  # Found one for this n, move to next n
                if found:
                    break
            if best_n is not None and best_n <= n:
                # We found a valid graph, and since we're searching in increasing n,
                # we can stop early if we've found one
                break
        
        if best_n is not None:
            return best_n, best_edges
    
    # Fallback: LUW construction
    q = k
    d = None
    for candidate_d in range(3, 20):
        if candidate_d % 2 == 1:
            if candidate_d + 5 >= g:
                d = candidate_d
                break
        else:
            if candidate_d + 4 >= g:
                d = candidate_d
                break
    if d is None:
        d = g - 4
    if d < 3:
        d = 3

    if q == 4:
        _add_t = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        _mul_t = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        def f_add(a, b): return a ^ b
        def f_sub(a, b): return a ^ b
        def f_mul(a, b): return _mul_t[a][b]
    else:
        def f_add(a, b): return (a + b) % q
        def f_sub(a, b): return (a - b) % q
        def f_mul(a, b): return (a * b) % q

    rules = {}
    rules[2] = (1, 1)
    rules[3] = (2, 1)
    rules[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        base = c - r
        candidates = [(1, base - 2), (base - 1, 1), (base, 1), (1, base + 1)]
        rules[c] = candidates[r]

    start_key = (0, (0,) * (d + 1))
    label = {start_key: 0}
    queue = [start_key]
    edge_list = []
    
    for vkey in queue:
        if time.time() > deadline:
            break
        side, coords = vkey
        vidx = label[vkey]
        for first_val in range(q):
            ncoords = [0] * (d + 1)
            ncoords[1] = first_val
            for c in range(2, d + 1):
                a, b = rules[c]
                if side == 0:
                    ncoords[c] = f_add(coords[c], f_mul(ncoords[a], coords[b]))
                else:
                    ncoords[c] = f_sub(coords[c], f_mul(coords[a], ncoords[b]))
            nkey = (1 - side, tuple(ncoords))
            if nkey not in label:
                label[nkey] = len(queue)
                queue.append(nkey)
            if side == 0:
                edge_list.append((vidx, label[nkey]))
    
    n = len(queue)
    return n, edge_list
# EVOLVE-BLOCK-END
