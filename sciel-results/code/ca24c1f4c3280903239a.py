"""Fresh design: branch-and-bound cochromatic partition via target-graph homomorphism."""
import time
import itertools

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    start = time.time()
    DEADLINE = 55.0          # safety margin before the 60s (or 90s) kill
    es = set()
    for e in P['edges']:
        es.add((int(e[0]), int(e[1])))
        es.add((int(e[1]), int(e[0])))
    adj = [0] * n
    for i in range(n):
        m = 0
        for j in range(n):
            if i != j and (i, j) in es:
                m |= (1 << j)
        adj[i] = m

    # A vertex v can be mapped to a clique-type target vertex only if its
    # neighbourhood is a clique; to an independent-type vertex only if its
    # neighbourhood is an independent set.
    clique_ok = [False] * n
    indep_ok = [False] * n
    for v in range(n):
        N = adj[v]
        okc = True
        while okc:
            if N == 0:
                break
            low = N & (-N)
            u = (low.bit_length() - 1)
            if adj[u] & N & ~low == 0:
                okc = False
            N ^= low
        oki = True
        while oki:
            if N == 0:
                break
            low = N & (-N)
            u = (low.bit_length() - 1)
            if adj[u] & N & ~low:
                oki = False
            N ^= low
        clique_ok[v] = okc
        indep_ok[v] = oki

    # Upper bound from a greedy construction (also a valid fallback answer).
    def greedy():
        used = [False] * n
        parts = []
        for v in range(n):
            placed = False
            for part in parts:
                okc = all((a, b) in es for a in part for b in part if a < b)
                oki = not any((a, b) in es for a in part for b in part if a < b)
                can = True
                for u in part:
                    edge = (u, v) in es
                    if okc and not edge:
                        can = False
                    if oki and edge:
                        can = False
                    if not okc and not oki:
                        can = False
                if can:
                    part.append(v)
                    used[v] = True
                    placed = True
                    break
            if not placed:
                parts.append([v])
                used[v] = True
        return parts
    greedy_parts = greedy()
    best_parts = greedy_parts
    best_k = len(greedy_parts)

    def is_valid_partition(assignment):
        groups = {}
        for v in range(n):
            groups.setdefault(assignment[v], []).append(v)
        for g in groups.values():
            if not g:
                return False
            has_edge = False
            no_edge = True
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    if (g[i], g[j]) in es:
                        has_edge = True
                    else:
                        no_edge = False
            if not (has_edge or no_edge):
                return False
        return True

    # Branch-and-bound over target graphs T with k vertices.
    # target_type[t] in {1: clique, 0: independent}; edges only between clique-type.
    def search(k, target_type, target_edge, assignment, depth, target_vertices):
        nonlocal best_parts, best_k
        if time.time() > DEADLINE:
            return False
        if depth == n:
            if k < best_k:
                # build the partition from the assignment
                groups = {}
                for v in range(n):
                    groups.setdefault(assignment[v], []).append(v)
                parts = [sorted(groups[t]) for t in range(k)]
                if is_valid_partition(assignment):
                    best_parts = parts
                    best_k = k
            return False

        v = depth
        # candidate target vertices: must be compatible with v's type.
        for t in target_vertices:
            if target_type[t] == 1:
                if not clique_ok[v]:
                    continue
            else:
                if not indep_ok[v]:
                    continue
            # check consistency with already-assigned neighbours of v
            ok = True
            nb = adj[v]
            # iterate assigned neighbours (depth < v)
            cand = nb & ((1 << depth) - 1)
            while ok and cand:
                low = cand & (-cand)
                u = (low.bit_length() - 1)
                tu = assignment[u]
                # same target vertex: need same edge status as (v,u)
                if tu == t:
                    edge = bool(nb & (1 << u))
                    if target_type[t] == 1 and not edge:
                        ok = False
                    if target_type[t] == 0 and edge:
                        ok = False
                # different target vertices: target edge must match graph edge
                else:
                    edge = bool(nb & (1 << u))
                    tedge = bool(target_edge[t] & (1 << tu))
                    if edge != tedge:
                        ok = False
                cand ^= low
            if not ok:
                continue
            assignment[v] = t
            if search(k, target_type, target_edge, assignment, depth + 1, target_vertices):
                return True
        return False

    # Try k from 2 upward. For each k, enumerate type assignments and edge patterns.
    # We cap the number of target graphs explored per k by the deadline.
    for k in range(2, best_k):
        if time.time() > DEADLINE:
            break
        # choose which of the k vertices are clique-type (at least 1 each side not required)
        # iterate over all type assignments and edge subsets, but prune heavily.
        # To bound the space, iterate type patterns first.
        for type_mask in range(1 << k):
            if time.time() > DEADLINE:
                break
            target_type = [1 if (type_mask >> t) & 1 else 0 for t in range(k)]
            # clique-type vertices form a clique in T; independent-type form an
            # independent set; edges between the two groups are arbitrary.
            clique_vs = [t for t in range(k) if target_type[t] == 1]
            indep_vs = [t for t in range(k) if target_type[t] == 0]
            # edges among clique_vs are forced to be present; among indep_vs absent.
            base_edge = 0
            for i in range(len(clique_vs)):
                for j in range(i + 1, len(clique_vs)):
                    a, b = clique_vs[i], clique_vs[j]
                    base_edge |= (1 << b)
                    base_edge |= (1 << a)
            # free edges are between a clique vertex and an independent vertex
            free_pairs = [(a, b) for a in clique_vs for b in indep_vs]
            num_free = len(free_pairs)
            # iterate edge patterns on free pairs
            for emask in range(1 << num_free):
                if time.time() > DEADLINE:
                    break
                target_edge = [base_edge] * k
                for idx, (a, b) in enumerate(free_pairs):
                    if (emask >> idx) & 1:
                        target_edge[a] |= (1 << b)
                        target_edge[b] |= (1 << a)
                assignment = [-1] * n
                # simple lower bound: if k >= best_k, skip (we only try k < best_k)
                if search(k, target_type, target_edge, assignment, 0, list(range(k))):
                    break
                if time.time() > DEADLINE:
                    break
            if time.time() > DEADLINE:
                break

    # Return a valid partition as a list of lists of Python ints.
    result = [list(map(int, part)) for part in best_parts]
    return result
