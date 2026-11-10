import itertools
import random
import time
import sys

sys.setrecursionlimit(10000)


def _is_non_2_colorable(n, edges):
    """
    Exact check: returns True if the hypergraph is NOT 2-colorable.
    Uses bitmask backtracking with unit propagation.
    """
    if not edges:
        return False

    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)

    full_mask = (1 << n) - 1

    vertex_degree = [0] * n
    for mask in edge_masks:
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            vertex_degree[idx] += 1
            m ^= lsb

    def propagate(color1, unassigned):
        changed = True
        while changed:
            changed = False
            for mask in edge_masks:
                unassigned_in_edge = mask & unassigned
                assigned_in_edge = mask & ~unassigned

                if unassigned_in_edge == 0:
                    c1 = bin(mask & color1).count('1')
                    if c1 == 0 or c1 == 5:
                        return None, None, True
                    continue

                aie = assigned_in_edge
                c1a = 0
                m2 = aie
                while m2:
                    lsb = m2 & -m2
                    if color1 & lsb:
                        c1a += 1
                    m2 ^= lsb
                c0a = bin(aie).count('1') - c1a

                if c0a == 0 and c1a > 0 and bin(unassigned_in_edge).count('1') == 1:
                    unassigned ^= unassigned_in_edge
                    changed = True
                elif c1a == 0 and c0a > 0 and bin(unassigned_in_edge).count('1') == 1:
                    color1 |= unassigned_in_edge
                    unassigned ^= unassigned_in_edge
                    changed = True

        return color1, unassigned, False

    def backtrack(color1, unassigned):
        color1, unassigned, conflict = propagate(color1, unassigned)
        if conflict:
            return False
        if unassigned == 0:
            for mask in edge_masks:
                c1 = bin(mask & color1).count('1')
                if c1 == 0 or c1 == 5:
                    return False
            return True
        best_v_bit = 0
        best_deg = -1
        m = unassigned
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            if vertex_degree[idx] > best_deg:
                best_deg = vertex_degree[idx]
                best_v_bit = lsb
            m ^= lsb
        new_unassigned = unassigned ^ best_v_bit
        if backtrack(color1, new_unassigned):
            return True
        if backtrack(color1 | best_v_bit, new_unassigned):
            return True
        return False

    initial_color1 = 0
    initial_unassigned = full_mask ^ 1
    initial_color1, initial_unassigned, conflict = propagate(initial_color1, initial_unassigned)
    if conflict:
        return True
    if initial_unassigned == 0:
        for mask in edge_masks:
            c1 = bin(mask & initial_color1).count('1')
            if c1 == 0 or c1 == 5:
                return False
        return False
    if backtrack(initial_color1, initial_unassigned):
        return False
    return True


def _reduce_edges(n, edges, deadline):
    """Greedy edge removal with redundancy-based ordering."""
    edges = edges[:]
    
    while time.time() < deadline:
        if len(edges) <= 51:
            break
        # Score each edge by redundancy
        scores = []
        for i, e in enumerate(edges):
            e_set = frozenset(e)
            s = 0
            for j, other in enumerate(edges):
                if j == i:
                    continue
                ov = len(e_set.intersection(frozenset(other)))
                if ov >= 4:
                    s += 10
                elif ov == 3:
                    s += 3
            scores.append(s)
        
        # Try removing the most redundant edge
        order = sorted(range(len(edges)), key=lambda i: -scores[i])
        removed = False
        for idx in order:
            if time.time() > deadline:
                break
            new_edges = edges[:idx] + edges[idx+1:]
            if _is_non_2_colorable(n, new_edges):
                edges = new_edges
                removed = True
                break
        if not removed:
            break
    
    return edges


def construct():
    start_time = time.time()
    deadline = start_time + 110

    best_edges = None
    best_len = float('inf')
    best_n = 9

    # Primary: reduce K_9^5
    n = 9
    all_edges = [list(e) for e in itertools.combinations(range(n), 5)]
    reduced = _reduce_edges(n, all_edges, deadline - 10)
    best_edges = reduced
    best_len = len(reduced)
    best_n = n

    # If we got close to 51, try to squeeze more
    if best_len > 51 and time.time() < deadline - 10:
        # Try randomized restarts
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            rng = random.Random(seed * 13 + 7)
            shuffled = all_edges[:]
            rng.shuffle(shuffled)
            
            edges = shuffled[:]
            while time.time() < deadline - 5 and len(edges) > 51:
                rng.shuffle(edges)
                removed = False
                for i in range(len(edges) - 1, -1, -1):
                    if time.time() > deadline - 5:
                        break
                    new_edges = edges[:i] + edges[i+1:]
                    if _is_non_2_colorable(n, new_edges):
                        edges = new_edges
                        removed = True
                        break
                if not removed:
                    break
            
            if len(edges) < best_len:
                best_len = len(edges)
                best_edges = edges[:]
                best_n = n

    # Fallback
    if best_edges is None or best_len > 130:
        n = 9
        best_edges = [list(e) for e in itertools.combinations(range(n), 5)]
        best_n = n

    if not _is_non_2_colorable(best_n, best_edges):
        n = 9
        best_edges = [list(e) for e in itertools.combinations(range(n), 5)]
        best_n = n

    return {'n': best_n, 'edges': best_edges}
