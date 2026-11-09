import time
import random
from collections import defaultdict

def construct():
    n = 15
    start = time.time()
    deadline = start + 55.0  # Safety margin

    def build_adj_deg(edge_list):
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        return adj, deg

    def has_violation_fast(adj, deg):
        """Use bitmask for fast 3-edge path detection between same-degree vertices."""
        # Build adjacency bitmasks
        adj_mask = [0] * n
        for i in range(n):
            for j in adj[i]:
                adj_mask[i] |= (1 << j)
        
        # Group vertices by degree
        deg_map = defaultdict(list)
        for i in range(n):
            deg_map[deg[i]].append(i)
        
        for d, verts in deg_map.items():
            if len(verts) < 2:
                continue
            # For each pair of same-degree vertices, check if there's a 3-edge path
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    s, t = verts[i], verts[j]
                    # Check all middle edges (u,v) where u is neighbor of s, v is neighbor of t
                    # Path: s - u - v - t
                    # u must be in adj[s], v must be in adj[u] and adj[t]
                    # So: for each u in adj[s], check if adj[u] & adj[t] has any vertex v != s, v != t
                    us = adj[s]
                    if not us:
                        continue
                    for u in us:
                        if u == t:
                            continue
                        # v must be in adj[u] and adj[t], and v != s, v != t
                        common = adj[u] & adj[t]
                        # Remove s and t from consideration
                        if s in common:
                            common.discard(s)
                        if t in common:
                            common.discard(t)
                        if common:
                            return True
        return False

    def is_valid(edge_list):
        adj, deg = build_adj_deg(edge_list)
        return not has_violation_fast(adj, deg)

    def is_valid_bitmask(adj_mask, deg):
        """Check validity using precomputed adjacency bitmasks."""
        deg_map = defaultdict(list)
        for i in range(n):
            deg_map[deg[i]].append(i)
        
        for d, verts in deg_map.items():
            if len(verts) < 2:
                continue
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    s, t = verts[i], verts[j]
                    # For each u in adj[s], check if adj[u] & adj[t] has any vertex v != s, v != t
                    us_mask = adj_mask[s]
                    if us_mask == 0:
                        continue
                    # Iterate through set bits in us_mask
                    m = us_mask
                    while m:
                        u = (m & (-m)).bit_length() - 1
                        m &= m - 1
                        if u == t:
                            continue
                        common = adj_mask[u] & adj_mask[t]
                        # Remove s and t from consideration
                        common &= ~(1 << s)
                        common &= ~(1 << t)
                        if common:
                            return False
        return True

    # Base: complete bipartite K_{7,8} - always valid
    base_edges = [(i, j) for i in range(7) for j in range(7, 15)]
    best_edges = base_edges[:]
    best_count = len(base_edges)

    # All possible edges
    all_possible = [(i, j) for i in range(n) for j in range(i + 1, n)]
    base_set = set(base_edges)
    remaining = [e for e in all_possible if e not in base_set]

    # Greedy add with multiple random orderings
    for attempt in range(800):
        if time.time() > deadline - 12:
            break
        current = base_edges[:]
        current_set = set(current)
        shuffled = remaining[:]
        random.shuffle(shuffled)
        for edge in shuffled:
            if edge in current_set:
                continue
            test = current + [edge]
            if is_valid(test):
                current = test
                current_set.add(edge)
        if len(current) > best_count:
            best_count = len(current)
            best_edges = current[:]
            # If we've reached a high count, try more attempts
            if best_count >= 60:
                for extra in range(300):
                    if time.time() > deadline - 8:
                        break
                    cur2 = best_edges[:]
                    cur2_set = set(cur2)
                    rem2 = [e for e in all_possible if e not in cur2_set]
                    random.shuffle(rem2)
                    for edge in rem2:
                        test = cur2 + [edge]
                        if is_valid(test):
                            cur2 = test
                            cur2_set.add(edge)
                    if len(cur2) > best_count:
                        best_count = len(cur2)
                        best_edges = cur2[:]

    # Local search: edge swap to escape local optima
    if time.time() < deadline - 5:
        for ls_attempt in range(200):
            if time.time() > deadline - 5:
                break
            cur = best_edges[:]
            cur_set = set(cur)
            improved = False
            # Try removing one edge and adding two others
            edges_to_remove = cur[:]
            random.shuffle(edges_to_remove)
            for rem_edge in edges_to_remove:
                if time.time() > deadline - 5:
                    break
                without = [e for e in cur if e != rem_edge]
                without_set = set(without)
                candidates = [e for e in all_possible if e not in without_set]
                random.shuffle(candidates)
                for add1 in candidates[:30]:
                    if time.time() > deadline - 5:
                        break
                    test1 = without + [add1]
                    if not is_valid(test1):
                        continue
                    for add2 in candidates[:30]:
                        if add2 == add1:
                            continue
                        if time.time() > deadline - 5:
                            break
                        test2 = test1 + [add2]
                        if is_valid(test2) and len(test2) > len(cur):
                            cur = test2
                            cur_set = set(cur)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                if len(cur) > best_count:
                    best_count = len(cur)
                    best_edges = cur[:]

    # Final polish: try to add individual edges
    if time.time() < deadline - 2:
        cur = best_edges[:]
        cur_set = set(cur)
        rem = [e for e in all_possible if e not in cur_set]
        random.shuffle(rem)
        for edge in rem:
            if time.time() > deadline - 2:
                break
            test = cur + [edge]
            if is_valid(test):
                cur = test
                cur_set.add(edge)
        if len(cur) > best_count:
            best_count = len(cur)
            best_edges = cur[:]

    out = [[u, v] for u, v in best_edges]
    return out
