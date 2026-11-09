"""Fewest C4s in graphs with 31 vertices and 91 edges.
Strategy:
1. Build C4-free 90-edge core from polarity graph of PG(2,5).
2. Add 91st edge minimizing new C4s.
3. Local search with incremental C4 counting for edge swaps.
"""
import numpy as np
import time
import random
from itertools import combinations

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin under 150s kill

    n = 31
    q = 5

    # Step 1: Construct C4-free graph with 90 edges using polarity graph of PG(2,5)
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, q - 2, q)
                    nx, ny, nz = 1, (y * inv_x) % q, (z * inv_x) % q
                elif y != 0:
                    inv_y = pow(y, q - 2, q)
                    nx, ny, nz = 0, 1, (z * inv_y) % q
                else:
                    nx, ny, nz = 0, 0, 1
                key = (nx, ny, nz)
                if key not in seen:
                    seen.add(key)
                    points.append(key)

    assert len(points) == 31, f"Expected 31 points, got {len(points)}"

    def adjacent(i, j):
        x1, y1, z1 = points[i]
        x2, y2, z2 = points[j]
        return (x1*x2 + y1*y2 - z1*z2) % q == 0

    # Build core 90-edge graph
    edges_core = []
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if adjacent(i, j):
                edges_core.append((i, j))
                adj[i].add(j)
                adj[j].add(i)

    assert len(edges_core) == 90, f"Expected 90 core edges, got {len(edges_core)}"

    # Helper: count C4s using common-neighbor pairs
    def count_c4(adj_sets):
        total = 0
        for u in range(n):
            for v in range(u+1, n):
                cn = len(adj_sets[u] & adj_sets[v])
                if cn >= 2:
                    total += cn * (cn - 1) // 2
        return total // 4

    # Step 2: Find best edge to add (minimizes new C4s)
    best_edge = None
    best_add = float('inf')

    for u in range(n):
        if time.time() > deadline - 80:
            break
        for v in range(u+1, n):
            if v in adj[u]:
                continue
            # Count paths of length 3 from u to v
            add = 0
            for a in adj[u]:
                for b in adj[v]:
                    if b in adj[a]:
                        add += 1
            if add < best_add:
                best_add = add
                best_edge = (u, v)

    if best_edge is None:
        best_edge = (0, 1)
        if 1 in adj[0]:
            for j in range(n):
                if j != 0 and j not in adj[0]:
                    best_edge = (0, j)
                    break

    # Initialize with core + best edge
    edge_set = set(edges_core) | {best_edge}
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)

    current_c4 = count_c4(adj)
    best_c4 = current_c4
    best_edges = set(edge_set)

    # Precompute all possible edges
    all_possible_edges = set()
    for i in range(n):
        for j in range(i+1, n):
            all_possible_edges.add((i, j))

    non_edges = list(all_possible_edges - edge_set)
    random.seed(42)
    random.shuffle(non_edges)

    # Efficient incremental C4 update for edge swap (remove e1, add e2)
    def c4_delta_remove_add(adj_sets, e1, e2):
        """Compute change in C4 count when removing e1 and adding e2."""
        u1, v1 = e1
        u2, v2 = e2

        # C4s lost by removing (u1,v1):
        # C4s containing edge (u1,v1) = C4s where u1 and v1 share 2+ common neighbors
        # Each pair of common neighbors (a,b) with a~b forms a C4: u1-a-b-v1-u1
        # But we need to be careful: C4s are counted by pairs of vertices with 2+ common neighbors
        # A C4 containing edge (u1,v1) has the form u1-a-v1-b-u1 where a~v1, b~u1, a~b
        # Number of such C4s = number of edges between N(u1)\{v1} and N(v1)\{u1}
        # Wait, that's not quite right either. Let me think more carefully.
        
        # Actually, the standard formula: total C4s = (1/4) * sum_{u<v} C(|N(u)∩N(v)|, 2)
        # When we remove edge (u1,v1), the changes are:
        # - For pair (u1,v1): their common neighbor set changes (loses each other)
        # - For pairs (u1, x) where x in N(v1)\{u1}: common neighbors change
        # - For pairs (v1, x) where x in N(u1)\{v1}: common neighbors change
        # - For pairs (a, b) where a in N(u1), b in N(v1), a!=v1, b!=u1: 
        #   if a~b, then removing u1-v1 doesn't affect their common neighbors
        #   Actually it does: u1 is a common neighbor of (a, ?) only if a~u1 and ?~u1
        #   This is getting complex. Let me just do a full recount for safety but optimize.
        
        # For now, use full recount but make it fast with numpy
        # Actually, let's just do the full recount - it's O(n^2) = ~465 operations
        pass

    # Given the small n=31, full recount is fast enough. Let's optimize the local search loop.
    
    swap_count = 0
    no_improve_count = 0
    max_no_improve = 3000
    current_edges_list = list(edge_set)
    random.shuffle(current_edges_list)

    def try_swap(e1, e2):
        """Try swapping e1 (remove) with e2 (add). Returns new_c4 or None if invalid."""
        if e2 in edge_set:
            return None
        if e1 not in edge_set:
            return None

        # Tentatively swap
        edge_set.remove(e1)
        edge_set.add(e2)
        adj[u1].discard(v1)
        adj[v1].discard(u1)
        u2, v2 = e2
        adj[u2].add(v2)
        adj[v2].add(u2)

        new_c4 = count_c4(adj)

        if new_c4 < current_c4:
            # Accept
            return new_c4
        else:
            # Revert
            edge_set.remove(e2)
            edge_set.add(e1)
            adj[u2].discard(v2)
            adj[v2].discard(u2)
            adj[u1].add(v1)
            adj[v1].add(u1)
            return None

    # Main local search loop
    while time.time() < deadline - 5:
        improved = False
        u1 = v1 = None  # for try_swap closure

        for _ in range(min(500, len(current_edges_list))):
            if time.time() > deadline - 5:
                break

            idx1 = random.randint(0, len(current_edges_list)-1)
            e1 = current_edges_list[idx1]
            u1, v1 = e1

            idx2 = random.randint(0, len(non_edges)-1)
            e2 = non_edges[idx2]

            if e2 in edge_set:
                continue

            # Perform swap
            edge_set.remove(e1)
            edge_set.add(e2)
            adj[u1].discard(v1)
            adj[v1].discard(u1)
            u2, v2 = e2
            adj[u2].add(v2)
            adj[v2].add(u2)

            new_c4 = count_c4(adj)

            if new_c4 < current_c4:
                current_c4 = new_c4
                if new_c4 < best_c4:
                    best_c4 = new_c4
                    best_edges = set(edge_set)
                # Update structures
                non_edges[idx2] = e1
                current_edges_list[idx1] = e2
                improved = True
                no_improve_count = 0
            else:
                # Revert
                edge_set.remove(e2)
                edge_set.add(e1)
                adj[u2].discard(v2)
                adj[v2].discard(u2)
                adj[u1].add(v1)
                adj[v1].add(u1)

            swap_count += 1

        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perturbation: do random swaps regardless
                for _ in range(10):
                    if time.time() > deadline - 5:
                        break
                    idx1 = random.randint(0, len(current_edges_list)-1)
                    e1 = current_edges_list[idx1]
                    idx2 = random.randint(0, len(non_edges)-1)
                    e2 = non_edges[idx2]
                    if e2 in edge_set:
                        continue
                    u1, v1 = e1
                    edge_set.remove(e1)
                    edge_set.add(e2)
                    adj[u1].discard(v1)
                    adj[v1].discard(u1)
                    u2, v2 = e2
                    adj[u2].add(v2)
                    adj[v2].add(u2)
                    non_edges[idx2] = e1
                    current_edges_list[idx1] = e2
                    current_c4 = count_c4(adj)
                no_improve_count = 0
                max_no_improve = min(max_no_improve + 500, 15000)
            elif no_improve_count > max_no_improve * 2:
                break
        else:
            max_no_improve = min(max_no_improve + 100, 15000)

    return [list(e) for e in best_edges]

if __name__ == "__main__":
    result = construct()
    print(f"Edges: {len(result)}")
