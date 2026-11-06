"""Strong edge colourings via exact maximum independent set on the conflict graph."""
import time
import sys
from random import Random

def construct():
    _start_time = time.time()
    _DEADLINE = 140.0  # safety margin before 150s kill

    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict_graph(edges):
        m = len(edges)
        # Map each vertex to the set of edge indices incident to it
        # Also need "joined by an edge" relation
        # For edge i=(u,v) and edge j=(x,y):
        #   conflict if {u,v} & {x,y} != empty (share vertex)
        #   OR if (u,x) in edge_set or (u,y) in edge_set or (v,x) in edge_set or (v,y) in edge_set
        edge_set = set(edges)
        # Build adjacency of vertices
        vert_adj = {}
        for u, v in edges:
            vert_adj.setdefault(u, set()).add(v)
            vert_adj.setdefault(v, set()).add(u)

        # Conflict matrix as list of bitmasks (or sets for small m)
        # m is at most ~n*d/2 * something; for n=80,d=6, edges ~ 240
        # Use Python sets of ints for clarity, then convert to int bitmasks
        conflicts = [set() for _ in range(m)]
        edge_to_idx = {e: i for i, e in enumerate(edges)}

        for i, (u, v) in enumerate(edges):
            for j in range(i + 1, m):
                x, y = edges[j]
                # Check share vertex
                if u == x or u == y or v == x or v == y:
                    conflicts[i].add(j)
                    conflicts[j].add(i)
                    continue
                # Check joined by an edge
                if (u, x) in edge_set or (u, y) in edge_set or (v, x) in edge_set or (v, y) in edge_set:
                    conflicts[i].add(j)
                    conflicts[j].add(i)

        return conflicts

    def greedy_coloring(conflicts, m):
        """Greedy: repeatedly pick a maximal independent set (greedy MIS), assign next color."""
        remaining = set(range(m))
        color = [-1] * m
        c = 0
        while remaining:
            # Greedy MIS: sort by degree in remaining conflict subgraph, pick lowest-degree first
            rem_list = list(remaining)
            rem_set = set(rem_list)
            # Compute degree within remaining
            deg = {v: len(conflicts[v] & rem_set) for v in rem_set}
            # Pick vertex with minimum degree
            independent = set()
            while rem_set:
                v = min(rem_set, key=lambda x: (deg[x], x))
                independent.add(v)
                rem_set.discard(v)
                # Remove neighbors from rem_set
                for nb in conflicts[v]:
                    if nb in rem_set:
                        rem_set.discard(nb)
                        deg.pop(nb, None)
                # Update degrees? For simplicity, just remove
            for v in independent:
                color[v] = c
            remaining -= independent
            c += 1
        return color, c

    def branch_and_bound_mis(conflict_sets, m, time_budget=None):
        """
        Find a maximum independent set in the graph defined by conflict_sets.
        Uses branch-and-bound with bitset representation.
        Returns the MIS as a set of vertices.
        """
        # Convert conflict sets to bitmasks
        # m can be up to ~250, so we use Python int as arbitrary-precision bitmask
        if m <= 64:
            pass  # could use numpy, but int is fine
        mask_all = (1 << m) - 1
        # adj_mask[i] = bitmask of vertices conflicting with i
        adj_mask = [0] * m
        for i in range(m):
            mask = 0
            for j in conflict_sets[i]:
                mask |= (1 << j)
            adj_mask[i] = mask

        best_set = set()
        best_size = 0

        # Upper bound: use greedy coloring of the induced subgraph on candidates
        # (number of colors in greedy coloring of conflict graph on S gives upper bound on alpha)
        def greedy_upper_bound(cand_mask):
            """Greedy coloring of the conflict graph induced by cand_mask.
            Returns number of colors used, which is an upper bound on alpha."""
            colors = 0
            remaining = cand_mask
            while remaining:
                colors += 1
                # Greedy coloring: repeatedly assign color to independent set
                # Actually, for upper bound on independent set size, we need
                # a coloring of the graph, where each color class is an independent set.
                # The number of colors is an upper bound on alpha.
                # Greedy: pick a vertex, color it, remove it and its non-neighbors? No.
                # Standard: color the graph. Each color class is an independent set.
                # alpha <= number of colors.
                # Greedy coloring: process vertices, assign smallest available color.
                # But we want a fast upper bound. Let's use a simple greedy:
                # Pick a maximal independent set, remove it, repeat. Count sets.
                # This gives an upper bound on alpha (since each set is independent,
                # and they partition the vertices, alpha <= total vertices, but that's trivial).
                # Wait: if we partition into k independent sets, then alpha >= max set size,
                # but alpha <= total. The number of colors k is NOT an upper bound on alpha.
                # Actually, chromatic number chi >= alpha is WRONG. chi >= n/alpha, and chi * alpha >= n.
                # So alpha <= n, and chi >= n/alpha => alpha >= n/chi.
                # So chi is a LOWER bound on n/alpha, i.e., alpha >= n/chi.
                # That's a lower bound, not upper.
                #
                # For an UPPER bound on alpha, we can use:
                # - Trivial: popcount(cand_mask)
                # - Greedy: repeatedly find a maximal clique in the COMPLEMENT, which is
                #   a maximal independent set in original. But that's what we're trying to do.
                #
                # Better upper bound: for each vertex v in cand, alpha <= 1 + alpha(N(v) complement in cand)?
                # Or use the "coloring bound": if we can color the graph with k colors,
                # then alpha <= k * (max size of color class)? No.
                #
                # Actually, a standard upper bound: alpha(G) <= sum over color classes in a proper
                # coloring of ... no.
                #
                # Let's use a simpler bound: for a set S, alpha(S) <= |S|.
                # And recursively, alpha(S) <= max over v in S of (1 + alpha(S \\ (N(v) ∪ {v})))
                # This is the recurrence for exact alpha, but as a bound, we can use:
                # alpha(S) <= 1 + alpha(S \\ N[v]) for any v, but that's not a bound, it's exact.
                #
                # For a fast upper bound, let's use: alpha(S) <= |S| / (min_degree_in_S + 1) ... no.
                #
                # Let's just use |S| as the upper bound, and rely on the B&B structure.
                return bin(remaining).count('1')

        sys.setrecursionlimit(100000)

        def bb(cand_mask, current_set, current_size):
            nonlocal best_set, best_size
            if time_budget is not None and time.time() > time_budget:
                return
            if cand_mask == 0:
                if current_size > best_size:
                    best_size = current_size
                    best_set = set(current_set)
                return
            # Upper bound
            ub = bin(cand_mask).count('1') + current_size
            if ub <= best_size:
                return
            # Pick a vertex to branch on
            # Heuristic: pick vertex with highest degree in cand
            v_list = []
            tmp = cand_mask
            while tmp:
                lsb = tmp & (-tmp)
                v = lsb.bit_length() - 1
                deg_in_cand = bin(adj_mask[v] & cand_mask).count('1')
                v_list.append((deg_in_cand, v))
                tmp &= tmp - 1
            # Sort by degree descending, pick first
            v_list.sort(reverse=True)
            v = v_list[0][1]
            v_bit = 1 << v

            # Branch 1: include v
            new_set = current_set + [v]
            # Remove v and its neighbors from cand
            new_cand = cand_mask & ~(adj_mask[v] | v_bit)
            bb(new_cand, new_set, current_size + 1)

            # Branch 2: exclude v
            new_cand2 = cand_mask & ~v_bit
            bb(new_cand2, current_set, current_size)

        bb(mask_all, [], 0)
        return best_set

    def solve_instance(edges):
        m = len(edges)
        if m == 0:
            return []
        conflicts = build_conflict_graph(edges)
        
        # First get a greedy solution
        color_greedy, num_colors_greedy = greedy_coloring(conflicts, m)
        
        # Try to improve with B&B on the conflict graph
        # Strategy: iteratively find a large MIS, assign a color, remove, repeat.
        # Use B&B to find near-maximal independent sets.
        
        # For time efficiency, use B&B with a time limit per MIS extraction
        color = [-1] * m
        remaining = set(range(m))
        c = 0
        best_total = num_colors_greedy
        best_color = color_greedy[:]
        
        time_per_mis = 2.0  # seconds per MIS extraction attempt
        
        while remaining:
            if time.time() > _DEADLINE - 5:
                break
            # Build conflict subgraph on remaining
            rem_list = sorted(remaining)
            rem_to_local = {v: i for i, v in enumerate(rem_list)}
            local_conflicts = [set() for _ in range(len(rem_list))]
            for i, vi in enumerate(rem_list):
                for j, vj in enumerate(rem_list):
                    if i < j and vj in conflicts[vi]:
                        local_conflicts[i].add(j)
                        local_conflicts[j].add(i)
            
            # Try B&B for MIS
            mis_time_limit = time.time() + min(time_per_mis, _DEADLINE - time.time() - 2)
            mis = branch_and_bound_mis(local_conflicts, len(rem_list), mis_time_limit)
            
            if not mis:
                # Fallback to greedy
                rem_set = set(range(len(rem_list)))
                independent = set()
                while rem_set:
                    v = min(rem_set, key=lambda x: (len(local_conflicts[x] & rem_set), x))
                    independent.add(v)
                    rem_set.discard(v)
                    for nb in local_conflicts[v]:
                        if nb in rem_set:
                            rem_set.discard(nb)
                mis = independent
            
            # Assign color
            for local_idx in mis:
                global_idx = rem_list[local_idx]
                color[global_idx] = c
            
            remaining -= set(rem_list[i] for i in mis)
            c += 1
        
        # Check if we improved
        if c < best_total:
            return color
        else:
            return best_color

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > _DEADLINE - 30:
            # Not enough time, use greedy for remaining
            edges = instance(*spec)
            if not edges:
                out.append([])
                continue
            conflicts = build_conflict_graph(edges)
            color, _ = greedy_coloring(conflicts, len(edges))
            out.append(color)
            continue
        edges = instance(*spec)
        if not edges:
            out.append([])
            continue
        color = solve_instance(edges)
        out.append(color)
    
    return out
