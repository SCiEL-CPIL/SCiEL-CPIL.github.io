"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def build_partition(n, k, p, seed):
        deadline = time.time() + 140.0
        rng = Random(seed)

        # Generate cross edges
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))

        # Build adjacency for independent vertices
        indep_neighbors = {v: set() for v in range(k, n)}
        for (u, v) in cross_edges:
            indep_neighbors[v].add(u)

        # All edges in the graph
        all_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                all_edges.add((u, v))
        for (u, v) in cross_edges:
            all_edges.add((min(u, v), max(u, v)))

        # Structural approach:
        # 1. Assign each independent vertex v a subset S_v ⊆ N(v).
        # 2. Constraint: no internal edge (u,w) appears in two different S_v's.
        # 3. Cost = (# nonempty S_v) + (# internal edges not in any S_v)
        #    (each uncovered internal edge becomes a pair clique)
        #
        # Optimization: maximize sum_v C(|S_v|, 2) - (number of nonempty S_v)
        # because total = num_nonempty_Sv + (C(k,2) - sum_v C(|S_v|,2))
        #            = C(k,2) + num_nonempty_Sv - sum_v C(|S_v|,2)
        # So minimize: num_nonempty_Sv - sum_v C(|S_v|,2)
        # i.e., maximize: sum_v C(|S_v|,2) - num_nonempty_Sv

        # Phase 1: Greedy assignment with conflict resolution
        # Process vertices in order of decreasing |N(v)|
        indep_vertices = [v for v in range(k, n) if indep_neighbors[v]]
        indep_vertices.sort(key=lambda v: len(indep_neighbors[v]), reverse=True)

        # S_v for each independent vertex
        S_assign = {v: set() for v in range(k, n)}
        # Track which internal edge is claimed by which vertex
        edge_owner = {}  # (u,w) -> v

        for v in indep_vertices:
            if time.time() > deadline:
                break
            Nv = indep_neighbors[v]
            if not Nv:
                continue

            # Find largest S ⊆ Nv with no internal edge conflicts
            # Use greedy: start with all of Nv, remove conflicting vertices
            S = set(Nv)
            changed = True
            while changed:
                changed = False
                s_list = sorted(S)
                for i in range(len(s_list)):
                    for j in range(i + 1, len(s_list)):
                        e = (s_list[i], s_list[j])
                        if e in edge_owner:
                            # Conflict: remove the vertex that appears in fewer other S's
                            # For simplicity, remove s_list[j]
                            S.discard(s_list[j])
                            changed = True
                            break
                    if changed:
                        break

            S_assign[v] = S
            if S:
                s_list = sorted(S)
                for i in range(len(s_list)):
                    for j in range(i + 1, len(s_list)):
                        e = (s_list[i], s_list[j])
                        edge_owner[e] = v

        # Phase 2: Local search - try to improve by moving vertices between S_v's
        # or expanding S_v's
        best_cost = None
        # Compute initial cost
        def compute_cost():
            nonempty = sum(1 for v in range(k, n) if S_assign[v])
            covered_internal = sum(len(S_assign[v]) * (len(S_assign[v]) - 1) // 2 for v in range(k, n))
            return nonempty - covered_internal

        best_cost = compute_cost()

        # Local search: try moving a vertex from one S_v to another, or adding to S_v
        # We focus on: for each independent vertex v, try to add a vertex u to S_v
        # if u ∈ N(v) and no internal edge conflict
        # Also try: remove a vertex from S_v if it frees up edges for another vertex

        # Build reverse: for each u in [0,k), which S_v's contain u
        vertex_in_S = {u: [] for u in range(k)}
        for v in range(k, n):
            for u in S_assign[v]:
                vertex_in_S[u].append(v)

        improved = True
        max_iterations = 50
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            improved = False

            # Try adding vertices to S_v
            for v in indep_vertices:
                if time.time() > deadline:
                    break
                Nv = indep_neighbors[v]
                Sv = S_assign[v]
                # Try adding each u in Nv not in Sv
                candidates = [u for u in sorted(Nv) if u not in Sv]
                for u in candidates:
                    # Check if adding u to Sv creates conflicts
                    # New edges: (u, w) for w in Sv
                    conflict = False
                    for w in Sv:
                        e = (min(u, w), max(u, w))
                        if e in edge_owner and edge_owner[e] != v:
                            conflict = True
                            break
                    if not conflict:
                        # Adding u is valid. Check if it improves cost.
                        # Old contribution of v: C(|Sv|, 2)
                        # New contribution: C(|Sv|+1, 2)
                        # Improvement: |Sv|
                        # But we need to update edge_owner
                        old_size = len(Sv)
                        Sv.add(u)
                        for w in sorted(Sv - {u}):
                            e = (min(u, w), max(u, w))
                            edge_owner[e] = v
                        vertex_in_S[u].append(v)

                        new_cost = compute_cost()
                        if new_cost < best_cost:
                            best_cost = new_cost
                            improved = True
                        else:
                            # Revert
                            Sv.discard(u)
                            for w in sorted(Sv):
                                e = (min(u, w), max(u, w))
                                if e in edge_owner and edge_owner[e] == v:
                                    del edge_owner[e]
                            vertex_in_S[u].remove(v)

            # Try removing vertices from S_v to free up space
            # This is less likely to help, skip for now

            if not improved:
                break

        # Build final cliques
        cliques = []
        uncovered = set(all_edges)

        for v in range(k, n):
            Sv = S_assign[v]
            if Sv:
                clique = sorted([v] + list(Sv))
                cliques.append(clique)
                # Remove covered edges
                for u in Sv:
                    e = (min(v, u), max(v, u))
                    uncovered.discard(e)
                s_list = sorted(Sv)
                for i in range(len(s_list)):
                    for j in range(i + 1, len(s_list)):
                        e = (s_list[i], s_list[j])
                        uncovered.discard(e)

        # Cover remaining edges with pairs
        for e in sorted(uncovered):
            if time.time() > deadline:
                break
            u, w = e
            cliques.append([u, w])

        # Validate: ensure no clique has size < 2
        cliques = [c for c in cliques if len(c) >= 2]

        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(build_partition(*spec))

    return results
# EVOLVE-BLOCK-END
