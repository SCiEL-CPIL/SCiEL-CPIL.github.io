"""Clique partition solver: greedy clique growth with multi-start and local refinement."""
import time
import random
import sys
from itertools import combinations

def construct():
    t0 = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adjacency(n, edges):
        adj = [set() for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))
        return adj, edge_set

    def greedy_grow_clique(start_edge, remaining_edges, adj, n, rng):
        """Grow a clique starting from start_edge, picking vertices that maximize internal density."""
        u, v = start_edge
        clique = [u, v]
        # Candidates: common neighbors of u and v in remaining graph
        cand_set = set(adj[u]) & set(adj[v])
        cand_set.discard(u)
        cand_set.discard(v)
        # Filter: must have edges to all clique members in remaining_edges
        def valid(w, clique_list):
            for c in clique_list:
                e = (min(w, c), max(w, c))
                if e not in remaining_edges:
                    return False
            return True
        valid_cands = set(w for w in cand_set if valid(w, clique))
        while valid_cands:
            # Score each candidate: number of edges to other valid candidates in remaining graph
            best_w = None
            best_score = -1
            for w in valid_cands:
                score = 0
                for x in valid_cands:
                    if w == x:
                        continue
                    e = (min(w, x), max(w, x))
                    if e in remaining_edges:
                        score += 1
                # Small random tie-breaker for diversity
                score += rng.random() * 0.01
                if score > best_score:
                    best_score = score
                    best_w = w
            if best_w is None:
                break
            clique.append(best_w)
            valid_cands.discard(best_w)
            # Update valid_cands: must be connected to new vertex in remaining_edges
            valid_cands = set(w for w in valid_cands if (min(w, best_w), max(w, best_w)) in remaining_edges)
        return clique

    def solve_instance(n, p, seed, time_budget):
        edges = random_graph(n, p, seed)
        adj, edge_set = build_adjacency(n, edges)
        total_edges = len(edges)

        # Fallback: each edge is a clique
        best_partition = [list(e) for e in edges]
        best_count = len(best_partition)

        end_time = t0 + time_budget

        # Multi-start greedy
        max_attempts = 30
        for attempt in range(max_attempts):
            if time.time() > end_time:
                break
            attempt_rng = random.Random(seed + attempt * 7919 + 12345)
            remaining_edges = set(edge_set)
            partition = []

            while remaining_edges:
                if time.time() > end_time:
                    break
                # Pick a random seed edge from remaining
                edge_list = list(remaining_edges)
                attempt_rng.shuffle(edge_list)
                seed_edge = edge_list[0]

                # Try growing from this seed edge
                clique = greedy_grow_clique(seed_edge, remaining_edges, adj, n, attempt_rng)

                if len(clique) < 2:
                    # Just take the seed edge
                    clique = list(seed_edge)

                # Remove all edges covered by this clique from remaining
                cl_sorted = sorted(clique)
                for i in range(len(cl_sorted)):
                    for j in range(i + 1, len(cl_sorted)):
                        e = (cl_sorted[i], cl_sorted[j])
                        remaining_edges.discard(e)

                partition.append(cl_sorted)

            # Check completeness
            if not remaining_edges:
                count = len(partition)
                if count < best_count:
                    best_count = count
                    best_partition = [list(c) for c in partition]
            else:
                # Add remaining edges as singletons
                for e in remaining_edges:
                    partition.append(list(e))
                count = len(partition)
                if count < best_count:
                    best_count = count
                    best_partition = [list(c) for c in partition]

        # Local search: try to merge adjacent cliques if union is a clique
        improved = True
        while improved and time.time() < end_time:
            improved = False
            new_partition = []
            i = 0
            while i < len(best_partition):
                c1 = set(best_partition[i])
                j = i + 1
                merged_any = False
                while j < len(best_partition):
                    c2 = set(best_partition[j])
                    union = c1 | c2
                    # Check if union forms a clique
                    is_clique = True
                    ul = sorted(union)
                    for a in range(len(ul)):
                        for b in range(a + 1, len(ul)):
                            if (ul[a], ul[b]) not in edge_set:
                                is_clique = False
                                break
                        if not is_clique:
                            break
                    if is_clique:
                        new_partition.append(ul)
                        merged_any = True
                        c1 = union
                        j += 1
                    else:
                        j += 1
                if not merged_any:
                    new_partition.append(sorted(c1))
                i += 1
            if len(new_partition) < len(best_partition):
                best_partition = new_partition
                improved = True

        return best_partition

    results = []
    # Split time budget: give more to the larger instance
    time_for_32 = min(40.0, (DEADLINE - (time.time() - t0)) * 0.3)
    time_for_64 = min(DEADLINE - (time.time() - t0), (DEADLINE - (time.time() - t0)) * 0.7)

    results.append(solve_instance(32, 0.6, 601, time_for_32))
    results.append(solve_instance(64, 0.55, 602, time_for_64))

    return results
