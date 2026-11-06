import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve(edges, time_budget):
        if not edges:
            return []

        n = max(max(e) for e in edges) + 1
        edge_set = set()
        for u, v in edges:
            edge_set.add((u, v) if u < v else (v, u))

        # Precompute edge-to-vertices mapping for fast checks
        cliques = []
        remaining_edges = set(edge_set)

        def get_adj(rem_edges):
            adj = [set() for _ in range(n)]
            for u, v in rem_edges:
                adj[u].add(v)
                adj[v].add(u)
            return adj

        def bron_kerbosch_maximal(P, R, X, adj, n, time_limit):
            """Yield maximal cliques using Bron-Kerbosch with pivot."""
            if time.time() > time_limit:
                return
            if not P and not X:
                yield R
                return

            # Choose pivot u from P ∪ X to minimize branching
            U = P | X
            if not U:
                return
            # Pick pivot with max degree in P
            pivot = max(U, key=lambda u: len(adj[u] & P))
            candidates = P - adj[pivot]

            for v in list(candidates):
                bron_kerbosch_maximal(P & adj[v], R | {v}, X & adj[v], adj, n, time_limit)
                P.remove(v)
                X.add(v)

        def find_best_clique(rem_edges, adj, time_limit):
            """Find a clique using the minimum common neighborhood heuristic."""
            if time.time() > time_limit:
                return None

            if not rem_edges:
                return None

            # Find the edge with the minimum number of common neighbors
            # This edge is the most "constrained" and must be covered soon
            min_common = float('inf')
            target_edge = None
            for u, v in rem_edges:
                # Calculate common neighbors in the remaining graph
                # adj[u] and adj[v] contain neighbors in the remaining graph
                common = adj[u] & adj[v]
                count = len(common)
                if count < min_common:
                    min_common = count
                    target_edge = (u, v)
                    # Early exit if we find an edge with 0 common neighbors (isolated edge in remaining graph)
                    if count == 0:
                        break

            if target_edge is None:
                return None

            u, v = target_edge
            
            # We want to find a maximal clique containing u and v.
            # Candidates are common neighbors of u and v.
            R = {u, v}
            P = adj[u] & adj[v]
            
            # Greedily expand the clique
            # We pick a vertex from P that is connected to all vertices in R.
            # Since P starts as N(u) & N(v), all v in P are connected to u and v.
            # As we add vertices to R, we must intersect P with N(new_vertex).
            
            while P:
                # Pick a vertex from P. 
                # Heuristic: pick the one with the highest degree in P
                best_v = max(P, key=lambda x: len(adj[x] & P))
                R.add(best_v)
                P = P & adj[best_v]
                
                if time.time() > time_limit:
                    break

            if len(R) < 2:
                return list(target_edge)
            
            return list(R)

        def local_search(cliques, remaining_edges, adj, time_limit):
            """Try to improve the partition by replacing cliques."""
            if time.time() > time_limit:
                return cliques, remaining_edges

            # For each clique, try replacing it with a better one
            improved = True
            while improved and time.time() < time_limit:
                improved = False
                for idx in range(len(cliques)):
                    if time.time() > time_limit:
                        break
                    # Temporarily remove this clique's edges
                    clique = cliques[idx]
                    clique_edges = set()
                    for u, v in combinations(clique, 2):
                        e = (u, v) if u < v else (v, u)
                        if e in remaining_edges:
                            clique_edges.add(e)
                        # Also check if it was already removed
                    # Actually we need to track which edges each clique covers
                    # Let's use a different approach: try to find a clique that covers
                    # some subset of edges not yet covered, and see if we can reduce total count

            return cliques, remaining_edges

        # Main loop: iteratively find best clique and remove its edges
        while remaining_edges and time.time() < deadline:
            adj = get_adj(remaining_edges)
            best = find_best_clique(remaining_edges, adj, deadline)

            if best is None or len(best) < 2:
                # Fallback
                e = next(iter(remaining_edges))
                best = list(e)

            # Remove edges in best clique
            # Optimization: only remove edges that are actually in the clique and in remaining_edges
            clique_set = set(best)
            for u in best:
                for v in best:
                    if u < v:
                        e = (u, v)
                        remaining_edges.discard(e)
            # Note: The above loop is O(k^2). We can optimize by iterating combinations.
            # But we must ensure we only discard valid edges.
            # Since best is a clique, all pairs are edges in the original graph.
            # And we only remove if in remaining_edges.
            
            cliques.append(best)

        # Post-processing: try to merge adjacent cliques or find better partitions
        # Simple local search: for pairs of cliques, check if their union is a clique
        if time.time() < deadline - 10:
            # Repeat merge passes until no more merges possible
            changed = True
            while changed and time.time() < deadline - 10:
                changed = False
                new_cliques = []
                used = [False] * len(cliques)
                for i in range(len(cliques)):
                    if used[i]:
                        continue
                    # Try to merge with other cliques
                    merged = set(cliques[i])
                    for j in range(i + 1, len(cliques)):
                        if used[j]:
                            continue
                        # Check if union is a clique
                        is_clique = True
                        for v in cliques[j]:
                            for u in merged:
                                if v == u:
                                    continue
                                e = (u, v) if u < v else (v, u)
                                if e not in edge_set:
                                    is_clique = False
                                    break
                            if not is_clique:
                                break
                        if is_clique:
                            merged.update(cliques[j])
                            used[j] = True
                            changed = True
                    new_cliques.append(list(merged))
                    used[i] = True

                # Verify: every edge must be in exactly one clique
                covered = set()
                valid = True
                for clique in new_cliques:
                    if len(clique) < 2:
                        valid = False
                        break
                    for u, v in combinations(clique, 2):
                        e = (u, v) if u < v else (v, u)
                        if e not in edge_set:
                            valid = False
                            break
                        if e in covered:
                            valid = False
                            break
                        covered.add(e)
                    if not valid:
                        break

                if valid and covered == edge_set:
                    cliques = new_cliques
                else:
                    break

        return cliques

    c1 = solve(random_graph(32, 0.6, 601), deadline)
    c2 = solve(random_graph(64, 0.55, 602), deadline)

    return [c1, c2]
