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
            """Find the maximal clique covering the most remaining edges."""
            if time.time() > time_limit:
                return None

            best_clique = None
            best_edges_covered = 0

            # Get active vertices
            active = set()
            for u, v in rem_edges:
                active.add(u)
                active.add(v)

            if not active:
                return None

            # Try Bron-Kerbosch to find maximal cliques
            P = set(active)
            X = set()
            R = set()

            # Limit enumeration time
            enum_deadline = min(time_limit, time.time() + 2.0)
            all_maximal = []

            def bk(P, R, X):
                if time.time() > enum_deadline:
                    return
                if not P and not X:
                    if len(R) >= 2:
                        all_maximal.append(frozenset(R))
                    return
                U = P | X
                if not U:
                    return
                pivot = max(U, key=lambda u: len(adj[u] & P))
                candidates = P - adj[pivot]
                for v in list(candidates):
                    bk(P & adj[v], R | {v}, X & adj[v])
                    P.discard(v)
                    X.add(v)

            bk(P, R, X)

            # Also add some greedy cliques for diversity
            greedy_cliques = []
            for _ in range(3):
                # Greedy: start from random active vertex
                if not active:
                    break
                start = random.choice(list(active))
                clique = {start}
                candidates = set(adj[start] & active)
                while candidates:
                    # Pick candidate with max overlap with current clique neighbors
                    best_v = max(candidates, key=lambda u: len(adj[u] & clique))
                    clique.add(best_v)
                    candidates = candidates & adj[best_v]
                if len(clique) >= 2:
                    greedy_cliques.append(frozenset(clique))

            # Evaluate all candidate cliques by edges covered
            candidates_list = list(set(all_maximal + greedy_cliques))

            for clique in candidates_list:
                if len(clique) < 2:
                    continue
                # Count how many remaining edges are in this clique
                covered = 0
                clique_list = list(clique)
                for i in range(len(clique_list)):
                    for j in range(i + 1, len(clique_list)):
                        u, v = clique_list[i], clique_list[j]
                        e = (u, v) if u < v else (v, u)
                        if e in rem_edges:
                            covered += 1
                if covered > best_edges_covered:
                    best_edges_covered = covered
                    best_clique = list(clique)

            # Fallback: greedy if no good clique found
            if best_clique is None or best_edges_covered == 0:
                # Just take one edge
                e = next(iter(rem_edges))
                best_clique = list(e)

            return best_clique

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
            for u, v in combinations(best, 2):
                e = (u, v) if u < v else (v, u)
                remaining_edges.discard(e)

            cliques.append(best)

        # Post-processing: try to merge adjacent cliques or find better partitions
        # Simple local search: for pairs of cliques, check if their union is a clique
        if time.time() < deadline - 10:
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
                    candidate = merged | set(cliques[j])
                    is_clique = True
                    cand_list = list(candidate)
                    for a in range(len(cand_list)):
                        for b in range(a + 1, len(cand_list)):
                            u, v = cand_list[a], cand_list[b]
                            e = (u, v) if u < v else (v, u)
                            # Check if edge exists in original graph
                            if e not in edge_set:
                                is_clique = False
                                break
                        if not is_clique:
                            break
                    if is_clique:
                        merged = candidate
                        used[j] = True
                new_cliques.append(list(merged))
                used[i] = True

            # Verify: every edge must be in exactly one clique
            # Rebuild from new_cliques and check coverage
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
                # Keep original
                pass

        return cliques

    c1 = solve(random_graph(32, 0.6, 601), deadline)
    c2 = solve(random_graph(64, 0.55, 602), deadline)

    return [c1, c2]
