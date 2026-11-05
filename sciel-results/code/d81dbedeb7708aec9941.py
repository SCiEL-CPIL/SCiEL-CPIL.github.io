"""Clique partitioning for two fixed dense random graphs."""
import time
import random
from itertools import combinations

DEADLINE = 140  # seconds, safety margin before 150s kill


def _generate_graph(n, p, seed):
    """Generate a random graph using the same RNG as the verifier."""
    rng = random.Random(seed)
    adj = [[False] * n for _ in range(n)]
    edge_set = set()
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u][v] = adj[v][u] = True
                edge_set.add((u, v))
    return adj, edge_set


def _bron_kerbosch_max_clique(adj, n, seed_vertex, deadline):
    """
    Find a maximum clique containing seed_vertex using Bron-Kerbosch with pivot.
    Returns the largest clique found that includes seed_vertex.
    """
    if time.time() > deadline:
        return [seed_vertex]

    # Convert adj to set of neighbors for fast lookup
    neighbors = [set() for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if adj[u][v]:
                neighbors[u].add(v)

    best_clique = [seed_vertex]
    best_size = 1

    # We want max clique containing seed_vertex.
    # Use Bron-Kerbosch on the subgraph induced by {seed_vertex} ∪ N(seed_vertex),
    # but force seed_vertex to be in the clique.
    # Equivalent: find max clique in N(seed_vertex) within the graph, then add seed_vertex.

    P = set(neighbors[seed_vertex])  # candidates that can join
    X = set()  # excluded (already processed)
    R = {seed_vertex}  # current clique

    def bk(P, X, R, deadline):
        nonlocal best_clique, best_size
        if time.time() > deadline:
            return
        if not P and not X:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return
        # Pivot selection: choose u from P ∪ X with max |P ∩ N(u)|
        candidates = P | X
        if not candidates:
            return
        pivot = max(candidates, key=lambda u: len(P & neighbors[u]))
        # Only iterate over P \ N(pivot)
        for v in list(P - neighbors[pivot]):
            if time.time() > deadline:
                return
            new_R = R | {v}
            new_P = P & neighbors[v]
            new_X = X & neighbors[v]
            bk(new_P, new_X, new_R, deadline)
            P.discard(v)
            X.add(v)

    bk(P, X, R, deadline)
    return best_clique


def _partition_edges(adj, edge_set, n, deadline):
    """
    Partition edges into cliques using:
    1. Repeatedly pick the edge with the most common neighbors (hardest to cover later),
       find the maximum clique containing both endpoints (via Bron-Kerbosch on common neighbors),
       add that clique, remove its edges.
    2. Remaining edges become 2-cliques.
    """
    cliques = []
    remaining_edges = set(edge_set)

    # Precompute neighbor sets for the full graph
    full_neighbors = [set() for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if adj[u][v]:
                full_neighbors[u].add(v)

    while remaining_edges and time.time() < deadline:
        # Score each remaining edge by number of common neighbors in the remaining graph
        # An edge (u,v) has "hardness" = |N(u) ∩ N(v) ∩ V_remaining|
        # where V_remaining are vertices incident to remaining edges.

        # Build set of vertices involved in remaining edges
        involved = set()
        for (u, v) in remaining_edges:
            involved.add(u)
            involved.add(v)

        if not involved:
            break

        # Find the edge with maximum common neighbors among involved vertices
        # For efficiency, compute for each edge: |N(u) ∩ N(v) ∩ involved|
        best_edge = None
        best_score = -1
        # To avoid O(E * n) per iteration, sample or use a heuristic
        # For small n (32, 64), we can afford iterating over all edges
        for (u, v) in remaining_edges:
            common = full_neighbors[u] & full_neighbors[v] & involved
            score = len(common)
            if score > best_score:
                best_score = score
                best_edge = (u, v)
            elif score == best_score and best_edge is not None:
                # Tie-break: prefer edge with lower total degree (more constrained)
                du = sum(1 for w in involved if adj[u][w])
                dv = sum(1 for w in involved if adj[v][w])
                bu, bv = best_edge
                dbu = sum(1 for w in involved if adj[bu][w])
                dbv = sum(1 for w in involved if adj[bv][w])
                if (du + dv) < (dbu + dbv):
                    best_edge = (u, v)

        if best_edge is None:
            break

        u, v = best_edge

        # Find maximum clique containing both u and v
        # Common neighbors of u and v in the remaining graph
        common = full_neighbors[u] & full_neighbors[v] & involved

        if len(common) >= 1:
            # Use Bron-Kerbosch to find max clique in common (which will be a clique
            # containing u and v when we add them back)
            # Actually, we want max clique in the subgraph induced by {u,v} ∪ common
            # that contains both u and v.
            # This is equivalent to: max clique in common's induced subgraph, then add {u,v}.
            # But we need the clique to be a valid clique in the ORIGINAL graph.
            # Since u and v are connected to all of common, any clique in common
            # combined with {u,v} is a clique.

            # Use Bron-Kerbosch on the subgraph induced by common
            # Find max clique within common
            if common:
                # Pick a seed from common for BK
                seed = max(common, key=lambda x: len(full_neighbors[x] & common))
                # Build local adjacency for common vertices
                local_adj = [[False] * n for _ in range(n)]
                for a in common:
                    for b in common:
                        if adj[a][b]:
                            local_adj[a][b] = True

                # Run BK to find max clique in common
                # Simple BK without pivot for small sets
                best_local = [seed]
                best_local_size = 1

                P = set(common) - {seed}
                X = set()
                R = {seed}

                def bk_local(P, X, R, deadline):
                    nonlocal best_local, best_local_size
                    if time.time() > deadline:
                        return
                    if not P and not X:
                        if len(R) > best_local_size:
                            best_local_size = len(R)
                            best_local = list(R)
                        return
                    candidates = P | X
                    if not candidates:
                        return
                    pivot = max(candidates, key=lambda w: len(P & full_neighbors[w]))
                    for w in list(P - full_neighbors[pivot]):
                        if time.time() > deadline:
                            return
                        new_R = R | {w}
                        new_P = P & full_neighbors[w]
                        new_X = X & full_neighbors[w]
                        bk_local(new_P, new_X, new_R, deadline)
                        P.discard(w)
                        X.add(w)

                bk_local(P, X, R, deadline)

                clique = [u, v] + best_local
            else:
                clique = [u, v]
        else:
            clique = [u, v]

        # Verify clique validity
        valid = True
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                a, b = clique[i], clique[j]
                if (a, b) not in remaining_edges and (b, a) not in remaining_edges:
                    valid = False
                    break
            if not valid:
                break

        if valid and len(clique) >= 2:
            cliques.append(clique)
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    a, b = clique[i], clique[j]
                    remaining_edges.discard((a, b))
                    remaining_edges.discard((b, a))
        else:
            # Fallback: just take the edge
            cliques.append([u, v])
            remaining_edges.discard((u, v))
            remaining_edges.discard((v, u))

    # Any remaining edges become 2-cliques
    for (u, v) in remaining_edges:
        cliques.append([u, v])

    return cliques


def construct():
    """
    Return two lists of cliques, one per instance.
    Each clique is a list of vertices forming a clique.
    Every edge appears in exactly one clique.
    Minimize total number of cliques.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        adj, edge_set = _generate_graph(n, p, seed)
        cliques = _partition_edges(adj, edge_set, n, deadline)
        results.append(cliques)

    return results


if __name__ == "__main__":
    res = construct()
    print(f"Instance 1: {len(res[0])} cliques")
    print(f"Instance 2: {len(res[1])} cliques")
    print(f"Total cliques: {len(res[0]) + len(res[1])}")
