"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return edges, adj

    def is_clique(vertices, adj):
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                if vertices[j] not in adj[vertices[i]]:
                    return False
        return True

    def clique_edges(clique):
        return set((min(u, v), max(u, v)) for u, v in combinations(clique, 2))

    def greedy_partition(edges, adj, n, rng):
        """Greedy: repeatedly pick a random large clique, remove its edges."""
        remaining = set(edges)
        cliques = []
        # Precompute all edges for fast lookup
        while remaining:
            if time.time() > DEADLINE:
                break
            # Try to find a large clique by greedy expansion from a random vertex
            best_clique = None
            best_size = 0
            # Sample multiple starting vertices to find a good clique
            candidates = list(remaining)
            rng.shuffle(candidates)
            for _ in range(min(50, len(candidates))):
                edge = candidates[_]
                start_v = edge[0]
                # Greedy expand clique
                clique = set()
                # Try adding start_v and start_v's neighbors
                cand = {start_v}
                # Greedy: pick vertex with max degree within current candidate set
                while True:
                    if not cand:
                        break
                    # Find vertex in cand with most connections to all others in cand
                    best_v = None
                    best_deg = -1
                    for v in cand:
                        deg = sum(1 for u in cand if u != v and u in adj[v])
                        if deg > best_deg:
                            best_deg = deg
                            best_v = v
                    if best_v is None or best_deg != len(cand) - 1:
                        # Can't add more, or cand is empty
                        break
                    clique.add(best_v)
                    # Remove vertices not connected to best_v
                    cand = {u for u in cand if u != best_v and u in adj[best_v]}
                    if not cand:
                        break
                if len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = clique
            if best_clique is None or len(best_clique) < 2:
                # Fallback: just take one edge
                edge = next(iter(remaining))
                best_clique = {edge[0], edge[1]}
            # Remove edges covered by this clique
            ce = clique_edges(best_clique)
            remaining -= ce
            cliques.append(list(best_clique))
        return cliques

    def local_search(cliques, edges, adj, n, rng):
        """Try to merge or swap to reduce clique count."""
        cliques = [list(c) for c in cliques]
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            # Try merging pairs of cliques
            for i in range(len(cliques)):
                if time.time() > DEADLINE:
                    return cliques
                for j in range(i + 1, len(cliques)):
                    # Can we merge cliques[i] and cliques[j] into one clique?
                    merged = set(cliques[i]) | set(cliques[j])
                    if is_clique(list(merged), adj):
                        # Check that merged clique's edges are subset of union of their edges
                        merged_edges = clique_edges(list(merged))
                        union_edges = clique_edges(cliques[i]) | clique_edges(cliques[j])
                        if merged_edges <= union_edges:
                            # Safe merge
                            cliques = [c for k, c in enumerate(cliques) if k != i and k != j]
                            cliques.append(list(merged))
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                # Try edge swaps: move an edge from one clique to another if valid
                for i in range(len(cliques)):
                    if time.time() > DEADLINE:
                        return cliques
                    if len(cliques[i]) < 3:
                        continue
                    for j in range(len(cliques)):
                        if i == j:
                            continue
                        # Try removing one vertex from clique i and adding it to clique j
                        for v in cliques[i]:
                            new_i = [x for x in cliques[i] if x != v]
                            new_j = list(set(cliques[j]) | {v})
                            if len(new_i) < 2 or len(new_j) < 2:
                                continue
                            if not is_clique(new_i, adj) or not is_clique(new_j, adj):
                                continue
                            # Check edge coverage
                            new_i_edges = clique_edges(new_i)
                            new_j_edges = clique_edges(new_j)
                            old_i_edges = clique_edges(cliques[i])
                            old_j_edges = clique_edges(cliques[j])
                            # The moved vertex v's edges in old_i must be covered by new_j
                            # And v's edges in old_j (none, since v not in old_j) 
                            # Actually we need: old_i_edges - new_i_edges = edges incident to v in old_i
                            # These must be subset of new_j_edges
                            moved_edges = old_i_edges - new_i_edges
                            if moved_edges <= new_j_edges:
                                # Also check new_j_edges - old_j_edges = edges from v to old_j vertices
                                # These must not conflict... they're new edges in the partition
                                # But we need every edge in exactly one clique. 
                                # The edges from v to old_j vertices were NOT in any clique before
                                # (since v wasn't in clique j). So this is invalid unless
                                # those edges are already covered by another clique.
                                # This is complex, so skip for now.
                                pass
        return cliques

    def solve_instance(n, p, seed, rng):
        edges, adj = random_graph(n, p, seed)
        # Multiple restarts of greedy
        best_cliques = None
        best_count = float('inf')
        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() > DEADLINE:
                break
            cliques = greedy_partition(edges, adj, n, rng)
            # Verify coverage
            covered = set()
            for c in cliques:
                covered |= clique_edges(c)
            if covered == edges:
                if len(cliques) < best_count:
                    best_count = len(cliques)
                    best_cliques = cliques
                    # Local search
                    improved_cliques = local_search(best_cliques, edges, adj, n, rng)
                    if len(improved_cliques) < best_count:
                        best_count = len(improved_cliques)
                        best_cliques = improved_cliques
        if best_cliques is None:
            # Fallback: each edge as its own clique
            best_cliques = [[u, v] for u, v in edges]
        return best_cliques

    rng = random.Random(42)
    result1 = solve_instance(32, 0.6, 601, rng)
    result2 = solve_instance(64, 0.55, 602, rng)
    return [result1, result2]
# EVOLVE-BLOCK-END
