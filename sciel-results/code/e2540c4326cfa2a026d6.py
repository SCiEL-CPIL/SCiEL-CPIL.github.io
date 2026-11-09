import time
import random

def construct():
    start_time = time.time()
    deadline = 118  # safety margin before 150s kill

    def get_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def greedy_partition(n, edges, adj, strategy, rng):
        """Greedy clique packing with a given edge-selection strategy."""
        remaining_edges = set(edges)
        work_adj = [set(s) for s in adj]
        cliques = []

        while remaining_edges:
            if time.time() > deadline:
                break

            # Select next edge based on strategy
            if strategy == 0:  # min common neighbors
                best_edge = None
                best_common = float('inf')
                for e in remaining_edges:
                    u, v = e
                    common = len(work_adj[u] & work_adj[v])
                    if common < best_common:
                        best_common = common
                        best_edge = e
                        if common <= 0:
                            break
            elif strategy == 1:  # random edge
                best_edge = rng.choice(list(remaining_edges))
            elif strategy == 2:  # min degree endpoint
                best_edge = None
                best_deg = float('inf')
                for e in remaining_edges:
                    u, v = e
                    d = min(len(work_adj[u]), len(work_adj[v]))
                    if d < best_deg:
                        best_deg = d
                        best_edge = e
            elif strategy == 3:  # max common neighbors (dense first)
                best_edge = None
                best_common = -1
                for e in remaining_edges:
                    u, v = e
                    common = len(work_adj[u] & work_adj[v])
                    if common > best_common:
                        best_common = common
                        best_edge = e
            else:  # strategy 4: weighted random by common neighbors
                weights = []
                for e in remaining_edges:
                    u, v = e
                    weights.append(1 + len(work_adj[u] & work_adj[v]))
                total = sum(weights)
                r = rng.random() * total
                cum = 0
                best_edge = None
                for e, w in zip(remaining_edges, weights):
                    cum += w
                    if cum >= r:
                        best_edge = e
                        break
                if best_edge is None:
                    best_edge = next(iter(remaining_edges))

            if best_edge is None:
                break

            u, v = best_edge
            C = [u, v]
            candidates = work_adj[u] & work_adj[v]
            candidates = list(candidates)
            candidates.sort(key=lambda x: len(work_adj[x]), reverse=True)

            # Greedily expand clique
            changed = True
            while changed:
                changed = False
                for idx in range(len(candidates)):
                    cand = candidates[idx]
                    if all(cand in work_adj[c] for c in C):
                        C.append(cand)
                        changed = True
                        candidates = [x for x in candidates if x != cand and x in work_adj[cand]]
                        candidates.sort(key=lambda x: len(work_adj[x]), reverse=True)
                        break

            cliques.append(C)

            # Remove edges within C
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    a, b = C[i], C[j]
                    if a > b:
                        a, b = b, a
                    edge = (a, b)
                    if edge in remaining_edges:
                        remaining_edges.discard(edge)
                        work_adj[a].discard(b)
                        work_adj[b].discard(a)

        return cliques, remaining_edges

    def local_improve(cliques, n, adj):
        """Merge cliques that form a larger clique together."""
        if not cliques:
            return cliques

        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for i in range(len(cliques)):
                for j in range(i + 1, len(cliques)):
                    can_merge = True
                    for a in cliques[i]:
                        for b in cliques[j]:
                            if b not in adj[a]:
                                can_merge = False
                                break
                        if not can_merge:
                            break
                    if can_merge:
                        cliques[i] = cliques[i] + cliques[j]
                        del cliques[j]
                        improved = True
                        break
                if improved:
                    break
        return cliques

    def solve_instance(n, p, seed):
        edges = set(get_graph(n, p, seed))
        if not edges:
            return []

        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Multiple restarts with different strategies
        best_cliques = None
        best_count = float('inf')

        num_restarts = 12 if n <= 32 else 8
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            rng = random.Random(seed * 1000 + restart)
            strategy = restart % 5
            cliques, remaining = greedy_partition(n, edges, adj, strategy, rng)

            # If we have remaining edges, we need to handle them
            if remaining:
                # Add remaining edges as individual 2-cliques
                for e in remaining:
                    cliques.append(list(e))

            count = len(cliques)
            if count < best_count:
                best_count = count
                best_cliques = [list(c) for c in cliques]

        if best_cliques is None:
            # Fallback: just pair up edges
            best_cliques = [list(e) for e in edges]

        # Local improvement pass
        best_cliques = local_improve(best_cliques, n, adj)

        return best_cliques

    results = []
    # Instance 1
    results.append(solve_instance(32, 0.6, 601))

    # Instance 2
    results.append(solve_instance(64, 0.55, 602))

    return results
