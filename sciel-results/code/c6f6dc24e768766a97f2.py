"""Direct construction for dense odd-order graphs avoiding equal-degree path endpoints."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 85.0

    def is_valid(edge_set, n):
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for (u, v) in edge_set:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        for (u, v) in edge_set:
            degs_u = set(deg[s] for s in adj[u] if s != v)
            degs_v = set(deg[t] for t in adj[v] if t != u)
            if degs_u & degs_v:
                return False
        return True

    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def make_bipartite(a_size):
        edges = set()
        for i in range(a_size):
            for j in range(a_size, n):
                edges.add((i, j))
        return edges

    # Start with K_{7,8}
    best_edges = make_bipartite(7)
    best_count = len(best_edges)

    # Aggressive local search with 1-swap moves
    current = set(best_edges)
    current_count = len(current)
    
    max_iter = 0
    while time.time() < deadline - 5.0:
        max_iter += 1
        improved = False
        # Try all possible 1-swap moves systematically
        for rem in list(current):
            if time.time() > deadline - 5.0:
                break
            trial = current - {rem}
            for add in all_possible:
                if add in trial:
                    continue
                if is_valid(trial | {add}, n):
                    current = trial | {add}
                    current_count = len(current)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            # Try random 2-swap moves
            for _ in range(200):
                if time.time() > deadline - 5.0:
                    break
                if len(current) < 2:
                    break
                rem1, rem2 = random.sample(list(current), 2)
                trial = current - {rem1, rem2}
                missing = [e for e in all_possible if e not in trial]
                if len(missing) < 2:
                    continue
                add1, add2 = random.sample(missing, 2)
                new_graph = trial | {add1, add2}
                if len(new_graph) > current_count and is_valid(new_graph, n):
                    current = new_graph
                    current_count = len(current)
                    improved = True
                    break
        if not improved:
            # Try removing an edge and adding a different one (net zero, but may enable further gains)
            found = False
            for _ in range(50):
                if time.time() > deadline - 5.0:
                    break
                rem = random.choice(list(current))
                trial = current - {rem}
                missing = [e for e in all_possible if e not in trial]
                if not missing:
                    continue
                add = random.choice(missing)
                new_graph = trial | {add}
                if is_valid(new_graph, n):
                    # Check if this enables a subsequent gain
                    for add2 in all_possible:
                        if add2 in new_graph:
                            continue
                        if is_valid(new_graph | {add2}, n):
                            current = new_graph | {add2}
                            current_count = len(current)
                            found = True
                            break
                    if found:
                        break
            if not found:
                # No improvement possible, stop
                break

    if current_count > best_count:
        best_edges = current
        best_count = current_count

    # Final: try adding single edges
    for e in all_possible:
        if e in best_edges:
            continue
        if is_valid(best_edges | {e}, n):
            best_edges = best_edges | {e}
            best_count = len(best_edges)

    if not is_valid(best_edges, n):
        best_edges = make_bipartite(7)
        best_count = len(best_edges)

    result = [list(e) for e in sorted(best_edges)]
    final = []
    for e in result:
        final.append([int(e[0]), int(e[1])])
    return final

# EVOLVE-BLOCK-END
