import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill

    n = 15
    all_possible = set((i, j) for i in range(n) for j in range(i + 1, n))

    def is_valid(adj, degrees):
        """
        Check if the graph is valid: no simple path s-u-v-t with deg(s) == deg(t).
        adj: list of sets of neighbors
        degrees: list of degrees
        """
        for s in range(n):
            for t in range(n):
                if s == t:
                    continue
                if degrees[s] != degrees[t]:
                    continue
                # Check if there exists a path s-u-v-t (simple, length 3)
                for u in adj[s]:
                    if u == t:
                        continue
                    for v in adj[u]:
                        if v == s or v == t or v == u:
                            continue
                        if t in adj[v]:
                            return False
        return True

    def compute_adj_degrees(edges_set):
        adj = [set() for _ in range(n)]
        degrees = [0] * n
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
            degrees[u] += 1
            degrees[v] += 1
        return adj, degrees

    def try_add_edge(edges_set, edge):
        """Try adding an edge; return True if valid."""
        u, v = edge
        if u in edges_set and v in edges_set:
            pass
        new_edges = edges_set | {edge}
        adj, degrees = compute_adj_degrees(new_edges)
        return is_valid(adj, degrees)

    def greedy_add(edges_set):
        """Greedily add edges one by one."""
        improved = True
        while improved:
            improved = False
            missing = [e for e in all_possible if e not in edges_set]
            random.shuffle(missing)
            for e in missing:
                if try_add_edge(edges_set, e):
                    edges_set = edges_set | {e}
                    improved = True
                    break
        return edges_set

    def swap_search(edges_set, max_iters=50):
        """Try swap moves to escape local optima."""
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1

            if time.time() - start_time > deadline:
                return edges_set

            missing = [e for e in all_possible if e not in edges_set]
            existing = list(edges_set)
            random.shuffle(missing)
            random.shuffle(existing)

            for e_remove in existing:
                if time.time() - start_time > deadline:
                    return edges_set
                edges_without = edges_set - {e_remove}
                for e_add in missing:
                    if e_add == e_remove:
                        continue
                    new_edges = edges_without | {e_add}
                    adj, degrees = compute_adj_degrees(new_edges)
                    if is_valid(adj, degrees):
                        edges_set = new_edges
                        improved = True
                        break
                if improved:
                    break

            # After a swap, try to add more edges
            if improved:
                edges_set = greedy_add(edges_set)

        return edges_set

    def local_search(initial_edges):
        edges = set(tuple(sorted(e)) for e in initial_edges)

        # Ensure validity first by removing edges
        if not is_valid(*compute_adj_degrees(edges)):
            attempts = 0
            while not is_valid(*compute_adj_degrees(edges)) and attempts < 3000:
                e_to_remove = random.choice(list(edges))
                edges = edges - {e_to_remove}
                attempts += 1
            if not is_valid(*compute_adj_degrees(edges)):
                return None

        edges = greedy_add(edges)
        edges = swap_search(edges)
        return edges

    best_edges = None
    best_score = -1

    num_restarts = 40
    for i in range(num_restarts):
        if time.time() - start_time > deadline:
            break

        # Different initial densities
        if i < num_restarts // 3:
            density = 0.6 + 0.4 * random.random()
        elif i < 2 * num_restarts // 3:
            density = 0.4 + 0.6 * random.random()
        else:
            density = 0.2 + 0.8 * random.random()

        num_initial = int(len(all_possible) * density)
        initial_edges = random.sample(list(all_possible), num_initial)

        res_edges = local_search(initial_edges)

        if res_edges is not None:
            score = len(res_edges)
            if score > best_score:
                best_score = score
                best_edges = res_edges

        if best_score > 95:
            break

    if best_edges is None:
        return []

    result = [[int(u), int(v)] for u, v in best_edges]
    return result
