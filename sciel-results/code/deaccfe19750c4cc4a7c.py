import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110.0

    best_edges = []
    best_count = 0

    def count_edges(adj):
        return sum(len(adj[i]) for i in range(n)) // 2

    def check_violation(adj):
        # Returns True if valid (no K3,3)
        for u in range(n):
            for v in range(u + 1, n):
                inter_uv = adj[u] & adj[v]
                if len(inter_uv) < 3:
                    continue
                for w in inter_uv:
                    # w is in N(u) and N(v)
                    # Check if |N(u) \cap N(v) \cap N(w)| >= 3
                    # This is |inter_uv \cap N(w)| >= 3
                    if len(inter_uv & adj[w]) >= 3:
                        return False
        return True

    def find_violation_edge(adj):
        # Returns an edge (u, v) whose removal might help, or None
        # We return an edge from a violating K3,3
        for u in range(n):
            for v in range(u + 1, n):
                inter_uv = adj[u] & adj[v]
                if len(inter_uv) < 3:
                    continue
                for w in inter_uv:
                    common = inter_uv & adj[w]
                    if len(common) >= 3:
                        # Pick a random neighbor c from common to break the link with u
                        # Removing (u, c) breaks the K3,3
                        c = next(iter(common))
                        return (u, c)
        return None

    def improve(adj):
        # Greedy addition of edges
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Get missing edges
            missing = [(i, j) for i in range(n) for j in range(i + 1, n) if j not in adj[i]]
            random.shuffle(missing)
            for u, v in missing:
                if time.time() > deadline:
                    break
                # Try adding
                adj[u].add(v)
                adj[v].add(u)
                if check_violation(adj):
                    improved = True
                else:
                    # Revert
                    adj[u].remove(v)
                    adj[v].remove(u)
        return adj

    # Main loop: Multi-restart
    while time.time() < deadline:
        # Random initialization
        p = random.uniform(0.4, 0.8)
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i].add(j)
                    adj[j].add(i)

        # Remove violations
        # Loop until valid
        while True:
            if time.time() > deadline:
                break
            edge_to_remove = find_violation_edge(adj)
            if edge_to_remove is None:
                break
            u, v = edge_to_remove
            adj[u].remove(v)
            adj[v].remove(u)

        # Improve
        adj = improve(adj)

        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_edges = [[u, v] for u in range(n) for v in adj[u] if u < v]

    # If no valid graph found (should not happen), return seed
    if not best_edges:
        return [[a, b] for a in range(2) for b in range(2, 20)]

    return best_edges
