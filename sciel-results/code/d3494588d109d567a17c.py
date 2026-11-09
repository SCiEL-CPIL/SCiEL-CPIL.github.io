import random
import time

def construct():
    start = time.time()
    deadline = 140  # safety margin before 150s kill

    def is_c4_free_complement(n, adj_bar):
        # adj_bar: list of sets, representing complement graph
        # G is C4-free iff for all u != v, |N_G(u) cap N_G(v)| <= 1
        # N_G(u) = V \ ({u} U N_bar(u))
        # |N_G(u) cap N_G(v)| = n - 2 - |N_bar(u) U N_bar(v)|
        # We need n - 2 - |N_bar(u) U N_bar(v)| <= 1
        # => |N_bar(u) U N_bar(v)| >= n - 3
        # => |N_bar(u)| + |N_bar(v)| - |N_bar(u) cap N_bar(v)| >= n - 3
        # => |N_bar(u) cap N_bar(v)| <= |N_bar(u)| + |N_bar(v)| - n + 3
        
        for u in range(n):
            for v in adj_bar[u]:
                if v > u:
                    common_bar = adj_bar[u] & adj_bar[v]
                    limit = len(adj_bar[u]) + len(adj_bar[v]) - n + 3
                    if len(common_bar) > limit:
                        return False
        return True

    def min_degree_g(n, adj_bar):
        # min degree in G = n - 1 - max degree in bar G
        max_deg_bar = max(len(adj_bar[i]) for i in range(n))
        return n - 1 - max_deg_bar

    def valid(n, adj_bar):
        if min_degree_g(n, adj_bar) < n - 39:
            return False
        return is_c4_free_complement(n, adj_bar)

    def to_edges(adj_bar, n):
        # Return edges of G
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if v not in adj_bar[u]:
                    edges.append([u, v])
        return edges

    best = {'n': 39, 'edges': []}

    # Try descending n from 47 to 40
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break
        
        found = False
        max_attempts = 5000
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Initialize random complement graph with max degree 38
            adj_bar = [set() for _ in range(n)]
            # Randomly add edges, respecting degree bound 38
            # To ensure good distribution, we can try to make it regular-ish
            # But random is fine for local search start
            
            # Let's try a random graph with edge probability p such that expected degree is ~38
            p = 38.0 / (n - 1)
            
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < p:
                        if len(adj_bar[u]) < 38 and len(adj_bar[v]) < 38:
                            adj_bar[u].add(v)
                            adj_bar[v].add(u)
            
            # Check if valid
            if valid(n, adj_bar):
                best = {'n': n, 'edges': to_edges(adj_bar, n)}
                found = True
                break
            
            # Local Search / Simulated Annealing
            # Cost function: number of violations in C4 condition
            def cost(adj_bar):
                c = 0
                for u in range(n):
                    for v in adj_bar[u]:
                        if v > u:
                            common_bar = adj_bar[u] & adj_bar[v]
                            limit = len(adj_bar[u]) + len(adj_bar[v]) - n + 3
                            if len(common_bar) > limit:
                                c += len(common_bar) - limit
                return c
            
            current_cost = cost(adj_bar)
            T = 1.0
            cooling = 0.999
            steps = 10000
            
            for step in range(steps):
                if time.time() > deadline:
                    break
                
                # Pick a random edge to remove or non-edge to add
                if random.random() < 0.5 and any(len(s) > 0 for s in adj_bar):
                    # Remove an edge
                    u = random.randint(0, n - 1)
                    if not adj_bar[u]:
                        continue
                    v = random.choice(list(adj_bar[u]))
                    adj_bar[u].remove(v)
                    adj_bar[v].remove(u)
                    new_cost = cost(adj_bar)
                    delta = new_cost - current_cost
                    if delta < 0 or random.random() < pow(2.718, -delta / T):
                        current_cost = new_cost
                    else:
                        # Revert
                        adj_bar[u].add(v)
                        adj_bar[v].add(u)
                else:
                    # Add an edge
                    # Find a pair with degree < 38
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u == v or v in adj_bar[u]:
                        continue
                    if len(adj_bar[u]) >= 38 or len(adj_bar[v]) >= 38:
                        continue
                    adj_bar[u].add(v)
                    adj_bar[v].add(u)
                    new_cost = cost(adj_bar)
                    delta = new_cost - current_cost
                    if delta < 0 or random.random() < pow(2.718, -delta / T):
                        current_cost = new_cost
                    else:
                        # Revert
                        adj_bar[u].remove(v)
                        adj_bar[v].remove(u)
                
                T *= cooling
                
                if current_cost == 0:
                    if valid(n, adj_bar):
                        best = {'n': n, 'edges': to_edges(adj_bar, n)}
                        found = True
                        break
            
            if found:
                break
        
        if found:
            break

    return best

# EVOLVE-BLOCK-END
