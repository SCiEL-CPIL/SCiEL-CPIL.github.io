import time
import random
from functools import lru_cache

def construct():
    start = time.time()
    DEADLINE = start + 135.0
    N = 16
    FULL = (1 << N) - 1

    def time_ok():
        return time.time() < DEADLINE

    def is_3_colorable_fast(adj, edge_count):
        """
        Optimized 3-colorability check using bitmask DP with memoization.
        """
        if edge_count <= 2 * N - 3:
            return True

        adj_mask = list(adj)
        
        # Order vertices by degree (descending)
        deg = [bin(adj_mask[i]).count('1') for i in range(N)]
        order = list(range(N))
        order.sort(key=lambda i: -deg[i])

        pos_in_order = [0] * N
        for i, v in enumerate(order):
            pos_in_order[v] = i

        ordered_adj = [0] * N
        for i, v in enumerate(order):
            m = 0
            for u in range(N):
                if (adj_mask[v] >> u) & 1:
                    m |= (1 << pos_in_order[u])
            ordered_adj[i] = m

        memo = {}

        def dfs(i, c0, c1, c2):
            if i == N:
                return True

            key = (i, c0, c1, c2)
            if key in memo:
                return memo[key]

            mask = ordered_adj[i]

            # Try color 0
            if (c0 & mask) == 0:
                if dfs(i + 1, c0 | (1 << i), c1, c2):
                    memo[key] = True
                    return True
            # Try color 1
            if (c1 & mask) == 0:
                if dfs(i + 1, c0, c1 | (1 << i), c2):
                    memo[key] = True
                    return True
            # Try color 2
            if (c2 & mask) == 0:
                if dfs(i + 1, c0, c1, c2 | (1 << i)):
                    memo[key] = True
                    return True

            memo[key] = False
            return False

        return dfs(0, 0, 0, 0)

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def edge_count_of(adj):
        cnt = 0
        for u in range(N):
            cnt += bin(adj[u]).count('1')
        return cnt // 2

    def edges_from_adj(adj):
        return [[u, v] for u in range(N) for v in range(u + 1, N) if (adj[u] >> v) & 1]

    best_edges = None
    best_count = -1

    # Clebsch graph seed
    clebsch_vertices = [i for i in range(32) if bin(i).count('1') % 2 == 0]
    clebsch_adj = [0] * N
    for i in range(N):
        for j in range(i + 1, N):
            xor_val = clebsch_vertices[i] ^ clebsch_vertices[j]
            if bin(xor_val).count('1') == 4:
                add_edge(clebsch_adj, i, j)

    if not is_3_colorable_fast(clebsch_adj, 40):
        best_count = 40
        best_edges = edges_from_adj(clebsch_adj)

    def local_search(adj, max_iter=1000):
        nonlocal best_count, best_edges
        ec = edge_count_of(adj)
        
        for _ in range(max_iter):
            if not time_ok():
                break
            improved = False
            swapped = False
            
            # 1. Try to add edges
            non_edges = [(u, v) for u in range(N) for v in range(u + 1, N) if not ((adj[u] >> v) & 1)]
            random.shuffle(non_edges)
            
            for u, v in non_edges:
                if (adj[u] & adj[v]) == 0:
                    add_edge(adj, u, v)
                    if not is_3_colorable_fast(adj, ec + 1):
                        ec += 1
                        improved = True
                        if ec > best_count:
                            best_count = ec
                            best_edges = edges_from_adj(adj)
                    else:
                        remove_edge(adj, u, v)
                        
            if improved:
                continue

            # 2. If no edge can be added, try swaps to escape local optimum
            # Pick a random non-edge and a random existing edge
            edges_list = edges_from_adj(adj)
            if non_edges and edges_list:
                u, v = random.choice(non_edges)
                x, y = random.choice(edges_list)
                
                # Check if adding (u,v) is triangle-free
                if (adj[u] & adj[v]) == 0:
                    # Perform swap
                    remove_edge(adj, x, y)
                    add_edge(adj, u, v)
                    
                    # Check 3-colorability
                    if not is_3_colorable_fast(adj, ec):
                        swapped = True
                        # After a successful swap, we might be able to add edges
                        # The loop will restart and try adding edges again
                        # We don't update best_count here as edge count is same
                    else:
                        # Revert
                        remove_edge(adj, u, v)
                        add_edge(adj, x, y)
            
            if not improved and not swapped:
                # Stuck, break
                break
                
        return adj

    rng = random.Random(42)
    attempts = 0

    while time_ok() and attempts < 5000:
        attempts += 1

        # Randomized greedy construction
        adj = [0] * N
        pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
        rng.shuffle(pairs)
        for u, v in pairs:
            if not ((adj[u] >> v) & 1):
                if (adj[u] & adj[v]) == 0:
                    add_edge(adj, u, v)

        ec = edge_count_of(adj)
        if is_3_colorable_fast(adj, ec):
            continue

        if ec > best_count:
            best_count = ec
            best_edges = edges_from_adj(adj)

        # Local search with swaps
        adj = local_search(adj, max_iter=500)
        ec = edge_count_of(adj)
        if ec > best_count:
            best_count = ec
            best_edges = edges_from_adj(adj)

        # Also try local search from the Clebsch graph with random perturbations
        if attempts % 10 == 0:
            adj2 = clebsch_adj[:]
            for _ in range(rng.randint(0, 5)):
                edges_list = edges_from_adj(adj2)
                if edges_list:
                    u, v = rng.choice(edges_list)
                    remove_edge(adj2, u, v)
            adj2 = local_search(adj2, max_iter=200)
            ec2 = edge_count_of(adj2)
            if ec2 > best_count:
                best_count = ec2
                best_edges = edges_from_adj(adj2)

    if best_edges is not None:
        return best_edges

    return edges_from_adj(clebsch_adj)
