import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 18

    def is_independent(mask, adj):
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if adj[v] & (mask ^ lsb):
                return False
            m ^= lsb
        return True

    def can_add_edge(adj, u, v):
        if (adj[u] >> v) & 1:
            return True
        common = adj[u] & adj[v]
        return is_independent(common, adj)

    def largest_tf(adj, time_limit=None):
        best = [0]
        full_mask = (1 << n) - 1
        
        # Greedy Lower Bound
        degrees = [bin(adj[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda x: -degrees[x])
        
        greedy_set = 0
        for v in order:
            if is_independent(adj[v] & greedy_set, adj):
                greedy_set |= (1 << v)
        best[0] = bin(greedy_set).count('1')
        
        def bnb(S_mask, C_mask):
            if time_limit and time.time() > time_limit:
                return
            sz = bin(S_mask).count('1')
            if sz + bin(C_mask).count('1') <= best[0]:
                return
            
            if not C_mask:
                if sz > best[0]:
                    best[0] = sz
                return

            # Select vertex with max degree in C
            max_deg = -1
            chosen_v = -1
            c = C_mask
            while c:
                lsb = c & -c
                vid = lsb.bit_length() - 1
                deg_in_c = bin(adj[vid] & C_mask).count('1')
                if deg_in_c > max_deg:
                    max_deg = deg_in_c
                    chosen_v = vid
                c ^= lsb
                
            v = chosen_v
            v_bit = 1 << v
            
            bnb(S_mask, C_mask & ~v_bit)
            
            if is_independent(adj[v] & S_mask, adj):
                bnb(S_mask | v_bit, C_mask & ~v_bit)
                
        bnb(0, full_mask)
        return best[0]

    def get_edge_list(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def build_two_c9():
        """Build two disjoint 9-cycles: C9 on vertices 0-8 and C9 on vertices 9-17.
        This is K4-free (max degree 2, no triangles at all).
        Triangle-free induced subgraph = independent set.
        For C9, alpha = 4 (floor(9/2) = 4).
        For two disjoint C9, alpha = 4 + 4 = 8.
        """
        adj = [0] * n
        # First C9: vertices 0-8
        for i in range(9):
            u = i
            v = (i + 1) % 9
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        # Second C9: vertices 9-17
        for i in range(9):
            u = 9 + i
            v = 9 + ((i + 1) % 9)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def add_random_edges(adj, count, seed=None):
        """Try to add edges while maintaining K4-free property."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        all_missing = [(i, j) for i in range(n) for j in range(i + 1, n) 
                       if not ((adj[i] >> j) & 1)]
        rng.shuffle(all_missing)
        added = 0
        for u, v in all_missing:
            if added >= count:
                break
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                added += 1
        return adj

    def local_search_add_edges(adj, time_limit):
        """Try to add edges that reduce the triangle-free number."""
        score = largest_tf(adj, time_limit=time_limit)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            all_missing = [(i, j) for i in range(n) for j in range(i + 1, n) 
                           if not ((adj[i] >> j) & 1)]
            random.shuffle(all_missing)
            for u, v in all_missing:
                if time.time() > time_limit:
                    break
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    new_score = largest_tf(adj, time_limit=time_limit)
                    if new_score < score:
                        score = new_score
                        improved = True
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
        return adj, score

    random.seed(42)
    best_adj = None
    best_score = n

    # Strategy 1: Start with two disjoint C9 (score should be 8)
    adj_c9 = build_two_c9()
    score_c9 = largest_tf(adj_c9, time_limit=deadline - 5.0)
    best_score = score_c9
    best_adj = adj_c9[:]
    print(f"Two C9 baseline: {best_score}")

    # Try to improve by adding edges to the C9 structure
    if time.time() < deadline - 10.0:
        adj_work = adj_c9[:]
        adj_work, score_work = local_search_add_edges(adj_work, deadline - 5.0)
        if score_work < best_score:
            best_score = score_work
            best_adj = adj_work[:]
            print(f"Local search on C9: {best_score}")

    # Strategy 2: Random restarts with K4-free greedy construction
    def generate_random_k4_free():
        adj = [0] * n
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # Try multiple random restarts
    restarts = 0
    while time.time() < deadline - 10.0:
        restarts += 1
        adj = generate_random_k4_free()
        
        # Initial Evaluation
        score = largest_tf(adj, time_limit=deadline - 5.0)
        
        if score < best_score:
            best_score = score
            best_adj = adj[:]
            print(f"Restart {restarts}: New best {best_score}")
            
        # Local Search: try adding edges that reduce triangle-free number
        if time.time() < deadline - 5.0:
            adj_ls, score_ls = local_search_add_edges(adj, deadline - 5.0)
            if score_ls < best_score:
                best_score = score_ls
                best_adj = adj_ls[:]
                print(f"  Local Search Improvement: {best_score}")

        if time.time() > deadline - 10.0:
            break

    if best_adj is None:
        # Fallback to C9 construction
        best_adj = build_two_c9()

    edges = get_edge_list(best_adj)
    return edges
