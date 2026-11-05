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

    def greedy_tf(adj):
        """Fast heuristic for triangle-free induced subgraph size."""
        degrees = [bin(adj[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda x: -degrees[x])
        
        greedy_set = 0
        for v in order:
            # Check if v can be added to greedy_set
            # v can be added if it has no neighbors in greedy_set
            # Wait, triangle-free induced subgraph means no edges between any pair in the set.
            # So it IS an independent set.
            # The current code in the prompt uses `is_independent(adj[v] & greedy_set, adj)`.
            # `adj[v] & greedy_set` is the set of neighbors of v that are in greedy_set.
            # If this set is independent, it means v has no neighbors in greedy_set?
            # No. `is_independent(S)` checks if S contains no edges.
            # If `adj[v] & greedy_set` is independent, it doesn't mean v is isolated from greedy_set.
            # It means the *neighbors* of v in greedy_set don't form a triangle among themselves?
            # That is NOT the condition for v to be added to an independent set.
            # Condition for v to be added to independent set I: v has no neighbors in I.
            # i.e., `adj[v] & I == 0`.
            
            # Let's re-read the prompt's `largest_tf` greedy part:
            # `if is_independent(adj[v] & greedy_set, adj):`
            # This logic seems flawed for finding an independent set.
            # If I want to build an independent set, I just check `if adj[v] & greedy_set == 0`.
            # Why did the prompt use `is_independent`?
            # Maybe it was trying to build a triangle-free set, not an independent set?
            # "Triangle-free induced subgraph" = Independent Set.
            # So the greedy check should be `if adj[v] & greedy_set == 0`.
            
            # Let's fix the greedy heuristic to be correct and fast.
            if adj[v] & greedy_set == 0:
                greedy_set |= (1 << v)
        return bin(greedy_set).count('1')

    def largest_tf(adj, time_limit=None):
        best = [0]
        full_mask = (1 << n) - 1
        
        # Greedy Lower Bound (Corrected)
        best[0] = greedy_tf(adj)
        
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
            
            # Branch 1: Exclude v
            bnb(S_mask, C_mask & ~v_bit)
            
            # Branch 2: Include v
            # If we include v, we must exclude all its neighbors in C
            # And we must ensure v is not adjacent to any vertex in S (independent set constraint)
            if not (adj[v] & S_mask):
                bnb(S_mask | v_bit, C_mask & ~v_bit & ~adj[v])
                
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
        adj = [0] * n
        for i in range(9):
            u = i
            v = (i + 1) % 9
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        for i in range(9):
            u = 9 + i
            v = 9 + ((i + 1) % 9)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def generate_random_k4_free():
        adj = [0] * n
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def local_search_add_edges(adj, time_limit):
        """Optimized local search using greedy filter."""
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
                    # Tentatively add
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
                    # Fast check
                    g_score = greedy_tf(adj)
                    if g_score < score:
                        # Slow check
                        new_score = largest_tf(adj, time_limit=time_limit)
                        if new_score < score:
                            score = new_score
                            improved = True
                        else:
                            # Revert
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                    else:
                        # Revert
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
        return adj, score

    def shake_and_refill(adj, time_limit):
        """Remove some edges and refill greedily."""
        adj_work = adj[:]
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n) 
                     if (adj_work[i] >> j) & 1]
        if not all_edges:
            return adj_work
        # Remove 5-10 random edges
        k = random.randint(5, 10)
        to_remove = random.sample(all_edges, min(k, len(all_edges)))
        for u, v in to_remove:
            adj_work[u] &= ~(1 << v)
            adj_work[v] &= ~(1 << u)
        
        # Refill greedily
        all_missing = [(i, j) for i in range(n) for j in range(i + 1, n) 
                       if not ((adj_work[i] >> j) & 1)]
        random.shuffle(all_missing)
        for u, v in all_missing:
            if can_add_edge(adj_work, u, v):
                adj_work[u] |= (1 << v)
                adj_work[v] |= (1 << u)
                
        return adj_work

    random.seed(42)
    best_adj = None
    best_score = n

    # Strategy 1: Start with two disjoint C9 (score should be 8)
    adj_c9 = build_two_c9()
    score_c9 = largest_tf(adj_c9, time_limit=deadline - 5.0)
    best_score = score_c9
    best_adj = adj_c9[:]
    print(f"Two C9 baseline: {best_score}")

    # Local search on C9
    if time.time() < deadline - 10.0:
        adj_work = adj_c9[:]
        adj_work, score_work = local_search_add_edges(adj_work, deadline - 5.0)
        if score_work < best_score:
            best_score = score_work
            best_adj = adj_work[:]
            print(f"Local search on C9: {best_score}")

    # Strategy 2: Random restarts
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
            
        # Local Search
        if time.time() < deadline - 5.0:
            adj_ls, score_ls = local_search_add_edges(adj, deadline - 5.0)
            if score_ls < best_score:
                best_score = score_ls
                best_adj = adj_ls[:]
                print(f"  Local Search Improvement: {best_score}")
        
        # Shake and Refill
        if time.time() < deadline - 5.0 and best_score > 5:
            adj_shake = shake_and_refill(best_adj, deadline - 5.0)
            score_shake = largest_tf(adj_shake, time_limit=deadline - 5.0)
            if score_shake < best_score:
                best_score = score_shake
                best_adj = adj_shake[:]
                print(f"  Shake Improvement: {best_score}")
                # Local search on shaken
                adj_ls2, score_ls2 = local_search_add_edges(adj_shake, deadline - 5.0)
                if score_ls2 < best_score:
                    best_score = score_ls2
                    best_adj = adj_ls2[:]
                    print(f"  Post-Shake LS: {best_score}")

    if best_adj is None:
        best_adj = build_two_c9()

    edges = get_edge_list(best_adj)
    return edges
