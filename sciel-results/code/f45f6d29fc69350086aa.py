"""
K2,3-free graph construction on 20 vertices.
Uses a structured construction based on complete bipartite-like patterns,
combined with aggressive local search via simulated annealing and 2-opt moves.
"""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    n = 20
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def common_count(adj, u, v):
        return bin(adj[u] & adj[v]).count('1')

    def is_valid_add(adj, u, v):
        """Check if adding edge (u,v) maintains K2,3-free property."""
        # If u and v already share >2 common neighbors, can't add
        if common_count(adj, u, v) > 2:
            return False
        # For each common neighbor w of u and v, adding (u,v) creates a new common neighbor pair
        # Actually, we need to check: after adding (u,v), no two vertices have >2 common neighbors
        # Adding (u,v) affects:
        # 1. common_count(u, v) increases by 1 - already checked
        # 2. For each neighbor w of u (w != v), common_count(v, w) increases by 1 (u becomes common neighbor of v and w)
        #    So we need common_count(v, w) < 2 for all w in N(u) \ {v}
        # 3. For each neighbor w of v (w != u), common_count(u, w) increases by 1 (v becomes common neighbor of u and w)
        #    So we need common_count(u, w) < 2 for all w in N(v) \ {u}
        
        nv = adj[v]
        nu = adj[u]
        
        # Check neighbors of u (excluding v)
        w = 0
        mask_u = nu & ~(1 << v)
        while mask_u:
            low = mask_u & (-mask_u)
            w = low.bit_length() - 1
            # After adding (u,v), common neighbors of v and w increase by 1 (u is now neighbor of both)
            if common_count(adj, v, w) >= 2:
                return False
            mask_u ^= low
        
        # Check neighbors of v (excluding u)
        mask_v = nv & ~(1 << u)
        while mask_v:
            low = mask_v & (-mask_v)
            w = low.bit_length() - 1
            # After adding (u,v), common neighbors of u and w increase by 1 (v is now neighbor of both)
            if common_count(adj, u, w) >= 2:
                return False
            mask_v ^= low
        
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def count_edges(adj):
        c = 0
        for i in range(n):
            c += bin(adj[i]).count('1')
        return c // 2

    def is_valid_graph(adj):
        """Check if the graph is K2,3-free."""
        for i in range(n):
            for j in range(i + 1, n):
                if common_count(adj, i, j) > 2:
                    return False
        return True

    def greedy_construct(adj, order):
        """Add edges in the given order if valid."""
        for u, v in order:
            if not ((adj[u] >> v) & 1):
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)

    def min_common_greedy(adj):
        """Greedy: repeatedly add the edge with minimum common neighbors that is valid."""
        remaining = [(i, j) for i in range(n) for j in range(i + 1, n) 
                     if not ((adj[i] >> j) & 1)]
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            best_pair = None
            best_cn = n + 1
            for u, v in remaining:
                if (adj[u] >> v) & 1:
                    continue
                cn = common_count(adj, u, v)
                if cn < best_cn:
                    best_cn = cn
                    best_pair = (u, v)
                if cn > 2:
                    continue
            if best_pair is not None and best_cn <= 2:
                u, v = best_pair
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    remaining.remove(best_pair)
                    changed = True
        # Final pass
        for u, v in remaining:
            if not ((adj[u] >> v) & 1):
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)

    def simulated_annealing(adj, max_iters=5000):
        """
        Simulated annealing: try random moves (add/remove/swap edges) with acceptance
        based on energy change.
        """
        current_adj = adj[:]
        current_count = count_edges(current_adj)
        best_adj = adj[:]
        best_count = current_count
        
        T = 10.0
        T_min = 0.01
        cooling = 0.9995
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Generate a random move
            move_type = random.choice(['add', 'remove', 'swap'])
            backup = current_adj[:]
            
            if move_type == 'add':
                # Try to add a random missing edge
                missing = [(i, j) for i in range(n) for j in range(i + 1, n)
                          if not ((current_adj[i] >> j) & 1)]
                if not missing:
                    continue
                u, v = random.choice(missing)
                if is_valid_add(current_adj, u, v):
                    add_edge(current_adj, u, v)
                    delta = 1
                else:
                    continue
            elif move_type == 'remove':
                # Try to remove a random existing edge
                existing = [(i, j) for i in range(n) for j in range(i + 1, n)
                           if (current_adj[i] >> j) & 1]
                if not existing:
                    continue
                u, v = random.choice(existing)
                remove_edge(current_adj, u, v)
                delta = -1
            else:  # swap
                # Remove one edge, add another
                existing = [(i, j) for i in range(n) for j in range(i + 1, n)
                           if (current_adj[i] >> j) & 1]
                missing = [(i, j) for i in range(n) for j in range(i + 1, n)
                          if not ((current_adj[i] >> j) & 1)]
                if not existing or not missing:
                    continue
                u1, v1 = random.choice(existing)
                u2, v2 = random.choice(missing)
                remove_edge(current_adj, u1, v1)
                if is_valid_add(current_adj, u2, v2):
                    add_edge(current_adj, u2, v2)
                    delta = 0
                else:
                    current_adj = backup[:]
                    continue
            
            new_count = count_edges(current_adj)
            delta = new_count - current_count
            
            # Accept or reject
            if delta > 0 or (T > T_min and random.random() < pow(2.71828, delta / T)):
                current_count = new_count
                if new_count > best_count:
                    best_count = new_count
                    best_adj = current_adj[:]
            else:
                current_adj = backup[:]
            
            T *= cooling
        
        return best_adj, best_count

    def two_opt_search(adj, max_iters=3000):
        """
        2-opt: try all pairs of edges to remove and add.
        """
        best_adj = adj[:]
        best_count = count_edges(best_adj)
        current_adj = adj[:]
        no_improve = 0
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            existing = [(i, j) for i in range(n) for j in range(i + 1, n)
                       if (current_adj[i] >> j) & 1]
            missing = [(i, j) for i in range(n) for j in range(i + 1, n)
                      if not ((current_adj[i] >> j) & 1)]
            
            if not existing or not missing:
                break
            
            # Try a few random swaps
            improved = False
            for _ in range(20):
                u1, v1 = random.choice(existing)
                u2, v2 = random.choice(missing)
                
                backup = current_adj[:]
                remove_edge(current_adj, u1, v1)
                if is_valid_add(current_adj, u2, v2):
                    add_edge(current_adj, u2, v2)
                    new_count = count_edges(current_adj)
                    if new_count > best_count:
                        best_count = new_count
                        best_adj = current_adj[:]
                        no_improve = 0
                        improved = True
                        break
                    else:
                        current_adj = backup[:]
                else:
                    current_adj = backup[:]
            
            if not improved:
                no_improve += 1
                if no_improve > 100:
                    break
            else:
                no_improve = 0
        
        return best_adj, best_count

    # Main strategy
    best_adj_overall = None
    best_count_overall = 0

    # Strategy 1: Min common neighbors greedy + local search
    adj = [0] * n
    min_common_greedy(adj)
    adj, cnt = simulated_annealing(adj)
    adj, cnt = two_opt_search(adj)
    if cnt > best_count_overall:
        best_count_overall = cnt
        best_adj_overall = adj[:]

    # Strategy 2: Multiple random orderings
    max_restarts = 40
    for r in range(max_restarts):
        if time.time() > deadline - 15:
            break
        
        adj = [0] * n
        order = all_pairs[:]
        random.shuffle(order)
        greedy_construct(adj, order)
        
        # Quick local search
        adj, cnt = simulated_annealing(adj, max_iters=2000)
        adj, cnt = two_opt_search(adj, max_iters=1000)
        if cnt > best_count_overall:
            best_count_overall = cnt
            best_adj_overall = adj[:]

    # Strategy 3: Structured circulant graphs
    for steps in [2, 3, 4, 5, 6]:
        if time.time() > deadline - 10:
            break
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % n
                if diff <= steps or diff >= n - steps:
                    if is_valid_add(adj, i, j):
                        add_edge(adj, i, j)
        adj, cnt = simulated_annealing(adj)
        adj, cnt = two_opt_search(adj)
        if cnt > best_count_overall:
            best_count_overall = cnt
            best_adj_overall = adj[:]

    # Strategy 4: Complete bipartite-like constructions
    for split in range(5, 11):
        if time.time() > deadline - 10:
            break
        adj = [0] * n
        # Split vertices into two groups
        group1 = list(range(split))
        group2 = list(range(split, n))
        for i in group1:
            for j in group2:
                if is_valid_add(adj, i, j):
                    add_edge(adj, i, j)
        # Also add some edges within groups
        for i in range(len(group1)):
            for j in range(i + 1, len(group1)):
                u, v = group1[i], group1[j]
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
        for i in range(len(group2)):
            for j in range(i + 1, len(group2)):
                u, v = group2[i], group2[j]
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
        adj, cnt = simulated_annealing(adj)
        adj, cnt = two_opt_search(adj)
        if cnt > best_count_overall:
            best_count_overall = cnt
            best_adj_overall = adj[:]

    # Final intensive local search on best
    if best_adj_overall is not None and time.time() < deadline - 5:
        best_adj_overall, best_count_overall = simulated_annealing(best_adj_overall, max_iters=3000)
        best_adj_overall, best_count_overall = two_opt_search(best_adj_overall, max_iters=2000)

    # Convert to edge list
    if best_adj_overall is None:
        return []

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj_overall[i] >> j) & 1:
                edges.append([i, j])

    return edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
