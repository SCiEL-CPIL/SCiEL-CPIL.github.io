import time
import random

def construct():
    n = 15
    start = time.perf_counter()
    deadline = start + 85  # safety margin before 90s kill

    # Precompute all potential edges
    edge_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(edge_list)

    # Bitmask adjacency representation
    # adj_mask[v] is a bitmask of neighbors of v

    def check_valid(adj_mask, n):
        """Check if no simple 3-edge path has endpoints with equal degree."""
        deg = [bin(adj_mask[v]).count('1') for v in range(n)]
        
        # Map degree to bitmask of vertices with that degree
        deg_to_mask = {}
        for v in range(n):
            d = deg[v]
            if d not in deg_to_mask:
                deg_to_mask[d] = 0
            deg_to_mask[d] |= (1 << v)
        
        # For each edge (u, v)
        for u in range(n):
            mask_u = adj_mask[u]
            for v in range(u + 1, n):
                if not (mask_u >> v) & 1:
                    continue
                mask_v = adj_mask[v]
                
                s_mask = mask_u & ~(1 << v)
                t_mask = mask_v & ~(1 << u)
                
                # Check for each degree d if there are s in N(u)\{v} and t in N(v)\{u}
                # with deg(s)==deg(t)==d and s != t
                for d, d_mask in deg_to_mask.items():
                    A = s_mask & d_mask
                    B = t_mask & d_mask
                    if A and B:
                        # If A and B are not identical, there must be s in A\B or t in B\A
                        # If s in A\B, then s is in N(u) and not in N(v).
                        # t can be any in B.
                        # If A != B, it's possible that A={s}, B={t} with s!=t.
                        # Actually, if A != B, does it guarantee a violation?
                        # If A={1}, B={2}. s=1, t=2. s!=t. Violation.
                        # If A={1,2}, B={1}. s=2, t=1. s!=t. Violation.
                        # If A={1}, B={1}. s=1, t=1. s==t. No violation for this pair.
                        # But if A={1}, B={1}, is it possible there are other pairs?
                        # No, A and B are the *sets* of all such vertices.
                        # So if A == B, the only pairs (s,t) with s in A, t in B are (x,x) for x in A.
                        # Since s must != t, these are all invalid paths (not simple paths with distinct endpoints).
                        # So A == B is SAFE.
                        # If A != B, is it always UNSAFE?
                        # If A={1}, B={1,2}. s=1, t=2. s!=t. Unsafe.
                        # So the condition for VALIDITY is:
                        # For all d, either A is empty, B is empty, or A == B.
                        # Wait, if A == B, is it possible that |A| > 1?
                        # If A = B = {1, 2}.
                        # Pairs (s,t) with s in A, t in B:
                        # (1,1) - invalid (s=t)
                        # (1,2) - valid path? s=1, t=2. deg(1)=deg(2)=d. s!=t. VIOLATION.
                        # So A == B is ONLY safe if |A| == 1.
                        
                        if A != B:
                            return False
                        if A & (A - 1): # More than 1 bit
                            return False
        return True

    def count_edges(adj_mask, n):
        total = 0
        for v in range(n):
            total += bin(adj_mask[v]).count('1')
        return total // 2

    # Start with K_{7,8} bipartite graph
    def make_bipartite(adj_mask, n, split):
        for v in range(n):
            adj_mask[v] = 0
        for i in range(split):
            for j in range(split, n):
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)

    def try_add_edges_greedy(adj_mask, n, deadline, order=None):
        """Greedy: try to add each missing edge if it maintains validity."""
        if order is None:
            order = list(range(m))
            random.shuffle(order)
        
        for idx in order:
            if time.perf_counter() > deadline:
                break
            i, j = edge_list[idx]
            if (adj_mask[i] >> j) & 1:
                continue
            # Try adding edge (i, j)
            adj_mask[i] |= (1 << j)
            adj_mask[j] |= (1 << i)
            if check_valid(adj_mask, n):
                pass  # keep it
            else:
                adj_mask[i] ^= (1 << j)
                adj_mask[j] ^= (1 << i)
        return adj_mask

    # Initial: try different bipartite splits and randomized greedy
    best_adj = [0] * n
    best_count = 0

    # Try a few more splits and randomized starts
    splits_to_try = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    random.shuffle(splits_to_try)
    
    for split in splits_to_try:
        if time.perf_counter() > deadline - 20:
            break
        adj_mask = [0] * n
        make_bipartite(adj_mask, n, split)
        adj_mask = try_add_edges_greedy(adj_mask, n, deadline)
        cnt = count_edges(adj_mask, n)
        if cnt > best_count:
            best_count = cnt
            best_adj = list(adj_mask)

    # Also try randomized greedy from empty
    # Increase trials if we have time
    trials = 0
    while time.perf_counter() < deadline - 15:
        trials += 1
        adj_mask = [0] * n
        adj_mask = try_add_edges_greedy(adj_mask, n, deadline)
        cnt = count_edges(adj_mask, n)
        if cnt > best_count:
            best_count = cnt
            best_adj = list(adj_mask)
        # Limit trials to prevent infinite loop if check is very fast
        if trials > 100: 
            break

    # Local search: edge swaps (remove one edge, add another)
    def local_search(adj_mask, n, deadline, max_rounds=200):
        """Try edge swaps to improve edge count."""
        improved = True
        rounds = 0
        while improved and time.perf_counter() < deadline - 5 and rounds < max_rounds:
            rounds += 1
            improved = False
            # Get current edge set
            edges = []
            non_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj_mask[i] >> j) & 1:
                        edges.append((i, j))
                    else:
                        non_edges.append((i, j))

            random.shuffle(edges)
            random.shuffle(non_edges)

            # Try adding a single edge first (cheapest improvement)
            for e2 in non_edges:
                if time.perf_counter() > deadline - 5:
                    break
                if improved:
                    break
                i2, j2 = e2
                adj_mask[i2] |= (1 << j2)
                adj_mask[j2] |= (1 << i2)
                if check_valid(adj_mask, n):
                    improved = True
                    break
                else:
                    adj_mask[i2] ^= (1 << j2)
                    adj_mask[j2] ^= (1 << i2)

            if improved:
                continue

            # Try swaps: remove one edge, add another
            for e1 in edges:
                if time.perf_counter() > deadline - 5:
                    break
                if improved:
                    break
                i1, j1 = e1
                adj_mask[i1] ^= (1 << j1)
                adj_mask[j1] ^= (1 << i1)
                for e2 in non_edges:
                    if time.perf_counter() > deadline - 5:
                        break
                    if improved:
                        break
                    i2, j2 = e2
                    adj_mask[i2] |= (1 << j2)
                    adj_mask[j2] |= (1 << i2)
                    if check_valid(adj_mask, n):
                        improved = True
                        break
                    else:
                        adj_mask[i2] ^= (1 << j2)
                        adj_mask[j2] ^= (1 << i2)
                # Restore removed edge if no swap found
                if not improved:
                    adj_mask[i1] |= (1 << j1)
                    adj_mask[j1] |= (1 << i1)

        return adj_mask

    # Run local search from best
    best_adj = local_search(best_adj, n, deadline)
    best_count = count_edges(best_adj, n)

    # Try 2-edge swaps (remove 2, add 2) for more exploration
    if time.perf_counter() < deadline - 15:
        adj_mask = list(best_adj)
        improved = True
        while improved and time.perf_counter() < deadline - 5:
            improved = False
            edges = []
            non_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj_mask[i] >> j) & 1:
                        edges.append((i, j))
                    else:
                        non_edges.append((i, j))

            random.shuffle(edges)
            random.shuffle(non_edges)

            # Try removing 2 edges and adding 2 edges
            # Limit search space to avoid O(N^4) explosion
            max_e = min(len(edges), 30)
            max_ne = min(len(non_edges), 30)
            
            # Optimize: Only try swaps that might improve degree distribution?
            # For now, random sampling.
            
            for idx1 in range(max_e):
                if time.perf_counter() > deadline - 5:
                    break
                if improved:
                    break
                for idx2 in range(idx1 + 1, max_e):
                    if time.perf_counter() > deadline - 5:
                        break
                    if improved:
                        break
                    e1 = edges[idx1]
                    e2 = edges[idx2]
                    # Try adding 2 non-edges
                    for idx3 in range(max_ne):
                        if time.perf_counter() > deadline - 5:
                            break
                        if improved:
                            break
                        for idx4 in range(idx3 + 1, max_ne):
                            if time.perf_counter() > deadline - 5:
                                break
                            if improved:
                                break
                            f1 = non_edges[idx3]
                            f2 = non_edges[idx4]
                            # Remove e1, e2; add f1, f2
                            for (a, b) in [e1, e2]:
                                adj_mask[a] ^= (1 << b)
                                adj_mask[b] ^= (1 << a)
                            for (a, b) in [f1, f2]:
                                adj_mask[a] |= (1 << b)
                                adj_mask[b] |= (1 << a)
                            if check_valid(adj_mask, n):
                                improved = True
                                break
                            else:
                                # Revert
                                for (a, b) in [f1, f2]:
                                    adj_mask[a] ^= (1 << b)
                                    adj_mask[b] ^= (1 << a)
                                for (a, b) in [e1, e2]:
                                    adj_mask[a] |= (1 << b)
                                    adj_mask[b] |= (1 << a)

        cnt = count_edges(adj_mask, n)
        if cnt > best_count:
            best_count = cnt
            best_adj = list(adj_mask)

    # Final: try to add any remaining edges greedily
    if time.perf_counter() < deadline - 2:
        adj_mask = list(best_adj)
        adj_mask = try_add_edges_greedy(adj_mask, n, deadline)
        cnt = count_edges(adj_mask, n)
        if cnt > best_count:
            best_count = cnt
            best_adj = list(adj_mask)

    # Convert to edge list
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                result.append([int(i), int(j)])

    return result
