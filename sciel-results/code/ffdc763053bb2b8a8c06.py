import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    def get_edges(adj, n):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def count_k4(n, adj):
        count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    deg_sum = 0
                    c = common
                    while c:
                        lsb = c & -c
                        x = lsb.bit_length() - 1
                        c ^= lsb
                        deg_sum += (adj[x] & common).bit_count()
                    count += deg_sum // 2
        return count

    def count_i6_exact(adj, n):
        """Count independent sets of size 6 in G using backtracking with pruning"""
        # Independent set in G = clique in complement
        # Use bitset-based backtracking
        mask = (1 << n) - 1
        
        def count_independent(candidates, depth):
            if depth == 6:
                return 1
            if len(candidates) < depth:
                return 0
            total = 0
            for idx in range(len(candidates)):
                v = candidates[idx]
                # New candidates: vertices after idx that are not adjacent to v
                new_cands = []
                for u in candidates[idx + 1:]:
                    if not ((adj[v] >> u) & 1):
                        new_cands.append(u)
                total += count_independent(new_cands, depth + 1)
            return total
        
        return count_independent(list(range(n)), 0)

    def has_i6(adj, n):
        """Check if there's an independent set of size 6 in G"""
        mask = (1 << n) - 1
        
        def bt(candidates, depth):
            if depth == 6:
                return True
            if len(candidates) < depth:
                return False
            for idx in range(len(candidates)):
                v = candidates[idx]
                new_cands = []
                for u in candidates[idx + 1:]:
                    if not ((adj[v] >> u) & 1):
                        new_cands.append(u)
                if bt(new_cands, depth + 1):
                    return True
            return False
        
        return bt(list(range(n)), 0)

    def is_valid(n, adj):
        if count_k4(n, adj) != 0:
            return False
        if has_i6(adj, n):
            return False
        return True

    def compute_cost(adj, n):
        """Compute total violation count: K4 count + I6 count"""
        k4_count = count_k4(n, adj)
        if k4_count > 1000:  # Too many K4s, skip I6 count for speed
            return k4_count * 10
        i6_count = count_i6_exact(adj, n)
        return k4_count * 10 + i6_count * 100  # Weight I6 more heavily

    best_n = 0
    best_adj = None

    # Try multiple target sizes and densities
    target_sizes = [35, 34, 33, 32, 31, 30, 28, 26, 24, 22, 20, 18, 17]
    densities = [0.3, 0.4, 0.5, 0.6, 0.7]

    for target_n in target_sizes:
        if time.time() > deadline - 5:
            break
        
        for density in densities:
            if time.time() > deadline - 10:
                break
            
            # Initialize with random graph
            adj = [0] * target_n
            for u in range(target_n):
                for v in range(u + 1, target_n):
                    if random.random() < density:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            current_cost = compute_cost(adj, target_n)
            best_local_cost = current_cost
            best_local_adj = adj[:]
            
            # Simulated Annealing
            T = 5.0
            cooling = 0.999
            min_T = 0.1
            
            step = 0
            while T > min_T and time.time() < deadline - 15:
                step += 1
                
                u = random.randint(0, target_n - 1)
                v = random.randint(0, target_n - 1)
                if u == v:
                    continue
                
                # Flip edge
                is_edge = bool((adj[u] >> v) & 1)
                if is_edge:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                
                new_cost = compute_cost(adj, target_n)
                delta = new_cost - current_cost
                
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_cost = new_cost
                    if current_cost < best_local_cost:
                        best_local_cost = current_cost
                        best_local_adj = adj[:]
                        
                        # Check if valid
                        if current_cost == 0 and time.time() < deadline - 20:
                            if is_valid(target_n, best_local_adj):
                                return {'n': target_n, 'edges': get_edges(best_local_adj, target_n)}
                else:
                    # Revert
                    if is_edge:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                
                T *= cooling
            
            # After SA, do greedy repair on best_local_adj
            if best_local_cost < 500 and time.time() < deadline - 20:
                adj = best_local_adj[:]
                improved = True
                while improved and time.time() < deadline - 15:
                    improved = False
                    best_delta = 0
                    best_u = -1
                    best_v = -1
                    
                    for u in range(target_n):
                        for v in range(u + 1, target_n):
                            # Try flipping and compute cost delta
                            is_edge = bool((adj[u] >> v) & 1)
                            if is_edge:
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                            else:
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
                            
                            new_cost = compute_cost(adj, target_n)
                            delta = new_cost - best_local_cost
                            
                            # Revert
                            if is_edge:
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
                            else:
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                            
                            if delta < best_delta:
                                best_delta = delta
                                best_u = u
                                best_v = v
                                improved = True
                    
                    if best_u != -1:
                        if (adj[best_u] >> best_v) & 1:
                            adj[best_u] &= ~(1 << best_v)
                            adj[best_v] &= ~(1 << best_u)
                        else:
                            adj[best_u] |= (1 << best_v)
                            adj[best_v] |= (1 << best_u)
                        best_local_cost = compute_cost(adj, target_n)
                
                if is_valid(target_n, adj):
                    return {'n': target_n, 'edges': get_edges(adj, target_n)}
                
                # Update best if valid or better
                if compute_cost(adj, target_n) < best_local_cost:
                    best_local_adj = adj[:]
                    best_local_cost = compute_cost(adj, target_n)
            
            # Track best valid graph
            if best_local_cost == 0 and is_valid(target_n, best_local_adj):
                if target_n > best_n:
                    best_n = target_n
                    best_adj = best_local_adj[:]

    if best_adj is not None:
        return {'n': best_n, 'edges': get_edges(best_adj, best_n)}
    
    # Fallback to Paley 17
    squares = {x * x % 17 for x in range(1, 17)}
    adj = [0] * 17
    for i in range(17):
        for j in range(i + 1, 17):
            if (j - i) % 17 in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return {'n': 17, 'edges': get_edges(adj, 17)}
