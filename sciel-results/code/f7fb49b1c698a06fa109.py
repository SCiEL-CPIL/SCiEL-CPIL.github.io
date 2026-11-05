import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def bitmask_to_edges(n, adj):
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
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        c ^= lsb
                        count += (adj[w] & c).bit_count()
        return count // 6

    def has_ind6(n, adj):
        mask = (1 << n) - 1
        comp_adj = [0] * n
        for i in range(n):
            comp_adj[i] = mask & ~(adj[i] | (1 << i))
        
        # Check if any vertex has < 5 non-neighbors in complement (i.e., > n-1-5 neighbors in graph)
        # Actually, we just need to find a clique of size 6 in comp_adj.
        
        def find_clique(candidates, size):
            if size == 0:
                return True
            if candidates.bit_count() < size:
                return False
            
            # Order vertices by degree in the induced subgraph to prune better?
            # Simple version: pick lowest index
            v = (candidates & -candidates).bit_length() - 1
            new_cands = candidates & ~(1 << v) & comp_adj[v]
            
            if find_clique(new_cands, size - 1):
                return True
            
            return find_clique(candidates & ~(1 << v), size)

        return find_clique((1 << n) - 1, 6)

    def is_valid(n, adj):
        if count_k4(n, adj) > 0:
            return False
        if has_ind6(n, adj):
            return False
        return True

    def local_search(n, adj, time_limit):
        current_adj = adj[:]
        
        def get_penalty(adj):
            k4 = count_k4(n, adj)
            if k4 > 0:
                return k4 * 10
            ind6 = has_ind6(n, adj)
            if ind6:
                return 100
            return 0

        current_penalty = get_penalty(current_adj)
        
        if current_penalty == 0:
            return current_adj

        temp = 50.0
        cooling = 0.9995
        
        while time.time() < time_limit:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            if (current_adj[u] >> v) & 1:
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
            else:
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
            
            new_penalty = get_penalty(current_adj)
            
            if new_penalty < current_penalty:
                current_penalty = new_penalty
                if current_penalty == 0:
                    return current_adj
            else:
                delta = new_penalty - current_penalty
                if random.random() < pow(2.71828, -delta / temp):
                    current_penalty = new_penalty
                else:
                    # Revert
                    if (current_adj[u] >> v) & 1:
                        current_adj[u] &= ~(1 << v)
                        current_adj[v] &= ~(1 << u)
                    else:
                        current_adj[u] |= (1 << v)
                        current_adj[v] |= (1 << u)
            
            temp *= cooling

        return current_adj

    best_n = 17
    best_adj = None
    
    # Fallback: Paley graph on 17
    squares = {x * x % 17 for x in range(1, 17)}
    adj17 = [0] * 17
    for i in range(17):
        for j in range(i + 1, 17):
            if (j - i) % 17 in squares:
                adj17[i] |= (1 << j)
                adj17[j] |= (1 << i)
    best_adj = adj17

    targets = [35, 34, 33, 32, 31, 30, 28, 25, 22, 20]
    
    for target_n in targets:
        if time.time() > deadline - 20:
            break
        
        # Try multiple random starts for better chance
        found = False
        for attempt in range(3):
            if time.time() > deadline - 20:
                break
                
            adj = [0] * target_n
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            slice_end = time.time() + 10
            if slice_end > deadline - 10:
                slice_end = deadline - 10
                
            result_adj = local_search(target_n, adj, slice_end)
            
            if is_valid(target_n, result_adj):
                best_n = target_n
                best_adj = result_adj
                found = True
                break

        if found:
            break

    return {'n': best_n, 'edges': bitmask_to_edges(best_n, best_adj)}
