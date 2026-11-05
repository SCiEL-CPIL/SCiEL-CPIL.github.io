import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    def bitmask_to_edges(n, adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def edges_in_induced_subgraph(n, adj, mask):
        c = mask
        cnt = 0
        while c:
            lsb = c & -c
            w = lsb.bit_length() - 1
            c ^= lsb
            cnt += (adj[w] & c).bit_count()
        return cnt

    def count_k4_optimized(n, adj):
        count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    count += edges_in_induced_subgraph(n, adj, common)
        return count // 6

    def has_k6_in_comp_fast(n, comp_adj):
        mask_all = (1 << n) - 1
        
        def dfs(size, candidates):
            if size == 6:
                return True
            if candidates.bit_count() < 6 - size:
                return False
            
            # Pick lowest set bit
            v = (candidates & -candidates).bit_length() - 1
            v_bit = 1 << v
            candidates_without_v = candidates ^ v_bit
            
            # Include v
            if dfs(size + 1, candidates_without_v & comp_adj[v]):
                return True
            
            # Exclude v
            return dfs(size, candidates_without_v)
            
        return dfs(0, mask_all)

    best_n = 17
    best_edges = None
    
    for target_n in [20, 25, 30, 35]:
        if time.time() > deadline - 5:
            break
            
        adj = [0] * target_n
        for i in range(target_n):
            for j in range(i + 1, target_n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        def calc_penalty(adj):
            k4s = count_k4_optimized(target_n, adj)
            if k4s > 0:
                # If there are K4s, we don't even need to check Ind6 for the penalty magnitude?
                # But we might need to know if Ind6 exists to guide the search away?
                # If we have K4s, penalty is huge.
                # Let's just return 1000 * k4s + 100 * has_ind6
                pass
            mask = (1 << target_n) - 1
            comp_adj = [0] * target_n
            for u in range(target_n):
                comp_adj[u] = (~adj[u]) & mask & ~(1 << u)
            has_ind6 = has_k6_in_comp_fast(target_n, comp_adj)
            return 1000 * k4s + (100 if has_ind6 else 0)
        
        current_penalty = calc_penalty(adj)
        best_penalty = current_penalty
        best_adj = adj[:]
        
        temp = 50.0
        cooling = 0.99
        
        while time.time() < deadline - 5:
            u = random.randint(0, target_n - 1)
            v = random.randint(0, target_n - 1)
            if u == v: continue
            
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            
            new_penalty = calc_penalty(adj)
            
            if new_penalty < current_penalty or (new_penalty > current_penalty and random.random() < pow(2.718, -(new_penalty - current_penalty) / temp)):
                current_penalty = new_penalty
                if new_penalty < best_penalty:
                    best_penalty = new_penalty
                    best_adj = adj[:]
                    if best_penalty == 0:
                        edges = bitmask_to_edges(target_n, best_adj)
                        if target_n > best_n:
                            best_n = target_n
                            best_edges = edges
                        break
            else:
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                
            temp *= cooling
            if temp < 1.0:
                break

    if best_edges is None:
        squares = {x * x % 17 for x in range(1, 17)}
        return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
    else:
        return {'n': best_n, 'edges': best_edges}
