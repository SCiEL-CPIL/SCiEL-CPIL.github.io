import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    def find_k4(adj, n):
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        c ^= lsb
                        others = adj[w] & common
                        if others:
                            lsb2 = others & -others
                            x = lsb2.bit_length() - 1
                            return [u, v, w, x]
        return None

    def find_ind6(adj, n):
        mask = (1 << n) - 1
        comp_adj = [0] * n
        for i in range(n):
            comp_adj[i] = mask & ~(adj[i] | (1 << i))
        
        # To speed up, we can use a simple recursive search with pruning
        # Sort vertices by degree in comp_adj? 
        # For simplicity, we use the bitset recursion.
        
        def find_clique(candidates, size, current_clique):
            if size == 0:
                return current_clique
            if candidates.bit_count() < size:
                return None
            
            # Heuristic: pick vertex with highest degree in candidates?
            # For speed, picking the first one is okay, but ordering helps.
            # Let's just pick the first one to avoid sorting overhead in every call.
            v = (candidates & -candidates).bit_length() - 1
            
            # Branch 1: Include v
            new_cands = candidates & comp_adj[v] & ~(1 << v)
            res = find_clique(new_cands, size - 1, current_clique + [v])
            if res:
                return res
            
            # Branch 2: Exclude v
            return find_clique(candidates & ~(1 << v), size, current_clique)

        return find_clique((1 << n) - 1, 6, [])

    def is_valid(adj, n):
        if find_k4(adj, n):
            return False
        if find_ind6(adj, n):
            return False
        return True

    def repair(adj, n, time_limit):
        while time.time() < time_limit:
            k4 = find_k4(adj, n)
            if k4:
                # Remove a random edge from the K4
                idx = random.randint(0, 5)
                u, v = k4[idx // 2], k4[(idx // 2) ^ 1] # This mapping is wrong for pairs
                # K4 is [u,v,w,x]. Edges are (u,v), (u,w), (u,x), (v,w), (v,x), (w,x)
                # Let's just pick two random distinct vertices from the K4
                a, b = random.sample(k4, 2)
                if (adj[a] >> b) & 1:
                    adj[a] &= ~(1 << b)
                    adj[b] &= ~(1 << a)
                    continue
            
            ind6 = find_ind6(adj, n)
            if ind6:
                # Add a random edge within the Ind6
                a, b = random.sample(ind6, 2)
                if not ((adj[a] >> b) & 1):
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
                    continue
            
            # If no violations found, it is valid
            return adj
        
        return adj

    def local_search(adj, n, time_limit):
        # Combine repair with random flips if stuck
        best_adj = adj[:]
        best_valid = is_valid(best_adj, n)
        
        while time.time() < time_limit:
            if best_valid:
                return best_adj
                
            # Try repair
            new_adj = adj[:]
            repaired = repair(new_adj, n, time.time() + 1.0) # Give 1s to repair
            
            if is_valid(repaired, n):
                return repaired
            
            # Try random flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v:
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                
                if is_valid(adj, n):
                    return adj
                    
                # If not valid, check if penalty improved? 
                # For now, just keep the state if it doesn't get worse significantly?
                # Simple hill climbing: accept if valid. Otherwise revert?
                # Let's revert if not valid to avoid getting stuck in bad states?
                # Or keep it if it "looks" better?
                # Let's just revert for simplicity, relying on repair to make progress.
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)

        return adj

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

    targets = [35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25]
    
    for target_n in targets:
        if time.time() > deadline - 10:
            break
        
        # Try multiple restarts
        found = False
        attempts = 5
        if target_n > 32:
            attempts = 10
            
        for _ in range(attempts):
            if time.time() > deadline - 5:
                break
                
            # Random init
            adj = [0] * target_n
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        
            result_adj = local_search(adj, target_n, deadline - 2)
            
            if is_valid(result_adj, target_n):
                best_n = target_n
                best_adj = result_adj
                found = True
                break

        if found:
            break

    def bitmask_to_edges(n, adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    return {'n': best_n, 'edges': bitmask_to_edges(best_n, best_adj)}
