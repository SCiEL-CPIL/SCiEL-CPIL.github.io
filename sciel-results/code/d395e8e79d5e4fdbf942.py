import numpy as np
import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 115.0

    def time_left():
        return deadline - (time.time() - start_time)

    def adj_to_orient(n, adj):
        orient = []
        for i in range(n):
            for j in range(i+1, n):
                orient.append(0 if adj[i][j] == 1 else 1)
        return orient

    def count_violations(n, adj, out_mask):
        count = 0
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            quad_mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
            found = False
            for v in range(n):
                if (out_mask[v] & quad_mask) == quad_mask:
                    found = True
                    break
            if not found:
                count += 1
        return count

    def is_valid(n, adj, out_mask):
        return count_violations(n, adj, out_mask) == 0

    def simulate_annealing(n, initial_adj, max_iters=50000):
        adj = [row[:] for row in initial_adj]
        out_mask = [0] * n
        for v in range(n):
            mask = 0
            for w in range(n):
                if adj[v][w]:
                    mask |= (1 << w)
            out_mask[v] = mask
            
        current_cost = count_violations(n, adj, out_mask)
        best_cost = current_cost
        best_adj = [row[:] for row in adj]
        
        T = 1.0
        cooling = 0.9999
        
        for i in range(max_iters):
            if time_left() < 5:
                break
            
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
                
            old_uv = adj[u][v]
            adj[u][v] = 1 - old_uv
            adj[v][u] = 1 - adj[u][v]
            
            if adj[u][v]:
                out_mask[u] |= (1 << v)
                out_mask[v] &= ~(1 << u)
            else:
                out_mask[u] &= ~(1 << v)
                out_mask[v] |= (1 << u)
                
            new_cost = count_violations(n, adj, out_mask)
            
            delta = new_cost - current_cost
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_adj = [row[:] for row in adj]
                    if best_cost == 0:
                        break
            else:
                adj[u][v] = old_uv
                adj[v][u] = 1 - old_uv
                if old_uv:
                    out_mask[u] |= (1 << v)
                    out_mask[v] &= ~(1 << u)
                else:
                    out_mask[u] &= ~(1 << v)
                    out_mask[v] |= (1 << u)
                    
            T *= cooling
            
        return best_adj, best_cost

    # 1. Construct Paley Tournament of order 83 (Known to be valid for k=4)
    p = 83
    squares = set(x * x % p for x in range(1, p))
    adj = [[0]*p for _ in range(p)]
    for i in range(p):
        for j in range(i+1, p):
            if (j - i) % p in squares:
                adj[i][j] = 1
            else:
                adj[j][i] = 1

    best_n = p
    best_adj = [row[:] for row in adj]
    
    # 2. Greedy Deletion Strategy
    # Try to remove vertices one by one while maintaining validity
    # A vertex v can be removed if every 4-set containing v is already "covered" by some other vertex in the set
    # Wait, if we remove v, the 4-sets that included v are gone. We only need to check 4-sets NOT containing v.
    # Actually, if we remove v, the remaining tournament must satisfy the property for all 4-sets in the remaining vertices.
    # So we just check if the induced subgraph on V \ {v} is valid.
    
    current_n = p
    current_adj = [row[:] for row in adj]
    
    # Precompute out_masks for speed
    def get_out_masks(n, adj):
        masks = [0] * n
        for v in range(n):
            mask = 0
            for w in range(n):
                if adj[v][w]:
                    mask |= (1 << w)
            masks[v] = mask
        return masks

    current_masks = get_out_masks(current_n, current_adj)
    
    # Optimization: Instead of full check every time, we can be smart.
    # But full check is O(N^4). N=83 -> 83^4 ~ 47M ops. Python is slow.
    # We need a faster validity check or a smarter deletion criterion.
    
    # Alternative: 
    # A vertex v is "redundant" if for every 3-set S in V\{v}, there is a vertex u in S such that u -> all in S.
    # This is because the only 4-sets affected by removing v are those containing v.
    # Wait, no. If we remove v, we don't care about 4-sets containing v anymore.
    # We only care about 4-sets in V\{v}.
    # So we just need to check if the tournament on V\{v} is valid.
    
    # Let's try a few random restarts of deletion from the Paley graph.
    # Since Paley is highly symmetric, we can try removing vertices in different orders.
    
    for attempt in range(20):
        if time_left() < 10:
            break
            
        current_n = p
        current_adj = [row[:] for row in adj]
        current_masks = get_out_masks(current_n, current_adj)
        
        # Shuffle vertex order for deletion attempt
        vertices = list(range(current_n))
        random.shuffle(vertices)
        
        for v in vertices:
            if time_left() < 5:
                break
            if v >= current_n:
                continue
            
            # Check if we can remove v
            # We need to check if the induced subgraph on V \ {v} is valid.
            # Instead of building a new adjacency matrix, we can just check the property on the existing structure ignoring v.
            
            # Fast check: 
            # For every 4-set Q in V \ {v}, is there a dominator?
            # This is still expensive.
            
            # Heuristic: If v has high out-degree, it might be more "useful" as a dominator? 
            # No, we are removing it. If v is a dominator for many sets, removing it is dangerous.
            # If v is NOT a dominator for any set (i.e., it's not the unique dominator for any 4-set it belongs to), 
            # then removing it is safe.
            
            # Let's calculate: For each 4-set Q containing v, does Q have a dominator in Q \ {v}?
            # If yes for all such Q, then v is redundant.
            
            redundant = True
            # Iterate over all 3-sets S in V \ {v}
            # Q = S U {v}
            # We need to check if S U {v} has a dominator in S U {v} \ {v} = S? 
            # No, the dominator must be in Q. If the dominator is v, then removing v breaks it.
            # If the dominator is in S, then it's fine.
            
            # So: For every 3-set S in V\{v}, does S U {v} have a dominator in S?
            # Note: The dominator of S U {v} must be in S U {v}.
            # If the dominator is v, then v -> all in S.
            # If the dominator is u in S, then u -> all in S and u -> v.
            
            # So we check: For every 3-set S in V\{v}, is there a u in S such that u dominates S U {v}?
            # u dominates S U {v} means u -> s for all s in S, and u -> v.
            
            # This check is O(N^3) per vertex v. Total O(N^4).
            # N=83. 83^3 = 571k. Times 83 vertices = 47M.
            # In Python, this might take a few seconds per deletion step.
            # 83 steps * few seconds = too slow.
            
            # We need a faster approximation or a different strategy.
            
            # Strategy: Try to remove vertices with the "least" impact.
            # Impact = number of 4-sets where v is the UNIQUE dominator.
            # If impact is 0, remove it.
            
            # Calculate unique dominator count for v.
            unique_count = 0
            for a in range(current_n):
                if a == v: continue
                for b in range(a+1, current_n):
                    if b == v: continue
                    for c in range(b+1, current_n):
                        if c == v: continue
                        # S = {a, b, c}
                        # Q = {a, b, c, v}
                        # Check if v is the unique dominator of Q.
                        # v is dominator if v->a, v->b, v->c.
                        if not (current_adj[v][a] and current_adj[v][b] and current_adj[v][c]):
                            continue
                        # v is a dominator. Is it unique?
                        # Check if any of a, b, c is also a dominator.
                        # a is dominator if a->b, a->c, a->v.
                        if current_adj[a][b] and current_adj[a][c] and current_adj[a][v]:
                            continue
                        if current_adj[b][a] and current_adj[b][c] and current_adj[b][v]:
                            continue
                        if current_adj[c][a] and current_adj[c][b] and current_adj[c][v]:
                            continue
                        # If none of a,b,c dominate Q, then v is the unique dominator.
                        unique_count += 1
                        if unique_count > 0:
                            break
                    if unique_count > 0:
                        break
                if unique_count > 0:
                    break
                    
            if unique_count == 0:
                # v is redundant, remove it
                # Rebuild adjacency matrix without v
                new_adj = [[0]*0 for _ in range(0)]
                # This is slow to rebuild. 
                # Let's just mark v as removed and adjust indices? 
                # Or just build a new matrix.
                new_n = current_n - 1
                new_adj = [[0]*new_n for _ in range(new_n)]
                idx_map = {}
                new_idx = 0
                for i in range(current_n):
                    if i == v:
                        continue
                    idx_map[i] = new_idx
                    new_idx += 1
                
                for i in range(current_n):
                    if i == v: continue
                    for j in range(current_n):
                        if j == v or j == i: continue
                        if i < j:
                            new_adj[idx_map[i]][idx_map[j]] = current_adj[i][j]
                        else:
                            new_adj[idx_map[i]][idx_map[j]] = current_adj[i][j]
                            
                current_n = new_n
                current_adj = new_adj
                current_masks = get_out_masks(current_n, current_adj)
                
                if current_n < best_n:
                    best_n = current_n
                    best_adj = [row[:] for row in current_adj]
                
                # Restart the scan from the beginning? Or continue?
                # If we continue, we might have removed a vertex that made another one redundant.
                # Let's restart the vertex list scan for efficiency in finding more deletions.
                # But we only have 115s.
                # Let's just break and restart the whole deletion process with the new graph.
                break
        
        if current_n < best_n:
            best_n = current_n
            best_adj = [row[:] for row in current_adj]
            
        if best_n <= 67:
            break

    return {'n': best_n, 'orientation': adj_to_orient(best_n, best_adj)}
