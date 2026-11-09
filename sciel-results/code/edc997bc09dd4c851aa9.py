import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin for 150s limit
    
    def popcount(x):
        return bin(x).count('1')
        
    def is_reg_fast(mask, adj, verts):
        # verts is tuple of vertices in mask
        d0 = popcount(adj[verts[0]] & mask)
        for v in verts[1:]:
            if popcount(adj[v] & mask) != d0:
                return False
        return True

    # Pre-compute data for n=20
    N_MAX = 20
    all_masks_6 = []
    mask_verts_map = {}
    for subset in itertools.combinations(range(N_MAX), 6):
        mask = 0
        for v in subset:
            mask |= (1 << v)
        all_masks_6.append(mask)
        mask_verts_map[mask] = tuple(subset)
        
    # Pre-compute affected masks for each pair (u,v)
    # affected_masks[u][v] = list of masks containing u and v
    affected_masks = [[None for _ in range(N_MAX)] for _ in range(N_MAX)]
    for u in range(N_MAX):
        for v in range(u+1, N_MAX):
            remaining = [x for x in range(N_MAX) if x != u and x != v]
            masks = []
            for comb in itertools.combinations(remaining, 4):
                mask = (1 << u) | (1 << v)
                for x in comb:
                    mask |= (1 << x)
                masks.append(mask)
            affected_masks[u][v] = masks
            affected_masks[v][u] = masks

    best_result = {'n': 5, 'edges': []}
    
    for n in [20, 19, 18]:
        if time.time() > deadline - 5:
            break
            
        # Filter masks for current n
        valid_masks = [m for m in all_masks_6 if m < (1 << n)]
        # Filter affected masks
        # We can just check if mask < (1<<n) inside the loop or pre-filter
        # Pre-filtering is better
        valid_affected = [[None for _ in range(n)] for _ in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                valid_affected[u][v] = [m for m in affected_masks[u][v] if m < (1 << n)]
                valid_affected[v][u] = valid_affected[u][v]

        # Random init
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        cur_viol = 0
        for mask in valid_masks:
            if is_reg_fast(mask, adj, mask_verts_map[mask]):
                cur_viol += 1
                
        best_viol = cur_viol
        best_adj = adj[:]
        
        # Local Search
        steps = 0
        while time.time() < deadline - 2:
            if cur_viol == 0:
                # Check for larger violations (7..n)
                # This is expensive, so we do it carefully
                has_larger = False
                for k in range(7, n+1):
                    for subset in itertools.combinations(range(n), k):
                        mask = 0
                        for v in subset:
                            mask |= (1 << v)
                        verts = tuple(subset)
                        if is_reg_fast(mask, adj, verts):
                            has_larger = True
                            break
                    if has_larger:
                        break
                            
                if not has_larger:
                    edges = []
                    for i in range(n):
                        for j in range(i+1, n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
                else:
                    # Stuck, break to try next n or restart
                    break

            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
                
            masks = valid_affected[u][v]
            
            old_viol_aff = 0
            for mask in masks:
                if is_reg_fast(mask, adj, mask_verts_map[mask]):
                    old_viol_aff += 1
                    
            # Flip
            if adj[u] & (1 << v):
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
            else:
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                
            new_viol_aff = 0
            for mask in masks:
                if is_reg_fast(mask, adj, mask_verts_map[mask]):
                    new_viol_aff += 1
                    
            new_viol = cur_viol - old_viol_aff + new_viol_aff
            
            if new_viol <= cur_viol:
                cur_viol = new_viol
                if cur_viol < best_viol:
                    best_viol = cur_viol
                    best_adj = adj[:]
            else:
                # Revert
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                
            steps += 1

    # Return best found or fallback
    # If best_viol was 0 for some n, we should have returned.
    # If not, return the graph with min violations?
    # The task requires a VALID graph.
    # If we didn't find one, we return n=5.
    
    return best_result
