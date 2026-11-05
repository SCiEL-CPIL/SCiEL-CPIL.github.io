import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Precompute regularity for all 2^15 6-vertex graph patterns
    edge_pairs = [(0,1),(0,2),(0,3),(0,4),(0,5),(1,2),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),(3,4),(3,5),(4,5)]
    universal_incident = [0]*6
    for idx, (i,j) in enumerate(edge_pairs):
        universal_incident[i] |= (1 << idx)
        universal_incident[j] |= (1 << idx)
    
    reg_table = bytearray(1 << 15)  # 1 if regular, 0 otherwise
    for p in range(1 << 15):
        d0 = bin(p & universal_incident[0]).count('1')
        if all(bin(p & universal_incident[v]).count('1') == d0 for v in range(1, 6)):
            reg_table[p] = 1
    
    for target_n in [20, 19, 18]:
        if time.time() > deadline - 10:
            break
        n = target_n
        
        all_six = list(combinations(range(n), 6))
        num_six = len(all_six)
        
        # For each subset, list of (edge_idx_in_subset, global_edge)
        # and for each global edge, list of (subset_idx, edge_idx_in_subset)
        subset_edges = []  # [si] = list of 15 (i,j)
        edge_pos = {}  # (si, (i,j)) -> position 0-14
        
        for si, sub in enumerate(all_six):
            es = []
            for a in range(6):
                for b in range(a+1, 6):
                    es.append((sub[a], sub[b]))
            subset_edges.append(es)
            for ep, e in enumerate(es):
                edge_pos[(si, e)] = ep
        
        # For each global edge, list of (si, ep)
        edge_affected = {}
        for si in range(num_six):
            for ep, e in enumerate(subset_edges[si]):
                if e not in edge_affected:
                    edge_affected[e] = []
                edge_affected[e].append((si, ep))
        
        for restart in range(40):
            if time.time() > deadline - 5:
                break
            
            # Random init
            adj = [[0]*n for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj[i][j] = 1
                        adj[j][i] = 1
            
            # Compute patterns
            patterns = [0]*num_six
            for si in range(num_six):
                p = 0
                for ep, (i,j) in enumerate(subset_edges[si]):
                    if adj[i][j]:
                        p |= (1 << ep)
                patterns[si] = p
            
            bad = sum(1 for p in patterns if reg_table[p])
            
            if bad == 0:
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if adj[i][j]:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
            
            # Local search
            for step in range(2000):
                if time.time() > deadline - 5:
                    break
                if bad == 0:
                    edges = []
                    for i in range(n):
                        for j in range(i+1, n):
                            if adj[i][j]:
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
                
                # Pick random edge
                u = random.randint(0, n-1)
                v = random.randint(u+1, n-1)
                e = (u, v)
                aff = edge_affected.get(e, [])
                if not aff:
                    continue
                
                # Compute old bad in affected
                old_bad_aff = 0
                for si, ep in aff:
                    if reg_table[patterns[si]]:
                        old_bad_aff += 1
                
                # Flip
                adj[u][v] ^= 1
                adj[v][u] = adj[u][v]
                
                # Update patterns
                new_bad_aff = 0
                for si, ep in aff:
                    patterns[si] ^= (1 << ep)
                    if reg_table[patterns[si]]:
                        new_bad_aff += 1
                
                delta = new_bad_aff - old_bad_aff
                if delta > 0:
                    # Revert
                    adj[u][v] ^= 1
                    adj[v][u] = adj[u][v]
                    for si, ep in aff:
                        patterns[si] ^= (1 << ep)
                else:
                    bad += delta
                    if delta < 0:
                        # Good move
                        pass
                    # If delta == 0, keep it (neutral move for diversity)
        
        # After all restarts, check if we found anything valid
        # (already returned above if found)
    
    return {'n': 5, 'edges': []}
