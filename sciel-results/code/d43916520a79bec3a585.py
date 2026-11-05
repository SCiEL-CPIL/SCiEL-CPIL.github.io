import time
import random
import numpy as np

def _check_triangle_free(n, S):
    D = set()
    for s in S:
        D.add(s)
        D.add((-s) % n)
    D.discard(0)
    D_list = list(D)
    D_set = set(D_list)
    for x in D_list:
        for y in D_list:
            if x == 0 or y == 0:
                continue
            z = (x + y) % n
            if z == 0:
                continue
            if z in D_set and x != z:
                return False
    return True

def _get_adj_list(n, S):
    D = set()
    for s in S:
        D.add(s)
        D.add((-s) % n)
    D.discard(0)
    adj = [[] for _ in range(n)]
    for i in range(n):
        for d in D:
            j = (i + d) % n
            adj[i].append(j)
    return adj

def _find_max_independent_set(n, adj):
    # Greedy + local search for MIS
    # Or exact B&B for n<=40
    best_size = 0
    best_set = set()
    
    # Use a simple B&B
    adj_masks = [0]*n
    for i in range(n):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        adj_masks[i] = mask
            
    full_mask = (1 << n) - 1
    
    def dfs(candidates, current_set, size):
        nonlocal best_size, best_set
        if size > best_size:
            best_size = size
            best_set = set(current_set)
            
        if candidates == 0:
            return
            
        # Bound
        if size + bin(candidates).count('1') <= best_size:
            return
            
        # Pick vertex
        temp = candidates
        best_v = -1
        max_deg = -1
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = bin(candidates & adj_masks[v]).count('1')
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp ^= lsb
            
        v = best_v
        
        # Include v
        new_cands = candidates & ~adj_masks[v] & ~(1 << v)
        current_set.add(v)
        dfs(new_cands, current_set, size + 1)
        current_set.remove(v)
        
        # Exclude v
        dfs(candidates & ~(1 << v), current_set, size)
        
    dfs(full_mask, set(), 0)
    return best_set

def _can_add_edge(n, adj, u, v):
    # Check if adding (u,v) creates a triangle
    # i.e., if N(u) and N(v) have common neighbor
    nu = set(adj[u])
    nv = set(adj[v])
    if nu & nv:
        return False
    return True

def _repair_alpha(n, S):
    # Try to reduce alpha by adding edges
    adj = _get_adj_list(n, S)
    for _ in range(100): # Limit iterations
        if not _check_triangle_free(n, S):
            break # Should not happen if we only add safe edges
            
        # Find MIS
        mis = _find_max_independent_set(n, adj)
        if len(mis) < 10:
            return S, True
            
        # Try to add an edge within MIS
        found = False
        mis_list = list(mis)
        random.shuffle(mis_list)
        for i in range(len(mis_list)):
            for j in range(i+1, len(mis_list)):
                u = mis_list[i]
                v = mis_list[j]
                if _can_add_edge(n, adj, u, v):
                    # Add edge
                    # Determine which s to add to S
                    # Edge (u,v) corresponds to diff = (v-u)%n or (u-v)%n
                    diff = (v - u) % n
                    # We need to add a representative to S
                    # S contains representatives in 1..n/2
                    # If diff <= n/2, add diff. Else add n-diff.
                    if diff <= n // 2:
                        s_new = diff
                    else:
                        s_new = n - diff
                        
                    if s_new not in S:
                        S.add(s_new)
                        adj = _get_adj_list(n, S)
                        found = True
                        break
            if found:
                break
                
        if not found:
            # Stuck, break
            break
            
    # Final check
    if _check_triangle_free(n, S):
        mis = _find_max_independent_set(n, adj)
        if len(mis) < 10:
            return S, True
    return S, False

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_n = 0
    best_edges = []
    
    for n in range(39, 29, -1):
        if time.time() > deadline - 5:
            break
            
        found = False
        # Try multiple random starts
        for restart in range(50):
            if time.time() > deadline - 2:
                break
                
            half = n // 2
            candidates = list(range(1, half + 1))
            # Start with a random S
            # Size heuristic: n/3
            size = max(3, n // 3)
            S = set(random.sample(candidates, size))
            
            # Phase 1: Remove edges to make triangle-free
            # Greedy removal
            while not _check_triangle_free(n, S):
                # Find a triangle and remove one of its edges
                # Simple: just remove a random s
                # Better: find s involved in triangle
                # For speed, just remove random s if S is not empty
                if S:
                    s_to_remove = random.choice(list(S))
                    S.remove(s_to_remove)
                else:
                    break
                    
            if not S:
                continue
                
            # Phase 2: Repair Alpha
            S_valid, success = _repair_alpha(n, S)
            if success:
                # Verify
                if _check_triangle_free(n, S_valid):
                    # Check alpha one last time
                    adj = _get_adj_list(n, S_valid)
                    mis = _find_max_independent_set(n, adj)
                    if len(mis) < 10:
                        # Build edges
                        D = set()
                        for s in S_valid:
                            D.add(s)
                            D.add((-s) % n)
                        D.discard(0)
                        edges = []
                        for i in range(n):
                            for d in D:
                                j = (i + d) % n
                                if j > i:
                                    edges.append([i, j])
                        best_n = n
                        best_edges = edges
                        found = True
                        break
                        
        if found:
            break
            
    return {'n': best_n, 'edges': best_edges}
