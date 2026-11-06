import time
import random
import math

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin before 90s kill
    
    n = 256
    
    # Build conflict graph using bitmasks for speed
    # conflict_mask[v] is a bitmask where bit u is set if v and u conflict
    conflict_mask = [0] * (n + 1)
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)
    
    # Precompute adjacency lists for local search
    adj = [set() for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                adj[a].add(b)
                adj[b].add(a)
    
    def is_independent(S):
        """Check if S is an independent set using bitmask."""
        S_list = list(S)
        for i, a in enumerate(S_list):
            for b in S_list[i+1:]:
                if conflict_mask[a] & (1 << b):
                    return False
        return True
    
    def greedy_mis(adj, n, random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        
        remaining = set(range(1, n + 1))
        result = []
        
        while remaining:
            min_v = None
            min_deg = float('inf')
            best_cands = []
            
            for v in remaining:
                deg = sum(1 for u in adj[v] if u in remaining)
                if deg < min_deg:
                    min_deg = deg
                    best_cands = [v]
                elif deg == min_deg:
                    best_cands.append(v)
            
            if random_seed is not None and best_cands:
                min_v = random.choice(best_cands)
            else:
                min_v = best_cands[0]
                
            result.append(min_v)
            
            to_remove = {min_v}
            for u in adj[min_v]:
                if u in remaining:
                    to_remove.add(u)
            remaining -= to_remove
            
            if time.time() - start_time > timeout - 5:
                break
                
        return set(result)
    
    def try_improve(S, adj, n):
        improved = False
        S = set(S)
        
        verts = list(S)
        random.shuffle(verts)
        
        for v in verts:
            if time.time() - start_time > timeout:
                return S, False
                
            S_new = S - {v}
            candidates = [u for u in adj[v] if u not in S_new]
            
            valid_cands = []
            for u in candidates:
                ok = True
                for w in adj[u]:
                    if w in S_new:
                        ok = False
                        break
                if ok:
                    valid_cands.append(u)
            
            if valid_cands:
                rem_c = set(valid_cands)
                add_set = set()
                while rem_c:
                    min_v_c = None
                    min_deg_c = float('inf')
                    for vc in rem_c:
                        deg = sum(1 for uc in adj[vc] if uc in rem_c)
                        if deg < min_deg_c:
                            min_deg_c = deg
                            min_v_c = vc
                    add_set.add(min_v_c)
                    rem_c -= {min_v_c}
                    for uc in adj[min_v_c]:
                        if uc in rem_c:
                            rem_c.discard(uc)
                
                if len(add_set) > 0:
                    if len(S_new) + len(add_set) > len(S):
                        S = S_new | add_set
                        improved = True
                        break 
                        
        return S, improved
    
    # Initial solution
    best_sol = greedy_mis(adj, n)
    best_size = len(best_sol)
    
    # Local search phase
    while time.time() - start_time < timeout:
        sol, improved = try_improve(best_sol, adj, n)
        if len(sol) > best_size:
            best_size = len(sol)
            best_sol = sol
        elif not improved:
            if random.random() < 0.1:
                break
            new_sol = greedy_mis(adj, n, random_seed=int(time.time() * 1000))
            if len(new_sol) > best_size:
                best_size = len(new_sol)
                best_sol = new_sol
            sol2, _ = try_improve(best_sol, adj, n)
            if len(sol2) > best_size:
                best_size = len(sol2)
                best_sol = sol2

    return list(best_sol)
