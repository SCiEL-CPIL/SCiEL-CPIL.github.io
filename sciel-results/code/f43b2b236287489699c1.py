import numpy as np
import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    def build_adj(n, orientation):
        """Build adjacency matrix from orientation list."""
        adj = np.zeros((n, n), dtype=bool)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if orientation[idx] == 0:
                    adj[i, j] = True
                else:
                    adj[j, i] = True
                idx += 1
        return adj

    def count_violations(n, adj):
        """Count 4-sets without a common dominator. Vectorized inner loop."""
        violations = 0
        # Precompute out-neighbors for each vertex as a set for fast intersection? 
        # No, let's stick to numpy boolean ops for speed.
        # For each 4-set {a,b,c,d}, check if any vertex dominates all others.
        # This is still O(n^4) which is slow for n=60 (4.5M sets).
        # We need a faster check or a smarter search.
        
        # Alternative: Precompute for each pair (u,v) if u->v.
        # For a quad (a,b,c,d), a dominates if adj[a,b] and adj[a,c] and adj[a,d].
        # Let's optimize the loop with numpy where possible, but pure Python loop over quads is the bottleneck.
        
        # For n=60, C(60,4) ~ 487k. This is manageable if the inner check is fast.
        # Let's try to make the inner check very fast using precomputed bitmasks or simple bools.
        
        # Precompute out_adj as list of sets? Or just use the matrix.
        # Matrix access adj[i,j] is fast.
        
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    for d in range(c+1, n):
                        # Check a
                        if adj[a,b] and adj[a,c] and adj[a,d]:
                            continue
                        # Check b
                        if adj[b,a] and adj[b,c] and adj[b,d]:
                            continue
                        # Check c
                        if adj[c,a] and adj[c,b] and adj[c,d]:
                            continue
                        # Check d
                        if adj[d,a] and adj[d,b] and adj[d,c]:
                            continue
                        violations += 1
        return violations

    # This is too slow for many iterations. We need a smarter strategy.
    # Strategy: "Dominator Cover"
    # Let D be a set of dominators. Let L be leaves.
    # Orient edges between D and L such that each leaf l is dominated by a subset S_l of D.
    # Orient edges within D as a tournament T_D.
    # Orient edges within L randomly (or as a tournament T_L).
    # A 4-set Q has a dominator if:
    # 1. It contains a dominator d that dominates all other 3 in Q.
    # 2. It contains no dominators (all leaves). Then we need a leaf to dominate the other 3.
    
    # To simplify, let's force all 4-sets with leaves to be dominated by a dominator if possible.
    # This requires that for any 3 leaves l1, l2, l3, there is a dominator d that dominates all three.
    # If we have k dominators, and each leaf is dominated by ~k/2 dominators, the probability that 3 specific leaves share a common dominator is low if k is small.
    # However, if we make the dominators "strong", i.e., each dominator dominates most leaves, then it's easier.
    
    # Let's try a different structural approach:
    # Random tournament + Local Search with **incremental** violation counting is hard to implement correctly and fast in Python.
    
    # Let's try **Simulated Annealing** or **Tabu Search** on the orientation bits, but with a **faster violation check**.
    # The bottleneck is the O(n^4) check.
    # Can we update violations incrementally when flipping a bit?
    # Flipping edge (u,v) affects all 4-sets containing u and v.
    # Number of such sets is C(n-2, 2) ~ n^2/2.
    # For n=60, that's ~870 sets.
    # For each such set, we need to re-evaluate if it has a dominator.
    # This is O(n^2) per flip.
    # Total time per step: O(n^2).
    # Total steps: maybe 1000?
    # Total time: 1000 * 3600 = 3.6M ops. This is feasible!
    
    # Let's implement incremental local search.
    
    def incremental_local_search(n, orientation, max_steps=5000, time_limit=100):
        adj = build_adj(n, orientation)
        
        # Precompute list of 4-sets? No, too much memory.
        # Instead, for a flipped edge (u,v), we need to check all 4-sets containing u and v.
        # A 4-set is {u, v, x, y}.
        
        # We need a function to check if a specific 4-set has a dominator.
        def quad_has_dominator(a, b, c, d):
            # Check a
            if adj[a,b] and adj[a,c] and adj[a,d]: return True
            # Check b
            if adj[b,a] and adj[b,c] and adj[b,d]: return True
            # Check c
            if adj[c,a] and adj[c,b] and adj[c,d]: return True
            # Check d
            if adj[d,a] and adj[d,b] and adj[d,c]: return True
            return False
            
        # Initial violation count
        current_violations = count_violations(n, adj)
        
        # Map from edge index to (u, v)
        edge_map = {}
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                edge_map[idx] = (i, j)
                idx += 1
                
        # To speed up, we can precompute the set of 4-sets for each edge?
        # No, just generate them on the fly.
        
        # We need to know the *change* in violations when flipping an edge.
        # Let V(S) be 1 if S has no dominator, 0 otherwise.
        # When flipping (u,v), the status of any 4-set NOT containing u or v is unchanged.
        # The status of 4-sets containing u and v may change.
        # Let S be a 4-set containing u and v.
        # Let old_status = V(S) before flip.
        # Let new_status = V(S) after flip.
        # Delta = new_status - old_status.
        
        # We can iterate through all 4-sets containing u and v, calculate old_status and new_status, and sum the deltas.
        
        # Optimization: 
        # For a 4-set {u, v, x, y}, the only edges involved in the dominator check for vertices u, v, x, y are:
        # (u,v), (u,x), (u,y), (v,x), (v,y), (x,y).
        # Flipping (u,v) only changes the direction of u->v or v->u.
        # This affects whether u dominates v (and vice versa).
        # It does NOT affect whether u dominates x, v dominates x, etc.
        # So, for a fixed 4-set {u, v, x, y}:
        # - If u was a dominator before, it required adj[u,v]=True. If we flip to False, u is no longer a dominator.
        # - If v was a dominator before, it required adj[v,u]=True. If we flip to False, v is no longer a dominator.
        # - x and y dominance is unaffected by the flip of (u,v) UNLESS x or y is u or v (which they aren't).
        # Wait, x and y dominate u or v?
        # If x dominates u, it needs adj[x,u]=True. This is unchanged.
        # So, the only vertices whose "dominator status" can change are u and v.
        
        # Let's refine:
        # For a 4-set S = {u, v, x, y}:
        # Let D_old be the set of dominators in S before flip.
        # Let D_new be the set of dominators in S after flip.
        # The only vertices in D_old \ D_new or D_new \ D_old are u and v.
        # Specifically:
        # - u is in D_new iff (u in D_old and adj[u,v] was True and remains True) OR (u not in D_old and adj[u,v] becomes True and u dominates x,y).
        # Actually, simpler:
        # u dominates S\{u} iff adj[u,v] and adj[u,x] and adj[u,y].
        # v dominates S\{v} iff adj[v,u] and adj[v,x] and adj[v,y].
        # x dominates S\{x} iff adj[x,u] and adj[x,v] and adj[x,y]. (Unchanged by flip of u-v)
        # y dominates S\{y} iff adj[y,u] and adj[y,v] and adj[y,x]. (Unchanged by flip of u-v)
        
        # So, let:
        # ux = adj[u,x], uy = adj[u,y]
        # vx = adj[v,x], vy = adj[v,y]
        # xy = adj[x,y] (not needed for x/y dominance of u/v, but needed for x/y dominance of each other? No, x dominates y if adj[x,y].)
        
        # Let's precompute for each pair (u,v) and each pair (x,y):
        # The "potential" for u to dominate {u,v,x,y} given adj[u,x], adj[u,y] is just (adj[u,x] and adj[u,y]).
        # Let's call this P_u.
        # Similarly P_v = (adj[v,x] and adj[v,y]).
        # Let X_dom = adj[x,u] and adj[x,v] and adj[x,y].
        # Let Y_dom = adj[y,u] and adj[y,v] and adj[y,x].
        
        # Before flip:
        # u_dom_old = P_u and adj[u,v]
        # v_dom_old = P_v and adj[v,u]
        # x_dom = X_dom
        # y_dom = Y_dom
        # old_has_dom = u_dom_old or v_dom_old or x_dom or y_dom
        
        # After flip (adj[u,v] becomes not adj[u,v]):
        # u_dom_new = P_u and (not adj[u,v])
        # v_dom_new = P_v and (not adj[v,u])  # which is P_v and adj[u,v]
        # x_dom, y_dom unchanged.
        # new_has_dom = u_dom_new or v_dom_new or x_dom or y_dom
        
        # This allows O(1) update per 4-set!
        
        # We need to iterate over all x, y such that x,y != u,v and x < y (and x,y > v? No, just distinct).
        # Actually, the 4-set is {u,v,x,y}. We can just iterate x in range(n), y in range(x+1, n) where x,y not in {u,v}.
        
        # Let's implement this incremental check.
        
        best_orientation = orientation[:]
        best_violations = current_violations
        
        # Precompute P_u, P_v, X_dom, Y_dom for all pairs? 
        # That's O(n^4) memory. Too much.
        # But we can compute them on the fly for a specific edge flip.
        
        # For a flip of edge (u,v):
        # We iterate x, y.
        # Cost: O(n^2) per flip.
        
        # Let's try it.
        
        t_start = time.time()
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Pick a random edge
            idx = random.randint(0, len(orientation) - 1)
            u, v = edge_map[idx]
            
            old_uv = adj[u, v]
            
            # Calculate delta violations
            delta = 0
            # Iterate over all x, y
            # We need to consider all pairs (x,y) from the remaining n-2 vertices.
            # To avoid double counting or missing, let's just iterate all x != u,v and y != u,v, x < y.
            # There are C(n-2, 2) such pairs.
            
            # Precompute lists of x,y? No, just nested loops.
            for x in range(n):
                if x == u or x == v: continue
                for y in range(x+1, n):
                    if y == u or y == v: continue
                    
                    # Compute P_u, P_v
                    # adj is the CURRENT adjacency (before flip)
                    pu = adj[u, x] and adj[u, y]
                    pv = adj[v, x] and adj[v, y]
                    
                    # Compute x_dom, y_dom (unchanged by flip)
                    # x dominates u, v, y
                    x_dom = adj[x, u] and adj[x, v] and adj[x, y]
                    # y dominates u, v, x
                    y_dom = adj[y, u] and adj[y, v] and adj[y, x]
                    
                    # Old status
                    u_dom_old = pu and old_uv
                    v_dom_old = pv and (not old_uv) # adj[v,u] is not adj[u,v]
                    old_has = u_dom_old or v_dom_old or x_dom or y_dom
                    
                    # New status
                    # Flip u->v
                    new_uv = not old_uv
                    u_dom_new = pu and new_uv
                    v_dom_new = pv and (not new_uv) # adj[v,u] becomes new_uv
                    new_has = u_dom_new or v_dom_new or x_dom or y_dom
                    
                    if old_has and not new_has:
                        delta += 1
                    elif new_has and not old_has:
                        delta -= 1
            
            if delta < 0:
                # Accept flip
                orientation[idx] = 1 - orientation[idx]
                adj[u, v] = not adj[u, v]
                adj[v, u] = not adj[v, u]
                current_violations += delta
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_orientation = orientation[:]
                
                if current_violations == 0:
                    break
            else:
                # Reject (or accept with prob exp(-delta/T) for SA, but let's stick to greedy first)
                pass
        
        return best_orientation, best_violations

    # Main Loop
    best_n = None
    best_orientation = None
    
    # Try n from 60 down to 35
    for n in range(60, 34, -1):
        if time.time() > deadline:
            break
        
        # Multiple restarts
        for restart in range(10):
            if time.time() > deadline:
                break
            
            # Random tournament
            orientation = [random.randint(0, 1) for _ in range(n * (n-1) // 2)]
            
            # Local search
            orientation, violations = incremental_local_search(n, orientation, max_steps=2000)
            
            if violations == 0:
                best_n = n
                best_orientation = orientation
                break
        
        if best_n is not None:
            break
    
    # If we found a valid tournament, return it
    if best_n is not None:
        return {'n': best_n, 'orientation': best_orientation}
    
    # Fallback: return Paley on 83
    p = 83
    squares = {x * x % p for x in range(1, p)}
    orientation = [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]
    return {'n': p, 'orientation': orientation}
