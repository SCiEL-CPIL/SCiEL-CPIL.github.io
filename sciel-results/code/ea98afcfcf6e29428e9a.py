import time
from math import gcd

def construct():
    start = time.time()
    deadline = start + 85
    
    n = 256
    num_vertices = n - 1  # 255 vertices (2..256)
    
    # Precompute Bad[a][b] as bitmask of c
    bad = [[0] * (n + 1) for _ in range(n + 1)]
    
    for a in range(2, n + 1):
        for b in range(2, n + 1):
            if a == b:
                continue
            mask = 0
            for c in range(2, n + 1):
                if c == a or c == b:
                    continue
                if (b * c) % a == 0 or (a * c) % b == 0 or (a * b) % c == 0:
                    mask |= (1 << (c - 2))
            bad[a][b] = mask
            bad[b][a] = mask
            
    full_mask = (1 << num_vertices) - 1
    best_set = []
    
    def popcount(x):
        return bin(x).count('1')
        
    def mask_to_list(mask):
        return [i + 2 for i in range(num_vertices) if (mask >> i) & 1]
        
    def greedy_color_bound(cand_mask, S):
        # Build conflict graph on candidates
        # M_u = union of bad[u][w] for w in S, intersected with cand_mask
        M = [0] * (n + 1)
        for i in range(num_vertices):
            if not (cand_mask >> i) & 1:
                continue
            u = i + 2
            mask_u = 0
            for w in S:
                mask_u |= bad[u][w]
            M[u] = mask_u & cand_mask
            
        # Get list of candidates
        verts = []
        for i in range(num_vertices):
            if (cand_mask >> i) & 1:
                verts.append(i + 2)
                
        if not verts:
            return 0
            
        # Order by degree descending
        def degree(v):
            return popcount(M[v])
            
        verts.sort(key=degree, reverse=True)
        
        # Greedy Coloring
        colors = {}
        max_color = 0
        for v in verts:
            # Find smallest color not used by neighbors
            used = set()
            # Neighbors are u in verts such that u in M[v] or v in M[u]
            # Since graph is undirected (conflict is symmetric?), check symmetry.
            # bad[u][w] contains v iff {u,w,v} is bad.
            # {u,w,v} is bad is symmetric in u,v?
            # Conditions: u|wv, v|uw, w|uv.
            # If u|wv, then v|uw? Not necessarily.
            # But the triple is the same. So yes, if {u,w,v} is bad, it's bad regardless of order.
            # So M[u] contains v iff M[v] contains u?
            # M[u] contains v if v in bad[u][w].
            # bad[u][w] contains v if {u,w,v} is bad.
            # bad[v][w] contains u if {v,w,u} is bad.
            # Since the set {u,w,v} is the same, the condition is the same.
            # So yes, symmetric.
            
            # Iterate neighbors
            # To be fast, we can just check all previous vertices?
            # Or iterate all verts and check if in M[v].
            for u in verts:
                if u == v:
                    continue
                if M[v] & (1 << (u - 2)):
                    used.add(colors.get(u, -1))
                    
            # Assign smallest available color
            c = 0
            while c in used:
                c += 1
            colors[v] = c
            if c > max_color:
                max_color = c
                
        return max_color + 1 # Number of colors is upper bound on MIS
        
    def search(S, cand_mask):
        nonlocal best_set
        if time.time() > deadline:
            return
            
        current_size = len(S)
        
        # Upper bound
        # Simple bound
        ub_simple = current_size + popcount(cand_mask)
        if ub_simple <= len(best_set):
            return
            
        # Coloring bound (more expensive but tighter)
        # Only use if cand_mask is not too large to avoid overhead?
        # Or always use?
        # Let's try always.
        ub_color = current_size + greedy_color_bound(cand_mask, S)
        if ub_color <= len(best_set):
            return
            
        if cand_mask == 0:
            if current_size > len(best_set):
                best_set = list(S)
            return
            
        # Select v
        # Compute forbidden_mask for S
        forbidden_mask = 0
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                a = S[i]
                b = S[j]
                forbidden_mask |= bad[a][b]
                
        best_v = -1
        best_score = -1
        best_v_compatible = False
        best_v_removal = 0
        
        for i in range(num_vertices):
            if not (cand_mask >> i) & 1:
                continue
            v = i + 2
            v_bit = 1 << i
            
            compatible = not (forbidden_mask & v_bit)
            
            removal_mask = 0
            for w in S:
                removal_mask |= bad[v][w]
            removal_mask &= cand_mask
            
            score = 0
            if compatible:
                score += 1000
            score += popcount(removal_mask)
            
            if score > best_score:
                best_score = score
                best_v = v
                best_v_compatible = compatible
                best_v_removal = removal_mask
                
        if best_v == -1:
            return
            
        # Branch Include
        if best_v_compatible:
            new_S = S + [best_v]
            new_cand_mask = cand_mask & ~(1 << (best_v - 2))
            new_cand_mask &= ~best_v_removal
            search(new_S, new_cand_mask)
            
        # Branch Exclude
        new_cand_mask = cand_mask & ~(1 << (best_v - 2))
        search(S, new_cand_mask)
        
    # Greedy Init
    greedy_S = []
    greedy_mask = full_mask
    while greedy_mask:
        if time.time() > deadline - 5:
            break
        # Pick first compatible
        best_v = -1
        for i in range(num_vertices):
            if not (greedy_mask >> i) & 1:
                continue
            v = i + 2
            # Check compatible with greedy_S
            ok = True
            for a in greedy_S:
                for b in greedy_S:
                    if a == b: continue
                    if bad[a][b] & (1 << (v-2)):
                        ok = False
                        break
                if not ok: break
            if ok:
                best_v = v
                break
                
        if best_v == -1:
            break
            
        greedy_S.append(best_v)
        greedy_mask &= ~(1 << (best_v - 2))
        
        removal = 0
        for w in greedy_S:
            removal |= bad[best_v][w]
        greedy_mask &= ~removal
        
    if len(greedy_S) > len(best_set):
        best_set = list(greedy_S)
        
    search([], full_mask)
    
    return list(best_set)
