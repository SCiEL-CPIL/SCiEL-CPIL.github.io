import sys
import time
import math

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    full_mask = (1 << n) - 1
    
    # Build adjacency bitmask
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Precompute is_ind and is_clique
    is_ind = [False] * (1 << n)
    is_ind[0] = True
    is_clique = [False] * (1 << n)
    is_clique[0] = True
    
    for s in range(1, 1 << n):
        lsb = s & -s
        v = lsb.bit_length() - 1
        rest = s ^ lsb
        is_ind[s] = is_ind[rest] and ((adj[v] & rest) == 0)
        is_clique[s] = is_clique[rest] and ((adj[v] & rest) == rest)

    # Find all maximal independent sets
    maximal_ind = []
    for s in range(1, 1 << n):
        if is_ind[s]:
            # Check if s is maximal
            is_max = True
            # Check if any vertex outside s can be added
            # A vertex v outside s can be added if (adj[v] & s) == 0
            # We iterate over vertices not in s
            not_in_s = full_mask ^ s
            temp = not_in_s
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                if (adj[v] & s) == 0:
                    is_max = False
                    break
                temp ^= lsb
            if is_max:
                maximal_ind.append(s)
                
    # Find all maximal cliques
    maximal_cliques = []
    for s in range(1, 1 << n):
        if is_clique[s]:
            # Check if s is maximal
            is_max = True
            not_in_s = full_mask ^ s
            temp = not_in_s
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                # v can be added if it is connected to all in s
                # i.e., (adj[v] & s) == s
                if (adj[v] & s) == s:
                    is_max = False
                    break
                temp ^= lsb
            if is_max:
                maximal_cliques.append(s)

    # Combine all candidate sets
    # Each candidate is (mask, type) where type 0 is independent, 1 is clique
    candidates = []
    for m in maximal_ind:
        candidates.append((m, 0))
    for m in maximal_cliques:
        candidates.append((m, 1))
        
    # Remove duplicates? A set cannot be both a maximal independent set and a maximal clique unless n=1 or specific cases.
    # Actually, if a set is both, it's a singleton (in a simple graph without loops, a set of size >1 cannot be both).
    # Singletons are both. So we might have duplicates if a singleton is maximal in both.
    # Let's just keep them all, branch and bound will handle it.
    # Or better, deduplicate by mask? No, the type matters for validity check?
    # No, if a mask is valid as independent, it's valid. If valid as clique, it's valid.
    # The constraint is "clique OR independent".
    # So if a mask is both, it's fine.
    # We just need to cover the vertices with disjoint sets from this list.
    # Wait, if I use a set that is an independent set, I don't care if it's also a clique.
    # So we can just treat the list as a set of "valid homogeneous sets".
    # Let's deduplicate masks to reduce search space?
    # If mask M is in maximal_ind, and also in maximal_cliques, it appears twice.
    # In B&B, picking M twice is impossible (disjointness).
    # Picking M once covers the same vertices.
    # So yes, deduplicate masks.
    
    unique_masks = list(set([m for m, t in candidates]))
    
    # Sort candidates for better pruning?
    # Larger sets first might be better?
    # Or sets that cover "hard" vertices first?
    # Let's just sort by size descending.
    unique_masks.sort(key=lambda x: bin(x).count('1'), reverse=True)
    
    m_count = len(unique_masks)
    
    # Lower bound for a set of remaining vertices R:
    # We need at least ceil(|R| / max_size) parts?
    # Better: Coloring number of the subgraph?
    # Since we can use cliques or independent sets, it's the cochromatic number of the induced subgraph.
    # Hard to compute on the fly.
    # Simple LB: 1 if R non-empty.
    # Or: If R has an edge, we need at least 1 part.
    # Let's use a simple heuristic:
    # Max independent set size in R?
    # Precompute max independent set size for all subsets? 2^n * n is too slow for on-the-fly?
    # We have is_ind.
    # max_ind_size[R] = max { |I| : I subset R, is_ind[I] }
    # This is 3^n precomputation. 3^14 = 4.7M. Acceptable.
    
    max_ind_size = [0] * (1 << n)
    # max_clique_size[R] = max { |C| : C subset R, is_clique[C] }
    max_clique_size = [0] * (1 << n)
    
    # Actually, we just need the max size of any valid part in R.
    # valid_size[R] = max( max_ind_size[R], max_clique_size[R] )
    # LB = ceil(|R| / valid_size[R])
    
    # Precompute valid_size for all R
    # This is expensive if done naively.
    # But we can compute max_ind_size using DP?
    # max_ind_size[R] = max( max_ind_size[R without v], max_ind_size[R without v] + 1 if v can be added? No.
    # Standard DP for max independent set is O(n 2^n).
    
    # Let's just precompute max_ind_size and max_clique_size.
    # max_ind_size[0] = 0
    # for s in 1..2^n:
    #   v = lsb
    #   rest = s ^ v
    #   if is_ind[s]:
    #       max_ind_size[s] = max(max_ind_size[s], popcount(s)) # If s itself is ind, it's a candidate
    #   # Also consider subsets of s that don't include v
    #   # This is getting complicated.
    
    # Alternative:
    # Iterate all independent sets I. For each I, update all supersets R of I?
    # That is 3^n.
    # For each I (independent), for each R superset of I:
    #    max_ind_size[R] = max(max_ind_size[R], |I|)
    
    # This is efficient enough.
    
    # Initialize
    for i in range(1 << n):
        max_ind_size[i] = 0
        max_clique_size[i] = 0
        
    # Update for independent sets
    for i in range(1 << n):
        if is_ind[i]:
            size_i = bin(i).count('1')
            if size_i > 0:
                # Iterate supersets of i
                # R = i | X where X is subset of full_mask ^ i
                # This is 2^(n-|i|) iterations.
                # Total sum over i of 2^(n-|i|) = 3^n.
                not_i = full_mask ^ i
                x = not_i
                while True:
                    r = i | x
                    if size_i > max_ind_size[r]:
                        max_ind_size[r] = size_i
                    if x == 0:
                        break
                    x = (x - 1) & not_i
                    
    # Update for cliques
    for i in range(1 << n):
        if is_clique[i]:
            size_i = bin(i).count('1')
            if size_i > 0:
                not_i = full_mask ^ i
                x = not_i
                while True:
                    r = i | x
                    if size_i > max_clique_size[r]:
                        max_clique_size[r] = size_i
                    if x == 0:
                        break
                    x = (x - 1) & not_i
                    
    # Precompute valid_max_size[R] = max(max_ind_size[R], max_clique_size[R])
    # And popcount[R]
    popcount = [0] * (1 << n)
    for i in range(1, 1 << n):
        popcount[i] = popcount[i >> 1] + (i & 1)
        
    valid_max_size = [0] * (1 << n)
    for i in range(1 << n):
        valid_max_size[i] = max(max_ind_size[i], max_clique_size[i])
        
    # Branch and Bound
    # State: current_cover (mask of vertices covered), current_parts (list of masks), current_count
    # We want to minimize current_count.
    
    best_count = n + 1
    best_solution = None
    
    # Order candidates?
    # We have unique_masks.
    # We need to ensure we don't pick the same mask twice.
    # Since we pick disjoint sets, and we iterate through a list, we can just keep an index.
    # But the order matters for pruning.
    
    # Let's sort unique_masks by some heuristic.
    # Maybe by "difficulty" or just size.
    
    # Recursive search
    # To avoid recursion depth issues (though n=14 is small, depth <= 14), recursion is fine.
    
    start_time = time.time()
    deadline = start_time + 50.0 # 50s limit, 10s margin
    
    # Heuristic: Try larger sets first?
    # Or sets that cover high-degree vertices?
    
    # Let's use a simple DFS.
    # We maintain a list of available candidate indices.
    # To allow any order, we can just iterate through all candidates at each step?
    # No, that's exponential in m.
    # Standard set cover B&B:
    # 1. Pick a vertex v in R (remaining).
    # 2. Try all candidates that contain v.
    # 3. Recurse.
    
    # This is much better.
    
    # Precompute for each vertex v, the list of candidate indices that contain v.
    # candidates_idx = list(range(len(unique_masks)))
    # v_to_cands = {v: [] for v in range(n)}
    # for idx, mask in enumerate(unique_masks):
    #     for v in range(n):
    #         if mask & (1 << v):
    #             v_to_cands[v].append(idx)
            
    # But we must ensure we don't reuse a candidate.
    # We can track used candidates in a set or bitmask?
    # m can be up to 2^n. Bitmask is fine.
    
    # However, the "pick a vertex" strategy requires that we only consider candidates that contain that vertex.
    # This naturally avoids reusing candidates if we remove them from consideration?
    # No, we need to track which candidates are already used.
    
    # Let's try this approach:
    # def search(rem_mask, used_mask, count, parts):
    #   if time.time() > deadline: return
    #   if rem_mask == 0:
    #      update best
    #      return
    #   if count >= best_count: return
    #   
    #   # Lower bound
    #   lb = (popcount[rem_mask] + valid_max_size[rem_mask] - 1) // valid_max_size[rem_mask]
    #   if count + lb >= best_count: return
    #   
    #   # Pick a vertex v in rem_mask
    #   # Heuristic: pick vertex with fewest candidates?
    #   v_mask = rem_mask
    #   # Just pick lsb
    #   v = (v_mask & -v_mask).bit_length() - 1
    #   
    #   # Iterate candidates containing v
    #   for idx in v_to_cands[v]:
    #       if used_mask & (1 << idx): continue
    #       m = unique_masks[idx]
    #       # Check if m is subset of rem_mask
    #       if (m & rem_mask) != m: continue
    #       
    #       # Recurse
    #       new_rem = rem_mask ^ m
    #       new_used = used_mask | (1 << idx)
    #       parts.append(m)
    #       search(new_rem, new_used, count + 1, parts)
    #       parts.pop()
        
    # This looks good.
    
    # Precompute v_to_cands
    v_to_cands = [[] for _ in range(n)]
    for idx, mask in enumerate(unique_masks):
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            v_to_cands[v].append(idx)
            m ^= lsb
            
    # Sort candidates for each vertex?
    # Maybe by size descending?
    for v in range(n):
        v_to_cands[v].sort(key=lambda idx: bin(unique_masks[idx]).count('1'), reverse=True)
        
    # Global variables for best
    best_count = n + 1
    best_parts_mask = []
    
    def search(rem_mask, used_mask, count, parts):
        nonlocal best_count, best_parts_mask
        if time.time() > deadline:
            return
        if rem_mask == 0:
            if count < best_count:
                best_count = count
                best_parts_mask = parts.copy()
            return
            
        # Pruning
        if count >= best_count:
            return
            
        # Lower bound
        # valid_max_size[rem_mask] is the size of the largest valid part in rem_mask.
        # If it's 0, it's a bug (shouldn't happen as singletons are valid).
        vmax = valid_max_size[rem_mask]
        if vmax == 0:
             return
        lb = (popcount[rem_mask] + vmax - 1) // vmax
        if count + lb >= best_count:
            return
            
        # Pick a vertex
        # Heuristic: pick vertex with minimum degree in the "conflict graph" of candidates?
        # Or just pick the vertex with the fewest available candidates.
        # Let's try picking the vertex with the fewest available candidates.
        v_mask = rem_mask
        best_v = -1
        min_cands = n + 1
        temp_v = v_mask
        while temp_v:
            lsb = temp_v & -temp_v
            v = lsb.bit_length() - 1
            # Count available candidates for v
            avail = 0
            for idx in v_to_cands[v]:
                if not (used_mask & (1 << idx)):
                    # Check if candidate is subset of rem_mask
                    if (unique_masks[idx] & rem_mask) == unique_masks[idx]:
                        avail += 1
            if avail < min_cands:
                min_cands = avail
                best_v = v
            temp_v ^= lsb
            
        if best_v == -1 or min_cands == 0:
            return
            
        # Iterate candidates for best_v
        for idx in v_to_cands[best_v]:
            if used_mask & (1 << idx):
                continue
            m = unique_masks[idx]
            if (m & rem_mask) != m:
                continue
                
            new_rem = rem_mask ^ m
            new_used = used_mask | (1 << idx)
            parts.append(m)
            search(new_rem, new_used, count + 1, parts)
            parts.pop()
            
    search(full_mask, 0, 0, [])
    
    if best_parts_mask:
        # Convert masks to lists of vertices
        result = []
        for m in best_parts_mask:
            verts = []
            temp = m
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                verts.append(v)
                temp ^= lsb
            result.append(verts)
        return result
    else:
        # Fallback
        return [[i] for i in range(n)]
