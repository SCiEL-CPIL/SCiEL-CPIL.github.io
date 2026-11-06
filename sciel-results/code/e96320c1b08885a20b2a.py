import time
import random
import math

def construct():
    n = 16
    target_edges = 65
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Helper to create seed
    def get_seed_masks():
        masks = [0] * n
        # K_8,8
        for u in range(8):
            for v in range(8, 16):
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
        # Add (0,1)
        masks[0] |= (1 << 1)
        masks[1] |= (1 << 0)
        return masks

    def masks_to_edges(masks):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if (masks[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def calc_score(masks):
        deg = [0] * n
        for u in range(n):
            deg[u] = bin(masks[u]).count('1')
        
        max_sum = 0
        has_tri = False
        for u in range(n):
            mu = masks[u]
            if mu == 0: continue
            # Iterate neighbors of u
            v = (mu & -mu)
            while v:
                v_idx = v.bit_length() - 1
                mv = masks[v_idx]
                # Common neighbors of u and v are neighbors of u (excluding v) AND neighbors of v
                # We want w > v to avoid duplicates? 
                # Actually, just check all w.
                # To be efficient:
                common = mu & mv
                # Remove u and v from common? 
                # u is not in mu (no self loop), v is in mu.
                # We want w such that u-w and v-w exist.
                # So w must be in mu and mv.
                # We can iterate w in common.
                # But we need to ensure we don't double count? 
                # The sum is symmetric. We just need the max.
                # So we can just iterate all w in common.
                w = common
                while w:
                    w_idx = w.bit_length() - 1
                    # Check if w is u or v? 
                    # u is not in mu.
                    # v is in mu. If w == v, then v-v edge? No.
                    # So w != u, w != v is guaranteed if graph is simple and no self loops?
                    # Wait, v is a neighbor of u, so bit v is set in mu.
                    # u is a neighbor of v, so bit u is set in mv.
                    # common = mu & mv.
                    # Does common contain u? 
                    # mu has bit v set. mv has bit u set.
                    # So common has bit v (from mu) and bit u (from mv)?
                    # Yes.
                    # So w could be u or v.
                    # If w == u: triangle u-v-u? No, simple graph.
                    # If w == v: triangle u-v-v? No.
                    # So we must exclude u and v from common.
                    common_no_uv = common & ~(1 << u) & ~(1 << v_idx)
                    w = common_no_uv
                    while w:
                        w_idx = w.bit_length() - 1
                        s = deg[u] + deg[v_idx] + deg[w_idx]
                        if s > max_sum:
                            max_sum = s
                        has_tri = True
                        w &= w - 1 # clear lowest set bit
                    v &= v - 1 # clear lowest set bit of v
        if not has_tri:
            return 1000
        return max_sum

    best_masks = get_seed_masks()
    best_score = calc_score(best_masks)
    
    current_masks = [m for m in best_masks]
    current_score = best_score
    
    T = 5.0
    cooling = 0.9995
    min_T = 0.05
    
    # Precompute list of all possible edges for random add
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Cache non-edges? 
    # Just sample.
    
    no_improve_count = 0
    max_no_improve = 10000
    
    while time.time() < deadline:
        # Move: remove random edge, add random non-edge
        # To be smarter, remove edge from a high-sum triangle if possible
        
        # Find a bad triangle
        # We can just do random swap for simplicity, but biased is better.
        
        # Let's stick to random swap first, it's robust.
        
        # Pick edge to remove
        # We need a list of edges in current_masks
        # Rebuilding list is O(n^2) = 256. Cheap.
        current_edge_list = []
        for u in range(n):
            for v in range(u+1, n):
                if (current_masks[u] >> v) & 1:
                    current_edge_list.append((u, v))
        
        rem_u, rem_v = random.choice(current_edge_list)
        
        # Pick edge to add
        while True:
            add_u, add_v = random.choice(all_edges)
            if not ((current_masks[add_u] >> add_v) & 1):
                break
        
        # Apply move
        new_masks = current_masks[:]
        new_masks[rem_u] ^= (1 << rem_v)
        new_masks[rem_v] ^= (1 << rem_u)
        new_masks[add_u] ^= (1 << add_v)
        new_masks[add_v] ^= (1 << add_u)
        
        new_score = calc_score(new_masks)
        
        if new_score < current_score:
            current_masks = new_masks
            current_score = new_score
            no_improve_count = 0
            if current_score < best_score:
                best_score = current_score
                best_masks = current_masks[:]
        else:
            no_improve_count += 1
            delta = new_score - current_score
            if delta > 0:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_masks = new_masks
                    current_score = new_score
        
        T *= cooling
        if T < min_T:
            T = min_T
            
        # Restart if stuck
        if no_improve_count > max_no_improve:
            # Restart from best
            current_masks = best_masks[:]
            current_score = best_score
            no_improve_count = 0
            T = 2.0 # Reset temp
            
    edges_list = masks_to_edges(best_masks)
    return {'n': n, 'edges': edges_list}
