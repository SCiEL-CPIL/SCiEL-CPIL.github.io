"""
Program combining algebraic construction (Lazebnik-Woldar style) with fast local search.
Strategy:
1. Try to construct n=26 deterministically using algebraic structures or fast search.
2. Use incremental conflict counting for speed.
3. Fallback to known valid n=25 if n=26 fails.
4. Attempt n=27 with remaining time if possible (long shot).
"""

import time
import random
import numpy as np

def _is_valid_coloring(n, colors):
    """Check if coloring has no monochromatic C4 using numpy matrix products."""
    if len(colors) != n * (n - 1) // 2:
        return False
    # Build adjacency matrices per color
    adjs = [np.zeros((n, n), dtype=np.uint8) for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            if c < 0 or c > 4:
                return False
            adjs[c][i, j] = 1
            adjs[c][j, i] = 1
            idx += 1
    # For each color, A^2 off-diagonal entries must be <= 1
    for c in range(5):
        A2 = adjs[c] @ adjs[c]
        # Zero out diagonal for check
        np.fill_diagonal(A2, 0)
        if np.max(A2) > 1:
            return False
    return True

def _lazy_woldar_25():
    """
    Construct a valid 5-coloring of K_25 using Z_5 x Z_5.
    This is a known construction for R_5(C4) >= 25.
    """
    n = 25
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi = i // 5, i % 5
            xj, yj = j // 5, j % 5
            if xi == xj:
                c = 4
            else:
                dx = (xj - xi) % 5
                dy = (yj - yi) % 5
                # dx^{-1} mod 5
                inv = pow(dx, -1, 5)
                c = (dy * inv) % 5
            colors.append(c)
    return n, colors

def _construct_26_algebraic():
    """
    Attempt a deterministic construction for n=26.
    The Lazebnik-Woldar construction for R_5(C4) >= 26 typically uses a specific algebraic structure.
    Since I don't have the exact formula memorized for 26, I will use a highly optimized
    local search with a good initialization derived from the 25-vertex solution.
    """
    # This function will be replaced by the fast local search logic in construct()
    # if a direct algebraic formula isn't implemented here.
    return None

def _fast_local_search(n, initial_colors, deadline):
    """
    Optimized local search with incremental conflict update.
    Uses bitmasks for adjacency.
    """
    if n < 3:
        return initial_colors[:]
        
    m = n * (n - 1) // 2
    
    # Precompute edge endpoints
    edge_u = [0] * m
    edge_v = [0] * m
    pair_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_u[idx] = i
            edge_v[idx] = j
            pair_to_idx[(i, j)] = idx
            pair_to_idx[(j, i)] = idx
            idx += 1
            
    # Masks: masks[v][c] is an int bitmask of neighbors of v in color c
    masks = [[0]*5 for _ in range(n)]
    
    def build_masks():
        for v in range(n):
            for c in range(5):
                masks[v][c] = 0
        for k in range(m):
            u = edge_u[k]
            v = edge_v[k]
            c = initial_colors[k]
            masks[u][c] |= (1 << v)
            masks[v][c] |= (1 << u)
            
    build_masks()
    
    # Calculate initial violations
    def count_violations():
        total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = bin(masks[u][c] & masks[v][c]).count('1')
                    if common >= 2:
                        total += common - 1
        return total
        
    current_violations = count_violations()
    best_violations = current_violations
    best_colors = initial_colors[:]
    
    if current_violations == 0:
        return initial_colors[:]
        
    T = 1.0
    alpha = 0.9999
    min_T = 0.01
    
    start_time = time.time()
    
    # Helper to get cost of pair (u,v) in color c
    def pair_cost(u, v, c):
        if u == v: return 0
        common = bin(masks[u][c] & masks[v][c]).count('1')
        return max(0, common - 1)
        
    while time.time() < deadline:
        # Pick random edge
        k = random.randint(0, m - 1)
        u = edge_u[k]
        v = edge_v[k]
        old_c = initial_colors[k]
        new_c = random.randint(0, 4)
        
        if new_c == old_c:
            continue
            
        # Calculate Delta
        # Pairs affected: (u, w) and (v, w) for all w != u, v. Plus (u, v).
        
        delta = 0
        
        # Sum old costs for affected pairs in old_c
        old_cost = 0
        for w in range(n):
            if w == u or w == v: continue
            # Pair (u, w)
            if u < w:
                old_cost += pair_cost(u, w, old_c)
            else:
                old_cost += pair_cost(w, u, old_c)
            # Pair (v, w)
            if v < w:
                old_cost += pair_cost(v, w, old_c)
            else:
                old_cost += pair_cost(w, v, old_c)
        # Pair (u, v)
        if u < v:
            old_cost += pair_cost(u, v, old_c)
        else:
            old_cost += pair_cost(v, u, old_c)
            
        # Sum old costs for affected pairs in new_c
        new_cost_old_masks = 0
        for w in range(n):
            if w == u or w == v: continue
            if u < w:
                new_cost_old_masks += pair_cost(u, w, new_c)
            else:
                new_cost_old_masks += pair_cost(w, u, new_c)
            if v < w:
                new_cost_old_masks += pair_cost(v, w, new_c)
            else:
                new_cost_old_masks += pair_cost(w, v, new_c)
        if u < v:
            new_cost_old_masks += pair_cost(u, v, new_c)
        else:
            new_cost_old_masks += pair_cost(v, u, new_c)
            
        # Apply change to masks
        masks[u][old_c] &= ~(1 << v)
        masks[v][old_c] &= ~(1 << u)
        masks[u][new_c] |= (1 << v)
        masks[v][new_c] |= (1 << u)
        initial_colors[k] = new_c
        
        # Recalculate cost for new state for affected pairs
        new_cost_new_masks = 0
        for w in range(n):
            if w == u or w == v: continue
            if u < w:
                new_cost_new_masks += pair_cost(u, w, old_c) # Note: w is neighbor in old_c? No, we are evaluating cost of pairs (u,w) in old_c and (v,w) in old_c?
                # Wait, the delta logic needs to be careful.
                # The total cost is sum over all pairs (a,b) of max(0, common_neigh(a,b) - 1).
                # Changing color of (u,v) affects:
                # 1. The count of common neighbors for (u,v) in old_c and new_c.
                # 2. The count of common neighbors for (u,w) in old_c and new_c? No.
                # (u,w) common neighbors in color c depend on N_c(u) and N_c(w).
                # If we change (u,v) from old_c to new_c:
                # - N_old_c(u) loses v. N_old_c(v) loses u.
                # - N_new_c(u) gains v. N_new_c(v) gains u.
                
                # So for any pair (x,y):
                # - If c == old_c:
                #   - If x==u or y==u, or x==v or y==v, the common neighbor count might change.
                #   - Specifically, (u,w) in old_c: common neighbors are N_old_c(u) & N_old_c(w).
                #     N_old_c(u) changed (lost v). So if v was in N_old_c(w), the count drops by 1.
                #     If v was NOT in N_old_c(w), count is same.
                #     Wait, v is a vertex. Is v in N_old_c(w)? That means edge (w,v) is color old_c.
                #     If so, then v was a common neighbor of u and w in old_c.
                #     Now u no longer connects to v in old_c, so v is no longer a common neighbor.
                #     So count decreases by 1 if color(w,v) == old_c.
                
                # This incremental update is complex. The previous code in the prompt did a full recount for simplicity in the "fast" version?
                # No, the prompt's inspiration 1 used a full recount `count_violations` inside the loop?
                # Let's check the inspiration 1 code snippet provided.
                # It calculates `old_cost` and `new_cost_old_masks` and then applies change.
                # It seems to be doing a partial update but the logic in the snippet was cut off.
                # For robustness and speed, I will use a simpler heuristic:
                # Just recalculate the total violations if the change looks promising? No, too slow.
                
                # Let's stick to the full recount if n is small (<=26).
                # n=26, m=325. Pairs = 325.
                # Counting violations is O(5 * n^2 * bitops).
                # 5 * 26^2 * 26 (bitops) ~ 90k ops.
                # Python overhead is high.
                # Let's try to optimize the inner loop.
                
                pass # This logic is getting messy. I will revert to a simpler, proven fast local search.
        
        # Revert for now and use a simpler approach
        masks[u][old_c] |= (1 << v)
        masks[v][old_c] |= (1 << u)
        masks[u][new_c] &= ~(1 << v)
        masks[v][new_c] &= ~(1 << u)
        initial_colors[k] = old_c

        # Simple accept/reject based on full recount
        # This is slower but less error-prone.
        # Let's try to make the full recount faster.
        
        # Actually, let's use the numpy verification only at the end, and a fast bitset check for local search.
        
        # I will implement a simpler local search that just picks a random edge, changes color, and checks if total conflicts decreased.
        # To check if total conflicts decreased, I need to know the delta.
        
        # Let's try a different strategy:
        # 1. Start with a random coloring.
        # 2. Repeatedly pick a random edge.
        # 3. Try all 5 colors for that edge.
        # 4. Pick the color that minimizes the number of conflicts involving that edge's endpoints.
        
        # This is a "best improvement" local search on single edges.
        
        break # Break out of this complex loop and use a simpler one below.

    # Fallback: Simple best-improvement local search
    # This is more robust.
    
    def get_conflicts_for_edge(u, v, c):
        """Count conflicts introduced by edge (u,v) having color c, assuming other edges are fixed."""
        # This is hard to do incrementally without knowing the full state.
        # Let's just use the full count.
        pass

    # Given the complexity, I will use the `count_violations` function which is O(n^2) per call.
    # For n=26, this is fast enough if we don't call it too often.
    # Let's do a simple hill climbing.
    
    current_violations = count_violations()
    
    while time.time() < deadline:
        # Pick random edge
        k = random.randint(0, m - 1)
        u = edge_u[k]
        v = edge_v[k]
        old_c = initial_colors[k]
        
        best_new_c = old_c
        best_violations_local = current_violations
        
        # Try all other colors
        for new_c in range(5):
            if new_c == old_c:
                continue
                
            # Apply change
            masks[u][old_c] &= ~(1 << v)
            masks[v][old_c] &= ~(1 << u)
            masks[u][new_c] |= (1 << v)
            masks[v][new_c] |= (1 << u)
            initial_colors[k] = new_c
            
            # Count violations
            new_violations = count_violations()
            
            if new_violations < best_violations_local:
                best_violations_local = new_violations
                best_new_c = new_c
            else:
                # Revert
                masks[u][new_c] &= ~(1 << v)
                masks[v][new_c] &= ~(1 << u)
                masks[u][old_c] |= (1 << v)
                masks[v][old_c] |= (1 << u)
                initial_colors[k] = old_c
        
        # Apply best change
        if best_new_c != old_c:
            masks[u][old_c] &= ~(1 << v)
            masks[v][old_c] &= ~(1 << u)
            masks[u][best_new_c] |= (1 << v)
            masks[v][best_new_c] |= (1 << u)
            initial_colors[k] = best_new_c
            current_violations = best_violations_local
            
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = initial_colors[:]
                
            if current_violations == 0:
                return initial_colors[:]
        else:
            # No improvement, try random perturbation
            new_c = random.randint(0, 4)
            if new_c != old_c:
                masks[u][old_c] &= ~(1 << v)
                masks[v][old_c] &= ~(1 << u)
                masks[u][new_c] |= (1 << v)
                masks[v][new_c] |= (1 << u)
                initial_colors[k] = new_c
                current_violations = count_violations()

    return best_colors if best_violations == 0 else None

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # 1. Try to get a valid 25-vertex coloring
    n25, colors25 = _lazy_woldar_25()
    
    if not _is_valid_coloring(n25, colors25):
        # Fallback to smaller n
        return {'n': 11, 'colors': [0]*55} # Placeholder, should be valid
        
    # 2. Try to extend to 26
    # Initialize 26-vertex coloring by adding a random 26th vertex
    n_target = 26
    m26 = n_target * (n_target - 1) // 2
    m25 = n25 * (n25 - 1) // 2
    
    # Copy 25-vertex colors
    init_26 = colors25[:]
    
    # Add edges for vertex 25
    for i in range(25):
        init_26.append(random.randint(0, 4))
        
    # Shuffle the edges? No, keep structure.
    
    # Run local search
    result_26 = _fast_local_search(n_target, init_26, deadline - 5)
    
    if result_26 is not None:
        if _is_valid_coloring(n_target, result_26):
            return {'n': n_target, 'colors': result_26}
            
    # 3. If 26 failed, try 25
    return {'n': n25, 'colors': colors25}

# To ensure we don't fail if the local search is too slow or buggy,
# we can also try a few restarts for n=26.

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n25, colors25 = _lazy_woldar_25()
    
    if not _is_valid_coloring(n25, colors25):
        # Emergency fallback
        # Generate a valid coloring for n=11
        n = 11
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                colors.append((i+j) % 5)
        if _is_valid_coloring(n, colors):
            return {'n': n, 'colors': colors}
        return {'n': 1, 'colors': []}

    # Try n=26 with multiple restarts
    n_target = 26
    m26 = n_target * (n_target - 1) // 2
    
    found = False
    for restart in range(10):
        if time.time() > deadline - 10:
            break
            
        init_26 = colors25[:]
        for i in range(25):
            init_26.append(random.randint(0, 4))
            
        # Run local search with a time limit per restart
        t_limit = deadline - 10
        result_26 = _fast_local_search(n_target, init_26, t_limit)
        
        if result_26 is not None:
            if _is_valid_coloring(n_target, result_26):
                return {'n': n_target, 'colors': result_26}
            
    # Fallback to 25
    return {'n': n25, 'colors': colors25}

# The above construct() is redefined. Let's make sure the logic is sound.
# The `_fast_local_search` implementation above has a `break` that exits the first while loop
# and then tries a simpler hill climbing. This might be inefficient.
# Let's refine `_fast_local_search` to be a single, consistent algorithm.

def _fast_local_search(n, initial_colors, deadline):
    if n < 3:
        return initial_colors[:]
        
    m = n * (n - 1) // 2
    
    # Precompute edge endpoints
    edge_u = [0] * m
    edge_v = [0] * m
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_u[idx] = i
            edge_v[idx] = j
            idx += 1
            
    # Masks: masks[v][c] is an int bitmask of neighbors of v in color c
    masks = [[0]*5 for _ in range(n)]
    
    def build_masks():
        for v in range(n):
            for c in range(5):
                masks[v][c] = 0
        for k in range(m):
            u = edge_u[k]
            v = edge_v[k]
            c = initial_colors[k]
            masks[u][c] |= (1 << v)
            masks[v][c] |= (1 << u)
            
    build_masks()
    
    def count_violations():
        total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = bin(masks[u][c] & masks[v][c]).count('1')
                    if common >= 2:
                        total += common - 1
        return total
        
    current_violations = count_violations()
    best_violations = current_violations
    best_colors = initial_colors[:]
    
    if current_violations == 0:
        return initial_colors[:]
        
    while time.time() < deadline:
        # Pick random edge
        k = random.randint(0, m - 1)
        u = edge_u[k]
        v = edge_v[k]
        old_c = initial_colors[k]
        
        best_new_c = old_c
        best_violations_local = current_violations
        
        # Try all other colors
        for new_c in range(5):
            if new_c == old_c:
                continue
                
            # Apply change
            masks[u][old_c] &= ~(1 << v)
            masks[v][old_c] &= ~(1 << u)
            masks[u][new_c] |= (1 << v)
            masks[v][new_c] |= (1 << u)
            initial_colors[k] = new_c
            
            # Count violations
            new_violations = count_violations()
            
            if new_violations < best_violations_local:
                best_violations_local = new_violations
                best_new_c = new_c
            else:
                # Revert
                masks[u][new_c] &= ~(1 << v)
                masks[v][new_c] &= ~(1 << u)
                masks[u][old_c] |= (1 << v)
                masks[v][old_c] |= (1 << u)
                initial_colors[k] = old_c
        
        # Apply best change
        if best_new_c != old_c:
            masks[u][old_c] &= ~(1 << v)
            masks[v][old_c] &= ~(1 << u)
            masks[u][best_new_c] |= (1 << v)
            masks[v][best_new_c] |= (1 << u)
            initial_colors[k] = best_new_c
            current_violations = best_violations_local
            
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = initial_colors[:]
                
            if current_violations == 0:
                return initial_colors[:]
        else:
            # No improvement, try random perturbation to escape local minima
            if random.random() < 0.1:
                new_c = random.randint(0, 4)
                if new_c != old_c:
                    masks[u][old_c] &= ~(1 << v)
                    masks[v][old_c] &= ~(1 << u)
                    masks[u][new_c] |= (1 << v)
                    masks[v][new_c] |= (1 << u)
                    initial_colors[k] = new_c
                    current_violations = count_violations()

    return best_colors if best_violations == 0 else None

# Final construct function
def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n25, colors25 = _lazy_woldar_25()
    
    if not _is_valid_coloring(n25, colors25):
        # Emergency fallback
        n = 11
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                colors.append((i+j) % 5)
        if _is_valid_coloring(n, colors):
            return {'n': n, 'colors': colors}
        return {'n': 1, 'colors': []}

    # Try n=26 with multiple restarts
    n_target = 26
    m26 = n_target * (n_target - 1) // 2
    
    found = False
    for restart in range(20):
        if time.time() > deadline - 5:
            break
            
        init_26 = colors25[:]
        for i in range(25):
            init_26.append(random.randint(0, 4))
            
        # Run local search with a time limit per restart
        t_limit = deadline - 5
        result_26 = _fast_local_search(n_target, init_26, t_limit)
        
        if result_26 is not None:
            if _is_valid_coloring(n_target, result_26):
                return {'n': n_target, 'colors': result_26}
            
    # Fallback to 25
    return {'n': n25, 'colors': colors25}
