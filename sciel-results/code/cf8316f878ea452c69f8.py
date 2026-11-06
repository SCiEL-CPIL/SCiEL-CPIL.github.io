import time
import random
import sys
import numpy as np

def construct():
    """
    Finds a 5-coloring of K_n without monochromatic C4 using Simulated Annealing
    with incremental cost updates.
    Maximizes n.
    """
    deadline = time.time() + 115.0  # Safety margin for 120s limit (evaluator kills at 150s, but 120s is the stated limit in prompt)
    
    # Helper to verify validity (slow, used only for final check)
    def is_valid(n, colors):
        if len(colors) != n * (n - 1) // 2:
            return False
        
        neighbors = [[0] * n for _ in range(5)]
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                if c < 0 or c > 4:
                    return False
                neighbors[c][i] |= (1 << j)
                neighbors[c][j] |= (1 << i)
                idx += 1
        
        for c in range(5):
            for u in range(n):
                nu = neighbors[c][u]
                if nu == 0:
                    continue
                for v in range(u + 1, n):
                    nv = neighbors[c][v]
                    common = nu & nv
                    if common & (common - 1):
                        return False
        return True

    def solve_sa(n, time_limit, seed=None):
        """
        Runs Simulated Annealing to find a valid coloring for K_n.
        Uses incremental updates for the objective function.
        """
        if seed is not None:
            random.seed(seed)
            
        total_edges = n * (n - 1) // 2
        
        # Precompute edge endpoints
        edge_map = [None] * total_edges
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map[idx] = (i, j)
                idx += 1
        
        # State:
        # neighbors[c][u] is a bitset of neighbors of u in color c
        neighbors = [[0] * n for _ in range(5)]
        # bad_pair_count[c][u][v] is 1 if u and v have >= 2 common neighbors in color c, else 0
        # We store this as a list of dicts or just recompute? 
        # For n=26, n^2 = 676. 5 * 676 is small.
        # Let's use a 2D list for bad_pair_count for each color.
        # bad_pairs[c][u] is a bitset of vertices v such that u and v form a bad pair in color c.
        # This makes it easy to count total bad pairs: sum(popcount(bad_pairs[c][u]) for u) / 2.
        # But updating it is tricky.
        
        # Alternative: Maintain total_cost directly.
        # When we add/remove an edge, we update the common neighbor counts for all pairs involving the endpoints.
        
        # Let's maintain:
        # common_count[c][u][v] = number of common neighbors of u and v in color c.
        # This is n x n matrix for each color. 5 * 26 * 26 = 3380 integers. Very small.
        common_count = [[[0] * n for _ in range(n)] for _ in range(5)]
        
        # bad_pair_count[c] = number of pairs (u,v) with u<v and common_count[c][u][v] >= 2
        bad_pair_count = [0] * 5
        total_cost = 0
        
        # Initialize colors randomly
        colors = [random.randint(0, 4) for _ in range(total_edges)]
        
        # Build initial state
        for e_idx in range(total_edges):
            u, v = edge_map[e_idx]
            c = colors[e_idx]
            
            # Add edge (u, v) to color c
            # Update common_count for all pairs involving u and v?
            # Actually, adding edge (u,v) to color c increases common_count[c][x][y] 
            # for all pairs (x,y) such that {x,y} intersects {u,v}? No.
            # A common neighbor of x and y is z if (x,z) and (y,z) are edges.
            # If we add (u,v), it affects common_count[c][u][w] for all w != u,v (if (w,v) exists? No, (w,u) and (w,v) need to exist).
            # Let's just do a full rebuild for initialization since it's only once.
            
        # Rebuild initial state efficiently
        for e_idx in range(total_edges):
            u, v = edge_map[e_idx]
            c = colors[e_idx]
            neighbors[c][u] |= (1 << v)
            neighbors[c][v] |= (1 << u)
            
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = neighbors[c][u] & neighbors[c][v]
                    cnt = bin(common).count('1')
                    common_count[c][u][v] = cnt
                    common_count[c][v][u] = cnt
                    if cnt >= 2:
                        bad_pair_count[c] += 1
                        total_cost += 1
        
        if total_cost == 0:
            return colors
            
        best_cost = total_cost
        best_colors = colors[:]
        
        # SA Parameters
        T = 2.0
        cooling_rate = 0.999
        start_time = time.time()
        
        # Helper to update state when changing edge (u,v) from c_old to c_new
        def update_state_remove(u, v, c):
            """Remove edge (u,v) from color c. Returns delta_cost."""
            nonlocal bad_pair_count, total_cost, common_count, neighbors
            
            # Remove from neighbors
            neighbors[c][u] &= ~(1 << v)
            neighbors[c][v] &= ~(1 << u)
            
            delta = 0
            
            # The edge (u,v) itself was a pair.
            # Before removal, common_count[c][u][v] was some value.
            # After removal, it changes.
            # Also, for any other vertex w, the pair (u,w) and (v,w) might change their common counts.
            
            # 1. Update pair (u, v)
            # Old common count: common_count[c][u][v]
            # New common count: popcount(neighbors[c][u] & neighbors[c][v])
            # Since we just removed the edge, the bitsets are already updated.
            old_cnt_uv = common_count[c][u][v]
            new_cnt_uv = bin(neighbors[c][u] & neighbors[c][v]).count('1')
            common_count[c][u][v] = new_cnt_uv
            common_count[c][v][u] = new_cnt_uv
            
            if old_cnt_uv >= 2 and new_cnt_uv < 2:
                bad_pair_count[c] -= 1
                delta -= 1
            elif old_cnt_uv < 2 and new_cnt_uv >= 2:
                bad_pair_count[c] += 1
                delta += 1
            
            # 2. Update pairs (u, w) and (v, w) for all w != u, v
            # For a pair (u, w), the common neighbors are neighbors[c][u] & neighbors[c][w].
            # Removing (u,v) means v is no longer a neighbor of u in color c.
            # So if v was a common neighbor of u and w, the count decreases by 1.
            # v is a common neighbor of u and w iff (v,w) is an edge in color c.
            
            # Iterate over all w
            for w in range(n):
                if w == u or w == v:
                    continue
                
                # Check if (v, w) is an edge in color c
                # If yes, then v was a common neighbor of u and w.
                if (neighbors[c][v] >> w) & 1:
                    # Update pair (u, w)
                    old_cnt_uw = common_count[c][u][w]
                    new_cnt_uw = old_cnt_uw - 1
                    common_count[c][u][w] = new_cnt_uw
                    common_count[c][w][u] = new_cnt_uw
                    
                    if old_cnt_uw >= 2 and new_cnt_uw < 2:
                        bad_pair_count[c] -= 1
                        delta -= 1
                    elif old_cnt_uw < 2 and new_cnt_uw >= 2:
                        bad_pair_count[c] += 1
                        delta += 1

                # Check if (u, w) is an edge in color c
                # If yes, then u was a common neighbor of v and w.
                if (neighbors[c][u] >> w) & 1:
                    # Update pair (v, w)
                    old_cnt_vw = common_count[c][v][w]
                    new_cnt_vw = old_cnt_vw - 1
                    common_count[c][v][w] = new_cnt_vw
                    common_count[c][w][v] = new_cnt_vw
                    
                    if old_cnt_vw >= 2 and new_cnt_vw < 2:
                        bad_pair_count[c] -= 1
                        delta -= 1
                    elif old_cnt_vw < 2 and new_cnt_vw >= 2:
                        bad_pair_count[c] += 1
                        delta += 1
            
            return delta

        def update_state_add(u, v, c):
            """Add edge (u,v) to color c. Returns delta_cost."""
            nonlocal bad_pair_count, total_cost, common_count, neighbors
            
            # Add to neighbors
            neighbors[c][u] |= (1 << v)
            neighbors[c][v] |= (1 << u)
            
            delta = 0
            
            # 1. Update pair (u, v)
            old_cnt_uv = common_count[c][u][v]
            new_cnt_uv = bin(neighbors[c][u] & neighbors[c][v]).count('1')
            common_count[c][u][v] = new_cnt_uv
            common_count[c][v][u] = new_cnt_uv
            
            if old_cnt_uv >= 2 and new_cnt_uv < 2:
                bad_pair_count[c] -= 1
                delta -= 1
            elif old_cnt_uv < 2 and new_cnt_uv >= 2:
                bad_pair_count[c] += 1
                delta += 1
            
            # 2. Update pairs (u, w) and (v, w) for all w != u, v
            for w in range(n):
                if w == u or w == v:
                    continue
                
                # If (v, w) is an edge in color c, v is now a common neighbor of u and w.
                if (neighbors[c][v] >> w) & 1:
                    old_cnt_uw = common_count[c][u][w]
                    new_cnt_uw = old_cnt_uw + 1
                    common_count[c][u][w] = new_cnt_uw
                    common_count[c][w][u] = new_cnt_uw
                    
                    if old_cnt_uw >= 2 and new_cnt_uw < 2:
                        bad_pair_count[c] -= 1
                        delta -= 1
                    elif old_cnt_uw < 2 and new_cnt_uw >= 2:
                        bad_pair_count[c] += 1
                        delta += 1

                # If (u, w) is an edge in color c, u is now a common neighbor of v and w.
                if (neighbors[c][u] >> w) & 1:
                    old_cnt_vw = common_count[c][v][w]
                    new_cnt_vw = old_cnt_vw + 1
                    common_count[c][v][w] = new_cnt_vw
                    common_count[c][w][v] = new_cnt_vw
                    
                    if old_cnt_vw >= 2 and new_cnt_vw < 2:
                        bad_pair_count[c] -= 1
                        delta -= 1
                    elif old_cnt_vw < 2 and new_cnt_vw >= 2:
                        bad_pair_count[c] += 1
                        delta += 1
            
            return delta

        # Main SA Loop
        while time.time() - start_time < time_limit:
            e_idx = random.randint(0, total_edges - 1)
            u, v = edge_map[e_idx]
            old_c = colors[e_idx]
            
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue
                
            # Calculate delta cost
            delta_remove = update_state_remove(u, v, old_c)
            delta_add = update_state_add(u, v, new_c)
            delta = delta_remove + delta_add
            
            # Acceptance
            if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                total_cost += delta
                colors[e_idx] = new_c
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_colors = colors[:]
                    if best_cost == 0:
                        return best_colors
            else:
                # Reject: Revert state
                update_state_remove(u, v, new_c)
                update_state_add(u, v, old_c)
                
            T *= cooling_rate
            if T < 0.05:
                T = 1.0
                
        return best_colors if best_cost == 0 else None

    # Search Plan
    # We prioritize larger n.
    # n=26 is the known target.
    search_plan = [
        (28, 5.0),
        (27, 10.0),
        (26, 40.0),
        (25, 10.0),
        (24, 5.0),
    ]
    
    # Fallback seed
    def get_seed_solution():
        n = 11
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = min(j - i, n - (j - i))
                colors.append(d - 1)
        return n, colors
        
    best_n, best_colors = get_seed_solution()
    
    for n, t_alloc in search_plan:
        if time.time() > deadline - 2.0:
            break
            
        # Try a few seeds if time permits?
        # For n=26, 40s is a lot. Let's use one long run.
        # For n=27/28, short runs.
        
        # Use a fixed seed for reproducibility in testing? No, random is fine.
        # But maybe a few restarts for smaller n?
        
        res = solve_sa(n, t_alloc)
        
        if res is not None:
            if is_valid(n, res):
                best_n = n
                best_colors = res
                # If we found a valid solution for a large n, return it.
                # Should we continue to try even larger?
                # The loop is ordered from largest to smallest.
                # So if we find n=26, we have already tried 28 and 27.
                # So we can return immediately.
                return {'n': best_n, 'colors': best_colors}
            
    return {'n': best_n, 'colors': best_colors}

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
