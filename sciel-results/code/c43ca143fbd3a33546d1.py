import time
import random
import sys
import numpy as np

def construct():
    """
    Finds a 5-coloring of K_n without monochromatic C4.
    Maximizes n.
    """
    deadline = time.time() + 145.0  # Safety margin for 150s limit

    # Known seed solution for n=11
    def get_seed_solution():
        n = 11
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = min(j - i, n - (j - i))
                colors.append(d - 1)
        return n, colors

    best_n, best_colors = get_seed_solution()

    def get_edge_index(i, j, n):
        if i > j:
            i, j = j, i
        return i * (i - 1) // 2 + (j - i - 1)

    def is_valid(n, colors):
        """
        Verifies if the coloring has no monochromatic C4.
        """
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
        
        # Check for C4
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

    def solve_sa_incremental(n, time_limit):
        """
        Runs Simulated Annealing with incremental cost updates.
        """
        total_edges = n * (n - 1) // 2
        
        # Initial random coloring
        colors = [random.randint(0, 4) for _ in range(total_edges)]
        
        # Precompute edge endpoints
        edge_u = [0] * total_edges
        edge_v = [0] * total_edges
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_u[idx] = i
                edge_v[idx] = j
                idx += 1

        # Initialize neighbors bitsets
        neighbors = [[0] * n for _ in range(5)]
        for e in range(total_edges):
            u = edge_u[e]
            v = edge_v[e]
            c = colors[e]
            neighbors[c][u] |= (1 << v)
            neighbors[c][v] |= (1 << u)

        # Initialize common neighbor counts and cost
        # common_counts[c][u][v] for u < v
        # We can store this in a 3D array or just a dict, but for n=26, a 3D list of lists is fine.
        # common_counts[c] is a 2D list n x n.
        common_counts = [[[0] * n for _ in range(n)] for _ in range(5)]
        
        current_cost = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = neighbors[c][u] & neighbors[c][v]
                    cnt = bin(common).count('1')
                    common_counts[c][u][v] = cnt
                    common_counts[c][v][u] = cnt
                    if cnt >= 2:
                        current_cost += 1

        if current_cost == 0:
            return colors

        best_cost = current_cost
        best_colors = colors[:]
        
        # SA Parameters
        T = 1.0
        cooling_rate = 0.99995
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            # Pick a random edge
            e_idx = random.randint(0, total_edges - 1)
            u = edge_u[e_idx]
            v = edge_v[e_idx]
            old_c = colors[e_idx]
            
            # Pick a new color different from old
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue
                
            # Calculate delta cost
            delta = 0
            
            # Remove from old_c
            # Update neighbors
            neighbors[old_c][u] &= ~(1 << v)
            neighbors[old_c][v] &= ~(1 << u)
            
            # Update common counts for pairs involving u or v in old_c
            # Pairs (u, w) where w in N_old_c(v) (excluding u)
            # N_old_c(v) before removal included u. Now it doesn't.
            # But we need to update C(u, w) and C(v, w).
            
            # Let's iterate over all w != u, v
            # Actually, only w that were neighbors of v (in old_c) or u (in old_c) are affected.
            # C(u, w) changes if w is in N_old_c(v). Because v was a common neighbor of u and w.
            # C(v, w) changes if w is in N_old_c(u). Because u was a common neighbor of v and w.
            
            # Get neighbors of v in old_c (before removal, which is current neighbors[old_c][v] with bit u set? No, we already removed it).
            # So we need the list of neighbors BEFORE removal.
            # Let's reconstruct or store.
            # Easier: iterate over all w? No, O(n^2) is too slow? n=26, n^2=676. 5 colors.
            # O(n) per move is target.
            
            # Let's re-do the logic carefully.
            # State: neighbors[old_c] has already had u, v removed.
            # We need to know who was connected to u and v.
            # Let N_u_old = neighbors[old_c][u] | (1 << v)  # Since v was neighbor
            # Let N_v_old = neighbors[old_c][v] | (1 << u)
            
            N_u_old = neighbors[old_c][u] | (1 << v)
            N_v_old = neighbors[old_c][v] | (1 << u)
            
            # For each w in N_v_old \ {u, v}:
            #   C(u, w) decreases by 1 (because v is no longer common neighbor)
            #   C(v, w) decreases by 1 (because u is no longer common neighbor? No, u was neighbor of w? No.)
            #   Wait.
            #   C(x, y) = |N(x) \cap N(y)|.
            #   Change in N(u): removed v.
            #   Change in N(v): removed u.
            #   
            #   Consider pair (u, w).
            #   Old C(u, w) = |N_old(u) \cap N_old(w)|.
            #   New C(u, w) = |N_new(u) \cap N_new(w)|.
            #   N_new(u) = N_old(u) \ {v}. N_new(w) = N_old(w) (w is not u or v).
            #   So New C(u, w) = Old C(u, w) - 1 if v \in N_old(w), else Old C(u, w).
            #   v \in N_old(w) iff w \in N_old(v).
            #   So for w in N_old(v) \ {u, v}, C(u, w) decreases by 1.
            #   
            #   Consider pair (v, w).
            #   New C(v, w) = Old C(v, w) - 1 if u \in N_old(w).
            #   u \in N_old(w) iff w \in N_old(u).
            #   So for w in N_old(u) \ {u, v}, C(v, w) decreases by 1.
            
            # We need to update delta cost based on these changes.
            # We must be careful not to double count or miss.
            # Also, C(u, v) itself?
            # Old C(u, v) = |N_old(u) \cap N_old(v)|.
            # New C(u, v) = |N_new(u) \cap N_new(v)|.
            # N_new(u) = N_old(u) \ {v}. N_new(v) = N_old(v) \ {u}.
            # Intersection: (N_old(u) \ {v}) \cap (N_old(v) \ {u}) = (N_old(u) \cap N_old(v)) \ {u, v}.
            # Since u \notin N_old(u) and v \notin N_old(v), u, v are not in the intersection anyway.
            # So C(u, v) does NOT change.
            
            # So we update C(u, w) for w in N_v_old \ {u, v} and C(v, w) for w in N_u_old \ {u, v}.
            
            # Helper to update cost for a specific (c, x, y) change
            def update_cost(c, x, y, delta_cnt):
                nonlocal current_cost
                old_cnt = common_counts[c][x][y]
                new_cnt = old_cnt + delta_cnt
                if old_cnt >= 2 and new_cnt < 2:
                    current_cost -= 1
                elif old_cnt < 2 and new_cnt >= 2:
                    current_cost += 1
                common_counts[c][x][y] = new_cnt
                common_counts[c][y][x] = new_cnt

            # Apply removals
            # Iterate w in N_v_old
            w_mask = N_v_old & ~(1 << u) & ~(1 << v)
            while w_mask:
                lsb = w_mask & -w_mask
                w = lsb.bit_length() - 1
                w_mask ^= lsb
                # C(u, w) decreases by 1
                update_cost(old_c, u, w, -1)
                
            # Iterate w in N_u_old
            w_mask = N_u_old & ~(1 << u) & ~(1 << v)
            while w_mask:
                lsb = w_mask & -w_mask
                w = lsb.bit_length() - 1
                w_mask ^= lsb
                # C(v, w) decreases by 1
                update_cost(old_c, v, w, -1)
                
            # Add to new_c
            # N_u_new = neighbors[new_c][u] | (1 << v) ? No, v is not neighbor yet.
            # N_u_old_new = neighbors[new_c][u]
            # N_v_old_new = neighbors[new_c][v]
            
            N_u_new = neighbors[new_c][u]
            N_v_new = neighbors[new_c][v]
            
            # Update neighbors
            neighbors[new_c][u] |= (1 << v)
            neighbors[new_c][v] |= (1 << u)
            
            # For w in N_v_new (before adding u? No, w is neighbor of v in new_c before adding u? 
            # Wait, the logic is symmetric.
            # We add edge (u, v).
            # N(u) gains v. N(v) gains u.
            # C(u, w) increases by 1 if v \in N_old(w) in new_c. i.e., w \in N_old(v).
            # So for w in N_v_new (which is N_old(v)), C(u, w) increases by 1.
            # C(v, w) increases by 1 if u \in N_old(w) in new_c. i.e., w \in N_old(u).
            # So for w in N_u_new (which is N_old(u)), C(v, w) increases by 1.
            
            # Note: N_v_new is the set of neighbors of v in new_c BEFORE adding u.
            # N_u_new is the set of neighbors of u in new_c BEFORE adding v.
            
            # Iterate w in N_v_new
            w_mask = N_v_new & ~(1 << u) & ~(1 << v)
            while w_mask:
                lsb = w_mask & -w_mask
                w = lsb.bit_length() - 1
                w_mask ^= lsb
                # C(u, w) increases by 1
                update_cost(new_c, u, w, 1)
                
            # Iterate w in N_u_new
            w_mask = N_u_new & ~(1 << u) & ~(1 << v)
            while w_mask:
                lsb = w_mask & -w_mask
                w = lsb.bit_length() - 1
                w_mask ^= lsb
                # C(v, w) increases by 1
                update_cost(new_c, v, w, 1)
                
            # Check acceptance
            if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                colors[e_idx] = new_c
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
                    if best_cost == 0:
                        return best_colors
            else:
                # Reject: revert changes
                colors[e_idx] = old_c
                
                # Revert new_c additions
                neighbors[new_c][u] &= ~(1 << v)
                neighbors[new_c][v] &= ~(1 << u)
                
                w_mask = N_v_new & ~(1 << u) & ~(1 << v)
                while w_mask:
                    lsb = w_mask & -w_mask
                    w = lsb.bit_length() - 1
                    w_mask ^= lsb
                    update_cost(new_c, u, w, -1)
                    
                w_mask = N_u_new & ~(1 << u) & ~(1 << v)
                while w_mask:
                    lsb = w_mask & -w_mask
                    w = lsb.bit_length() - 1
                    w_mask ^= lsb
                    update_cost(new_c, v, w, -1)
                    
                # Revert old_c removals
                neighbors[old_c][u] |= (1 << v)
                neighbors[old_c][v] |= (1 << u)
                
                w_mask = N_v_old & ~(1 << u) & ~(1 << v)
                while w_mask:
                    lsb = w_mask & -w_mask
                    w = lsb.bit_length() - 1
                    w_mask ^= lsb
                    update_cost(old_c, u, w, 1)
                    
                w_mask = N_u_old & ~(1 << u) & ~(1 << v)
                while w_mask:
                    lsb = w_mask & -w_mask
                    w = lsb.bit_length() - 1
                    w_mask ^= lsb
                    update_cost(old_c, v, w, 1)

            T *= cooling_rate
            if T < 0.01:
                T = 1.0

        return best_colors if best_cost == 0 else None

    for n in range(29, 11, -1):
        if time.time() > deadline - 5:
            break
            
        if n >= 27:
            t_alloc = 10.0
        elif n == 26:
            t_alloc = 40.0
        else:
            t_alloc = 5.0
            
        t_alloc = min(t_alloc, deadline - time.time() - 2.0)
        if t_alloc <= 0:
            break
            
        res = solve_sa_incremental(n, t_alloc)
        
        if res:
            if is_valid(n, res):
                best_n = n
                best_colors = res
                return {'n': best_n, 'colors': best_colors}
                
    return {'n': best_n, 'colors': best_colors}

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
