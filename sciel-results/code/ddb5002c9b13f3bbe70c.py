import time
from math import gcd
import sys

N = 512
DEADLINE = 85.0

def construct():
    start_time = time.time()
    
    # 1. Precompute LCMs and build conflict graph
    # adj[i] is a bitmask of nodes j such that lcm(i+1, j+1) > N
    adj = [0] * N
    
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            g = gcd(a, b)
            if (a * b) // g > N:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # 2. Reductions
    # We will work with a list of active vertices.
    # Initially all are active.
    # We maintain a mapping from original index to active index? 
    # Or just work with the original indices and a bitmask of active nodes.
    
    active_mask = (1 << N) - 1
    
    # Greedy initial solution
    def greedy_solution(active_mask):
        remaining = active_mask
        sol = []
        # Sort vertices by degree in active graph? 
        # Or just pick lowest index with lowest degree?
        # Let's try picking the vertex with minimum degree in the remaining graph.
        # This is O(N^2) per step, O(N^3) total. 512^3 is ~134M, might be too slow in Python if not careful.
        # Let's just pick the first available vertex with lowest degree among all available?
        # Or simply: iterate 0..N-1, if active and not conflicting with current sol, add.
        # This gives a valid set but maybe not large.
        
        # Better greedy:
        # Repeat N times:
        #   Find v in remaining with min degree in remaining.
        #   Add v to sol.
        #   Remove v and its neighbors from remaining.
        
        # Optimization: Precompute degrees? Degrees change.
        # For N=512, this is acceptable.
        
        rem = active_mask
        while rem:
            # Find min degree vertex in rem
            min_deg = N
            best_v = -1
            temp = rem
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                deg = (adj[v] & rem).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                temp ^= lsb
            
            if best_v == -1: break
            
            sol.append(best_v + 1)
            # Remove best_v and its neighbors
            to_remove = adj[best_v] | (1 << best_v)
            rem &= ~to_remove
            
        return sol

    init_sol = greedy_solution(active_mask)
    best_size = len(init_sol)
    best_set = init_sol
    
    # Reduction: Dominated vertices
    # If N(u) subset N(v) and u != v, then v is dominated by u.
    # We can remove v from consideration.
    # Note: This applies to the "original" graph? Or the induced subgraph?
    # In MIS, if N(u) subset N(v), v is never needed.
    # We can apply this to the full graph first.
    
    active_mask = (1 << N) - 1
    
    # Apply dominated vertex removal
    # We do this in a loop until no more changes, or just once.
    # Let's do a few passes.
    for _ in range(3):
        changed = False
        for v in range(N):
            if not (active_mask & (1 << v)):
                continue
            # Check if v is dominated by any other active u
            # N(u) subset N(v)  <=>  (adj[u] & active_mask) & ~(adj[v] & active_mask) == 0
            # Actually, we should check neighbors within active_mask.
            nv = adj[v] & active_mask
            dominated = False
            for u in range(N):
                if u == v or not (active_mask & (1 << u)):
                    continue
                nu = adj[u] & active_mask
                # Check if nu is subset of nv
                # nu & ~nv == 0
                if (nu & ~nv) == 0:
                    # u dominates v
                    # But we must ensure u and v are not adjacent?
                    # If they are adjacent, u in N(v) and v in N(u).
                    # If N(u) subset N(v), then v in N(v) which is false.
                    # So they cannot be adjacent.
                    dominated = True
                    break
            if dominated:
                active_mask &= ~(1 << v)
                changed = True
        if not changed:
            break
            
    # Now solve MIS on the reduced graph (active_mask)
    # We need to map back to original values.
    # The B&B will work on the bitmask of active nodes.
    
    # B&B with coloring bound
    # We need a function to compute a greedy coloring of the subgraph induced by mask.
    # Returns the number of colors.
    
    def greedy_color_bound(mask):
        # Returns upper bound on independent set size in mask
        # Greedy coloring:
        # Assign color 1 to first available, remove neighbors.
        # Assign color 2 to next available, etc.
        # The number of colors is the bound.
        
        rem = mask
        colors = 0
        while rem:
            colors += 1
            # Pick a vertex, remove it and its neighbors from rem
            # To get a tight bound, we should pick vertices carefully?
            # Standard greedy coloring for bound:
            # Just pick any vertex? Or max degree?
            # Picking max degree usually gives fewer colors (tighter bound).
            # Let's pick max degree in rem.
            max_deg = -1
            best_v = -1
            temp = rem
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                deg = (adj[v] & rem).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                temp ^= lsb
            
            if best_v == -1: break
            
            to_remove = adj[best_v] | (1 << best_v)
            rem &= ~to_remove
            
        return colors

    def solve(mask, current_size, current_set):
        nonlocal best_size, best_set, start_time
        
        if time.time() - start_time > DEADLINE:
            return
            
        if mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set[:]
            return
            
        # Bound
        # If current_size + greedy_color_bound(mask) <= best_size, prune
        # This is expensive. Only do it if mask is small or we are close to deadline?
        # Or always?
        # For N=512, coloring is O(N^2) roughly.
        # Let's do it.
        
        # Simple bound: popcount
        rem_count = mask.bit_count()
        if current_size + rem_count <= best_size:
            return
            
        # Coloring bound
        # If rem_count is small, it's fast.
        # If rem_count is large, it's slow but might prune deep branches.
        # Let's use it.
        
        color_bound = greedy_color_bound(mask)
        if current_size + color_bound <= best_size:
            return
            
        # Branching
        # Pick vertex with max degree in mask
        max_deg = -1
        best_v = -1
        temp = mask
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = (adj[v] & mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp ^= lsb
            
        if best_v == -1:
            return
            
        # Branch 1: Exclude best_v
        mask_excl = mask & ~(1 << best_v)
        solve(mask_excl, current_size, current_set)
        
        # Branch 2: Include best_v
        mask_incl = mask & ~(adj[best_v] | (1 << best_v))
        current_set.append(best_v + 1)
        solve(mask_incl, current_size + 1, current_set)
        current_set.pop()
        
    # Initial call
    solve(active_mask, 0, [])
    
    if best_size == 0:
        return [1]
        
    return best_set

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            a, b = res[i], res[j]
            if (a // gcd(a, b)) * b > 512:
                print(f"Invalid pair: {a}, {b}")
                valid = False
                break
        if not valid:
            break
    if valid:
        print("All pairs valid.")
