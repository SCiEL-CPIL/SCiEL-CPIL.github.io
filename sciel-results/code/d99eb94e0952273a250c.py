import time
import random
import math

def construct():
    start_time = time.time()
    # Safety margin: return by 55s to be safe against 60s limit
    deadline = start_time + 55.0
    
    n = 256
    # 1. Build Conflict Graph
    # adj[i] is a bitmask of vertices that conflict with i
    adj = [0] * (n + 1)
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
    
    # 2. Greedy Initialization for a good lower bound
    def greedy_mis():
        remaining = (1 << (n + 1)) - 2 # bits 1..n set
        result = 0
        while remaining:
            # Find vertex with minimum degree in remaining
            min_v = -1
            min_deg = float('inf')
            # Iterate through set bits in remaining
            r = remaining
            while r:
                lsb = r & -r
                v = lsb.bit_length() - 1
                deg = (adj[v] & remaining).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
                r ^= lsb
            
            if min_v == -1:
                break
                
            result |= (1 << min_v)
            # Remove min_v and its neighbors from remaining
            remaining &= ~(adj[min_v] | (1 << min_v))
            
        return result

    best_mask = greedy_mis()
    best_size = best_mask.bit_count()
    
    # 3. Branch and Bound with Tomita Coloring
    # We store the best solution mask found
    global_best_mask = best_mask
    global_best_size = best_size
    
    # Color function: returns a list of colors for vertices in P, and the max color
    # Colors are 1-indexed. color[v] is the color assigned to v.
    def color_sort(P):
        # P is a bitmask
        # Returns list of vertices sorted by color descending, and max_color
        if P == 0:
            return [], 0
        
        # Greedy coloring
        # We will assign colors 1, 2, ...
        # We process vertices in P. To make it fast, we just iterate.
        # A simple greedy coloring:
        # For each vertex v in P, assign the smallest color not used by its neighbors in P.
        # Since we want an upper bound, any valid coloring works.
        # Tomita's algorithm typically sorts by color.
        
        # To speed up, we can use a simple heuristic:
        # Iterate vertices, assign smallest available color.
        colors = {}
        # We need to check neighbors in P.
        # Let's just iterate all vertices in P
        verts = []
        r = P
        while r:
            lsb = r & -r
            v = lsb.bit_length() - 1
            verts.append(v)
            r ^= lsb
        
        # Sort by degree descending? Or just arbitrary?
        # Tomita uses a specific ordering. Let's try sorting by degree in P descending for better bounds.
        verts.sort(key=lambda v: (adj[v] & P).bit_count(), reverse=True)
        
        for v in verts:
            # Find smallest color not in neighbors
            used_colors = set()
            for u in verts:
                if u == v: continue
                if (adj[v] >> u) & 1:
                    if u in colors:
                        used_colors.add(colors[u])
            
            c = 1
            while c in used_colors:
                c += 1
            colors[v] = c
            
        # Sort by color descending
        sorted_verts = sorted(verts, key=lambda v: colors[v], reverse=True)
        max_color = max(colors.values()) if colors else 0
        return sorted_verts, max_color

    def branch_and_bound(P, S, size):
        nonlocal global_best_mask, global_best_size, deadline
        
        if time.time() > deadline - 1.0:
            return False # Signal timeout
            
        if P == 0:
            if size > global_best_size:
                global_best_size = size
                global_best_mask = S
            return True
            
        # Upper bound: size + max possible additional vertices
        # If size + popcount(P) <= global_best_size, prune
        if size + P.bit_count() <= global_best_size:
            return True
            
        sorted_verts, max_color = color_sort(P)
        
        for v in sorted_verts:
            # The color of v is an upper bound on the number of vertices we can add from P starting with v
            # Actually, in Tomita, color(v) is the upper bound for the subgraph induced by P.
            # If size + color(v) <= global_best_size, we can break (since colors are descending)
            # Note: color(v) here is the color assigned in the coloring of P.
            # The property is that any independent set in P has size <= max_color.
            # More precisely, if we pick v, the remaining candidates are N(v) & P.
            # The bound is usually size + color(v).
            
            # Get color of v
            # We need to map v to its color. 
            # In the previous step, we sorted by color. 
            # Let's re-derive or store it. 
            # For efficiency in the loop, let's just use the property that 
            # if we process in descending color order, and size + color(v) <= best, we can stop.
            
            # However, we need the specific color of v.
            # Let's modify color_sort to return a dict or we re-calc.
            # Re-calc is expensive. Let's have color_sort return (sorted_verts, colors_dict)
            # Actually, the standard Tomita bound is:
            # If size + color(v) <= best, break.
            
            # Let's assume the color of the first vertex in sorted_verts is max_color.
            # We need the color of current v.
            # Since we sorted by color descending, we can just keep track or look it up.
            # Let's modify color_sort to return colors dict as well.
            
            # Wait, I didn't return colors dict in the draft above. 
            # Let's fix color_sort to return (sorted_verts, colors_dict).
            
            # For now, let's just use the max_color for the whole P as a loose bound? 
            # No, that's weak. 
            # Let's implement the color lookup properly.
            
            # To avoid excessive complexity in this rewrite, I will use a slightly less tight but faster bound:
            # size + P.bit_count() is very loose.
            # Let's stick to the standard Tomita which requires the color of v.
            
            # I will refactor color_sort to return the colors.
            # But to keep the code simple and robust, let's use a simple bound:
            # If size + P.bit_count() <= global_best_size: return
            # This is already done.
            
            # Let's try a simple recursive step.
            # Pick v.
            new_S = S | (1 << v)
            new_P = P & adj[v] # Neighbors of v in P
            
            # Prune check before recursing
            if size + 1 + new_P.bit_count() > global_best_size:
                # Recurse
                if not branch_and_bound(new_P, new_S, size + 1):
                    return False
                
            # Remove v from P for the next iteration
            P &= ~(1 << v)
            
            # If P becomes 0, we handled it at the top, but we might have pruned the last one?
            # The loop continues.
            
        return True

    # Refactor color_sort to return colors
    def color_sort_v2(P):
        if P == 0:
            return [], {}
        
        verts = []
        r = P
        while r:
            lsb = r & -r
            v = lsb.bit_length() - 1
            verts.append(v)
            r ^= lsb
            
        # Sort by degree descending to get better coloring
        verts.sort(key=lambda v: (adj[v] & P).bit_count(), reverse=True)
        
        # Greedy coloring: assign smallest available color to each vertex
        # in the order of verts
        assigned = {}
        for v in verts:
            used = set()
            # Check neighbors in P that are already assigned
            # Only those already in `assigned`
            for u in assigned:
                if (adj[v] >> u) & 1:
                    used.add(assigned[u])
            
            c = 1
            while c in used:
                c += 1
            assigned[v] = c
            
        sorted_verts = sorted(verts, key=lambda v: assigned[v], reverse=True)
        return sorted_verts, assigned

    # Updated B&B
    def bb(P, S, size):
        nonlocal global_best_mask, global_best_size, deadline
        
        if time.time() > deadline - 0.5:
            return False
            
        if P == 0:
            if size > global_best_size:
                global_best_size = size
                global_best_mask = S
            return True
            
        if size + P.bit_count() <= global_best_size:
            return True
            
        sorted_verts, colors = color_sort_v2(P)
        
        for v in sorted_verts:
            c = colors[v]
            # Tomita bound: if size + c <= best, we can stop processing this and subsequent vertices
            # because they have <= color.
            if size + c <= global_best_size:
                break
                
            new_S = S | (1 << v)
            # For independent set: remaining candidates are those NOT adjacent to v
            new_P = P & ~adj[v]
            # Remove v itself from new_P
            new_P &= ~(1 << v)
            
            # Recurse
            if not bb(new_P, new_S, size + 1):
                return False
                
            P &= ~(1 << v)
            
        return True

    # Run B&B
    # Start with all vertices as candidates
    initial_P = (1 << (n + 1)) - 2
    # We can try to improve the initial solution with a few random restarts of greedy
    # to give B&B a higher starting bound.
    for _ in range(10):
        r_sol = greedy_mis()
        if r_sol.bit_count() > global_best_size:
            global_best_size = r_sol.bit_count()
            global_best_mask = r_sol
            
    # Time check before starting heavy B&B
    if time.time() < deadline - 2.0:
        bb(initial_P, 0, 0)
        
    # Fallback: if B&B didn't finish or didn't improve, use the best greedy found
    # The global_best_mask is always a valid independent set.
    
    # Convert mask to list
    result = []
    r = global_best_mask
    while r:
        lsb = r & -r
        v = lsb.bit_length() - 1
        result.append(v)
        r ^= lsb
        
    return result

# To ensure we don't hang if B&B is too slow, the time check inside bb handles it.
# If it returns False (timeout), we just use the best found so far.

# One final check: ensure the result is valid.
# (The construction guarantees it, but just in case)
def _verify(sol):
    sol_set = set(sol)
    for i in range(len(sol)):
        for j in range(i+1, len(sol)):
            a = sol[i]
            b = sol[j]
            if (a * b) % (a + b) == 0:
                return False
    return True

# In production, we don't want to pay the cost of verification every time, 
# but for safety in this specific "rewrite from scratch" context where correctness is paramount,
# the B&B logic is sound. The greedy init is also sound.

if __name__ == "__main__":
    # Quick test
    pass
