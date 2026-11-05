import math
import time

def construct():
    N = 512
    start_time = time.time()
    time_limit = 85.0  # Leave 5s margin for 90s limit

    # 1. Precompute LCM and Conflict Graph
    # adj[i] is a bitmask of j such that lcm(i, j) > N
    adj = [0] * (N + 1)
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            # lcm(i, j) = i * j // gcd(i, j)
            g = math.gcd(i, j)
            l = i * j // g
            if l > N:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Precompute compatibility graph (complement of adj, excluding self)
    full_mask = (1 << (N + 1)) - 2
    comp_adj = [0] * (N + 1)
    for i in range(1, N + 1):
        comp_adj[i] = full_mask & ~adj[i] & ~(1 << i)

    # 2. Greedy Initialization
    # We want a good initial lower bound.
    # A simple greedy: repeatedly pick the vertex with the highest degree in the compatibility graph
    # among the remaining vertices. This tends to build a dense clique.
    
    def greedy_init():
        remaining = full_mask
        chosen = []
        while remaining:
            best_v = -1
            max_deg = -1
            temp = remaining
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                deg = (comp_adj[v] & remaining).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                temp ^= lsb
            if best_v == -1:
                break
            chosen.append(best_v)
            # Remove best_v and all its non-neighbors (conflicts) from remaining?
            # No, we want to find a clique.
            # If we pick v, the next vertex must be compatible with v.
            # So we restrict remaining to neighbors of v in the compatibility graph.
            remaining &= comp_adj[best_v]
            remaining &= ~(1 << best_v)
        
        return chosen

    init_set = greedy_init()
    best_size = len(init_set)
    best_set = list(init_set)
    
    # 3. Branch and Bound for Maximum Clique in Compatibility Graph
    # We use a recursive solver with a strong upper bound based on greedy coloring.
    
    def greedy_color_bound(mask):
        """
        Computes an upper bound on the size of the maximum clique in the subgraph induced by mask.
        Uses a greedy coloring heuristic.
        The number of colors used is an upper bound on the clique size.
        """
        colors = 0
        uncolored = mask
        
        while uncolored:
            colors += 1
            # Pick a vertex with the highest degree in the uncolored subgraph
            # This helps create larger color classes (independent sets in the complement, i.e., cliques in the original? No.)
            # Wait. We are coloring the Compatibility Graph H.
            # A color class in H is an independent set in H.
            # An independent set in H is a set of vertices with no edges in H, i.e., a set of vertices that are pairwise INCOMPATIBLE.
            # We want to color H such that adjacent vertices (compatible pairs) have different colors.
            # Then any clique in H must have distinct colors.
            # So the number of colors is an upper bound on the clique size.
            
            # To minimize the number of colors, we should pick vertices that have many neighbors?
            # Actually, a standard heuristic for coloring is to pick the vertex with the highest degree.
            # But here, we want to bound the clique size.
            # The bound is tighter if the color classes are larger.
            # A color class is an independent set in H.
            # To make the color class large, we should pick a vertex that has few neighbors?
            # No, if we pick a vertex v, the color class can include all uncolored vertices that are NOT neighbors of v.
            # So to maximize the size of the color class (and thus potentially reduce the total number of colors),
            # we should pick a vertex v with the FEWEST neighbors in the uncolored set.
            
            best_v = -1
            min_deg = N + 1
            temp = uncolored
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                deg = (comp_adj[v] & uncolored).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                temp ^= lsb
            
            if best_v == -1:
                break
            
            # The color class consists of best_v and all uncolored vertices not adjacent to best_v in H.
            # i.e., uncolored & ~comp_adj[best_v]
            color_class = uncolored & ~comp_adj[best_v]
            
            uncolored &= ~color_class
            
        return colors

    def solve(mask, current_size, current_set):
        nonlocal best_size, best_set
        
        if time.time() - start_time > time_limit:
            return
        
        # Prune 1: Simple count
        count = mask.bit_count()
        if current_size + count <= best_size:
            return
        
        # Prune 2: Coloring bound
        # We only need to know if the bound is <= best_size - current_size
        # If so, we prune.
        # To save time, we can stop the coloring early if the number of colors exceeds the limit.
        limit = best_size - current_size
        if limit <= 0:
            return
            
        # Modified greedy_color_bound that stops if colors > limit
        colors = 0
        uncolored = mask
        while uncolored:
            colors += 1
            if colors > limit:
                break
            
            best_v = -1
            min_deg = N + 1
            temp = uncolored
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                deg = (comp_adj[v] & uncolored).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                temp ^= lsb
            
            if best_v == -1:
                break
            
            color_class = uncolored & ~comp_adj[best_v]
            uncolored &= ~color_class
        
        if colors <= limit:
            return
        
        if mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = list(current_set)
            return
        
        # Pick vertex to branch on.
        # Heuristic: Pick the vertex with the highest degree in the induced subgraph.
        # This is a common heuristic for Maximum Clique.
        best_v = -1
        max_deg = -1
        temp = mask
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = (comp_adj[v] & mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp ^= lsb
        
        if best_v == -1:
            return
        
        # Branch 1: Include best_v
        # New mask is the intersection of current mask and neighbors of best_v.
        new_mask_include = mask & comp_adj[best_v]
        current_set.append(best_v)
        solve(new_mask_include, current_size + 1, current_set)
        current_set.pop()
        
        # Branch 2: Exclude best_v
        new_mask_exclude = mask & ~(1 << best_v)
        solve(new_mask_exclude, current_size, current_set)

    # Start B&B
    init_mask = full_mask
    
    # Run B&B from the full set
    solve(init_mask, 0, [])
    
    # Ensure we return the best set found
    if len(best_set) < len(init_set):
        best_set = init_set

    # Final validation and cleanup
    result = sorted(best_set)
    return result

if __name__ == "__main__":
    # Test
    res = construct()
    print(f"Size: {len(res)}")
    # Verify
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            a, b = res[i], res[j]
            l = a * b // math.gcd(a, b)
            if l > 512:
                print(f"Invalid: {a}, {b}, lcm={l}")
                break
