import math
import time

def construct():
    N = 512
    start_time = time.time()
    time_limit = 85.0  # Leave 5s margin for 90s limit

    # 1. Precompute LCM and Conflict Graph
    # adj[i] is a bitmask of j such that lcm(i, j) > N
    adj = [0] * (N + 1)
    
    # Precompute primes up to N for faster LCM? 
    # Or just use math.gcd. math.gcd is fast in C.
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            # lcm(i, j) = i * j // gcd(i, j)
            g = math.gcd(i, j)
            l = i * j // g
            if l > N:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # 2. Greedy Initialization
    # Try to find a large independent set greedily
    def greedy_init():
        # Sort by degree (number of conflicts) ascending? 
        # Nodes with fewer conflicts are "safer" to pick? 
        # Or pick nodes with small values? 
        # Let's try picking nodes that have the most compatible neighbors.
        # Actually, a simple heuristic: iterate 1..N, if compatible with all picked, pick it.
        # This gives the seed solution {1..22}.
        # Let's try a better greedy: 
        # Repeatedly pick the node with the highest "score" (e.g., smallest value, or most compatible neighbors remaining).
        
        best_set = []
        remaining = list(range(1, N + 1))
        
        # Heuristic: Pick node with minimum value first? 
        # The seed {1..22} works because 1..22 are all pairwise compatible.
        # Let's check if we can add 23. 
        # lcm(23, 22) = 506 <= 512. 
        # lcm(23, 21) = 483 <= 512.
        # lcm(23, 20) = 460.
        # ...
        # 23 is prime. lcm(23, k) = 23*k if k not div by 23.
        # We need 23*k <= 512 => k <= 22.
        # So 23 is compatible with all 1..22.
        # So {1..23} is valid. Size 23.
        
        # Can we add 24?
        # lcm(24, 23) = 552 > 512. Conflict!
        # So we can't have both 23 and 24.
        
        # Let's just run a proper greedy that tries to maximize size.
        # We'll use a simple local search or greedy with backtracking if time permits.
        
        # For now, let's just return the seed if B&B is too complex to implement safely in time.
        # But B&B is the way to "maximize".
        
        # Let's implement a simple greedy: 
        # 1. Start with all nodes.
        # 2. Pick node v that minimizes (degree in remaining graph).
        # 3. Add v to set. Remove v and its neighbors from remaining.
        # 4. Repeat.
        
        # This is fast.
        remaining_mask = (1 << (N + 1)) - 2  # bits 1..N set
        # Note: 1 << (N+1) might be large but Python handles big ints.
        
        chosen = []
        rem = remaining_mask
        
        while rem:
            # Find node in rem with min degree in subgraph induced by rem
            best_v = -1
            min_deg = N + 1
            min_val = N + 1
            temp = rem
            while temp:
                # Get lowest set bit
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                # Count neighbors in rem
                # neighbors of v in rem = adj[v] & rem
                deg = (adj[v] & rem).bit_count()
                if deg < min_deg or (deg == min_deg and v < min_val):
                    min_deg = deg
                    best_v = v
                    min_val = v
                temp ^= lsb
                if time.time() - start_time > time_limit:
                    break
            
            if best_v == -1:
                break
                
            chosen.append(best_v)
            # Remove best_v and its neighbors
            rem &= ~(1 << best_v)
            rem &= ~adj[best_v]
            
        return chosen

    init_set = greedy_init()
    best_size = len(init_set)
    best_set = init_set
    
    # 3. Branch and Bound
    # We will try to improve upon init_set.
    # State: (mask, current_size)
    # We use a global variable for best_size and best_set to update.
    # To avoid deep recursion issues and overhead, we can use a stack or just recursion if depth is small.
    # Depth is at most N, but branching factor is reduced.
    
    # To make B&B effective, we need a good upper bound.
    # Upper bound for a set of nodes M: 
    # A simple bound is the number of nodes. 
    # A tighter bound is the size of a maximal independent set in M (which is what we are solving).
    # We can use a greedy coloring of the conflict graph induced by M. 
    # The number of colors is an upper bound on the size of the independent set? 
    # No, for Independent Set, the number of colors in a coloring of the *complement* graph? 
    # Wait. 
    # In a graph G, a set of vertices is independent if no two are adjacent.
    # If we color G such that adjacent vertices have different colors, then an independent set can contain at most one vertex of each color? No.
    # An independent set can contain multiple vertices of the same color only if they are not adjacent. But in a proper coloring, vertices of the same color are NOT adjacent. 
    # So, if we color the conflict graph G, any color class is an independent set.
    # Therefore, the size of the maximum independent set is at least the size of the largest color class.
    # This is a LOWER bound on the MIS size, not an upper bound.
    
    # We need an UPPER bound to prune.
    # Upper bound: The size of the maximum clique in the *complement* graph? 
    # The complement graph G' has edges where lcm <= 512. 
    # We want Max Independent Set in G, which is Max Clique in G'.
    # Upper bound for Max Clique in G': 
    # Greedy coloring of G' gives a lower bound on Chromatic Number, which is a lower bound on Clique Number? No.
    # Chromatic Number >= Clique Number. 
    # So a greedy coloring of G' gives a lower bound on the clique size? No.
    # The number of colors used is an upper bound on the Chromatic Number? No, it's an approximation.
    # Actually, a standard upper bound for Max Clique is obtained by coloring the graph. 
    # If we color G' with k colors, then the maximum clique size is at most k? 
    # No, a clique can use at most one vertex from each color class? 
    # In a proper coloring, all vertices in a clique must have distinct colors. 
    # So, if we use k colors, the clique size is at most k.
    # YES. So, if we greedily color the *compatibility* graph (G'), the number of colors is an upper bound on the size of the compatibility clique (which is our independent set in G).
    
    def color_bound(mask):
        # Greedily color the subgraph induced by mask in the COMPATIBILITY graph.
        # Compatibility graph edges: i-j if lcm(i,j) <= 512.
        # We don't have the compatibility graph bitsets precomputed, but we can derive them.
        # comp_adj[i] = ~(adj[i]) & full_mask
        # Let's precompute comp_adj.
        
        # To save time, we'll just use a simple bound: 
        # The number of nodes in mask. 
        # This is weak.
        
        # Let's try to compute a tighter bound using the precomputed adj.
        # We want to color the graph where edges exist if NOT adj.
        
        colors = 0
        uncolored = mask
        while uncolored:
            colors += 1
            # Find a maximal independent set in the compatibility graph? 
            # No, a color class in the compatibility graph is a set of vertices that are ALL pairwise compatible? 
            # No, a color class in a proper coloring of G' is a set of vertices with no edges between them in G'.
            # Edges in G' are compatibility. 
            # So a color class is a set of vertices that are NOT pairwise compatible? 
            # This is getting confusing.
            
            # Let's stick to the simpler Upper Bound: 
            # In the Conflict Graph G, if we pick a vertex v, we can pick at most 1 from its neighbors? No.
            
            # Let's use a different Upper Bound: 
            # For any subset of nodes, the size of the Max Independent Set is at most the size of the minimum vertex cover? No.
            
            # Let's just use the count of nodes as the bound, but prune aggressively if current_size + count <= best_size.
            # This is weak but safe.
            
            return mask.bit_count()
        
        return 0

    # Precompute compatibility bitsets for faster checks if needed
    full_mask = (1 << (N + 1)) - 2
    comp_adj = [0] * (N + 1)
    for i in range(1, N + 1):
        comp_adj[i] = full_mask & ~adj[i] & ~(1 << i)

    # Better Upper Bound using Greedy Coloring of Conflict Graph?
    # If we color the Conflict Graph G, the number of colors is an upper bound on the size of the Max Independent Set?
    # No. In a proper coloring of G, independent sets are color classes. 
    # So the size of the MIS is at least the max color class size.
    # It does NOT give an upper bound.
    
    # Upper bound for MIS in G:
    # The size of the Max Independent Set in G is equal to the size of the Max Clique in G_complement.
    # Let H = G_complement (Compatibility Graph).
    # We want Max Clique in H.
    # Upper bound for Max Clique in H:
    # Greedy coloring of H. 
    # If H is colored with k colors, any clique in H must have distinct colors. 
    # So Max Clique size <= k.
    # So, if we can quickly compute a greedy coloring of H induced by mask, the number of colors is an upper bound.
    
    def greedy_color_bound_H(mask, limit=None):
        """
        Upper bound on max clique size in H induced by mask.
        Uses greedy coloring: color classes in H are sets of mutually incompatible vertices.
        Any clique in H has at most one vertex per color class, so #colors >= clique size.
        Wait: that's wrong. In a proper coloring of H, vertices in a clique must all have distinct colors.
        So if we use k colors, clique size <= k. So k is an UPPER bound on clique size.
        """
        if limit is not None:
            # If we can already prove the bound is <= limit, return early
            pass
        
        colors = 0
        uncolored = mask
        while uncolored:
            colors += 1
            if limit is not None and colors >= limit:
                return colors
            
            # Pick vertex with max degree in uncolored for tighter coloring
            best_v = -1
            max_deg = -1
            temp = uncolored
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                deg = (comp_adj[v] & uncolored).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                temp ^= lsb
            
            v = best_v
            if v == -1:
                break
            
            # Color class: v and all uncolored vertices NOT adjacent to v in H
            # i.e., uncolored & ~comp_adj[v] & ~(1<<v)
            color_class = uncolored & ~comp_adj[v] & ~(1 << v)
            color_class |= (1 << v)
            
            uncolored &= ~color_class
            
        return colors

    # This coloring might be slow if done at every node.
    # We will only use it if the simple count bound doesn't prune.
    
    def solve(mask, current_size, current_set):
        nonlocal best_size, best_set
        
        if time.time() - start_time > time_limit:
            return
        
        # Prune 1: Simple count
        count = mask.bit_count()
        if current_size + count <= best_size:
            return
        
        # Prune 2: Coloring bound - use it more aggressively
        # The limit parameter lets us stop early if we already know we can't beat best_size
        needed = best_size - current_size + 1  # We need at least this many more to improve
        ub = greedy_color_bound_H(mask, limit=needed)
        if current_size + ub <= best_size:
            return
        
        if mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = list(current_set)
            return
        
        # Pick vertex: minimum degree in conflict graph G, tie-break by smaller value
        best_v = -1
        min_deg = N + 1
        min_val = N + 1
        temp = mask
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = (adj[v] & mask).bit_count()
            if deg < min_deg or (deg == min_deg and v < min_val):
                min_deg = deg
                best_v = v
                min_val = v
            temp ^= lsb
            
        if best_v == -1:
            return
        
        # Branch 1: Include best_v
        new_mask_include = mask & ~(1 << best_v) & ~adj[best_v]
        current_set.append(best_v)
        solve(new_mask_include, current_size + 1, current_set)
        current_set.pop()
        
        # Branch 2: Exclude best_v
        new_mask_exclude = mask & ~(1 << best_v)
        solve(new_mask_exclude, current_size, current_set)

    # Start B&B
    # Initial mask: all nodes
    init_mask = (1 << (N + 1)) - 2
    
    # Run B&B from the full set
    solve(init_mask, 0, [])
    
    # If B&B didn't find anything better, return init_set
    # Note: best_set might be empty if init_set was better? 
    # No, solve updates best_set only if it finds something strictly larger.
    # But init_set is a valid solution.
    # We should ensure best_set is at least init_set.
    
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
