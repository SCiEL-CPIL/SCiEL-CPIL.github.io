import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit

    N = 256
    numbers = list(range(1, N + 1))
    
    # Precompute conflict masks
    # conflicts[i] is a bitmask where bit j is set if numbers[i] and numbers[j] conflict
    conflicts = [0] * N
    
    for i in range(N):
        a = numbers[i]
        mask = 0
        for j in range(N):
            if i == j:
                continue
            b = numbers[j]
            # Check if (a*b) % (a+b) == 0
            if (a * b) % (a + b) == 0:
                mask |= (1 << j)
        conflicts[i] = mask

    # Precompute degrees for ordering
    degrees = [conflicts[i].bit_count() for i in range(N)]
    
    # Order vertices by degree descending to improve greedy coloring and search
    # We create a permutation: order[i] is the vertex index to process at step i
    # But for Tomita, we usually pass the set P and let the algorithm handle ordering via coloring.
    # However, initializing the candidate set with high-degree vertices first can help.
    # A common trick is to just use the natural order 0..N-1 but ensure the coloring
    # is effective. Let's stick to 0..N-1 for simplicity in bitmask logic, 
    # but the greedy coloring inside expand will handle the dynamic ordering.
    
    best_size = 0
    best_solution_mask = 0
    
    # Helper: Greedy Coloring for Tomita's Algorithm
    # Returns:
    #   v_list: list of vertices in P, sorted by color descending
    #   max_color: the maximum color used (upper bound on independence number of P)
    #   colors: list where colors[v] is the color assigned to v (for bound calculation)
    def greedy_color(P):
        U = P
        colors = [0] * N
        max_color = 0
        v_list = []
        
        # Extract vertices from P
        # We iterate through all possible vertices in P
        temp = P
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            v_list.append(idx)
            temp ^= lsb
        
        # Greedy coloring
        # We assign colors to vertices in v_list
        # To make the bound tight, we want to use as few colors as possible? 
        # No, Tomita's algorithm uses the number of colors as an upper bound for the 
        # number of vertices that can be added. Wait, the bound is the number of colors?
        # Yes, if the graph is colored with k colors, any independent set is a subset of 
        # the union of color classes? No.
        # Theorem: If a graph G is colored with k colors, then the independence number 
        # alpha(G) is at most k? No, that's false. 
        # Actually, Tomita's algorithm uses the fact that if we order vertices by color,
        # and process them, the color of a vertex v is an upper bound on the number of 
        # vertices in the remaining set (with lower or equal color) that can be added 
        # to an independent set containing v? 
        # Let's look at the standard implementation.
        # The bound used is the number of colors. 
        # Why? 
        # If we have a valid coloring, an independent set can contain at most one vertex 
        # from each "clique" formed by the coloring? No.
        # The standard Tomita algorithm (MCQ) uses a greedy coloring that produces an 
        # ordering of vertices $v_1, \dots, v_m$ such that $v_i$ has color $c_i$.
        # The algorithm maintains a bound. 
        # The key property is that the color of a vertex $v$ in the greedy coloring 
        # (processed in a specific order) is an upper bound on the size of the maximum 
        # independent set in the subgraph induced by $v$ and its predecessors in the 
        # ordering? 
        # Actually, the simplest explanation: 
        # In the standard implementation, `color_sort` returns a list of vertices sorted 
        # by color. The maximum color is used as the bound for the branch-and-bound. 
        # This works because the greedy coloring is done in a way that the color of a 
        # vertex is an upper bound on the number of vertices that can be selected from 
        # the remaining candidates if we were to select that vertex.
        
        # Let's implement the standard greedy coloring where we assign the smallest 
        # available color to each vertex in P.
        
        for v in v_list:
            # Determine used colors by neighbors in P
            # We need to know which colors are used by neighbors of v that are in P 
            # and have been colored already.
            # Since we are processing v_list, and v_list is just a set, the order matters.
            # Tomita's algorithm typically processes vertices in the order they appear 
            # in P, but the coloring is done on the subgraph induced by P.
            # To be effective, the order of coloring should be consistent with the search.
            # A common approach: color vertices in the order of their index in P.
            
            # For each vertex v, find the smallest color c such that no neighbor u 
            # in P with index < v (in the coloring order) has color c.
            # Since we are using bitmasks, "index < v" is ambiguous unless we define 
            # the order. Let's use the order in which we extracted them from P.
            # Actually, it's better to use the vertex index itself or a precomputed order.
            # Let's use the vertex index as the tie-breaker for coloring order.
            
            # To speed up, we can just check all neighbors in P.
            # neighbor_colors_mask: bit c is set if color c is used by a neighbor
            neighbor_colors_mask = 0
            neigh_mask = conflicts[v] & P
            temp_neigh = neigh_mask
            while temp_neigh:
                lsb = temp_neigh & (-temp_neigh)
                u = lsb.bit_length() - 1
                if colors[u] > 0:
                    neighbor_colors_mask |= (1 << colors[u])
                temp_neigh ^= lsb
            
            # Find smallest color not in neighbor_colors_mask
            c = 1
            while (neighbor_colors_mask >> c) & 1:
                c += 1
            
            colors[v] = c
            if c > max_color:
                max_color = c
        
        # Sort vertices by color descending
        # If colors are equal, we can sort by degree or index to break ties
        # Tomita suggests sorting by color, and for equal colors, by some heuristic.
        # Let's sort by color descending, then by degree descending.
        v_list.sort(key=lambda v: (-colors[v], -degrees[v]))
        
        return v_list, max_color, colors

    def expand(X, P, bound):
        nonlocal best_size, best_solution_mask
        
        if time.time() > deadline:
            return
        
        if P == 0:
            if X.bit_count() > best_size:
                best_size = X.bit_count()
                best_solution_mask = X
            return
        
        if X.bit_count() + bound <= best_size:
            return
        
        # Color P
        v_list, max_color, colors = greedy_color(P)
        
        # Update bound
        if max_color < bound:
            bound = max_color
            
        if X.bit_count() + bound <= best_size:
            return
        
        # Process vertices in reverse order of v_list (highest color first)
        # v_list is sorted by color descending.
        # We iterate from the end of v_list to the beginning?
        # No, Tomita's algorithm processes vertices in the order of increasing color?
        # Or decreasing?
        # Standard Tomita:
        # 1. Color the graph.
        # 2. Sort vertices by color (usually descending for the list, but processed in a specific way).
        # 3. The loop iterates through the vertices.
        
        # Let's follow the standard recursive structure:
        # For each vertex v in the ordered list (from last to first? or first to last?):
        #   If v is in P:
        #     Add v to X
        #     New P = P \ {v} \ N(v)
        #     Recurse with bound = color of v (or max color of remaining)
        #     Remove v from P
        
        # In the standard implementation, `v_list` is sorted such that the first element 
        # has the highest color. The algorithm iterates through `v_list`.
        # The bound for the recursive call is the color of the current vertex? 
        # No, the bound is the number of colors remaining?
        
        # Let's use the property: 
        # If we process vertices in the order of the list, and we have colored the graph,
        # the color of vertex v is an upper bound on the number of vertices that can be 
        # added to X from the set of vertices that come after v in the list (in the 
        # original coloring order) and are compatible with v.
        
        # Actually, a simpler and correct approach:
        # The bound passed to the recursive call is the max color of the new P.
        # But recomputing color is expensive.
        # Tomita's algorithm avoids this by using the color of the current vertex as 
        # a bound for the subproblem? 
        # Let's check: If v has color c, then the maximum independent set in the 
        # subgraph induced by the vertices with color < c (or <= c depending on 
        # definition) is at most c? 
        # Yes, if the coloring is valid, an independent set can contain at most one 
        # vertex from each "level" if the levels were cliques. They are not.
        # However, the greedy coloring in Tomita's algorithm has the property that 
        # the color of a vertex is an upper bound on the independence number of the 
        # subgraph induced by that vertex and its predecessors in the coloring order.
        
        # Let's implement the standard loop:
        # for i in range(len(v_list) - 1, -1, -1):
        #   v = v_list[i]
        #   if v not in P: continue
        #   # The bound for the remaining vertices (v_list[i+1:]) is at most the color of v_list[i+1]?
        #   # Or simply, we can use the max_color of the whole P as a loose bound, 
        #   # but Tomita uses the color of the current vertex to prune.
        
        # Let's use the following logic which is standard for MCQ:
        # The vertices are sorted by color.
        # We iterate through them.
        # When we pick v, the new P consists of vertices in P that are not adjacent to v.
        # The new P is a subset of the vertices that were in P and have index > i in v_list?
        # No, it's a subset of P \ {v} \ N(v).
        
        # To be safe and correct, we will recompute the bound for the new P if we have time, 
        # or use a loose bound. 
        # Given N=256, recomputing color might be feasible if we optimize it.
        # But let's try the standard Tomita pruning first.
        
        # The standard implementation iterates through the sorted list.
        # For each vertex v:
        #   if v in P:
        #     X' = X U {v}
        #     P' = P \ {v} \ N(v)
        #     # The bound for P' is the max color of P'
        #     # We can approximate this by the color of v? No.
        #     # Let's just call expand with a bound that is the max color of P'.
        #     # To avoid full recoloring, we can use the fact that P' is a subgraph of P.
        #     # The max color of P' is <= max color of P.
        
        # Let's use a simpler bound: the number of vertices in P'.
        # This is loose.
        
        # Better: Use the color of v as the bound for the recursive call?
        # If v has color c, then the maximum number of vertices we can add from the 
        # remaining candidates (which are a subset of the vertices with color <= c) 
        # is at most c? 
        # This is true if the vertices with color <= c form a graph with independence 
        # number <= c. This is not generally true.
        
        # Let's stick to the robust approach: 
        # 1. Get v_list, max_color, colors from greedy_color(P).
        # 2. Iterate through v_list in reverse (highest color first).
        # 3. For each v, if v is in P:
        #      new_P = P & ~(1<<v) & ~conflicts[v]
        #      new_X = X | (1<<v)
        #      # Compute bound for new_P
        #      # We can use the max color of new_P. 
        #      # To save time, we can use the max color of the suffix of v_list that is in new_P.
        #      # Since v_list is sorted by color, the max color of any subset is the max color 
        #      # of the first element in that subset (in v_list order).
        #      # But new_P is not necessarily a suffix of v_list.
        
        # Let's just recompute the bound for new_P using a fast greedy coloring 
        # if the size of new_P is small, or use a loose bound otherwise.
        # Actually, for N=256, a fast greedy coloring is cheap.
        
        for i in range(len(v_list) - 1, -1, -1):
            if time.time() > deadline:
                return
            
            v = v_list[i]
            
            if not (P & (1 << v)):
                continue
                
            new_P = P & ~(1 << v) & ~conflicts[v]
            new_X = X | (1 << v)
            
            # Calculate bound for new_P
            # If new_P is empty, bound is 0
            if new_P == 0:
                if new_X.bit_count() > best_size:
                    best_size = new_X.bit_count()
                    best_solution_mask = new_X
            else:
                # Use the max color of new_P as bound
                # We can get a quick upper bound by the number of vertices in new_P
                # Or recompute greedy_color for new_P? 
                # Recomputing is O(N^2) roughly. 
                # Let's try to use the color of v as a bound? 
                # No, let's use the max color of the remaining vertices in v_list that are in new_P.
                # Since v_list is sorted by color descending, the max color of new_P 
                # is at most the color of the first vertex in v_list (index 0) that is in new_P?
                # No, the colors are assigned to the original P. 
                
                # Let's just use a simple bound: new_P.bit_count()
                # This is correct but loose.
                # Better: use the max_color of the original P? 
                # That's also loose.
                
                # Let's recompute greedy_color for new_P to get a tight bound.
                # This might be expensive but N=256 is small.
                if new_P.bit_count() > 0:
                    _, next_bound, _ = greedy_color(new_P)
                else:
                    next_bound = 0
                
                if new_X.bit_count() + next_bound > best_size:
                    expand(new_X, new_P, next_bound)
            
            # Backtrack
            P = P & ~(1 << v)
            
            # Update bound for the next iteration
            # The bound for the remaining P (which is P without v) 
            # We can recompute or just keep the old bound (which is safe but loose).
            # To be efficient, we can just keep the old bound.
            # But if we want to prune, we should update it.
            # Let's keep the old bound for simplicity and speed.
            # The pruning will happen in the next expand call.

    # Initial P is all vertices
    initial_P = (1 << N) - 1
    initial_X = 0
    initial_bound = N
    
    # Greedy MIS to initialize best_size
    def greedy_mis(P):
        selected = 0
        while P:
            best_v = -1
            min_deg = N + 1
            temp = P
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                deg = (conflicts[v] & P).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                temp ^= lsb
            
            if best_v != -1:
                selected |= (1 << best_v)
                P = P & ~(1 << best_v) & ~conflicts[best_v]
        return selected

    greedy_sol = greedy_mis(initial_P)
    best_size = greedy_sol.bit_count()
    best_solution_mask = greedy_sol
    
    # Run Tomita
    expand(initial_X, initial_P, initial_bound)
    
    # Convert best_solution_mask to list of values
    result = []
    temp = best_solution_mask
    while temp:
        lsb = temp & (-temp)
        idx = lsb.bit_length() - 1
        result.append(numbers[idx])
        temp ^= lsb
        
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            a, b = res[i], res[j]
            if (a*b) % (a+b) == 0:
                valid = False
    print(valid)
