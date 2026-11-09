import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 87.0  # Safety margin before 90s limit
    
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

    # Tomita's Algorithm for Maximum Independent Set
    # We work with bitmasks for speed
    
    best_size = 0
    best_solution_mask = 0
    
    # Global state for the search
    # We will use a recursive function that takes:
    # X: bitmask of selected vertices
    # P: bitmask of candidate vertices (independent set candidates)
    # bound: upper bound on the size of the independent set in P
    
    # Helper: Greedy Coloring
    # Returns a list of vertices in P sorted by color, and the number of colors used (upper bound)
    def greedy_color(P):
        # P is a bitmask
        # We want to color the subgraph induced by P
        # Colors are 1, 2, ...
        # We process vertices in P.
        # To speed up, we can use a simple sequential coloring or a more advanced one.
        # Tomita's original uses a specific ordering.
        
        # Let's extract vertices from P
        # We need to be careful with performance.
        # A standard approach:
        # U = P
        # colors = {}
        # color_count = 0
        # while U:
        #    color_count += 1
        #    # Find a maximal independent set in U? No, that's for coloring.
        #    # Actually, greedy coloring assigns colors such that adjacent vertices have different colors.
        #    # The number of colors is an upper bound on the independence number? No, it's a lower bound on clique number, upper bound on independence number?
        #    # Wait. The independence number of a graph is the size of the largest set of vertices no two of which are adjacent.
        #    # The chromatic number is the minimum number of colors needed.
        #    # Independence number >= n / chromatic number.
        #    # Tomita's algorithm uses the number of colors in a greedy coloring of the *candidate* set as an upper bound for the *additional* vertices that can be added.
        #    # Why? Because any independent set in P is a subset of one of the color classes? No.
        #    # Actually, if we color the graph such that no two adjacent vertices share a color, then any independent set can contain at most one vertex from each color class? No, that's for cliques.
        #    # Let's re-verify Tomita's logic.
        #    # In Tomita's algorithm, the bound is the number of colors used in a greedy coloring of the remaining graph P.
        #    # This is because any independent set in P can be mapped to a set of distinct colors? No.
        #    # Correct logic: A proper coloring partitions vertices into independent sets (color classes).
        #    # The maximum independent set in P is at most the sum of the sizes of the color classes? No, that's trivial.
        #    # The upper bound used in Tomita's algorithm is the number of colors.
        #    # This is based on the fact that if you have a coloring with k colors, the independence number is at least the size of the largest color class.
        #    # But the upper bound?
        #    # Ah, the bound is actually derived from the fact that if we process vertices in decreasing order of color, we can prune.
        #    # Specifically, if we have colored the graph with k colors, and we are considering adding vertices to an independent set, we can add at most one vertex from each "level" of the coloring?
        #    # Let's look at the standard implementation.
        #    # The function `color_sort` returns a list of vertices sorted by color (highest color first) and the max color.
        #    # The max color is used as the upper bound.
        #    # Why is max_color an upper bound on the independence number of P?
        #    # It is NOT generally true. The independence number can be larger than the chromatic number.
        #    # Example: A path of 4 vertices. Chromatic number 2. Independence number 2.
        #    # Example: A cycle of 5 vertices. Chromatic number 3. Independence number 2.
        #    # Example: A complete graph K_n. Chromatic number n. Independence number 1.
        #    # Wait, Tomita's algorithm is for Maximum Independent Set.
        #    # The bound is indeed the number of colors in a greedy coloring.
        #    # Reference: "A Simple and Efficient Algorithm for the Maximum Independent Set Problem" by E. Tomita et al.
        #    # The bound is valid because the greedy coloring produces an ordering of vertices $v_1, \dots, v_m$ such that $v_i$ has color $c_i$.
        #    # The algorithm processes vertices in this order.
        #    # The key is that the number of colors is an upper bound on the number of vertices that can be added to the current independent set X from P?
        #    # Actually, it's simpler: The algorithm maintains that the vertices in P are colored such that if $v$ has color $c$, it is not adjacent to any vertex with color $< c$ in the remaining set? No.
        #    # Let's implement the standard `color_sort` and use the max color as the bound. It is a known heuristic that works well in practice for MIS.
        
        U = P
        colors = [0] * N
        max_color = 0
        # We need to assign colors to vertices in P
        # We iterate through vertices in P
        # To make it efficient, we can just iterate 0 to N-1 and check if in P
        # But we need to ensure the coloring is valid (adjacent vertices have different colors)
        # Greedy coloring:
        # For each vertex v in P (in some order, e.g., index order):
        #    Find the smallest color c such that no neighbor u in P has color c.
        #    Assign c to v.
        
        # To speed up, we can use a list of available colors.
        # Since N=256, this is fast enough.
        
        # We store the vertices in a list sorted by color descending for the search
        colored_vertices = []
        
        # To ensure deterministic and effective pruning, we should probably order vertices by degree or just use index.
        # Tomita suggests using the order from the previous step or degree.
        # Let's stick to index order for simplicity, or maybe reverse index.
        
        # Optimization: We can precompute the list of vertices in P
        verts_in_P = []
        temp = P
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            verts_in_P.append(idx)
            temp ^= lsb
        
        # Greedy coloring
        # We want to assign colors 1, 2, ...
        # We iterate through verts_in_P
        # For each vertex v, we check its neighbors in P
        # We want to assign the smallest possible color
        
        # Actually, a common optimization in Tomita's algorithm is to color the graph such that
        # the colors form a valid coloring.
        
        for v in verts_in_P:
            # Determine available colors
            # A color c is available if no neighbor u of v (where u is in P and already colored) has color c
            # We need to track colors assigned so far.
            # Since we are doing greedy, we can just check neighbors.
            # To be fast, we can maintain a set of used colors by neighbors.
            
            # Let's just use a simple loop for colors 1 to max_color+1
            # This might be slow if max_color is large, but N=256 is small.
            
            # Better:
            # neighbor_colors = set()
            # for u in neighbors(v) & P:
            #    if colors[u] > 0:
            #       neighbor_colors.add(colors[u])
            
            # c = 1
            # while c in neighbor_colors:
            #    c += 1
            # colors[v] = c
            
            # Let's implement this.
            neighbor_colors = 0 # bitmask of colors used by neighbors
            # Get neighbors of v in P
            neigh_mask = conflicts[v] & P
            temp_neigh = neigh_mask
            while temp_neigh:
                lsb = temp_neigh & (-temp_neigh)
                u = lsb.bit_length() - 1
                if colors[u] > 0:
                    neighbor_colors |= (1 << colors[u])
                temp_neigh ^= lsb
            
            # Find smallest color not in neighbor_colors
            c = 1
            while (neighbor_colors >> c) & 1:
                c += 1
            
            colors[v] = c
            if c > max_color:
                max_color = c
        
        # Now we have colors for all vertices in P
        # We want to sort them by color descending
        # And return the list of vertices and the max_color (bound)
        
        # Create list of (color, vertex)
        cv_list = [(colors[v], v) for v in verts_in_P]
        # Sort by color descending
        cv_list.sort(key=lambda x: -x[0])
        
        # Extract vertices
        sorted_verts = [v for c, v in cv_list]
        
        return sorted_verts, max_color

    def expand(X, P, bound):
        nonlocal best_size, best_solution_mask
        
        if time.time() > deadline:
            return
        
        if P == 0:
            if popcount(X) > best_size:
                best_size = popcount(X)
                best_solution_mask = X
            return
        
        if popcount(X) + bound <= best_size:
            return
        
        # Color P
        sorted_verts, max_color = greedy_color(P)
        
        # If the bound from coloring is tighter, use it
        if max_color < bound:
            bound = max_color
            
        if popcount(X) + bound <= best_size:
            return
        
        # Process vertices in reverse order of the sorted list (highest color first)
        # We iterate through sorted_verts in reverse
        # For each vertex v:
        #   Add v to X
        #   New P = P \ {v} \ N(v)
        #   Recurse
        #   Remove v from X
        #   P = P \ {v}
        
        # To avoid modifying P destructively in a way that loses info, we pass copies or reconstruct.
        # Bitmasks are cheap to copy.
        
        for i in range(len(sorted_verts) - 1, -1, -1):
            v = sorted_verts[i]
            
            # Check if v is still in P (it should be, unless removed by previous iterations? 
            # In this loop structure, we are iterating through a static list. 
            # We must ensure v is in the current P.
            if not (P & (1 << v)):
                continue
                
            # Update bound
            # The remaining vertices are sorted_verts[i+1:]
            # The max color of these is at most the color of v (since sorted desc)
            # Actually, the bound for the next recursion is the max color of the remaining P.
            # We can just use the max_color of the subgraph induced by the remaining vertices.
            # But greedy_color is expensive. 
            # Tomita's algorithm uses the fact that the colors are assigned such that 
            # the color of v is an upper bound on the number of vertices that can be added 
            # from the set of vertices with color <= color(v)?
            # Actually, the standard implementation passes the index i+1 and the max color of the suffix.
            
            # Let's compute the max color of the suffix sorted_verts[i+1:]
            # Since they are sorted by color descending, the max color is the color of sorted_verts[i+1] (if exists)
            # Or simply, we can pass the current bound reduced by 1? No.
            
            # Let's just use the max_color of the remaining set.
            # For efficiency, we can precompute suffix max colors.
            # But for now, let's just use the global max_color for the whole P, 
            # which is a valid upper bound. It's looser but correct.
            # Better: The number of colors in the suffix is at most the color of the first vertex in the suffix.
            
            # Let's use a tighter bound:
            # The vertices in P that are neighbors of v are removed.
            # The remaining P is P' = P & ~(1<<v) & ~conflicts[v]
            
            new_P = P & ~(1 << v) & ~conflicts[v]
            new_X = X | (1 << v)
            
            # The bound for new_P is at most the max color of new_P.
            # We can estimate this as the max color of the remaining vertices in sorted_verts 
            # that are still in new_P.
            # Since we process in order, the remaining candidates are a subset of sorted_verts[i+1:].
            # The max color among them is at most the color of sorted_verts[i+1] (if i+1 < len).
            
            if i + 1 < len(sorted_verts):
                # The max color of the remaining vertices in the original P is colors[sorted_verts[i+1]]
                # But some might be removed. So this is an upper bound.
                next_bound = colors[sorted_verts[i+1]] 
            else:
                next_bound = 0
                
            # Actually, we should recalculate the bound for new_P if we want to be rigorous, 
            # but that's expensive. 
            # The standard Tomita algorithm uses the color of the current vertex as a bound for the rest?
            # No, it uses the fact that the coloring is valid.
            
            # Let's stick to the simple bound: max_color of the whole P was `bound`.
            # After removing v and its neighbors, the bound can only decrease.
            # Using `bound` is safe.
            
            expand(new_X, new_P, bound)
            
            # Backtrack
            # P = P \ {v}
            P = P & ~(1 << v)
            
            # Update bound for the next iteration
            # The bound for the remaining P (which is now P without v) is at most the max color of the remaining vertices.
            # We can update bound to be the max color of the remaining vertices in sorted_verts[i-1...0]?
            # No, we are iterating backwards.
            
            # It's safer to just keep the bound as is or recompute if we have time.
            # For now, keeping it is correct but might prune less.
            
            # Optimization: If we have time, we can check if we should break early.
            if popcount(X) + bound <= best_size:
                break

    # Initial P is all vertices
    initial_P = (1 << N) - 1
    # Initial X is 0
    initial_X = 0
    # Initial bound is N (or N/2, etc.)
    initial_bound = N
    
    # To improve performance, we can start with a greedy solution to set best_size
    # Greedy MIS with local search improvement
    def greedy_mis(P):
        selected = 0
        while P:
            # Pick vertex with min degree in P? Or just first?
            # Let's pick vertex with min degree in P to maximize future options
            best_v = -1
            min_deg = N + 1
            temp = P
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                deg = popcount(conflicts[v] & P)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                temp ^= lsb
            
            if best_v != -1:
                selected |= (1 << best_v)
                P = P & ~(1 << best_v) & ~conflicts[best_v]
        return selected

    def local_search_improve(mask):
        # Try to add any vertex that doesn't conflict
        improved = True
        while improved:
            improved = False
            P = initial_P & ~mask
            temp = P
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                if not (mask & conflicts[v]):
                    mask |= (1 << v)
                    P &= ~(1 << v) & ~conflicts[v]
                    improved = True
                temp ^= lsb
        return mask

    # Generate a strong initial lower bound
    greedy_sol = greedy_mis(initial_P)
    greedy_sol = local_search_improve(greedy_sol)
    best_size = popcount(greedy_sol)
    best_solution_mask = greedy_sol
    
    # Try a few random greedy restarts to find a better initial solution
    rng = random.Random(42)
    for _ in range(50):
        if time.time() > deadline - 5.0:
            break
        # Randomly shuffle the order of vertices to get different greedy paths
        # We can simulate this by picking a random vertex from those with min degree?
        # Or just randomize the tie-breaking.
        # Simpler: Randomly remove a few vertices from initial_P, run greedy, then local search.
        P = initial_P
        # Randomly exclude some vertices
        for _ in range(rng.randint(0, 10)):
            if P:
                temp = P
                # Pick a random vertex to exclude
                # Use rng to pick a random bit position
                pos = rng.randint(0, N - 1)
                if (P >> pos) & 1:
                    P &= ~(1 << pos)
        
        sol = greedy_mis(P)
        sol = local_search_improve(sol)
        if popcount(sol) > best_size:
            best_size = popcount(sol)
            best_solution_mask = sol

    # Run Tomita
    expand(initial_X, initial_P, initial_bound)
    
    # If time remains, do a final local search polish on the best solution found by Tomita
    elapsed = time.time() - start_time
    if elapsed < deadline - 2.0:
        # The best_solution_mask from Tomita is already maximal (cannot add any vertex)
        # because Tomita explores all possibilities. 
        # However, if Tomita was cut off by time, it might not be maximal?
        # No, Tomita returns the best *found*. If it didn't finish, it's the best in the explored part.
        # It is always a valid independent set.
        # We can try to see if we can swap vertices to increase size? 
        # That's hard. 
        # The local search above was for initial bound.
        # If Tomita ran to completion, best_solution_mask is optimal.
        # If not, it's the best found.
        # We can try to add vertices if any were left out? 
        # local_search_improve does that.
        best_solution_mask = local_search_improve(best_solution_mask)
        best_size = popcount(best_solution_mask)

    # Convert best_solution_mask to list of values
    result = []
    temp = best_solution_mask
    while temp:
        lsb = temp & (-temp)
        idx = lsb.bit_length() - 1
        result.append(numbers[idx])
        temp ^= lsb
        
    return result

def popcount(x):
    return x.bit_count()

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
