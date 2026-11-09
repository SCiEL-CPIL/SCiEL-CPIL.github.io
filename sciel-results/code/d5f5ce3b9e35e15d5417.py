import time
import math
import sys

def construct():
    start_time = time.time()
    DEADLINE = start_time + 85  # 85s safety margin for 90s limit
    
    LIMIT = 512
    N = LIMIT
    
    # 1. Build Compatibility Graph
    # compat[i] is a bitmask of nodes j (1..N) such that lcm(i, j) <= LIMIT
    # We use 0-indexed bits for nodes 1..N, so bit j-1 corresponds to node j.
    compat = [0] * (N + 1)
    
    # Precompute gcds or just compute lcm directly
    # N=512 is small, O(N^2) is 262k operations, very fast.
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = math.gcd(i, j)
            l = (i // g) * j
            if l <= LIMIT:
                # Set bit j-1 in compat[i] and bit i-1 in compat[j]
                compat[i] |= (1 << (j - 1))
                compat[j] |= (1 << (i - 1))
    
    # 2. Tomita's Algorithm for Maximum Clique
    # We represent the candidate set P as a list of vertices.
    # We use a global best_clique.
    
    best_clique = []
    
    # Greedy Lower Bound to start with a decent lb
    def greedy_clique():
        # Sort nodes by degree (popcount of compat mask)
        degrees = [(i, bin(compat[i]).count('1')) for i in range(1, N+1)]
        degrees.sort(key=lambda x: -x[1])
        
        clique = []
        # Candidates must be compatible with all in clique
        # We can just iterate and check
        for node, deg in degrees:
            if time.time() > DEADLINE:
                break
            # Check if node is compatible with all in clique
            ok = True
            for c in clique:
                # Check if c is in compat[node]
                # c is 1-indexed, bit index is c-1
                if not (compat[node] & (1 << (c - 1))):
                    ok = False
                    break
            if ok:
                clique.append(node)
        return clique

    # Initial lower bound
    init_clique = greedy_clique()
    if init_clique:
        best_clique = init_clique
    
    # Helper to check if a vertex v is compatible with all vertices in a list C
    def is_compatible_with_all(v, C):
        mask = compat[v]
        for c in C:
            if not (mask & (1 << (c - 1))):
                return False
        return True

    # Tomita's Expand function
    # P: list of candidate vertices
    # C: current clique (list of vertices)
    # depth: current depth (len(C))
    
    sys.setrecursionlimit(10000)
    
    def expand(P, C, depth):
        nonlocal best_clique
        
        if time.time() > DEADLINE:
            return
            
        if not P:
            if depth > len(best_clique):
                best_clique = C[:]
            return
        
        # Greedy Coloring of P
        # We assign colors to vertices in P.
        # The color of a vertex is the smallest integer not used by its neighbors in the processed part.
        # Actually, standard Tomita:
        # Sort P by some heuristic (e.g. degree in P).
        # Then color greedily.
        # The color classes provide an upper bound on the clique size in P.
        
        # To speed up, we can just use a simple greedy coloring on the list P.
        # We don't need perfect coloring, just a valid one to get a bound.
        
        # Optimization: If depth + len(P) <= len(best_clique), return
        if depth + len(P) <= len(best_clique):
            return
            
        # Coloring Step
        # We'll create a list of (vertex, color)
        # Standard approach: 
        # 1. Order P (e.g., by degree in P, descending).
        # 2. For each v in ordered P, assign the smallest available color.
        
        # Calculate degree in P for sorting
        # To save time, we might skip dynamic degree calculation if P is small, 
        # but for N=512, it's fine.
        
        # Let's do a simpler bound first: if depth + len(P) <= best, prune.
        # Then do coloring.
        
        # Coloring implementation:
        # colors[v] = color index (0-indexed)
        # max_color = max color index + 1
        
        # We process P in a specific order. Let's sort by degree in P descending.
        # degree_in_P(v) = popcount(compat[v] & mask_P)
        
        # To avoid O(N^2) inside recursion, we can approximate or just use static degree.
        # Static degree is a valid upper bound on degree in P? No, but it's a good heuristic for ordering.
        # Let's use static degree for ordering P.
        
        P_sorted = sorted(P, key=lambda v: -bin(compat[v]).count('1'))
        
        color_of = {}
        used_colors = [set() for _ in range(len(P_sorted) + 1)] # color -> set of vertices with that color? No, we need vertex -> color.
        # Actually, we just need to know the max color.
        # And we need to process vertices in order of increasing color for the branching?
        # Tomita: 
        # 1. Color P greedily.
        # 2. The vertices are grouped by color.
        # 3. We branch on vertices from the last color class (highest color) to the first.
        
        # Greedy Coloring:
        # For v in P_sorted:
        #   find smallest color k such that no neighbor of v in P_sorted[processed] has color k.
        #   assign color k to v.
        
        # This is O(N^2) per node in P. With N=512, worst case 512^2 * 512 = 134M ops. Too slow for deep recursion.
        # We need a faster coloring or a simpler bound.
        
        # Alternative: Use a simpler bound. 
        # However, Tomita's coloring is key to performance.
        # Let's optimize the coloring:
        # We can maintain a list of vertices for each color.
        
        color_list = [] # list of lists, color_list[k] contains vertices with color k
        vertex_color = {}
        
        for v in P_sorted:
            # Find smallest color
            # A vertex v can take color k if no neighbor of v has color k.
            # We check neighbors of v in the already colored vertices.
            
            # To speed up, we can check:
            # Does v have any neighbor in color_list[0]? If not, can use color 0.
            # If yes, check color 1, etc.
            
            # This is still potentially slow.
            
            # Let's try a different strategy: 
            # Since N is small (512), we can use a bitset-based coloring?
            # Or just accept the O(N^2) per node if the depth is small.
            # The depth of the maximum clique is likely < 50.
            # The size of P decreases rapidly.
            
            # Let's implement the standard check.
            
            assigned = False
            for k in range(len(color_list)):
                # Check if v is compatible with all vertices in color_list[k]
                # If v is compatible with SOME vertex in color_list[k], it CANNOT have color k.
                # Wait, coloring rule: Adjacent vertices must have different colors.
                # So v cannot have color k if it is adjacent to ANY vertex in color_list[k].
                
                # Check adjacency:
                # mask_v = compat[v]
                # For any u in color_list[k], if mask_v & (1 << (u-1)), then conflict.
                
                conflict = False
                for u in color_list[k]:
                    if compat[v] & (1 << (u - 1)):
                        conflict = True
                        break
                if not conflict:
                    color_list[k].append(v)
                    vertex_color[v] = k
                    assigned = True
                    break
            
            if not assigned:
                color_list.append([v])
                vertex_color[v] = len(color_list) - 1
        
        max_color = len(color_list)
        
        # Bound: depth + max_color <= len(best_clique) -> Prune
        if depth + max_color <= len(best_clique):
            return
        
        # Branching
        # We process color classes from max_color down to 0.
        # Within a color class, the order doesn't matter much for correctness, but for speed, 
        # we can just iterate.
        
        # Tomita's algorithm branches on vertices in the order they were colored?
        # Actually, the standard implementation:
        # for k in range(max_color - 1, -1, -1):
        #    for v in color_list[k]:
        #       ...
        
        # But we must be careful with the P set update.
        # When we branch on v:
        #   New P = P \ {v} \ (V \ N(v))  =>  P \ (V \ N(v)) is not quite right.
        #   New P = P \ {v} intersected with N(v).
        #   Wait, the candidates for the next step must be neighbors of v.
        #   So new_P = [u for u in P if u != v and (compat[v] & (1 << (u-1)))]
        
        # However, we are iterating through color classes.
        # If we pick v from color_list[k], we remove v from P.
        # The remaining P for the "exclude v" branch is P without v.
        # The P for the "include v" branch is P \ {v} intersect N(v).
        
        # Implementation detail:
        # We can't just modify P in place easily while iterating color lists if we want to backtrack.
        # We should pass lists.
        
        # Let's restructure:
        # We have P (list).
        # We computed color_list.
        # We iterate k from max_color-1 down to 0.
        # For each v in color_list[k]:
        #   1. If depth + (number of vertices in color_list[k..max_color] that are still in P?) 
        #      Actually, the bound is simply depth + max_color.
        #      But as we remove vertices, the bound decreases.
        #      Tomita uses: if depth + (max_color - k) <= best, prune?
        #      No, the bound is based on the coloring of the *current* P.
        #      Since we are removing vertices, the coloring of the remaining P might change.
        #      Standard Tomita: 
        #      The coloring is done on the initial P for that node.
        #      The bound is depth + max_color.
        #      When we branch, we don't re-color immediately in the simple version, 
        #      but we do prune based on the current bound.
        
        # Let's stick to the simple version:
        # The bound is valid for the whole P.
        # As we remove vertices, the bound remains valid (upper bound).
        
        # To be more precise, we can update the bound.
        # But let's try the simple one first.
        
        # We need to be careful: 
        # When we pick v, we create new_P.
        # We recurse on new_P.
        # Then we remove v from P for the next iteration (exclude v).
        
        # However, the color_list was computed for the original P.
        # If we remove vertices, the color_list is no longer valid for the remaining P?
        # Yes, but the bound `depth + max_color` is still an upper bound.
        # It might be loose, but it's safe.
        
        # To improve, we can re-color if P is small.
        
        # Let's implement the loop:
        
        # We need to track which vertices are still in P.
        # P is a list.
        
        # We will iterate through the vertices in the order defined by the coloring.
        # Order: color_list[max_color-1], ..., color_list[0].
        
        # We need to be able to check if a vertex is still in P.
        # We can use a set or a boolean array.
        
        P_set = set(P)
        
        # Iterate colors in reverse
        for k in range(max_color - 1, -1, -1):
            # Vertices in this color class
            # We iterate through them.
            # Note: Some vertices in color_list[k] might have been removed from P in previous steps?
            # No, we are iterating the *original* P's color list.
            # We must check if v is still in P_set.
            
            for v in color_list[k]:
                if v not in P_set:
                    continue
                    
                # Bound check: 
                # The number of vertices in P with color >= k is an upper bound on the clique size in P.
                # Let count_ge_k = sum(len(color_list[i]) for i in range(k, max_color))
                # But we removed some vertices.
                # A tighter bound: count of vertices in P_set that have color >= k.
                # This is expensive to compute.
                # Let's just use the simple bound: if depth + (max_color - k) <= len(best_clique)?
                # No, that's not right.
                # The bound is depth + max_color.
                
                if depth + max_color <= len(best_clique):
                    return # Prune entire subtree? No, just this branch?
                # Actually, if depth + max_color <= best, we can return from the whole expand call.
                # We checked that at the beginning.
                
                # Include v
                C.append(v)
                
                # New P: neighbors of v in P_set \ {v}
                new_P = []
                mask_v = compat[v]
                for u in P_set:
                    if u != v and (mask_v & (1 << (u - 1))):
                        new_P.append(u)
                
                # Recurse
                expand(new_P, C, depth + 1)
                
                C.pop()
                
                # Exclude v
                P_set.remove(v)
                
                # Optional: if time is up, return
                if time.time() > DEADLINE:
                    return

    # Start the algorithm
    # Initial P is all nodes 1..N
    initial_P = list(range(1, N + 1))
    
    # To improve performance, we can sort initial_P by degree descending.
    # This helps the greedy coloring produce tighter bounds earlier.
    initial_P.sort(key=lambda v: -bin(compat[v]).count('1'))
    
    expand(initial_P, [], 0)
    
    # Return the best clique found
    if not best_clique:
        # Fallback
        return [1]
        
    return best_clique

# Run the construct function
if __name__ == "__main__":
    result = construct()
    print(f"Found clique of size {len(result)}")
    print(result)
