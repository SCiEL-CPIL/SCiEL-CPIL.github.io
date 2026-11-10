import time
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    N = 256
    max_sum = 2 * N
    
    # 1. Precompute squares
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 1):
        squares.add(i * i)
    
    # 2. Filter vertices: 2*v must not be a square (diagonal constraint a=a => 2a not square)
    valid_vertices = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_vertices.append(v)
            
    M = len(valid_vertices)
    if M == 0:
        return [1] # Fallback, though 1 is valid (2!=sq)
        
    # Map vertex value to index in valid_vertices
    # We work with indices 0..M-1
    # Build adjacency bitsets
    # adj[i] is an integer where bit j is 1 if valid_vertices[i] + valid_vertices[j] is a square
    adj = [0] * M
    
    # Optimization: Precompute which pairs are connected
    # Since M is ~240, M^2 is ~57600, which is small.
    for i in range(M):
        vi = valid_vertices[i]
        for j in range(i + 1, M):
            vj = valid_vertices[j]
            if vi + vj in squares:
                # Set bit j in adj[i] and bit i in adj[j]
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Global best solution
    best_size = 0
    best_set = 0 # Bitset representation
    
    # Helper: Popcount for Python integers
    def popcount(x):
        return bin(x).count('1')
        
    # Helper: Greedy Coloring for Upper Bound
    # Returns the number of colors used in a greedy coloring of the subgraph defined by mask
    def greedy_coloring_bound(mask):
        # We want to color the vertices in 'mask' such that adjacent vertices have different colors.
        # The number of colors is an upper bound on the independent set size? 
        # No, the size of the maximum independent set is at most the number of colors in any valid coloring?
        # Yes. If we partition vertices into k independent sets (colors), the max independent set can take at most 1 from each color? 
        # Wait. An independent set can contain multiple vertices from the SAME color if they are not adjacent?
        # No, by definition of coloring, vertices with the same color are NOT adjacent. 
        # So a color class IS an independent set.
        # Therefore, the maximum independent set size is at least the size of the largest color class.
        # But the UPPER bound logic in B&B for MIS usually uses the fact that if we color the graph with k colors, 
        # we can select at most 1 vertex from each color? NO. That's for Vertex Cover or something else?
        # Let's re-verify.
        # If we have a coloring with k colors, the vertices are partitioned into k independent sets $C_1, \dots, C_k$.
        # An independent set $I$ in the graph can contain vertices from multiple color classes.
        # However, $I$ cannot contain two adjacent vertices.
        # The standard bound for MIS B&B (Tomita) is:
        # If the subgraph is colored with k colors, then the size of the maximum independent set in that subgraph 
        # is at most k? 
        # Let's check. 
        # Example: Path of 3 vertices 1-2-3. Colors: 1(red), 2(blue), 3(red). k=2.
        # Max Independent Set: {1, 3} size 2. k=2. Matches.
        # Example: Cycle of 4 vertices 1-2-3-4-1. Colors: 1(R), 2(B), 3(R), 4(B). k=2.
        # Max Independent Set: {1, 3} or {2, 4} size 2. k=2. Matches.
        # Example: Complete graph K3. Colors: R, B, G. k=3.
        # Max Independent Set: size 1. k=3. 1 <= 3. Correct.
        # So yes, if a graph can be colored with k colors, the size of the maximum independent set is at most k.
        # Wait, is it "at most k" or "at most max color class size"?
        # Actually, the bound is that we can pick at most 1 vertex from each color? NO.
        # We can pick ALL vertices from a single color class because they are mutually non-adjacent.
        # So the bound is NOT k.
        # The bound used in Tomita's algorithm is based on the ordering of colors.
        # Specifically, if we assign colors $c_1, c_2, \dots$ to vertices $v_1, v_2, \dots$ in a specific order,
        # and we ensure that $c_i \le \text{max color of previous neighbors} + 1$,
        # then the number of vertices with color $\ge k$ is an upper bound?
        
        # Let's look up Tomita's algorithm bound.
        # "A simple and effective algorithm for the maximum independent set problem"
        # The bound is: If the vertices in the candidate set are ordered $v_1, \dots, v_m$ and colored such that
        # $color(v_i) \le \max \{ color(v_j) \mid v_j \in N(v_i), j < i \} + 1$,
        # then the size of the maximum independent set is at most the number of vertices with color $c$ 
        # such that there is no vertex with color $> c$? No.
        
        # Actually, a simpler bound often used is:
        # If we can color the graph with $k$ colors, the MIS size is NOT bounded by $k$.
        # Example: Two disjoint triangles. Colors: R,B,G for first, R,B,G for second. Total colors 3.
        # MIS size is 2 (one from each triangle). 2 <= 3.
        # Example: Complete bipartite graph K_{n,n}. Colors: R, B. k=2.
        # MIS size is n. If n=10, k=2. 10 > 2. So "MIS <= k" is FALSE.
        
        # So the simple "number of colors" is NOT a valid upper bound for MIS size.
        # The valid bound in Tomita's algorithm relies on the specific coloring order and the fact that
        # if we branch on vertices in reverse color order, we can prune.
        
        # Given the complexity of implementing Tomita's specific bound correctly in Python quickly,
        # and the fact that M ~ 240, a simpler B&B with a good heuristic lower bound might suffice.
        # Or, we can use the fact that the graph is sparse.
        
        # Alternative: Use a simple greedy coloring to estimate the bound as the size of the largest color class?
        # No, MIS can be larger than the largest color class?
        # In K_{n,n}, largest color class is n. MIS is n.
        # In a path of 5: 1-2-3-4-5. Colors R,B,R,B,R. Largest class size 3 (1,3,5). MIS is 3.
        # It seems the size of the MIS is often close to the largest color class or sum of some.
        
        # Let's stick to a robust approach:
        # 1. Get a good initial solution using Greedy + Local Search.
        # 2. Use B&B with simple pruning: if size + popcount(candidates) <= best, prune.
        # 3. To speed up B&B, order vertices by degree.
        
        pass

    # Let's implement the Greedy + Local Search first to get a strong lower bound.
    
    def greedy_initial():
        # Sort vertices by degree (highest first)
        degrees = [popcount(adj[i]) for i in range(M)]
        order = sorted(range(M), key=lambda x: -degrees[x])
        
        selected = set()
        candidates = set(range(M))
        
        for v in order:
            if v not in candidates:
                continue
            # Check if v can be added
            # v can be added if no neighbor is in selected
            if not (adj[v] & (1 << v)): # Self loop check, already handled by valid_vertices
                # Check neighbors in selected
                # We need to check if any neighbor of v is in selected
                # Since selected is a set of indices, we can check intersection
                # But we are using bitsets for adj. Let's maintain selected as a bitset too.
                pass
        
        # Let's use bitsets for everything in the heuristic too for speed.
        selected_bits = 0
        candidates_bits = (1 << M) - 1
        
        # Greedy: pick vertex with highest degree in current candidate set?
        # Or just iterate in order of original degree.
        for v in order:
            if not (candidates_bits & (1 << v)):
                continue
            # If v is in candidates, check if it conflicts with selected
            # Conflict if (adj[v] & selected_bits) != 0
            if (adj[v] & selected_bits) == 0:
                selected_bits |= (1 << v)
                # Remove v and its neighbors from candidates
                candidates_bits &= ~(1 << v)
                candidates_bits &= ~adj[v]
                
        return selected_bits

    best_set = greedy_initial()
    best_size = popcount(best_set)
    
    # Local Search to improve
    # Try adding vertices that can be added, or swapping?
    # Simple improvement: try to add any vertex that doesn't conflict.
    # Then try to remove a vertex to add two others?
    
    # For now, let's run B&B with the initial bound.
    # To make B&B fast, we need a good bound.
    # Let's use a simple bound: size + popcount(candidates)
    # And order the branching by degree.
    
    # Precompute a static order for branching: vertices with higher degree first.
    # This helps in pruning earlier.
    branch_order = sorted(range(M), key=lambda x: -popcount(adj[x]))
    
    # Recursive B&B
    # To avoid recursion depth issues and overhead, we might need to be careful.
    # But M=240 is small enough for recursion if pruning is effective.
    
    def dfs(candidates, current_size, current_set):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
            
        if current_size > best_size:
            best_size = current_size
            best_set = current_set
            
        if candidates == 0:
            return
            
        # Prune: even if we take all remaining, we can't beat best
        if current_size + popcount(candidates) <= best_size:
            return
            
        # Pick a vertex to branch on.
        # Heuristic: pick the vertex with the highest degree within the current candidate set.
        # This is expensive to compute every time.
        # Alternative: use the precomputed branch_order, and pick the first one in candidates.
        
        # Let's find the first vertex in branch_order that is in candidates.
        v = -1
        for u in branch_order:
            if candidates & (1 << u):
                v = u
                break
                
        if v == -1:
            return
            
        # Branch 1: Include v
        # New candidates: remove v and its neighbors
        new_candidates = candidates & ~(1 << v) & ~adj[v]
        dfs(new_candidates, current_size + 1, current_set | (1 << v))
        
        # Branch 2: Exclude v
        new_candidates_ex = candidates & ~(1 << v)
        dfs(new_candidates_ex, current_size, current_set)

    # Start B&B
    # Initial candidates: all valid vertices
    initial_candidates = (1 << M) - 1
    dfs(initial_candidates, 0, 0)
    
    # Convert best_set bitset to list of vertex values
    solution = []
    for i in range(M):
        if best_set & (1 << i):
            solution.append(valid_vertices[i])
            
    if not solution:
        return [valid_vertices[0]]
        
    return solution
