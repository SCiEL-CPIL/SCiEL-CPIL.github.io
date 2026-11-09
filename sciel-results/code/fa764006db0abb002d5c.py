import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s margin

    N = 256
    max_sum = 2 * N
    squares = set()
    i = 2
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
    
    # 1. Filter valid nodes
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)
            
    M = len(valid_nodes)
    if M == 0:
        return [1]
        
    # Map node value to index
    node_to_idx = {v: i for i, v in enumerate(valid_nodes)}
    
    # 2. Build Complement Graph (Compatibility Graph)
    # Edge (i, j) exists if valid_nodes[i] + valid_nodes[j] is NOT a square
    # We use Python integers as bitsets for adjacency
    adj = [0] * M
    
    for i in range(M):
        vi = valid_nodes[i]
        for j in range(i + 1, M):
            vj = valid_nodes[j]
            if vi + vj not in squares:
                # Compatible
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # 3. Tomita's Algorithm for Maximum Clique in the Compatibility Graph
    # This is equivalent to Maximum Independent Set in the Conflict Graph
    
    best_clique_size = 0
    best_clique = []
    
    # Greedy coloring for upper bound
    # Returns (coloring, order) where coloring[i] is the color of vertex i in the ordered list
    # and order is the list of vertices sorted by color
    def color_sort(candidates):
        # candidates: list of vertex indices
        # We perform greedy coloring
        # Returns a list of (color, vertex) sorted by color descending? 
        # Standard Tomita sorts by color, then iterates.
        
        # Simple greedy coloring
        # We want to order vertices such that we can prune.
        # The upper bound for a vertex v is its color.
        
        # Let's use a standard greedy coloring that returns the max color
        # and the ordering.
        
        # To be efficient, we can just calculate the max color.
        # But we need the ordering to iterate.
        
        # Implementation:
        # 1. Sort candidates by degree in the induced subgraph (optional, but helps)
        # 2. Assign colors greedily.
        
        # For simplicity and speed in Python, we'll do a simple greedy coloring
        # on the list.
        
        # We need to map vertex to its color.
        # And we need to iterate vertices in decreasing order of color.
        
        # Let's create a list of (vertex, color)
        
        # Greedy coloring algorithm:
        # For each vertex in some order, assign the smallest available color.
        
        # Order: We can use the order they are in `candidates` or sort by degree.
        # Sorting by degree (in the subgraph induced by candidates) is better for Tomita.
        
        # Calculate degrees within candidates
        # This is O(M^2) per call, which might be slow if called many times.
        # M=245. 245^2 = 60k. 
        # Number of calls? Could be high.
        
        # Optimization: Use a simpler bound if needed, but Tomita relies on coloring.
        
        # Let's try a simpler approach:
        # Just return the max color found by a greedy pass.
        
        # Actually, the standard Tomita implementation re-colors at every node.
        
        # Let's implement a fast greedy coloring.
        
        # We'll use a dictionary for colors
        # And a list of vertices sorted by color.
        
        # To speed up, we can pre-compute nothing, but use bit operations.
        
        # Let's stick to the list-based approach for clarity, and optimize if too slow.
        
        # Re-order candidates by degree in the subgraph
        # deg[v] = popcount(adj[v] & mask(candidates))
        
        # This is expensive. 
        # Alternative: Just use the order of indices?
        # Let's try simple greedy coloring on the given list order first.
        
        colors = {}
        used_colors = []
        
        # To make it deterministic and potentially better, sort by degree in full graph?
        # No, degree in subgraph is what matters.
        
        # Let's just do greedy on the list as is.
        for v in candidates:
            # Find smallest color not used by neighbors in candidates
            # Neighbors of v in candidates:
            # nbs = adj[v] & mask(candidates)
            
            # We need a mask of candidates
            # Creating mask is O(M)
            
            # This is getting complex to optimize in pure Python quickly.
            # Let's use a simpler upper bound: 
            # The size of the maximum matching? No.
            # The number of vertices? Too loose.
            
            # Let's try a different strategy: 
            # Branch and Bound with a simple greedy independent set size as lower bound (initialization)
            # and a simple upper bound like: 
            # 1. |P| (trivial)
            # 2. Greedy Coloring (standard)
            
            # I will implement the standard Tomita coloring.
            
            pass # Placeholder, will implement below
            
    # Given the complexity of implementing Tomita's coloring efficiently in Python without C extensions,
    # and the risk of timeout, let's consider the ILP approach mentioned in the findings.
    # SciPy's milp is available.
    
    # Formulation:
    # Maximize sum(x_i)
    # Subject to:
    # x_i + x_j <= 1 for all edges (i,j) in Conflict Graph
    # x_i in {0, 1}
    
    # Conflict Graph edges: i != j, valid_nodes[i] + valid_nodes[j] is square.
    # Note: Diagonal constraint already handled by filtering valid_nodes.
    
    # Number of variables: M ~ 245.
    # Number of constraints: Edges in conflict graph.
    # Conflict graph is sparse.
    
    # Let's build the constraint matrix.
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        
        # Variables x[0..M-1]
        # Objective: maximize sum(x) -> minimize -sum(x)
        c = -np.ones(M)
        
        # Constraints:
        # For each conflict edge (i, j): x_i + x_j <= 1
        # This can be written as A @ x <= b
        
        # Collect edges
        edges = []
        for i in range(M):
            vi = valid_nodes[i]
            for j in range(i + 1, M):
                vj = valid_nodes[j]
                if vi + vj in squares:
                    edges.append((i, j))
                    
        if not edges:
            return valid_nodes
            
        # Build A
        # Number of constraints = len(edges)
        # Sparse matrix is better.
        
        from scipy.sparse import lil_matrix, csc_matrix
        
        A = lil_matrix((len(edges), M), dtype=int)
        b = np.ones(len(edges))
        
        for k, (i, j) in enumerate(edges):
            A[k, i] = 1
            A[k, j] = 1
            
        A = csc_matrix(A)
        
        # Bounds: 0 <= x_i <= 1
        lb = np.zeros(M)
        ub = np.ones(M)
        
        # Integrality: all integers
        integrality = np.ones(M)
        
        # Solve
        # Time limit? milp doesn't have a direct time limit param in all versions,
        # but we can run it in a thread or just hope it's fast.
        # For N=245 and sparse constraints, it should be fast.
        
        # To be safe against timeout, we can run it with a timeout wrapper?
        # Or just trust it. The findings say ILP worked.
        
        res = milp(c, constraints=LinearConstraint(A, -np.inf, b), 
                   integrality=integrality, bounds=Bounds(lb, ub))
        
        if res.success:
            # res.x is the solution
            # Round to integers
            x = np.round(res.x).astype(int)
            
            solution = []
            for i in range(M):
                if x[i] == 1:
                    solution.append(valid_nodes[i])
                    
            if solution:
                return solution
        else:
            # Fallback to greedy if ILP fails
            pass
            
    except Exception as e:
        # Fallback
        pass
        
    # Fallback: Greedy + Local Search
    # This part is similar to the original but we ensure it's robust.
    
    # Greedy Initialization
    result = set()
    remaining = list(valid_nodes)
    
    # Sort by degree in conflict graph (fewest conflicts first)
    # Calculate degrees
    degrees = [0] * M
    for i, j in edges:
        degrees[i] += 1
        degrees[j] += 1
        
    remaining.sort(key=lambda v: degrees[node_to_idx[v]])
    
    for v in remaining:
        vi = node_to_idx[v]
        can_add = True
        for u in result:
            ui = node_to_idx[u]
            if (adj[vi] >> ui) & 1 == 0: # Wait, adj is compatibility.
                # If compatible, bit is 1.
                # We want to check if u is a CONFLICT.
                # Conflict if NOT compatible.
                # So if bit is 0 (and u != v), they conflict.
                # But adj[v] has 0 on diagonal.
                # So if u != v and bit is 0, conflict.
                
                # Actually, let's just check sum.
                if v + u in squares:
                    can_add = False
                    break
        if can_add:
            result.add(v)
            
    # Local Search: Swaps
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try to add
        for v in valid_nodes:
            if v not in result:
                can_add = True
                for u in result:
                    if v + u in squares:
                        can_add = False
                        break
                if can_add:
                    result.add(v)
                    improved = True
                    break
        # Try swap: remove one, add two
        if not improved:
            for remove in list(result):
                temp_result = result - {remove}
                # Try to add v1
                for v1 in valid_nodes:
                    if v1 in temp_result:
                        continue
                    can_add_v1 = True
                    for u in temp_result:
                        if v1 + u in squares:
                            can_add_v1 = False
                            break
                    if not can_add_v1:
                        continue
                    temp_result2 = temp_result | {v1}
                    # Try to add v2
                    for v2 in valid_nodes:
                        if v2 in temp_result2:
                            continue
                        can_add_v2 = True
                        for u in temp_result2:
                            if v2 + u in squares:
                                can_add_v2 = False
                                break
                        if can_add_v2:
                            # Check if size increases
                            if len(temp_result2) + 1 > len(result):
                                result = temp_result2 | {v2}
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
                    
    return list(result)
