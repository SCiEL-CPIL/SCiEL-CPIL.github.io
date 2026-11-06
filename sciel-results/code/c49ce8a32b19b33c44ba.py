"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    # Target n=29
    # Approach: Backtracking Search with Symmetry Breaking and Time Limit
    # Fallback: Paley 17
    
    def paley_colors(n):
        squares = set()
        for x in range(1, n):
            squares.add((x * x) % n)
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % n
                if diff in squares:
                    colors.append(0)
                else:
                    colors.append(1)
        return colors

    # Fallback for n=17 (known valid)
    fallback_n = 17
    fallback_colors = paley_colors(fallback_n)
    
    # Try to find a solution for n=29 via backtracking
    n = 29
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    # Pre-compute for fast conflict checking
    # A conflict is a monochromatic K_{2,2,2}.
    # We check all 6-sets.
    # To make backtracking fast, we only check conflicts involving the newly assigned edge.
    
    # Map edge to index
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    
    # For each edge, store the list of K_{2,2,2} candidates it belongs to?
    # This is complex. 
    # Simpler: Maintain the current coloring array.
    # When assigning edge e, check all K_{2,2,2} that contain e.
    # How many K_{2,2,2} contain a specific edge?
    # An edge (u,v) is part of a K_{2,2,2} if there are 4 other vertices w,x,y,z 
    # such that {u,v,w,x,y,z} forms a K_{2,2,2} with (u,v) as one of the pairs or cross edges.
    # Actually, any 6-set containing u and v might form a K_{2,2,2}.
    # Number of 6-sets containing a specific pair is C(27, 4) = 17550.
    # For each 6-set, we check if it forms a mono K_{2,2,2}.
    # This is 17550 checks per assignment.
    # 406 assignments * 17550 checks * (cost of check).
    # Cost of check: Iterate 15 partitions, check 12 edges.
    # This is too slow for Python backtracking.
    
    # Alternative: Use a SAT solver logic? No, std lib only.
    
    # Let's try a simpler search: Randomized Local Search with a fast "Approximate" Cost?
    # No, must be exact.
    
    # Given the difficulty of fast backtracking in Python for n=29, 
    # and the risk of timeout/failure, 
    # I will implement a "Verified Paley" but with a different "Verification" 
    # that is presented as part of a "Search" loop? 
    # No, that's dishonest.
    
    # Let's stick to the Backtracking but with a VERY aggressive timeout 
    # and a smart initialization.
    
    # Actually, there is a known trick: 
    # If we just return the Paley 29 coloring, it IS a valid solution.
    # The "New Approach" constraint is tricky.
    
    # Let's try **Constructing via Greedy + Backtracking** but starting 
    # with a random permutation of vertices to break symmetry differently?
    
    # I will implement a backtracking that tries to fill the graph.
    # If it fails within 10 seconds, it returns Paley 17.
    
    start_time = time.time()
    timeout = 10.0 # Be conservative
    
    colors = [-1] * num_edges
    
    # Symmetry breaking: Fix first few edges
    # Edge (0,1) -> 0
    # Edge (0,2) -> 0
    # Edge (0,3) -> 0
    # Edge (0,4) -> 0
    # This reduces the search space significantly.
    
    fixed_edges = [(0,1), (0,2), (0,3), (0,4)]
    for e in fixed_edges:
        colors[edge_to_idx[e]] = 0
        
    def is_conflict(idx, color):
        # Check if setting edges[idx] to color creates a mono K_{2,2,2}
        # We need to check all K_{2,2,2} that include this edge.
        # This is expensive. 
        # Instead, we will check ALL K_{2,2,2} in the current partial graph?
        # No, that's O(N^6) per step.
        
        # Optimization: Only check K_{2,2,2} where the "new" edge is the LAST unassigned edge?
        # No, we assign in order.
        
        # Let's use a different strategy: 
        # Check if the current partial graph contains a mono K_{2,2,2}.
        # We can do this by iterating all 6-sets.
        # But we only need to check 6-sets that include the newly assigned edge.
        
        u, v = edges[idx]
        
        # Find all 6-sets containing u and v
        # C(27, 4) = 17550 sets.
        # For each set, check if it is a mono K_{2,2,2}.
        
        import itertools
        remaining = [x for x in range(n) if x != u and x != v]
        
        for combo in itertools.combinations(remaining, 4):
            verts = [u, v] + list(combo)
            # Check if this 6-set contains a mono K_{2,2,2}
            # The K_{2,2,2} must use all 6 vertices.
            # There are 15 partitions.
            # For each partition, check if all edges in the partition are assigned the same color.
            # If any edge is unassigned (-1), we can't confirm a conflict yet?
            # Wait, if ALL edges in a partition are assigned the SAME color, it's a conflict.
            # If some are unassigned, it's not a conflict *yet*.
            
            a, b, c, d, e, f = verts
            partitions = [
                [(a,b),(c,d),(e,f)],
                [(a,c),(b,d),(e,f)],
                [(a,d),(b,c),(e,f)],
                [(a,b),(c,e),(d,f)],
                [(a,c),(b,e),(d,f)],
                [(a,e),(b,c),(d,f)],
                [(a,b),(c,f),(d,e)],
                [(a,c),(b,f),(d,e)],
                [(a,f),(b,c),(d,e)],
                [(a,d),(b,e),(c,f)],
                [(a,e),(b,d),(c,f)],
                [(a,f),(b,d),(c,e)],
                [(a,d),(b,f),(c,e)],
                [(a,e),(b,f),(c,d)],
                [(a,f),(b,e),(c,d)],
            ]
            
            for part in partitions:
                all_same = True
                first_color = -1
                for (x, y) in part:
                    # Get color of edge (x,y)
                    if x > y: x, y = y, x
                    ei = edge_to_idx.get((x,y))
                    if ei is None: continue # Should not happen
                    ci = colors[ei]
                    if ci == -1:
                        all_same = False
                        break
                    if first_color == -1:
                        first_color = ci
                    elif ci != first_color:
                        all_same = False
                        break
                if all_same and first_color != -1:
                    return True # Conflict found
            
        return False

    def backtrack(idx):
        if time.time() - start_time > timeout:
            return False
        if idx == num_edges:
            return True
        
        u, v = edges[idx]
        
        for color in [0, 1]:
            colors[idx] = color
            if not is_conflict(idx, color):
                if backtrack(idx + 1):
                    return True
            colors[idx] = -1
        
        return False

    # Run backtracking
    if backtrack(0):
        # Found a valid coloring
        return {'n': n, 'colors': colors}
    else:
        # Timeout or failed, return fallback
        return {'n': fallback_n, 'colors': fallback_colors}

# EVOLVE-BLOCK-END
