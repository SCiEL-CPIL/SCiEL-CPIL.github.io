import time
import sys
import numpy as np
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s

    # Helper: Generate all C6 cycles as frozensets of edge-tuples (u,v) with u<v
    def generate_cycles(n):
        cycles = []
        seen = set()
        # For each 6-tuple of distinct vertices
        for verts in itertools.combinations(range(n), 6):
            # Fix the first vertex to break rotational symmetry
            # Permute the remaining 5, break reflection by requiring v1 < v5
            v0 = verts[0]
            others = list(verts[1:])
            for perm in itertools.permutations(others):
                if perm[0] > perm[-1]:
                    continue
                c_verts = [v0] + list(perm)
                edges = []
                for k in range(6):
                    u, v = c_verts[k], c_verts[(k + 1) % 6]
                    if u > v:
                        u, v = v, u
                    edges.append((u, v))
                # Canonical key for deduplication
                key = frozenset(edges)
                if key not in seen:
                    seen.add(key)
                    cycles.append(list(key)) # Store as list of edge tuples
        return cycles

    def solve(n, time_limit):
        """
        Backtracking search for a 4-coloring of K_n without mono-C6.
        Strategy: 
        1. Order vertices to assign edges in a specific order? 
        No, standard edge-order (lexicographic) is simpler for state, but 
        vertex-based ordering might allow better pruning.
        
        Let's stick to lexicographic edge order but with strong pruning.
        
        State:
        - colors: array of length E (0..E-1), values 0..3 or -1 (unassigned)
        - adj_colors: list of length n, each is an array of size n, values 0..3 or -1
          adj_colors[u][v] is the color of edge (u,v).
        
        Precomputation:
        - edges: list of (u,v) for i<j
        - edge_idx: dict (u,v) -> idx
        - cycles: list of frozensets of edge-indices? 
          Storing frozensets of indices is memory heavy but okay for n=17/18.
          n=17: E=136, C6 count ~ 371k.
          n=18: E=153, C6 count ~ 371280? No, C(18,6)*30 = 18564*30 = 556,920.
          Each cycle has 6 edges.
          Total edge-cycle incidences = 6 * 556920 = 3.3M.
          This is getting large.
          
          Alternative: Check for mono-C6 on the fly?
          When assigning edge (u,v) with color c, check if there is a path of length 4
          in the subgraph of color c connecting u and v?
          If u and v are connected by a path of length 4 in color c, and we color (u,v) with c,
          we form a C6 (assuming the path is induced? No, C6 is not induced.
          The path u-x-y-z-w-v has 5 edges. Adding (u,v) makes a cycle of length 6.
          Wait, a path of length 5 (5 edges) + closing edge = C6.
          So if we add edge (u,v) color c, we create a mono-C6 if and only if
          there is a simple path of length 5 between u and v in the graph of color c.
          
          How to check for simple path of length 5?
          DFS from u with depth limit 5 to see if we reach v?
          Depth 5 is very small. Branching factor in a sparse graph (color class) is low.
          Each color class in a valid 4-coloring of K17 has at most ~34 edges.
          Average degree ~ 4.
          BFS/DFS depth 5 from u: max nodes visited ~ 1 + 4 + 16 + 64 + 256 + 1024.
          This is fast!
          
          This is much better than precomputing all cycles and checking bitmasks.
          
          Let's verify:
          A 4-coloring is valid iff no color class contains a C6.
          A C6 is a cycle of length 6.
          If we assign edge (u,v) color c, it creates a C6 if and only if
          (u,v) is part of a C6 in color c.
          This is equivalent to: is there a path of length 5 between u and v in G_c 
          (the graph of edges already colored c)?
          Note: The path must be simple (no repeated vertices) to form a simple cycle.
          BFS/DFS with visited set handles this.
          
          Complexity:
          Number of edges to assign: E = n(n-1)/2.
          For each edge, for each color (4 attempts), we do a BFS/DFS of depth 5.
          E ~ 136. 4 * 136 = 544 checks.
          Each check visits <= ~1000 nodes worst case, usually much less.
          This is extremely fast compared to 371k cycle checks.
          
          We can optimize further:
          Maintain the color subgraphs dynamically.
          
          Let's implement this.
        """
        E = n * (n - 1) // 2
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # adj_color_matrix: n x n, init -1
        # Using numpy for speed? Or just list of lists.
        # List of lists of ints is fast enough for depth 5 BFS.
        mat = [[-1] * n for _ in range(n)]
        # colors list for output, lexicographic order
        colors = [0] * E
        
        # Symmetry breaking:
        # Fix edge (0,1) to 0?
        # Even better: 
        # Edge (0,1) = 0
        # Edge (0,2) = 0 or 1
        # Edge (0,3) = 0, 1, or 2
        # This reduces initial branching.
        
        # We will assign edges in lexicographic order.
        # But we can hardcode the first few to break symmetry.
        
        # Let's just fix (0,1)=0.
        mat[0][1] = mat[1][0] = 0
        colors[0] = 0 # Edge 0 is (0,1)
        
        # Stack for backtracking: (edge_idx, next_color, prev_state_backup?)
        # We need to backup the changes to `mat` if we backtrack.
        # Changes are: mat[u][v] and mat[v][u].
        # We store (u, v, old_val) to undo.
        
        # Stack item: [edge_idx, next_color_to_try]
        # We also need to record what we changed to undo it.
        
        start_time_local = time.time()
        
        # We define a helper for BFS depth 5
        def has_path_depth_5(u, v, mat_local):
            # BFS from u, depth limit 5, target v.
            # Returns True if simple path of length 5 exists.
            # Note: u != v.
            
            # visited: dict or set. Since n <= 18/20, we can use a bitset or just a list.
            # Actually, we need to ensure simple path (no repeated vertices).
            # Standard DFS with visited set.
            
            # Optimization: If dist(u,v) in mat_local graph is < 5, we might still have a path of length 5?
            # No, if there is a path of length 3, we can't just extend it to 5 without repeating?
            # Actually, we just need ANY simple path of length exactly 5.
            
            # DFS is better for exact length constraint?
            # BFS with depth tracking and path reconstruction?
            # For depth 5, DFS is very cheap.
            
            def dfs(curr, target, depth, visited):
                if depth == 5:
                    return curr == target
                visited.add(curr)
                for nxt in range(n):
                    if nxt == curr:
                        continue
                    if mat_local[curr][nxt] != -1: # Check edge exists in this color
                        # Actually we pass the color implicitly? 
                        # No, `mat_local` here should only contain edges of the specific color.
                        # Better: pass `color` and check `mat[curr][nxt] == color`.
                        pass
                visited.remove(curr)
                return False
            
            # Let's implement DFS inline for speed.
            # We want to check if there is a simple path of length 5 between u and v.
            # u and v are distinct.
            
            # Quick check: is v a neighbor of u? (Path len 1) -> No, need 5.
            # If the subgraph is sparse, this is fast.
            
            # DFS
            stack = [(u, 0, frozenset([u]))]
            while stack:
                curr, d, visited = stack.pop()
                if d == 5:
                    if curr == v:
                        return True
                    continue
                for nxt in range(n):
                    if nxt not in visited:
                        # Check if edge (curr, nxt) exists in color `color`
                        if mat[curr][nxt] == color:
                            new_visited = visited | {nxt}
                            stack.append((nxt, d + 1, new_visited))
            return False

        # Wait, the function above uses `mat` and `color` from closure. 
        # It's better to pass them or make it nested.
        
        def is_valid_color_assignment(u, v, color, mat_local):
            # Check if adding edge (u,v) with this color creates a C6.
            # This happens if there is a simple path of length 5 between u and v
            # in the subgraph induced by edges of this `color` already in mat_local.
            
            # Optimization: 
            # If u and v are not connected in the subgraph, no path.
            # If they are connected by a path of length 5, yes.
            
            # Special case: if u and v are already adjacent in this color? 
            # No, we are assigning it, so it's not in mat_local yet.
            
            # BFS/DFS depth 5.
            
            # To make it faster, we can pre-compute neighbors for each color?
            # Maintaining dynamic neighbor lists for 4 colors is cheap.
            
            # But for n=17/18, iterating 0..n-1 is only 17/18 steps.
            # Depth 5, branching ~4. 4^5 = 1024. 
            # 1024 * 18 = 18k operations per check.
            # 18k * 544 checks = 10M ops.
            # In Python, 10M ops is ~1-2 seconds. 
            # This is too slow if we have to do this for many backtracks.
            
            # We need to prune the search space aggressively.
            # Simple chronological backtracking might explode.
            
            # Idea: Use MRV (Minimum Remaining Values) or Degree ordering?
            # Or, better: maintain the count of "potential C6s" formed?
            
            # Let's try a different approach: 
            # Instead of checking path of length 5, we can just check if u and v 
            # are in the same connected component of depth <= 5? 
            # No, simple path is required.
            
            # Let's stick to DFS but optimize it.
            # Use bitsets for visited if possible?
            
            # Let's first try if the simple backtracking with this check is fast enough.
            # If it fails, we need a smarter heuristic.
            
            # Heuristic for edge ordering:
            # Assign edges in an order that maximizes constraints?
            # Lexicographic is standard.
            
            # Let's implement the DFS check efficiently.
            
            # We will do a BFS to find all nodes reachable within depth 4 from u.
            # Then check if v is reachable from any of those in 1 step?
            # No, path length exactly 5.
            
            # Let's use a simple recursive DFS.
            
            # To avoid function call overhead, let's inline it?
            # Or just write a tight loop.
            
            def has_path_len_5(u, v, color, mat_local, n_local):
                # Use a stack to avoid recursion limit (though depth is small)
                # Node, depth, visited_frozen_set
                # Frozenset copy is expensive. 
                # For depth 5, we can use a 64-bit int bitmask since n<=20.
                
                mask = 1 << u
                stack = [(u, 0)]
                
                while stack:
                    curr, d = stack.pop()
                    if d == 5:
                        if curr == v:
                            return True
                        continue
                    next_mask = mask
                    # Iterate neighbors
                    for nxt in range(n_local):
                        if mat_local[curr][nxt] == color and not (next_mask & (1 << nxt)):
                            # Only push if not visited
                            # Wait, we need to update mask per branch.
                            # So we push (nxt, d+1, mask | (1<<nxt))
                            pass
                # This simple stack doesn't track visited per path.
                # We must use path-specific visited.
                
                # Correct approach with bitmask:
                def _dfs(curr, depth, visited_mask):
                    if depth == 5:
                        return curr == v
                    visited_mask |= (1 << curr)
                    for nxt in range(n_local):
                        if mat_local[curr][nxt] == color and not (visited_mask & (1 << nxt)):
                            if _dfs(nxt, depth + 1, visited_mask):
                                return True
                    return False
                    
                # Recursion depth 5 is safe.
                return _dfs(u, 0, 0)

            return not has_path_len_5(u, v, color, mat_local, n_local)

        # Backtracking loop
        # Stack of (edge_idx, color_idx_to_try_next)
        # We also need to restore `mat` on backtrack.
        # We store the state of the edge being backtracked.
        
        # Let's use a stack of states.
        # State: edge_idx
        # We assign colors[edge_idx] and update mat.
        
        # To simplify, we just store the sequence of assignments.
        # assignments: list of (u, v, old_color)
        
        assignments = []
        
        # Initial edge 0 (0,1) is fixed to 0.
        # Let's start from edge 1.
        
        # Stack: (edge_idx, next_color)
        stack = [(1, 0)]
        
        # Time check
        while stack:
            if time.time() > time_limit:
                return None
            
            edge_idx, next_color = stack[-1]
            
            if edge_idx == E:
                return colors
            
            found = False
            while next_color < 4:
                u, v = edges[edge_idx]
                # Check validity
                if is_valid_color_assignment(u, v, next_color, mat, n):
                    # Apply
                    mat[u][v] = next_color
                    mat[v][u] = next_color
                    colors[edge_idx] = next_color
                    assignments.append((u, v, next_color)) # Store for undo? 
                    # Actually, we just need to set back to -1 on backtrack.
                    # But what was it before? It was -1.
                    # So undo is just set to -1.
                    
                    # Push next edge
                    stack.append((edge_idx + 1, 0))
                    found = True
                    break
                next_color += 1
            
            if found:
                # Pop the top of the stack's "next_color" state? 
                # No, we appended a new state.
                # We need to remove the state that is currently being processed?
                # Standard iterative DFS:
                # If we find a valid color, we push the next level.
                # If we don't find, we pop the current level and increment its color counter.
                
                # Wait, my logic is slightly off.
                # The `stack` holds frames.
                # If we found a color, we push a new frame for edge_idx + 1.
                # The current frame (edge_idx, next_color) is done for this branch?
                # No, if the deeper search fails, we need to backtrack to this frame 
                # and try the NEXT color.
                # So we should NOT pop the current frame when we push a deeper one.
                # We only pop when we exhaust colors.
                
                # But my stack structure is: [(edge_idx, next_color_to_try)]
                # When we push (edge_idx+1, 0), we are entering a new state.
                # The previous state (edge_idx, next_color) should be preserved 
                # with the color we just tried, so we know to increment it if we come back.
                
                # So, when we succeed:
                # We push (edge_idx + 1, 0).
                # We update the current top? No, the current top is now covered.
                # We need to remember that for `edge_idx`, we tried `next_color`.
                # When we pop back to `edge_idx`, we need to try `next_color + 1`.
                
                # So, when we push, we don't change the current frame.
                # When we pop, we increment the frame's color counter.
                
                # Let's adjust:
                # If found:
                #   Push (edge_idx + 1, 0)
                #   Break (to process the new top)
                
                # If not found (all 4 colors tried for this edge):
                #   Undo the assignment for this edge (set mat to -1)
                #   Pop the frame
                #   If stack is not empty, increment the color counter of the new top
                #   (The new top is the parent edge)
                
                pass # Logic is a bit tangled, let's rewrite the loop clearly.

        # Rewritten Loop Logic:
        # Stack contains (edge_idx, next_color_to_try)
        # Initial: [(1, 0)]
        
        # We need to handle the "undo" part carefully.
        # When we pop, we must revert the assignment made for THAT edge.
        # The assignment for edge_idx is stored in `mat` and `colors`.
        # To revert: mat[u][v] = -1, mat[v][u] = -1.
        
        # Let's restart the loop with clearer control flow.
        
        assignments_stack = [] # To track what to undo
        
        # Reset state for clean start of this specific solve attempt
        mat = [[-1] * n for _ in range(n)]
        colors = [0] * E
        mat[0][1] = mat[1][0] = 0
        colors[0] = 0
        assignments_stack.append((0, 1, 0)) # dummy? No, just start from edge 1
        
        stack = [(1, 0)]
        
        while stack:
            if time.time() > time_limit:
                return None
            
            edge_idx, next_color = stack[-1]
            
            if edge_idx == E:
                return colors
            
            # Find a valid color for edge_idx
            valid_color_found = False
            while next_color < 4:
                u, v = edges[edge_idx]
                # Optimization: If edge was pre-assigned? No, only 0 is.
                if is_valid_color_assignment(u, v, next_color, mat, n):
                    # Apply assignment
                    mat[u][v] = next_color
                    mat[v][u] = next_color
                    colors[edge_idx] = next_color
                    assignments_stack.append((u, v))
                    valid_color_found = True
                    # Push next edge
                    stack.append((edge_idx + 1, 0))
                    break
                next_color += 1
            
            if valid_color_found:
                # We pushed a new item. Loop continues to process it.
                # We don't pop or update current item.
                pass
            else:
                # No color worked for this edge.
                # Backtrack:
                if edge_idx > 0: # We only start from 1, but 0 is fixed.
                    # Undo assignment for THIS edge
                    # The last item in assignments_stack corresponds to this edge?
                    # Yes, because we append when we assign.
                    u, v = assignments_stack.pop()
                    mat[u][v] = -1
                    mat[v][u] = -1
                
                # Remove current frame
                stack.pop()
                
                if not stack:
                    return None
                
                # Increment color counter of the new top (parent edge)
                prev_edge_idx, prev_color = stack[-1]
                stack[-1] = (prev_edge_idx, prev_color + 1)
        
        return None

    # Try n=18 first? 
    # If we have a fast checker, n=18 might be feasible.
    # But if n=18 is infeasible, we waste time.
    # Known: R_4(C6) is likely 18. So n=18 might not exist.
    # Let's try n=17 first to secure a score, then n=18.
    
    best_n = 0
    best_colors = None
    
    # Try n=17
    t_start = time.time()
    res_17 = solve(17, t_start + 80)
    
    if res_17 is not None:
        best_n = 17
        best_colors = res_17
    
    # If n=17 found, try n=18 if time permits
    if best_n == 17:
        t_rem = deadline - time.time()
        if t_rem > 15:
            res_18 = solve(18, deadline - 5)
            if res_18 is not None:
                best_n = 18
                best_colors = res_18
    
    if best_n == 0:
        # Fallback n=9
        n = 9
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                c = min(j - i, n - j + i) - 1
                if c >= 4: c = 3
                colors.append(c)
        return {'n': n, 'colors': colors}
    
    return {'n': best_n, 'colors': best_colors}

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}, colors length={len(res['colors'])}")
