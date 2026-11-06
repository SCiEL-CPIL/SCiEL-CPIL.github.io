import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    def is_2_colorable(n, edges):
        # edges: list of tuples of 5 vertices
        # Returns True if 2-colorable, False otherwise
        if not edges:
            return True
        
        # Pre-process: remove duplicate edges? 
        # We assume edges are distinct.
        
        # Adjacency structure for unit propagation
        # For each vertex, list of edges it belongs to
        # For each edge, list of vertices
        
        # State:
        # color[v] = -1 (unassigned), 0, 1
        # For unit propagation, we need to know if an edge is "satisfied" (mixed colors) or "forced"
        
        # Efficient representation:
        # edge_vertices: list of lists of 5 ints
        # vertex_edges: list of lists of edge indices
        
        # We will use a stack for backtracking.
        # Each state: (assignment, edge_status)
        # edge_status: 
        #   -1: mixed (satisfied)
        #   0: all unassigned or mixed unassigned/assigned but not forced? 
        #   Actually, simpler:
        #   For each edge, count colors 0, 1, and unassigned.
        #   If count[0] == 5 or count[1] == 5 -> Conflict (False)
        #   If count[0] == 4 -> The remaining one must be 1
        #   If count[1] == 4 -> The remaining one must be 0
        #   If count[0] + count[1] == 5 -> Satisfied (Mixed)
        
        # Optimization: 
        # We maintain counts for each edge.
        
        # Initialize
        colors = [-1] * n
        # edge_counts: list of [count0, count1, count_unassigned, list_of_unassigned_indices_in_edge]
        # Storing list of unassigned indices is expensive. 
        # Alternative: 
        # For each edge, store count0, count1.
        # If count0 + count1 < 5, it's active.
        
        # Let's use a recursive backtracking with propagation.
        
        # Data structures:
        # colors: list of int
        # edge_c0: list of int (count of 0s)
        # edge_c1: list of int (count of 1s)
        # edge_vertices: list of list of int
        
        # Propagation queue
        # When a vertex is assigned, update all its edges.
        # If an edge becomes forced, assign the remaining vertex.
        
        # To allow backtracking, we need to record changes.
        # Change: (vertex, old_color)
        # For edge counts, we can just recompute or store deltas.
        # Simpler: Store the full state of colors and edge counts? 
        # n <= 26, edges <= 200. State size is small.
        # But copying lists is O(n + E).
        
        # Let's try a standard backtracking with undo stack.
        
        # Global variables for the check
        colors = [-1] * n
        e_c0 = [0] * len(edges)
        e_c1 = [0] * len(edges)
        
        # Precompute vertex to edges
        v_to_e = [[] for _ in range(n)]
        for i, e in enumerate(edges):
            for v in e:
                v_to_e[v].append(i)
        
        # Initial counts are 0,0. All unassigned.
        
        # We need a function to check if a vertex assignment leads to conflict or propagation.
        # This is complex to implement correctly and fast in Python within the time limit if done naively.
        
        # Alternative: Use a simpler heuristic search for the whole problem?
        # No, we need a reliable 2-colorability checker.
        
        # Let's use a library if available? No, standard library only.
        # Let's write a very optimized backtracker.
        
        # State:
        # colors: array of int
        # e_c0, e_c1: arrays of int
        
        # We will use a stack of (vertex, old_color, list_of_edge_changes)
        # Actually, just storing the vertex and old color is enough if we re-derive?
        # No, re-deriving is O(E).
        # Let's store the changes to e_c0 and e_c1.
        
        # Better approach:
        # Use a dictionary or list for active edges?
        
        # Let's try this structure:
        # A recursive function `search(depth)`
        # 1. If all vertices assigned, return True (if no conflict).
        # 2. Check for conflicts in current state.
        # 3. Propagate:
        #    Find edges with 4 of one color. Assign the 5th.
        #    Repeat until no more propagation.
        #    If conflict during propagation, return False.
        # 4. If no unassigned vertices, return True.
        # 5. Pick an unassigned vertex.
        # 6. Try color 0, recurse.
        # 7. Try color 1, recurse.
        
        # To make propagation fast:
        # We maintain a queue of "touched" edges.
        # When a vertex changes, add its edges to the queue.
        # Process queue:
        #   For each edge, check counts.
        #   If forced, assign vertex, update counts, add new edges to queue.
        
        # Implementation detail:
        # We need to handle backtracking of assignments made by propagation.
        # So we record every assignment (vertex, old_color) in a stack.
        
        def check():
            # Initialize
            colors = [-1] * n
            e_c0 = [0] * len(edges)
            e_c1 = [0] * len(edges)
            
            # We will use a list `assign_stack` to store (vertex, old_color)
            # And we need to know which edges were updated? 
            # Actually, if we just re-run propagation from scratch on the current colors?
            # No, that's too slow.
            
            # Let's use the queue method with a stack for undo.
            
            # This is getting complex. Let's use a simpler but slightly slower method:
            # Randomized Restart with Unit Propagation only?
            # No, must be exact.
            
            # Let's use a standard backtracking that copies state.
            # n=20, E=50. Copying lists of size 20 and 50 is fast.
            # Depth of recursion might be up to 20.
            # 2^20 is 1M. With pruning, it should be fast.
            
            def bt(colors, e_c0, e_c1):
                # Check for conflicts
                for i in range(len(edges)):
                    if e_c0[i] == 5 or e_c1[i] == 5:
                        return False
                
                # Propagation
                changed = True
                while changed:
                    changed = False
                    for i in range(len(edges)):
                        c0 = e_c0[i]
                        c1 = e_c1[i]
                        unassigned = 5 - c0 - c1
                        if unassigned == 0:
                            continue
                        if c0 == 4:
                            # Must assign 1 to the unassigned vertex
                            for v in edges[i]:
                                if colors[v] == -1:
                                    # Assign
                                    colors[v] = 1
                                    e_c1[i] += 1
                                    # Update other edges containing v
                                    for j in v_to_e[v]:
                                        if j == i: continue
                                        if colors[v] == 1:
                                            e_c1[j] += 1
                                        # Note: if colors[v] was -1, it wasn't counted in e_c0 or e_c1 of j?
                                        # Yes, e_c0/e_c1 only count assigned colors.
                                    changed = True
                                    # Check if this caused a conflict in other edges?
                                    # The loop will catch it in the next iteration or conflict check.
                                    # But we should break to restart loop?
                                    # Or just continue.
                                    break
                        elif c1 == 4:
                            for v in edges[i]:
                                if colors[v] == -1:
                                    colors[v] = 0
                                    e_c0[i] += 1
                                    for j in v_to_e[v]:
                                        if j == i: continue
                                        e_c0[j] += 1
                                    changed = True
                                    break
                    # After propagation, check conflicts again
                    for i in range(len(edges)):
                        if e_c0[i] == 5 or e_c1[i] == 5:
                            return False
                
                # Find unassigned vertex
                unassigned = [v for v in range(n) if colors[v] == -1]
                if not unassigned:
                    return True
                
                # Heuristic: pick vertex in most constrained edge?
                # For now, pick first unassigned.
                v = unassigned[0]
                
                # Try 0
                new_colors = colors[:]
                new_e_c0 = e_c0[:]
                new_e_c1 = e_c1[:]
                new_colors[v] = 0
                for j in v_to_e[v]:
                    new_e_c0[j] += 1
                if bt(new_colors, new_e_c0, new_e_c1):
                    return True
                
                # Try 1
                new_colors = colors[:]
                new_e_c0 = e_c0[:]
                new_e_c1 = e_c1[:]
                new_colors[v] = 1
                for j in v_to_e[v]:
                    new_e_c1[j] += 1
                if bt(new_colors, new_e_c0, new_e_c1):
                    return True
                
                return False
            
            return bt(colors, e_c0, e_c1)

    best_n = 0
    best_edges = []
    
    # Search for a small non-2-colorable hypergraph
    # Try different n
    for n in range(12, 21):
        if time.time() > deadline:
            break
        for trial in range(20):
            if time.time() > deadline:
                break
            # Generate random edges
            num_edges = 50 + random.randint(0, 20)
            possible_edges = list(combinations(range(n), 5))
            random.shuffle(possible_edges)
            edges = [list(e) for e in possible_edges[:num_edges]]
            
            # Check if non-2-colorable
            if not is_2_colorable(n, edges):
                # Found one! Now reduce
                current_edges = edges[:]
                # Shuffle to remove random edges
                random.shuffle(current_edges)
                # Try to remove edges
                # We will iterate through a list of edges to try removing
                # If removal keeps it non-2-colorable, remove it.
                
                # Optimization: 
                # We can try to remove edges in random order repeatedly.
                # Or simply iterate through the list.
                
                # Let's do a greedy removal
                i = 0
                while i < len(current_edges):
                    if time.time() > deadline:
                        break
                    e_to_remove = current_edges[i]
                    # Try removing e_to_remove
                    test_edges = current_edges[:i] + current_edges[i+1:]
                    if not is_2_colorable(n, test_edges):
                        # Success, remove it
                        current_edges = test_edges
                        # Do not increment i, check next edge at same index
                    else:
                        i += 1
                
                # If this is better (fewer edges), update best
                if len(current_edges) < len(best_edges) or (len(current_edges) == len(best_edges) and n < best_n):
                    best_n = n
                    best_edges = current_edges[:]
                    print(f"Found non-2-colorable with {len(best_edges)} edges on {best_n} vertices", file=sys.stderr)
                    # If we found 51 or less, we can stop early?
                    # The task says best known is 51.
                    if len(best_edges) <= 51:
                        # Try to see if we can go lower?
                        # But 51 is a very good score.
                        # Let's continue for a bit to see if we can hit 50, 49...
                        pass

    if not best_edges:
        # Fallback: Return the complete 9-vertex graph (126 edges)
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
    else:
        return {'n': best_n, 'edges': best_edges}

# Need to import combinations
from itertools import combinations
