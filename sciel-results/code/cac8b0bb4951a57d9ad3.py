import time
import random
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Check if the hypergraph is 2-colorable using backtracking with unit propagation.
    Returns True if 2-colorable, False otherwise.
    """
    if not edges:
        return True
    
    m = len(edges)
    
    # Precompute incidence: for each vertex, list of edges it belongs to
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
            
    # State: color[v] = -1 (unassigned), 0, or 1
    color = [-1] * n
    
    # For each edge, track count of 0s and 1s
    count0 = [0] * m
    count1 = [0] * m
    
    # Stack for backtracking: list of (vertex, old_color, affected_edges, old_counts)
    # To save memory and time, we can just store the vertex and color, and recompute/reverse updates.
    # But storing the delta is safer.
    # Let's use a simple stack of (v, c) and reverse updates manually.
    
    # Actually, to make unit propagation work correctly with backtracking,
    # we need to be able to undo assignments.
    # Assignments can be:
    # 1. Manual (from backtracking choice)
    # 2. Forced (from unit propagation)
    
    # We will maintain a stack of assignments: (v, c, was_forced)
    # When backtracking, we pop assignments until we undo the manual choice.
    
    # However, a simpler approach for small n (<=26) and m (<=200) is to use recursion
    # and pass state by reference, undoing changes manually.
    
    # Let's implement a robust iterative solver with explicit undo.
    
    # To simplify, we will use a recursive solver but with a depth limit check.
    # Given n<=26, recursion depth is fine.
    
    def propagate():
        """
        Perform unit propagation.
        Returns False if contradiction, True otherwise.
        Appends forced assignments to a list 'forced_assignments' for undoing.
        """
        forced_assignments = []
        changed = True
        while changed:
            changed = False
            for i in range(m):
                if count0[i] == 5 or count1[i] == 5:
                    return False, forced_assignments
                
                if count0[i] == 4:
                    # Find unassigned vertex
                    for v in edges[i]:
                        if color[v] == -1:
                            # Assign 1
                            color[v] = 1
                            count1[i] += 1
                            # Update other edges containing v
                            for ei in vertex_edges[v]:
                                if ei != i:
                                    count1[ei] += 1
                            forced_assignments.append((v, 1))
                            changed = True
                            break # Move to next edge after assignment? 
                                 # Actually, we should continue checking other edges 
                                 # or restart. Restarting is safer.
                            # Let's break the for loop and restart the while loop
                            # by setting changed=True and breaking out of for loop
                            # to re-evaluate all edges.
                            # But we need to break out of the for loop cleanly.
                            # Let's just break and let the while loop handle it.
                            break
                elif count1[i] == 4:
                    # Find unassigned vertex
                    for v in edges[i]:
                        if color[v] == -1:
                            # Assign 0
                            color[v] = 0
                            count0[i] += 1
                            # Update other edges containing v
                            for ei in vertex_edges[v]:
                                if ei != i:
                                    count0[ei] += 1
                            forced_assignments.append((v, 0))
                            changed = True
                            break
            # If we broke out of for loop due to assignment, we continue while loop
            # If we finished for loop without assignment, changed is False, loop ends.
            # But if we broke due to assignment, we need to restart the for loop.
            # The current structure breaks the for loop, then checks `changed`.
            # If `changed` is True, it goes back to `while changed`.
            # This is correct.
            
        return True, forced_assignments

    def undo_assignments(assignments):
        """Undo a list of (v, c) assignments."""
        for v, c in reversed(assignments):
            color[v] = -1
            if c == 0:
                for ei in vertex_edges[v]:
                    count0[ei] -= 1
            else:
                for ei in vertex_edges[v]:
                    count1[ei] -= 1

    def solve():
        # Find first unassigned vertex
        v = -1
        for i in range(n):
            if color[i] == -1:
                v = i
                break
        
        if v == -1:
            # All assigned. Check if all edges are valid (not monochromatic)
            # If propagation didn't catch it, this is the final check.
            # But propagation should catch monochromatic edges (count==5).
            for i in range(m):
                if count0[i] == 5 or count1[i] == 5:
                    return False
            return True
        
        # Try color 0
        color[v] = 0
        for ei in vertex_edges[v]:
            count0[ei] += 1
        
        ok, forced0 = propagate()
        
        if ok:
            if solve():
                return True
            else:
                undo_assignments(forced0)
        
        # Undo manual assignment
        color[v] = -1
        for ei in vertex_edges[v]:
            count0[ei] -= 1
        
        # Try color 1
        color[v] = 1
        for ei in vertex_edges[v]:
            count1[ei] += 1
        
        ok, forced1 = propagate()
        
        if ok:
            if solve():
                return True
            else:
                undo_assignments(forced1)
        
        # Undo manual assignment
        color[v] = -1
        for ei in vertex_edges[v]:
            count1[ei] -= 1
        
        return False

    return solve()

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Known optimal construction: 13 vertices, 51 edges.
    # This is the Fano plane lift.
    # Vertices 0..6 form the Fano plane (7 vertices, 7 edges of size 3).
    # Vertices 7..12 are "duplicates" or related.
    # Actually, the standard minimal non-2-colorable 5-uniform hypergraph is on 13 vertices.
    # It is constructed by taking the Fano plane (7 vertices, 7 edges) and for each edge {a,b,c},
    # creating a 5-uniform edge {a, b, c, a', b'} where a', b' are new vertices?
    # No, that would add too many vertices.
    
    # Let's use a known specific 51-edge construction.
    # One such construction is the "Petersen hypergraph" or similar?
    # Actually, the lower bound 28 and upper bound 51 are for m(5).
    # The construction with 51 edges is often attributed to specific research.
    
    # Let's try to generate a candidate and verify.
    # If I don't have the exact 51 edges hardcoded, I can try to find one.
    # But generating a non-2-colorable hypergraph is hard.
    
    # Alternative: Use the complete hypergraph on 9 vertices (126 edges) and try to reduce it.
    # The seed program did this but was slow.
    # With a faster solver, I can do more deletions.
    
    n = 9
    all_edges = [list(e) for e in combinations(range(n), 5)]
    current_edges = all_edges[:]
    
    # Verify it's non-2-colorable
    if is_2_colorable(n, current_edges):
        return {'n': n, 'edges': current_edges}
    
    # Deletion process
    # Try to remove each edge if the graph remains non-2-colorable.
    # Do this in multiple passes with random ordering.
    
    for _ in range(5):  # 5 passes
        if time.time() > deadline - 10:
            break
            
        random.shuffle(current_edges)
        new_edges = []
        
        for edge in current_edges:
            if time.time() > deadline - 5:
                new_edges.append(edge)
                continue
                
            # Try to remove this edge
            candidate = [e for e in current_edges if e != edge]
            
            # Check if candidate is still non-2-colorable
            if not is_2_colorable(n, candidate):
                # Still non-2-colorable, so we can remove the edge
                continue
            else:
                # Became 2-colorable, so we must keep the edge
                new_edges.append(edge)
        
        current_edges = new_edges
        
        # If no edges were removed in this pass, we might be at a local minimum
        # But let's do a few more passes with different orderings to be sure.
    
    # Final check
    if is_2_colorable(n, current_edges):
        return {'n': n, 'edges': all_edges}
    
    return {'n': n, 'edges': current_edges}
