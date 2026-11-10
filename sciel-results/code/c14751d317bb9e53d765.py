import random
import time
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Check if the hypergraph is 2-colorable using backtracking with unit propagation.
    Returns True if 2-colorable, False otherwise.
    """
    if not edges:
        return True
    
    # Map vertex to list of edges it belongs to
    vertex_edges = {v: [] for v in range(n)}
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
    
    # State: assignment of vertices to colors (0, 1) or -1 (unassigned)
    # We use a dictionary for assignments to allow partial assignments
    # But for speed, we can use a list of length n
    assignment = [-1] * n
    edge_status = [0] * len(edges) # 0: uncolored, 1: has color 0, 2: has color 1, 3: both (conflict)
    
    # Actually, a simpler backtracking:
    # Try to assign colors to vertices.
    # If an edge has all vertices assigned the same color, it's a conflict.
    # If an edge has vertices of both colors, it's satisfied.
    
    # Let's use a recursive backtracking.
    # To speed up, we can use a heuristic: pick the vertex with the most constraints or involved in the most unsatisfied edges.
    
    # For simplicity and given n <= 26, a basic backtracking might be slow if not careful.
    # Let's use a standard backtracking with pruning.
    
    def backtrack(idx):
        if idx == n:
            # Check all edges
            for e in edges:
                colors = set(assignment[v] for v in e)
                if len(colors) < 2:
                    return False
            return True
        
        v = idx
        # Try color 0
        assignment[v] = 0
        # Check if any edge is now fully assigned and monochromatic
        conflict = False
        for e_idx in vertex_edges[v]:
            e = edges[e_idx]
            # Check if all vertices in e are assigned
            all_assigned = True
            has_0 = False
            has_1 = False
            for u in e:
                if assignment[u] == -1:
                    all_assigned = False
                    break
                if assignment[u] == 0:
                    has_0 = True
                else:
                    has_1 = True
            if all_assigned and not (has_0 and has_1):
                conflict = True
                break
        
        if not conflict and backtrack(idx + 1):
            return True
        
        # Try color 1
        assignment[v] = 1
        conflict = False
        for e_idx in vertex_edges[v]:
            e = edges[e_idx]
            all_assigned = True
            has_0 = False
            has_1 = False
            for u in e:
                if assignment[u] == -1:
                    all_assigned = False
                    break
                if assignment[u] == 0:
                    has_0 = True
                else:
                    has_1 = True
            if all_assigned and not (has_0 and has_1):
                conflict = True
                break
        
        if not conflict and backtrack(idx + 1):
            return True
        
        assignment[v] = -1
        return False

    return backtrack(0)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Start with a known non-2-colorable hypergraph.
    # The complete 5-uniform hypergraph on 9 vertices is non-2-colorable.
    # n=9, edges = all combinations of 5 from 9.
    n = 9
    all_edges = list(combinations(range(n), 5))
    edges = [list(e) for e in all_edges]
    
    # Try to remove edges to minimize the count while keeping it non-2-colorable.
    # We will randomly shuffle the edges and try to remove them one by one.
    # If removing an edge makes the hypergraph 2-colorable, we put it back.
    
    # To make it more effective, we can try multiple random orders.
    best_edges = edges[:]
    best_count = len(edges)
    
    # We can also try to reduce n if possible, but n=9 is small.
    # The known minimum is around 51 edges, likely on more vertices.
    # Let's try to find a hypergraph with fewer edges using a randomized search on a larger n?
    # No, larger n usually requires more edges to be non-2-colorable if sparse.
    # Dense hypergraphs on small n are a good starting point.
    
    # Let's try to improve the 126-edge solution by deletion.
    random.seed(42)
    
    for _ in range(50): # 50 iterations of random deletion
        if time.time() > deadline:
            break
            
        current_edges = best_edges[:]
        random.shuffle(current_edges)
        
        # Try to remove edges in this order
        i = 0
        while i < len(current_edges):
            if time.time() > deadline:
                break
                
            # Try removing current_edges[i]
            test_edges = current_edges[:i] + current_edges[i+1:]
            
            # Check if test_edges is 2-colorable
            if is_2_colorable(n, test_edges):
                # It is 2-colorable, so we must keep the edge
                i += 1
            else:
                # It is still non-2-colorable, so we can remove the edge
                current_edges = test_edges
                # Do not increment i, because the list shifted
        
        if len(current_edges) < best_count:
            best_edges = current_edges
            best_count = len(current_edges)
            print(f"Improved to {best_count} edges")
            
    return {'n': n, 'edges': best_edges}
