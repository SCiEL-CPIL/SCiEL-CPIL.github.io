import random
import time
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Checks if the hypergraph (n, edges) has a proper 2-coloring.
    Returns True if 2-colorable, False otherwise.
    Uses backtracking with simple unit propagation.
    """
    # Build adjacency or just use the list of edges
    # We assign colors 0 or 1.
    # Unassigned is None.
    assignment = [None] * n
    
    # Pre-process edges for speed: convert to tuples of indices
    edge_tuples = [tuple(e) for e in edges]
    
    # Simple backtracking solver
    def backtrack(idx):
        if idx == n:
            return True
        
        # Try 0
        assignment[idx] = 0
        if is_consistent(idx):
            if backtrack(idx + 1):
                return True
        # Try 1
        assignment[idx] = 1
        if is_consistent(idx):
            if backtrack(idx + 1):
                return True
        
        assignment[idx] = None
        return False

    def is_consistent(idx):
        # Check all edges involving vertices 0..idx
        # Optimization: Only check edges where idx is the max vertex?
        # No, changing idx might violate constraints on edges with max vertex < idx?
        # Actually, if we fix variables sequentially, an edge is only fully determined
        # when all its vertices are assigned.
        # We can just check all edges where all vertices are assigned.
        for e in edge_tuples:
            # If all vertices in e are assigned
            if all(assignment[v] is not None for v in e):
                colors = [assignment[v] for v in e]
                if len(set(colors)) == 1:
                    return False
        return True

    return backtrack(0)

def construct():
    # Start with the complete 5-uniform hypergraph on 9 vertices
    n = 9
    all_edges = [list(e) for e in combinations(range(n), 5)]
    
    # Shuffle edges to randomize the pruning order
    edges_to_try = all_edges.copy()
    random.seed(42) # Deterministic for reproducibility
    random.shuffle(edges_to_try)
    
    current_edges = [tuple(e) for e in all_edges]
    
    # Prune edges: if removing an edge keeps the graph non-2-colorable, remove it.
    # We iterate through a shuffled list of edges.
    # To be efficient, we can just iterate through the list once.
    # Since removing an edge makes the constraint set weaker (easier to satisfy),
    # if the graph was non-2-colorable, it might become 2-colorable.
    # We only keep an edge if the graph WITHOUT it is 2-colorable (i.e., the edge is necessary).
    
    # Note: The order of removal matters. Greedy removal yields a Minimal Unsatisfiable Set (MUS).
    # The size of a MUS is not necessarily the minimum size, but it is often much smaller than the full set.
    
    for edge in edges_to_try:
        if edge not in current_edges:
            continue
            
        # Temporarily remove
        current_edges.remove(edge)
        
        # Check if still non-2-colorable
        # is_2_colorable returns True if 2-colorable.
        # We want it to be NOT 2-colorable.
        if not is_2_colorable(n, current_edges):
            # It is still non-2-colorable, so we can keep it removed.
            pass
        else:
            # It became 2-colorable, so we must add the edge back.
            current_edges.append(edge)
            
    return {'n': n, 'edges': [list(e) for e in current_edges]}
