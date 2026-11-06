# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def is_2_colorable(edges, n):
    """
    Checks if the hypergraph is 2-colorable using backtracking with unit propagation.
    Returns True if 2-colorable, False otherwise.
    """
    colors = [-1] * n
    
    def propagate():
        changed = True
        while changed:
            changed = False
            for e in edges:
                c0 = sum(1 for v in e if colors[v] == 0)
                c1 = sum(1 for v in e if colors[v] == 1)
                cu = sum(1 for v in e if colors[v] == -1)
                
                if c0 == 4 and cu > 0:
                    for v in e:
                        if colors[v] == -1:
                            colors[v] = 1
                            changed = True
                elif c1 == 4 and cu > 0:
                    for v in e:
                        if colors[v] == -1:
                            colors[v] = 0
                            changed = True
                            
                if c0 == 5 or c1 == 5:
                    return False
        return True

    def backtrack():
        idx = -1
        for i in range(n):
            if colors[i] == -1:
                idx = i
                break
        
        if idx == -1:
            return True
            
        colors[idx] = 0
        if propagate() and backtrack():
            return True
        colors[idx] = -1
        
        colors[idx] = 1
        if propagate() and backtrack():
            return True
        colors[idx] = -1
        
        return False

    return backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit, leaving 40s margin
    
    # Start with complete 5-uniform hypergraph on 9 vertices
    n = 9
    all_edges = [list(e) for e in combinations(range(n), 5)]
    current_edges = all_edges[:]
    
    # Try to remove edges while maintaining non-2-colorability
    # We iterate until we can't remove any more or time runs out
    max_iterations = 10000
    for i in range(max_iterations):
        if time.time() > deadline:
            break
        
        if len(current_edges) == 0:
            break
            
        # Pick a random edge to try removing
        idx = random.randint(0, len(current_edges) - 1)
        edge_to_remove = current_edges[idx]
        
        # Temporarily remove
        current_edges.pop(idx)
        
        # Check if still non-2-colorable
        if not is_2_colorable(current_edges, n):
            # Keep it removed
            pass
        else:
            # Restore
            current_edges.insert(idx, edge_to_remove)
            
    return {'n': n, 'edges': current_edges}
# EVOLVE-BLOCK-END
