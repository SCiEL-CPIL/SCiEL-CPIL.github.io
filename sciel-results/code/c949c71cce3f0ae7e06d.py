"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_2_colorable(n, edges, deadline=None):
    """
    Check if the hypergraph is 2-colorable using backtracking with unit propagation.
    Returns True if 2-colorable, False if not.
    """
    if deadline is not None and time.time() > deadline:
        return None  # timeout sentinel
    
    m = len(edges)
    # Precompute for each vertex, which edges contain it
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
    
    # State: color[v] = -1 (unassigned), 0, or 1
    # For each edge, track how many of each color are assigned
    edge_counts = [[0, 0] for _ in range(m)]
    
    def propagate(color, edge_counts, changed):
        """
        Unit propagation: if an edge has 4 vertices of same color, the 5th must be opposite.
        Returns False if contradiction, True otherwise.
        """
        while changed:
            changed = False
            for v in range(n):
                if color[v] != -1:
                    continue
                # Check if any edge forces v
                forced = None
                for ei in vertex_edges[v]:
                    c0, c1 = edge_counts[ei]
                    # If 4 are color 0, v must be 1; if 4 are color 1, v must be 0
                    if c0 == 4:
                        forced = 1
                        break
                    elif c1 == 4:
                        forced = 0
                        break
                if forced is not None:
                    color[v] = forced
                    edge_counts[forced][0] += 0  # placeholder
                    for ei in vertex_edges[v]:
                        edge_counts[ei][forced] += 1
                        # Check for contradiction: if now 5 of one color
                        if edge_counts[ei][0] == 5 or edge_counts[ei][1] == 5:
                            return False
                    changed = True
        return True
    
    def backtrack(color, edge_counts, idx):
        """
        Backtracking search. idx is the index of the next vertex to assign.
        """
        if deadline is not None and time.time() > deadline:
            return None
        
        # Find first unassigned vertex
        v = -1
        for i in range(n):
            if color[i] == -1:
                v = i
                break
        if v == -1:
            return True  # all assigned, no contradiction found
        
        # Try both colors
        for c in [0, 1]:
            # Check if assigning c to v causes immediate contradiction
            ok = True
            for ei in vertex_edges[v]:
                if edge_counts[ei][c] == 4:
                    ok = False
                    break
            if not ok:
                continue
            
            # Save state
            old_color = color[v]
            old_counts = [ec[:] for ec in edge_counts]
            
            color[v] = c
            for ei in vertex_edges[v]:
                edge_counts[ei][c] += 1
                if edge_counts[ei][0] == 5 or edge_counts[ei][1] == 5:
                    # Restore and try other color
                    color[v] = old_color
                    for i2 in range(len(edge_counts)):
                        edge_counts[i2] = old_counts[i2]
                    break
            else:
                # No immediate contradiction, propagate
                if not propagate(color, edge_counts, True):
                    color[v] = old_color
                    for i2 in range(len(edge_counts)):
                        edge_counts[i2] = old_counts[i2]
                    continue
                
                result = backtrack(color, edge_counts, idx + 1)
                if result is not None:
                    return result
                
                # Restore
                color[v] = old_color
                for i2 in range(len(edge_counts)):
                    edge_counts[i2] = old_counts[i2]
        
        return False
    
    color = [-1] * n
    result = backtrack(color, edge_counts, 0)
    return result

def construct():
    """
    Construct a 5-uniform hypergraph with no proper 2-coloring, minimizing edges.
    """
    start_time = time.time()
    deadline = start_time + 130  # 130s deadline to be safe
    
    # Start with a known non-2-colorable hypergraph and try to reduce edges
    # The complete 5-uniform hypergraph on 9 vertices has C(9,5)=126 edges
    # We want to find a minimal subset
    
    # Strategy: Start with all C(9,5) edges, then greedily remove edges
    # that are not needed for non-2-colorability
    
    all_edges = list(combinations(range(9), 5))
    current_edges = list(all_edges)
    
    # First, verify the full hypergraph is not 2-colorable
    if not _is_2_colorable(9, current_edges, deadline):
        pass  # Not 2-colorable, good
    elif _is_2_colorable(9, current_edges, deadline) is None:
        # Timeout, just return what we have
        return {'n': 9, 'edges': [list(e) for e in current_edges]}
    
    # Now try to remove edges one by one
    # For each edge, check if removing it still leaves the hypergraph non-2-colorable
    # If so, remove it
    
    improved = True
    while improved:
        improved = False
        for i in range(len(current_edges)):
            if time.time() > deadline - 5:
                break
            
            # Try removing edge i
            test_edges = current_edges[:i] + current_edges[i+1:]
            if not _is_2_colorable(9, test_edges, deadline):
                current_edges = test_edges
                improved = True
                break
            elif _is_2_colorable(9, test_edges, deadline) is None:
                # Timeout during check, be conservative, don't remove
                break
        if time.time() > deadline - 5:
            break
    
    # If we have time, try a more aggressive search: random removal
    if time.time() < deadline - 10:
        # Try random subsets of edges
        best_edges = current_edges[:]
        best_count = len(best_edges)
        
        # Try to find a smaller set by random restarts
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            
            # Start with a random subset
            k = max(10, best_count - 5)
            random_subset = random.sample(current_edges, min(k, len(current_edges)))
            
            # Check if this subset is non-2-colorable
            if not _is_2_colorable(9, random_subset, deadline):
                if len(random_subset) < best_count:
                    best_edges = random_subset
                    best_count = len(best_edges)
                    current_edges = best_edges[:]
                else:
                    current_edges = random_subset
                    # Try to reduce further
                    improved2 = True
                    while improved2 and time.time() < deadline - 10:
                        improved2 = False
                        for j in range(len(current_edges)):
                            test_edges = current_edges[:j] + current_edges[j+1:]
                            if not _is_2_colorable(9, test_edges, deadline):
                                current_edges = test_edges
                                improved2 = True
                                break
                            elif _is_2_colorable(9, test_edges, deadline) is None:
                                break
                        if time.time() > deadline - 10:
                            break
                    if len(current_edges) < best_count:
                        best_edges = current_edges[:]
                        best_count = len(best_edges)
    
    return {'n': 9, 'edges': [list(e) for e in best_edges]}
# EVOLVE-BLOCK-END
