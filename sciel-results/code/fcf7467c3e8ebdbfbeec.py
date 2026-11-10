"""
5-uniform hypergraph without Property B (m(5)) - combined construction.

Strategy:
  1. Use a fast, array-based 2-colourability checker with unit propagation.
  2. Greedy edge removal from a complete hypergraph on small n.
  3. Local search: random restarts, add edges to break 2-colourability, then minimize.
  4. Hard-coded fallback if search fails.
"""

import time
import random
import itertools
from itertools import combinations

def _is_2_colorable(n, edges):
    """
    Check if a 5-uniform hypergraph on n vertices is 2-colorable (has Property B).
    Uses backtracking with unit propagation.
    Returns True if 2-colorable, False otherwise.
    """
    if not edges:
        return True
    
    # Precompute edge list as tuples of ints
    edge_list = [tuple(e) for e in edges]
    m = len(edge_list)
    
    # For speed, we'll use a list of lists for counts per edge
    # But for small n and m, simple iteration is fine.
    
    # colors: -1 unassigned, 0 or 1 assigned
    colors = [-1] * n
    
    # Precompute: for each vertex, list of edge indices containing it
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edge_list):
        for v in e:
            vertex_edges[v].append(i)
    
    def propagate():
        """
        Apply unit propagation.
        Returns False if contradiction (monochromatic edge), True otherwise.
        Modifies colors in place.
        """
        changed = True
        while changed:
            changed = False
            for i in range(m):
                e = edge_list[i]
                c0 = 0
                c1 = 0
                unassigned = -1
                uncount = 0
                for v in e:
                    c = colors[v]
                    if c == 0:
                        c0 += 1
                    elif c == 1:
                        c1 += 1
                    else:
                        unassigned = v
                        uncount += 1
                
                if c0 == 5 or c1 == 5:
                    return False
                if uncount == 1:
                    if c0 == 4:
                        if colors[unassigned] == -1:
                            colors[unassigned] = 1
                            changed = True
                    elif c1 == 4:
                        if colors[unassigned] == -1:
                            colors[unassigned] = 0
                            changed = True
        return True
    
    def find_unassigned():
        """Find an unassigned vertex with the most constraints."""
        best_v = -1
        best_deg = -1
        for v in range(n):
            if colors[v] == -1:
                deg = len(vertex_edges[v])
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
        return best_v
    
    def backtrack():
        """Try to complete the assignment. Returns True if 2-coloring exists."""
        if not propagate():
            return False
        
        v = find_unassigned()
        if v == -1:
            return True
        
        # Try color 0
        colors[v] = 0
        if backtrack():
            return True
        
        # Try color 1
        colors[v] = 1
        if backtrack():
            return True
        
        colors[v] = -1
        return False
    
    return backtrack()


def _greedy_minimize(n, initial_edges, deadline):
    """
    Start with initial_edges (non-2-colorable) and greedily remove edges
    while maintaining non-2-colorability.
    Returns the minimized edge list.
    """
    current_edges = list(initial_edges)
    random.shuffle(current_edges)
    
    i = 0
    while i < len(current_edges) and time.time() < deadline:
        e = current_edges[i]
        # Try removing edge i
        test_edges = current_edges[:i] + current_edges[i+1:]
        if not _is_2_colorable(n, test_edges):
            # Still non-2-colorable, remove it
            current_edges = test_edges
            # Don't increment i
        else:
            i += 1
    
    return current_edges


def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115 second budget, leave 5s margin for safety
    
    best_result = None
    best_edge_count = float('inf')
    
    # Try different n values
    for n in range(6, 15):
        if time.time() > deadline:
            break
        
        all_edges = list(combinations(range(n), 5))
        num_total = len(all_edges)
        
        if num_total > 200:
            continue
        
        # Check if complete hypergraph is non-2-colorable
        if _is_2_colorable(n, all_edges):
            continue
        
        # It's non-2-colorable. Now try to minimize.
        
        # Strategy 1: Greedy removal from complete hypergraph
        greedy_result = _greedy_minimize(n, all_edges, deadline)
        if len(greedy_result) < best_edge_count:
            best_edge_count = len(greedy_result)
            best_result = {'n': n, 'edges': [list(e) for e in greedy_result]}
        
        # Strategy 2: Local search with random restarts
        for restart in range(5):
            if time.time() > deadline:
                break
            
            # Start with a random subset
            # Try different sizes
            for target_size in [30, 40, 50, 60]:
                if time.time() > deadline:
                    break
                
                if target_size > num_total:
                    continue
                
                # Random subset
                subset = random.sample(all_edges, target_size)
                subset_set = set(subset)
                
                # Ensure non-2-colorable: add edges if needed
                while _is_2_colorable(n, list(subset_set)):
                    if time.time() > deadline:
                        break
                    remaining = [e for e in all_edges if e not in subset_set]
                    if not remaining:
                        break
                    e = random.choice(remaining)
                    subset_set.add(e)
                
                if time.time() > deadline:
                    break
                
                # Now minimize
                minimized = _greedy_minimize(n, list(subset_set), deadline)
                if len(minimized) < best_edge_count:
                    best_edge_count = len(minimized)
                    best_result = {'n': n, 'edges': [list(e) for e in minimized]}
        
        # If we found a good result, we can break early
        if best_edge_count <= 51:
            break
    
    # Fallback: complete hypergraph on 9 vertices
    if best_result is None:
        all_edges = list(combinations(range(9), 5))
        best_result = {'n': 9, 'edges': [list(e) for e in all_edges]}
    
    return best_result
