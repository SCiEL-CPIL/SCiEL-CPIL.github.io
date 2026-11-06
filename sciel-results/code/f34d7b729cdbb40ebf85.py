import time
import random
from itertools import combinations

def _is_not_2_colorable(n, edges, timeout=5.0):
    """Fast bitmask-based backtracking with unit propagation.
    Returns True if the hypergraph is NOT 2-colorable (i.e., no proper 2-coloring exists)."""
    start = time.time()
    
    m = len(edges)
    if m == 0:
        return False
    
    # Precompute edge vertex lists
    edge_verts = [list(e) for e in edges]
    
    # Build vertex-to-edges index
    vert_to_edges = [[] for _ in range(n)]
    for i in range(m):
        for v in edge_verts[i]:
            vert_to_edges[v].append(i)
    
    def check_not_2colorable():
        """Try to find a 2-coloring. Return False if none exists (not 2-colorable)."""
        nonlocal start
        color = [-1] * n  # -1 unassigned, 0 or 1
        edge_c0 = [0] * m
        edge_c1 = [0] * m
        edge_un = [5] * m
        
        def propagate():
            """Unit propagation. Return False on conflict, True on success, None on timeout."""
            nonlocal start
            if time.time() - start > timeout:
                return None
            changed = True
            while changed:
                changed = False
                for i in range(m):
                    if edge_un[i] == 0:
                        if edge_c0[i] == 5 or edge_c1[i] == 5:
                            return False
                    elif edge_un[i] == 1:
                        if edge_c0[i] == 4:
                            # Must be color 1
                            for v in edge_verts[i]:
                                if color[v] == -1:
                                    color[v] = 1
                                    edge_c1[i] += 1
                                    edge_un[i] -= 1
                                    changed = True
                                    break
                        elif edge_c1[i] == 4:
                            # Must be color 0
                            for v in edge_verts[i]:
                                if color[v] == -1:
                                    color[v] = 0
                                    edge_c0[i] += 1
                                    edge_un[i] -= 1
                                    changed = True
                                    break
            return True
        
        # Order vertices by degree descending
        vertex_order = sorted(range(n), key=lambda v: -len(vert_to_edges[v]))
        
        def backtrack(pos):
            nonlocal start
            if time.time() - start > timeout:
                return None
            if pos == n:
                return True
            
            v = vertex_order[pos]
            
            for c in (0, 1):
                # Assign color c to vertex v
                color[v] = c
                affected = []
                for ei in vert_to_edges[v]:
                    if c == 0:
                        edge_c0[ei] += 1
                    else:
                        edge_c1[ei] += 1
                    edge_un[ei] -= 1
                    affected.append(ei)
                
                result = propagate()
                if result is True:
                    res = backtrack(pos + 1)
                    if res is not None:
                        # Undo
                        color[v] = -1
                        for ei in affected:
                            if c == 0:
                                edge_c0[ei] -= 1
                            else:
                                edge_c1[ei] -= 1
                            edge_un[ei] += 1
                        return res
                elif result is None:
                    # Timeout
                    color[v] = -1
                    for ei in affected:
                        if c == 0:
                            edge_c0[ei] -= 1
                        else:
                            edge_c1[ei] -= 1
                        edge_un[ei] += 1
                    return None
                
                # Undo
                color[v] = -1
                for ei in affected:
                    if c == 0:
                        edge_c0[ei] -= 1
                    else:
                        edge_c1[ei] -= 1
                    edge_un[ei] += 1
            
            return None
        
        # Initial propagation
        result = propagate()
        if result is False:
            return True
        if result is None:
            return False
        
        result = backtrack(0)
        if result is True:
            return False
        elif result is None:
            return False
        else:
            return True
    
    return check_not_2colorable()

def _greedy_minimize(n, edges, deadline):
    """Greedy edge removal to minimize edges while maintaining non-2-colorability."""
    current = list(edges)
    i = 0
    while i < len(current):
        if time.time() > deadline - 3:
            break
        test = current[:i] + current[i+1:]
        if _is_not_2_colorable(n, test, timeout=1.5):
            current = test
        else:
            i += 1
    return current

def construct():
    deadline = time.time() + 115  # safety margin before 150s kill
    
    best_n = None
    best_edges = None
    best_count = float('inf')
    
    # Start with known small constructions
    # Fano plane based: 7 vertices, 7 edges (triples) - but we need 5-uniform
    # For 5-uniform, we need different approaches
    
    # Try n from 9 upward
    for n in range(9, 15):
        if time.time() > deadline - 15:
            break
        
        all_possible = list(combinations(range(n), 5))
        total_edges = len(all_possible)
        
        # Multiple random restarts with different strategies
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Strategy 1: Random subset
            num_edges = min(total_edges, max(25, n * 3))
            random.shuffle(all_possible)
            selected = all_possible[:num_edges]
            
            if _is_not_2_colorable(n, selected, timeout=2.0):
                current = _greedy_minimize(n, selected, deadline)
                if len(current) < best_count:
                    best_count = len(current)
                    best_n = n
                    best_edges = [list(e) for e in current]
                    if best_count <= 51:
                        break
            
            # Strategy 2: Dense core + random expansion
            if attempt >= 10:
                # Start with a dense core on first min(n, 12) vertices
                core_n = min(n, 12)
                core_edges = list(combinations(range(core_n), 5))
                random.shuffle(core_edges)
                core = core_edges[:min(len(core_edges), 40)]
                
                # Add random edges involving other vertices
                remaining = [e for e in all_possible if e not in set(core)]
                random.shuffle(remaining)
                extra = remaining[:min(len(remaining), 15)]
                selected = core + extra
                
                if _is_not_2_colorable(n, selected, timeout=2.0):
                    current = _greedy_minimize(n, selected, deadline)
                    if len(current) < best_count:
                        best_count = len(current)
                        best_n = n
                        best_edges = [list(e) for e in current]
                        if best_count <= 51:
                            break
        
        if best_count <= 51:
            break
    
    # If we haven't found something good, try larger n with dense random
    if best_count > 60:
        for n in range(15, 20):
            if time.time() > deadline - 10:
                break
            
            all_possible = list(combinations(range(n), 5))
            for attempt in range(20):
                if time.time() > deadline - 8:
                    break
                
                num_edges = min(len(all_possible), max(30, n * 2))
                random.shuffle(all_possible)
                selected = all_possible[:num_edges]
                
                if _is_not_2_colorable(n, selected, timeout=2.0):
                    current = _greedy_minimize(n, selected, deadline)
                    if len(current) < best_count:
                        best_count = len(current)
                        best_n = n
                        best_edges = [list(e) for e in current]
                        if best_count <= 51:
                            break
            
            if best_count <= 51:
                break
    
    # Fallback: complete 5-uniform on 9 vertices (126 edges)
    if best_edges is None or best_count > 126:
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
    
    return {'n': best_n, 'edges': best_edges}
