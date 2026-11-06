import time
import random
from itertools import combinations

def _is_2_colorable(n, edges_bitmasks):
    """Check if the hypergraph is 2-colourable using backtracking with unit propagation.
    Returns True if 2-colourable, False otherwise."""
    num_edges = len(edges_bitmasks)
    if num_edges == 0:
        return True
    
    # Precompute: for each edge, which vertices are in it
    # State: color assignment for vertices, -1 = unassigned, 0 or 1 = colour
    # Use unit propagation: if an edge has 4 vertices of same colour, the 5th must be opposite.
    # If an edge has all 5 same colour -> contradiction.
    
    # Represent state as two bitmasks: assigned0, assigned1, unassigned
    # Actually, let's use a dict or array for simplicity, but bitmask for speed.
    
    # For speed, use recursive backtracking with memoization on (assigned_mask, color_mask)
    # But n can be up to 26, so 2^26 states is too many. Use simple backtracking with ordering.
    
    # Better: use the fact that we can pick a vertex, try both colours, and propagate.
    
    # Let's use a stack-based approach with unit propagation.
    
    def propagate(assigned, colors):
        """Propagate unit constraints. Returns (success, assigned, colors) or (False, None, None) on contradiction."""
        changed = True
        while changed:
            changed = False
            for e in edges_bitmasks:
                # Count how many vertices in this edge are assigned to 0 or 1
                count0 = 0
                count1 = 0
                unassigned_in_edge = []
                for v in range(n):
                    if (e >> v) & 1:
                        if assigned[v]:
                            if colors[v] == 0:
                                count0 += 1
                            else:
                                count1 += 1
                        else:
                            unassigned_in_edge.append(v)
                
                if count0 == 5 or count1 == 5:
                    return False, None, None
                if count0 == 4 and len(unassigned_in_edge) == 1:
                    v = unassigned_in_edge[0]
                    if colors[v] != 1:
                        colors[v] = 1
                        assigned[v] = True
                        changed = True
                elif count1 == 4 and len(unassigned_in_edge) == 1:
                    v = unassigned_in_edge[0]
                    if colors[v] != 0:
                        colors[v] = 0
                        assigned[v] = True
                        changed = True
        
        return True, assigned, colors
    
    # Backtracking
    def backtrack(assigned, colors):
        # Find first unassigned vertex
        for v in range(n):
            if not assigned[v]:
                # Try colour 0
                assigned2 = assigned[:]
                colors2 = colors[:]
                assigned2[v] = True
                colors2[v] = 0
                ok, a, c = propagate(assigned2, colors2)
                if ok and backtrack(a, c):
                    return True
                
                # Try colour 1
                assigned2 = assigned[:]
                colors2 = colors[:]
                assigned2[v] = True
                colors2[v] = 1
                ok, a, c = propagate(assigned2, colors2)
                if ok and backtrack(a, c):
                    return True
                
                return False
        return True  # All assigned, no contradiction
    
    assigned = [False] * n
    colors = [0] * n
    return backtrack(assigned, colors)


def _check_non_2_colorable(n, edges):
    """Check if hypergraph is NOT 2-colourable."""
    if not edges:
        return False
    edges_bitmasks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edges_bitmasks.append(mask)
    return not _is_2_colorable(n, edges_bitmasks)


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    best_result = None
    best_edges_count = float('inf')
    
    # Try different random seeds and vertex counts
    rng = random.Random(42)
    
    for attempt in range(200):
        if time.time() > deadline - 5:
            break
        
        # Try various n values
        n = rng.randint(12, 22)
        # Start with a random set of edges
        max_edges = min(200, 60)
        
        # Generate random edges
        all_possible = list(combinations(range(n), 5))
        rng.shuffle(all_possible)
        
        edges = [list(e) for e in all_possible[:max_edges]]
        
        # Check if non-2-colourable
        if _check_non_2_colorable(n, edges):
            # Try to minimize
            current_edges = edges[:]
            current_n = n
            
            # Greedy removal
            changed = True
            while changed and time.time() < deadline - 10:
                changed = False
                for i in range(len(current_edges)):
                    if time.time() > deadline - 10:
                        break
                    # Try removing edge i
                    test_edges = current_edges[:i] + current_edges[i+1:]
                    if _check_non_2_colorable(current_n, test_edges):
                        current_edges = test_edges
                        changed = True
                        break
            
            if len(current_edges) < best_edges_count:
                best_edges_count = len(current_edges)
                best_result = {'n': current_n, 'edges': current_edges}
        
        if best_result is not None and best_edges_count <= 55:
            # We're getting close to the known best, try harder
            pass
    
    if best_result is None:
        # Fallback: use a known construction or the complete hypergraph on 9 vertices
        from itertools import combinations as comb
        return {'n': 9, 'edges': [list(e) for e in comb(range(9), 5)]}
    
    return best_result
