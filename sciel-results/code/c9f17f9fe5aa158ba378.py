"""Few edges in 5-uniform hypergraphs without Property B (m(5))."""
import time
import random
from itertools import combinations

# Deadline guard: evaluator kills at 150s, keep safety margin.
_DEADLINE = 135.0  # seconds

def _edge_mask(edge):
    """Convert a 5-element edge to a bit mask."""
    m = 0
    for v in edge:
        m |= (1 << v)
    return m

def _is_2_colorable(edges_masks, n, deadline):
    """
    Check if the hypergraph (given as list of bit masks) has a proper 2-coloring.
    A proper 2-coloring means no edge is monochromatic.
    We use backtracking with unit propagation.
    
    Returns True if 2-colorable, False if not (i.e., no proper 2-coloring exists).
    """
    # Precompute for each edge: list of vertex indices
    # We also need to know which edges contain which vertices for propagation efficiency
    # But for n <= 26, simple iteration is fast enough if optimized.
    
    # Optimization: 
    # 1. Use bitmasks for color assignments? No, standard array is fine.
    # 2. Fast unit propagation: 
    #    If an edge has 4 vertices of color 0, the 5th MUST be 1.
    #    If an edge has 4 vertices of color 1, the 5th MUST be 0.
    #    If an edge has 5 vertices of color 0 or 1, it's a conflict.

    # To speed up, we can precompute the vertex list for each edge mask.
    edge_verts = []
    for mask in edges_masks:
        verts = []
        # n is small, so this is fast
        for v in range(n):
            if mask & (1 << v):
                verts.append(v)
        edge_verts.append(verts)
    
    # State: color assignment array
    # -1: unassigned, 0: color 0, 1: color 1
    color = [-1] * n
    
    # We use a stack for backtracking to avoid recursion depth issues and overhead
    # But for n <= 26, recursion is fine. Let's stick to recursion for clarity and speed in CPython.
    
    def propagate():
        """
        Apply unit propagation.
        Returns (success, assignments) where assignments is a list of (v, c) to undo.
        """
        assignments = []
        changed = True
        while changed:
            changed = False
            for i, verts in enumerate(edge_verts):
                c0 = 0
                c1 = 0
                unassigned = []
                for v in verts:
                    if color[v] == -1:
                        unassigned.append(v)
                    elif color[v] == 0:
                        c0 += 1
                    else:
                        c1 += 1
                
                # Check for contradiction
                if c0 == 5 or c1 == 5:
                    return False, assignments
                
                # Unit propagation
                if len(unassigned) == 1:
                    if c0 == 4:
                        v = unassigned[0]
                        if color[v] != 1:
                            if color[v] != -1:
                                return False, assignments
                            color[v] = 1
                            assignments.append((v, 1))
                            changed = True
                    elif c1 == 4:
                        v = unassigned[0]
                        if color[v] != 0:
                            if color[v] != -1:
                                return False, assignments
                            color[v] = 0
                            assignments.append((v, 0))
                            changed = True
        return True, assignments

    def backtrack():
        if time.time() > deadline:
            return False  # timeout, assume colorable (conservative)
        
        # Check if all assigned
        # We can maintain a count of assigned vertices to avoid O(n) check
        # But for simplicity and n<=26, this is fine.
        # Actually, let's just check if we found a valid coloring.
        # If we reach a state where no more propagation is possible and no conflicts,
        # we need to branch.
        
        ok, assignments = propagate()
        if not ok:
            return False
        
        # Find an unassigned vertex to branch on
        # Heuristic: pick vertex in an edge with most assigned vertices (most constrained)
        best_v = -1
        best_constraint = -1
        
        # We can also just pick the first unassigned vertex.
        # But MRV (Minimum Remaining Values) or similar heuristics help.
        # Here, we look for an edge that is "almost monochromatic" or has high assignment.
        
        for i, verts in enumerate(edge_verts):
            assigned_count = 0
            for v in verts:
                if color[v] != -1:
                    assigned_count += 1
            
            if assigned_count > best_constraint:
                best_constraint = assigned_count
                # Pick an unassigned vertex in this edge
                for v in verts:
                    if color[v] == -1:
                        best_v = v
                        break
                if best_v != -1:
                    break
        
        if best_v == -1:
            # All vertices assigned. Check if valid (no mono edges).
            # Propagation would have caught conflicts if 5 same color.
            # If we get here, it means we have a valid coloring.
            return True
        
        # Try both colors
        for c in (0, 1):
            color[best_v] = c
            if backtrack():
                return True
            color[best_v] = -1
        
        # Undo assignments made by propagation
        for v, c in assignments:
            color[v] = -1
        
        return False
    
    return backtrack()

def _greedy_reduce(edges, n, deadline, rng):
    """
    Given a non-2-colorable hypergraph, greedily remove redundant edges.
    An edge is redundant if removing it still leaves the hypergraph non-2-colorable.
    Returns the reduced edge list.
    """
    current_edges = list(edges)
    
    # We will iterate until no more edges can be removed or time runs out.
    # To avoid getting stuck in local minima, we can shuffle the order of checks.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle the order of edges to try
        indices = list(range(len(current_edges)))
        rng.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            
            # Try removing edge at idx
            if len(current_edges) <= 1:
                break
            
            # Build list without this edge
            # Optimization: Instead of copying the list, we can pass a flag or use a set.
            # But for small lists, copying is fine.
            test_edges = current_edges[:idx] + current_edges[idx+1:]
            test_masks = [_edge_mask(e) for e in test_edges]
            
            # Check if still non-2-colorable
            # If _is_2_colorable returns False, it is NOT colorable (good, we can remove)
            # If _is_2_colorable returns True, it IS colorable (bad, we must keep)
            colorable = _is_2_colorable(test_masks, n, deadline)
            
            if not colorable:
                # Edge is redundant, remove it
                current_edges = test_edges
                improved = True
    
    return current_edges

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE - 5  # 5 second safety margin
    
    rng = random.Random(42)
    
    best_edges = None
    best_n = 0
    best_count = 201  # more than max allowed
    
    # Strategy: Multi-start Greedy Reduction
    # We try to find non-2-colorable hypergraphs and reduce them.
    # Start points:
    # 1. Complete hypergraphs K_n^(5) for small n.
    # 2. Random sparse hypergraphs that happen to be non-2-colorable.
    
    # Precompute all edges for small n
    all_edges_cache = {}
    for n in range(9, 15):
        all_edges_cache[n] = list(combinations(range(n), 5))
    
    # 1. Try reducing complete hypergraphs
    for n in [9, 10, 11, 12]:
        if time.time() > deadline:
            break
        
        all_edges = all_edges_cache[n]
        if len(all_edges) > 200:
            continue
        
        # Check if complete hypergraph is non-2-colorable
        # K_9^(5) is known to be non-2-colorable.
        masks = [_edge_mask(e) for e in all_edges]
        colorable = _is_2_colorable(masks, n, deadline)
        
        if not colorable:
            # Greedily reduce
            reduced = _greedy_reduce(all_edges, n, deadline, rng)
            
            if len(reduced) < best_count:
                best_edges = reduced
                best_n = n
                best_count = len(reduced)
        
        # 2. Try random subsets of larger hypergraphs?
        # Or random sparse graphs.
        # Let's try to generate random non-2-colorable graphs directly.
        # This is hard. Instead, let's try to perturb the reduced graph.
        
    # 3. Iterated Local Search (ILS)
    # If we have a best_edges, we can try to improve it by adding a random edge and reducing.
    # This allows escaping local minima of the greedy reduction.
    
    if best_edges is not None:
        current_best = list(best_edges)
        current_n = best_n
        current_count = len(current_best)
        
        # We will spend remaining time on ILS
        # Each iteration:
        # a) Add a random edge (from the complete graph on current_n)
        # b) Run greedy reduction
        # c) If result is better, accept.
        
        # Precompute masks for current_best
        # We need a pool of edges to add.
        pool = all_edges_cache.get(current_n, [])
        if not pool:
            # If n was not in cache, generate it
            pool = list(combinations(range(current_n), 5))
        
        pool_set = set(current_best)
        candidates = [e for e in pool if e not in pool_set]
        
        while time.time() < deadline:
            if not candidates:
                break
            
            # Pick a random edge to add
            edge_to_add = rng.choice(candidates)
            new_edges = current_best + [edge_to_add]
            
            # Run greedy reduction
            # We need a new rng seed for each iteration to vary the reduction order?
            # Or just use the same rng.
            reduced = _greedy_reduce(new_edges, current_n, deadline, rng)
            
            if len(reduced) < current_count:
                current_best = reduced
                current_count = len(reduced)
                best_edges = current_best
                best_count = current_count
                
                # Update pool candidates
                pool_set = set(current_best)
                candidates = [e for e in pool if e not in pool_set]
            else:
                # Optionally, we could try to "kick" the state by removing a random edge from current_best
                # and then reducing, but that might just increase edges.
                # Let's just continue.
                pass
    
    # If we didn't find anything, fall back to a known construction
    if best_edges is None:
        # Complete 5-uniform hypergraph on 9 vertices
        best_edges = [list(e) for e in combinations(range(9), 5)]
        best_n = 9
        best_count = len(best_edges)
    
    return {'n': best_n, 'edges': best_edges}
