"""Few edges in 5-uniform hypergraphs without Property B (m(5)).

Strategy:
  - We know a 51-edge construction exists (best known upper bound).
  - We know every 5-uniform hypergraph with <= 28 edges is 2-colourable.
  - So the answer m(5) is in {29, ..., 51}.
  - We will try to find a small non-2-colourable 5-uniform hypergraph by
    randomized search + exact verification via backtracking with unit propagation.

  The seed program returns the complete 5-uniform hypergraph on 9 vertices,
  which has C(9,5)=126 edges and scores 1.0 (seed baseline).

  We aim to beat 126 by finding a hypergraph with fewer edges that is still
  not 2-colourable. If we find one with e edges, the score is 126/e > 1.

  Approach:
    1. Start from a known small non-2-colourable hypergraph if we can hardcode one.
    2. Otherwise, use randomized construction: pick random 5-sets, check
       non-2-colourability, try to prune edges.

  Since hardcoding a verified 51-edge (or smaller) hypergraph is the most
  reliable path to a high score, and we have time to search at runtime,
  we do a time-bounded randomized search to find a small non-2-colourable
  5-uniform hypergraph, then verify it exactly.

  For robustness, we also include a fallback: the complete 5-graph on 9 vertices
  (126 edges) which is guaranteed non-2-colourable.

  Key insight: The property "not 2-colourable" for 5-uniform hypergraphs is
  equivalent to: the hypergraph contains a sub-hypergraph that is not 2-colourable.
  Small examples are known. The smallest known 5-uniform non-2-colourable
  hypergraph has 51 edges on some number of vertices.

  We'll do a runtime search:
    - Try n from 10 to 26.
    - For each n, try random subsets of C(n,5) edges of increasing size.
    - Check if the hypergraph is non-2-colourable using a fast backtracking
      checker with unit propagation.
    - Stop when we find a non-2-colourable hypergraph with few edges.

  To make this feasible in 120s, we use:
    - A very fast 2-colourability checker (bitmask-based backtracking).
    - Randomized edge selection with pruning.
    - Time limit with early return of best found.
"""

import random
import time
import itertools
from collections import deque

# Deadline: keep 10s safety margin
_DEADLINE = time.time() + 140


def _is_2_colorable(n, edges):
    """
    Check if a 5-uniform hypergraph is 2-colourable.
    Returns True if 2-colourable, False otherwise.
    Uses backtracking with unit propagation on bitmasks.
    """
    # Build incidence: for each vertex, list of edge indices containing it
    # For speed, we use a different formulation.
    # A 2-colouring is a partition of vertices into two sets A, B such that
    # no edge is entirely in A or entirely in B.
    # Equivalently, for each edge e, the edge is not monochromatic.
    #
    # Backtracking: assign vertices one by one. For each edge, once 4 vertices
    # are assigned, the 5th is forced (unit propagation).
    
    m = len(edges)
    if m == 0:
        return True
    
    # Map each vertex to list of edges it's in
    vert_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vert_edges[v].append(i)
    
    # We'll use a recursive backtracking with memoization on the state of
    # "forced" edges. Actually, let's use a simpler approach:
    # Try all 2^n colourings? No, n can be up to 26.
    # 
    # Better: use the fact that we can fix vertex 0 = colour 0 (symmetry).
    # Then backtrack. With unit propagation: if an edge has 4 vertices of 
    # colour c, the 5th must be 1-c.
    
    # State: assignment array (-1 = unassigned, 0 or 1 = colour)
    # We'll use iterative deepening with pruning.
    
    # Actually, for small n (<= 26) and m <= 200, a well-optimized
    # backtracking should work. Let's use a bitmask representation.
    
    # For each edge, store the 5 vertices as a tuple.
    edge_verts = [tuple(e) for e in edges]
    
    # Precompute for each vertex, the edges containing it
    # vert_edges already built above
    
    # Use recursion with bitmasks for speed
    # State: (assigned_mask, colour_mask) where assigned_mask has bits for
    # assigned vertices, colour_mask has bits for vertices coloured 1.
    
    # We'll use a stack-based DFS with unit propagation.
    
    # For efficiency, let's use a different approach:
    # Represent the problem as a SAT-like constraint.
    # Each edge e = {v1,...,v5}: NOT(all 0) AND NOT(all 1)
    # i.e., (x_v1 OR x_v2 OR x_v3 OR x_v4 OR x_v5) AND (NOT x_v1 OR ... OR NOT x_v5)
    
    # Let's just do a clean recursive backtracking with unit propagation.
    
    assignment = [-1] * n  # -1 unassigned, 0 or 1
    
    # Count how many 0s and 1s in each edge (for propagation)
    # edge_zero_count[e] = number of vertices in edge e assigned colour 0
    # edge_one_count[e] = number of vertices in edge e assigned colour 1
    
    edge_zero = [0] * m
    edge_one = [0] * m
    
    # For each vertex, which edges it affects
    # vert_edges already built
    
    def propagate(queue):
        """Unit propagation: if an edge has 4 of one colour, force the 5th."""
        while queue:
            v = queue.popleft()
            if assignment[v] != -1:
                continue
            # Try to determine v's colour from constraints
            # For each edge containing v, check if 4 other vertices are same colour
            forced = -1
            for e in vert_edges[v]:
                others = [u for u in edge_verts[e] if u != v]
                zeros = sum(1 for u in others if assignment[u] == 0)
                ones = sum(1 for u in others if assignment[u] == 1)
                unassigned = sum(1 for u in others if assignment[u] == -1)
                if unassigned == 0:
                    if zeros == 4:
                        # v must be 1
                        if forced != -1 and forced != 1:
                            return False
                        forced = 1
                    elif ones == 4:
                        # v must be 0
                        if forced != -1 and forced != 0:
                            return False
                        forced = 0
                elif unassigned == 1:
                    # One other vertex unassigned, can't force yet
                    pass
                else:
                    # Multiple unassigned, can't force
                    pass
            if forced != -1:
                assignment[v] = forced
                # Update edge counts and queue neighbors
                for e in vert_edges[v]:
                    if forced == 0:
                        edge_zero[e] += 1
                    else:
                        edge_one[e] += 1
                    for u in edge_verts[e]:
                        if assignment[u] == -1:
                            queue.append(u)
            else:
                # Can't force v yet, but check for contradictions
                for e in vert_edges[v]:
                    others = [u for u in edge_verts[e] if u != v]
                    zeros = sum(1 for u in others if assignment[u] == 0)
                    ones = sum(1 for u in others if assignment[u] == 1)
                    if zeros == 4 and ones == 0:
                        # v must be 1, but we said we can't force... 
                        # Actually this should have been caught above
                        pass
        return True
    
    # This propagation is getting complex. Let me use a simpler but effective
    # approach: recursive backtracking choosing the next vertex heuristically.
    
    # Actually, let me use a cleaner formulation. Since n <= 26, we can use
    # a bitmask for the colouring and check constraints incrementally.
    # But 2^26 is too large.
    
    # Let's use the standard approach: pick an unassigned vertex, try both colours,
    # propagate, recurse.
    
    def backtrack():
        # Find unassigned vertex with highest degree (most constraints)
        best_v = -1
        best_deg = -1
        for v in range(n):
            if assignment[v] == -1:
                deg = len(vert_edges[v])
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
        if best_v == -1:
            # All assigned, check all edges are not monochromatic
            for e in range(m):
                if edge_zero[e] == 5 or edge_one[e] == 5:
                    return False
            return True
        
        for col in (0, 1):
            # Try assigning best_v = col
            assignment[best_v] = col
            # Check if this creates a monochromatic edge
            valid = True
            for e in vert_edges[best_v]:
                if col == 0:
                    edge_zero[e] += 1
                    if edge_zero[e] == 5:
                        valid = False
                        break
                else:
                    edge_one[e] += 1
                    if edge_one[e] == 5:
                        valid = False
                        break
            if not valid:
                # Undo
                for e in vert_edges[best_v]:
                    if col == 0:
                        edge_zero[e] -= 1
                    else:
                        edge_one[e] -= 1
                assignment[best_v] = -1
                continue
            
            # Unit propagation
            queue = deque()
            for e in vert_edges[best_v]:
                for u in edge_verts[e]:
                    if assignment[u] == -1:
                        queue.append(u)
            
            prop_ok = True
            while queue and prop_ok:
                v = queue.popleft()
                if assignment[v] != -1:
                    continue
                forced = -1
                for e in vert_edges[v]:
                    others = [u for u in edge_verts[e] if u != v]
                    z = sum(1 for u in others if assignment[u] == 0)
                    o = sum(1 for u in others if assignment[u] == 1)
                    ua = sum(1 for u in others if assignment[u] == -1)
                    if ua == 0:
                        if z == 4:
                            if forced != -1 and forced != 1:
                                prop_ok = False
                                break
                            forced = 1
                        elif o == 4:
                            if forced != -1 and forced != 0:
                                prop_ok = False
                                break
                            forced = 0
                    elif ua == 1:
                        # Check if forcing is possible
                        if z == 3 and o == 0:
                            # If the unassigned one is 0, all 5 are 0 -> bad
                            # So it must be 1? No, that's not unit propagation.
                            # Unit propagation only when 4 are determined.
                            pass
                if not prop_ok:
                    break
                if forced != -1:
                    assignment[v] = forced
                    for e in vert_edges[v]:
                        if forced == 0:
                            edge_zero[e] += 1
                            if edge_zero[e] == 5:
                                prop_ok = False
                                break
                        else:
                            edge_one[e] += 1
                            if edge_one[e] == 5:
                                prop_ok = False
                                break
                        if not prop_ok:
                            break
                    if prop_ok:
                        for e in vert_edges[v]:
                            for u in edge_verts[e]:
                                if assignment[u] == -1:
                                    queue.append(u)
            
            if not prop_ok:
                # Undo all changes made in this branch
                # This is complex; let's just undo best_v and continue
                for e in vert_edges[best_v]:
                    if col == 0:
                        edge_zero[e] -= 1
                    else:
                        edge_one[e] -= 1
                # Undo propagation changes - this is a problem.
                # Let me restructure to use a copy-on-write approach or 
                # a simpler backtracking without in-place propagation.
                assignment[best_v] = -1
                continue
            
            if backtrack():
                return True
            
            # Undo: this is where it gets tricky with in-place changes.
            # For correctness, let's use a simpler approach without 
            # in-place propagation during backtracking.
            assignment[best_v] = -1
            for e in vert_edges[best_v]:
                if col == 0:
                    edge_zero[e] -= 1
                else:
                    edge_one[e] -= 1
        
        return False
    
    return backtrack()


def _fast_check_2col(n, edges):
    """
    Fast 2-colourability check using a cleaner recursive approach.
    Uses symmetry breaking: vertex 0 is always colour 0.
    """
    m = len(edges)
    if m == 0:
        return True
    
    edge_verts = [tuple(e) for e in edges]
    vert_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vert_edges[v].append(i)
    
    # Use a state-based approach with memoization on small states
    # For n up to ~20, we can use bitmask state.
    # For larger n, we rely on good branching.
    
    # Symmetry breaking: fix vertex 0 = 0
    # State: (assigned_mask, col_mask) where col_mask has 1 for colour 1
    # But 2^n states is too many for n > 20.
    
    # Let's use recursive backtracking with a good variable ordering.
    
    def _bt(assign, edge_zero, edge_one):
        # Find next vertex to assign
        best_v = -1
        best_score = -1
        for v in range(n):
            if assign[v] == -1:
                # Score: number of edges where v is the last unassigned
                score = 0
                for e in vert_edges[v]:
                    others = [u for u in edge_verts[e] if u != v]
                    ua = sum(1 for u in others if assign[u] == -1)
                    if ua == 0:
                        score += 10  # Can be forced
                    elif ua == 1:
                        score += 5
                if score > best_score:
                    best_score = score
                    best_v = v
            if time.time() > _DEADLINE:
                return None  # Timeout, unknown
        if best_v == -1:
            # All assigned
            for e in range(m):
                if edge_zero[e] == 5 or edge_one[e] == 5:
                    return False
            return True
        
        for col in (0, 1):
            assign[best_v] = col
            ok = True
            for e in vert_edges[best_v]:
                if col == 0:
                    edge_zero[e] += 1
                    if edge_zero[e] == 5:
                        ok = False
                        break
                else:
                    edge_one[e] += 1
                    if edge_one[e] == 5:
                        ok = False
                        break
            if ok:
                # Check unit propagation
                # For each edge containing best_v, if 4 others are same colour,
                # the 5th (if unassigned) is forced
                forced_assignments = []
                for e in vert_edges[best_v]:
                    others = [u for u in edge_verts[e] if u != best_v]
                    z = sum(1 for u in others if assign[u] == 0)
                    o = sum(1 for u in others if assign[u] == 1)
                    ua_list = [u for u in others if assign[u] == -1]
                    if len(ua_list) == 1:
                        if z == 4:
                            forced_assignments.append((ua_list[0], 1))
                        elif o == 4:
                            forced_assignments.append((ua_list[0], 0))
                
                # Apply forced assignments
                prop_ok = True
                for v, c in forced_assignments:
                    if assign[v] != -1:
                        if assign[v] != c:
                            prop_ok = False
                            break
                    else:
                        assign[v] = c
                        for e2 in vert_edges[v]:
                            if c == 0:
                                edge_zero[e2] += 1
                                if edge_zero[e2] == 5:
                                    prop_ok = False
                                    break
                            else:
                                edge_one[e2] += 1
                                if edge_one[e2] == 5:
                                    prop_ok = False
                                    break
                            if not prop_ok:
                                break
                    if not prop_ok:
                        break
                
                if prop_ok:
                    result = _bt(assign, edge_zero, edge_one)
                    if result is not None:
                        if result:
                            return True
                        # Else continue trying
                else:
                    pass  # Propagation failed, undo and try next col
            
            # Undo
            for v, c in forced_assignments:
                if assign[v] != -1:
                    assign[v] = -1
                    for e2 in vert_edges[v]:
                        if c == 0:
                            edge_zero[e2] -= 1
                        else:
                            edge_one[e2] -= 1
            assign[best_v] = -1
            for e in vert_edges[best_v]:
                if col == 0:
                    edge_zero[e] -= 1
                else:
                    edge_one[e] -= 1
            if time.time() > _DEADLINE:
                return None
        
        return False
    
    # Fix vertex 0 = 0
    assign = [0] * n
    edge_zero = [0] * m
    edge_one = [0] * m
    for e in range(m):
        for v in edge_verts[e]:
            if v == 0:
                edge_zero[e] += 1
            else:
                assign[v] = -1
    
    # Check if vertex 0 being 0 already creates a monochromatic edge
    for e in range(m):
        if edge_zero[e] == 5:
            return False  # Not 2-colourable (only colour 0 works for all)
        # If all other 4 are unassigned, fine
    
    result = _bt(assign, edge_zero, edge_one)
    return result if result is not None else True  # On timeout, assume 2-colourable (safe for search)


def construct():
    """
    Find a small 5-uniform hypergraph with no proper 2-colouring.
    Returns {'n': n, 'edges': [[v1,...,v5], ...]}
    """
    start_time = time.time()
    
    # Fallback: complete 5-graph on 9 vertices (126 edges)
    from itertools import combinations
    fallback_edges = [list(e) for e in combinations(range(9), 5)]
    best_n = 9
    best_edges = fallback_edges
    best_m = len(fallback_edges)
    
    # Try to find something better via randomized search
    # For each n, try random edge sets of increasing size
    
    rng = random.Random(42)
    
    for n in range(10, 27):
        if time.time() - start_time > 130:
            break
        
        all_edges = list(combinations(range(n), 5))
        total = len(all_edges)
        
        # Try different sizes
        for target_m in range(29, min(52, total + 1)):
            if time.time() - start_time > 130:
                break
            
            # Try a few random subsets of size target_m
            attempts = 3
            for _ in range(attempts):
                if time.time() - start_time > 130:
                    break
                
                subset = rng.sample(all_edges, target_m)
                # Check if non-2-colourable
                if not _fast_check_2col(n, subset):
                    # Found a non-2-colourable hypergraph
                    # Try to prune edges
                    edges = [list(e) for e in subset]
                    # Greedy edge removal
                    for i in range(len(edges) - 1, -1, -1):
                        if time.time() - start_time > 130:
                            break
                        test_edges = edges[:i] + edges[i+1:]
                        if test_edges and not _fast_check_2col(n, test_edges):
                            edges = test_edges
                    if len(edges) < best_m:
                        best_n = n
                        best_edges = edges
                        best_m = len(edges)
                    break  # Found one for this target_m, move to next n or size
    
    if best_m < 126:
        return {'n': best_n, 'edges': best_edges}
    else:
        return {'n': 9, 'edges': fallback_edges}
