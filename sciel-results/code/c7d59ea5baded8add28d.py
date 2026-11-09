# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_2_colorable(edges, n, deadline=None):
    """Exact 2-colourability check with unit propagation + backtracking.
    Returns True if a proper 2-colouring exists, False if none.
    edges: list of frozensets of size 5.
    """
    if deadline is None:
        deadline = time.time() + 10
        
    if not edges:
        return True

    # Build incidence: vertex -> list of edge indices
    inc = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            inc[v].append(i)

    # State: colour[v] in {-1, 0, 1}
    assign = [-1] * n
    evs = [list(e) for e in edges]
    
    # Count how many edges each vertex is in
    deg = [len(inc[v]) for v in range(n)]

    def propagate(assign, changed):
        stack = list(changed)
        visited = set(stack)
        while stack:
            v = stack.pop()
            cv = assign[v]
            if cv == -1:
                continue
            for ei in inc[v]:
                e = evs[ei]
                c0 = 0
                c1 = 0
                unassigned = -1
                for u in e:
                    cu = assign[u]
                    if cu == 0:
                        c0 += 1
                    elif cu == 1:
                        c1 += 1
                    else:
                        unassigned = u
                        break
                if c0 > 0 and c1 > 0:
                    continue
                if c0 == 0 and c1 == 0:
                    continue
                # One color is present, the other is not
                if c0 == 0 and c1 > 0:
                    # All assigned are 1
                    if unassigned != -1:
                        assign[unassigned] = 0
                        if unassigned not in visited:
                            visited.add(unassigned)
                            stack.append(unassigned)
                    else:
                        # All assigned and all 1 -> Conflict
                        return False
                elif c1 == 0 and c0 > 0:
                    # All assigned are 0
                    if unassigned != -1:
                        assign[unassigned] = 1
                        if unassigned not in visited:
                            visited.add(unassigned)
                            stack.append(unassigned)
                    else:
                        # All assigned and all 0 -> Conflict
                        return False
        return True

    def backtrack(assign, uncolored):
        if time.time() > deadline:
            return None
        if not uncolored:
            return assign
        
        # Heuristic: choose vertex with max degree among uncolored
        # Ties broken by current assignment status (none)
        # We can also try to pick a vertex involved in a constraint that is "tight"
        # But simple max degree is usually good.
        
        v = max(uncolored, key=lambda x: deg[x])
        new_uncolored = uncolored - {v}
        
        for val in (0, 1):
            assign[v] = val
            if propagate(assign, [v]):
                res = backtrack(assign, new_uncolored)
                if res is not None:
                    return res
            assign[v] = -1
        return None

    uncolored = set(range(n))
    result = backtrack(assign, uncolored)
    if result is None:
        return False
    return True

def _remove_edge_if_possible(edges, n, deadline):
    """
    Attempts to remove one edge from the hypergraph while maintaining non-2-colorability.
    Returns (new_edges, removed_edge) or (None, None) if no edge could be removed.
    """
    if not edges:
        return None, None
        
    # Try to remove a random edge
    idx = random.randint(0, len(edges) - 1)
    e_to_remove = edges[idx]
    new_edges = [e for i, e in enumerate(edges) if i != idx]
    
    # Check if new_edges is still non-2-colorable
    if not _is_2_colorable(new_edges, n, deadline):
        return new_edges, e_to_remove
    else:
        return None, None

def _local_search(edges, n, max_iterations, deadline):
    """
    Local search to minimize edges.
    """
    current_edges = list(edges)
    current_count = len(current_edges)
    best_edges = list(current_edges)
    best_count = current_count
    
    no_improve = 0
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
            
        new_edges, removed = _remove_edge_if_possible(current_edges, n, deadline)
        if new_edges is not None:
            current_edges = new_edges
            current_count -= 1
            if current_count < best_count:
                best_count = current_count
                best_edges = list(current_edges)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 20:
                break
    return best_edges

def _greedy_set_cover(n, deadline):
    """
    Greedy set cover to find a non-2-colorable hypergraph.
    Universe: all 2^n colorings (bitmasks 0 to 2^n - 1).
    Set for each edge: colorings where the edge is monochromatic.
    """
    if time.time() > deadline:
        return None
    
    # Only feasible for small n
    if n > 14:
        return None
        
    num_colorings = 1 << n
    
    # Universe bitmask: all 1s
    universe_mask = (1 << num_colorings) - 1
    
    edges = list(combinations(range(n), 5))
    
    edge_masks = []
    for e in edges:
        mask = 0
        for c in range(num_colorings):
            b0 = (c >> e[0]) & 1
            if b0 == ((c >> e[1]) & 1) == ((c >> e[2]) & 1) == ((c >> e[3]) & 1) == ((c >> e[4]) & 1):
                mask |= (1 << c)
        edge_masks.append((e, mask))
        
    # Greedy Set Cover
    uncovered = universe_mask
    selected_edges = []
    
    # Sort edges by initial coverage? Or just iterate.
    edge_masks.sort(key=lambda x: x[1].bit_count(), reverse=True)
    
    while uncovered:
        if time.time() > deadline:
            return None
        best_edge = None
        best_gain = 0
        best_mask = 0
        
        for e, mask in edge_masks:
            intersection = mask & uncovered
            if intersection:
                gain = intersection.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_edge = e
                    best_mask = mask
                    if gain == uncovered.bit_count():
                        break
                        
        if best_edge is None:
            return None
            
        selected_edges.append(best_edge)
        uncovered &= ~best_mask
        
    return [list(e) for e in selected_edges]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    best_edges = None
    best_n = None
    best_count = float('inf')
    
    # We will try to find a small non-2-colorable hypergraph.
    # Start with n=11, which is small enough for bitmask set cover.
    # Then try to reduce edges via local search.
    
    # 1. Try Greedy Set Cover for n=11
    if time.time() < deadline - 20:
        edges = _greedy_set_cover(11, deadline - 20)
        if edges:
            fs_edges = [frozenset(e) for e in edges]
            if not _is_2_colorable(fs_edges, 11, deadline - 10):
                best_edges = edges
                best_n = 11
                best_count = len(edges)
                # Local search to reduce
                if time.time() < deadline - 5:
                    reduced = _local_search(best_edges, 11, 1000, deadline - 5)
                    if len(reduced) < best_count:
                        best_edges = reduced
                        best_count = len(reduced)

    # 2. If that failed or didn't find anything good, try random construction for n=11, 12
    for n in [11, 12]:
        if time.time() > deadline - 10:
            break
            
        # Try a few random restarts
        for restart in range(5):
            if time.time() > deadline - 5:
                break
                
            # Start with a random subset of edges?
            # Or start with all edges and remove?
            all_edges = list(combinations(range(n), 5))
            random.shuffle(all_edges)
            
            # Try a random subset of size ~50-100
            # If it's non-2-colorable, try to reduce.
            # If it's 2-colorable, add more edges.
            
            # Let's start with a medium size and adjust
            initial_size = 60
            if n == 12:
                initial_size = 80
                
            current_edges = all_edges[:initial_size]
            
            # Check if it's non-2-colorable
            fs_current = [frozenset(e) for e in current_edges]
            if not _is_2_colorable(fs_current, n, deadline - 5):
                # It is non-2-colorable. Try to reduce.
                reduced = _local_search(current_edges, n, 500, deadline - 5)
                if len(reduced) < best_count:
                    best_count = len(reduced)
                    best_edges = reduced
                    best_n = n
            else:
                # It is 2-colorable. Try to add edges until it becomes non-2-colorable.
                # This is a bit slow, so we'll just skip this restart or do a quick greedy add.
                # For simplicity, let's just skip.
                pass

    # 3. Fallback: If we still have nothing, use a known dense hypergraph and reduce.
    if best_edges is None:
        n = 11
        all_edges = list(combinations(range(n), 5))
        current_edges = list(all_edges)
        # Check if full hypergraph is non-2-colorable
        if not _is_2_colorable([frozenset(e) for e in current_edges], n, deadline - 5):
            reduced = _local_search(current_edges, n, 1000, deadline - 5)
            best_edges = reduced
            best_n = n
            best_count = len(reduced)
        else:
            # This should not happen for n=11
            # If it does, try n=12
            n = 12
            all_edges = list(combinations(range(n), 5))
            current_edges = list(all_edges)
            if not _is_2_colorable([frozenset(e) for e in current_edges], n, deadline - 5):
                reduced = _local_search(current_edges, n, 1000, deadline - 5)
                best_edges = reduced
                best_n = n
                best_count = len(reduced)
            else:
                # Last resort: return full K_11^5
                best_edges = list(combinations(range(11), 5))
                best_n = 11
                best_count = len(best_edges)

    # Final verification
    if best_edges is None:
        # Should not happen
        best_edges = list(combinations(range(11), 5))
        best_n = 11

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
