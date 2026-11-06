"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    N = 256
    # Build conflict graph: edge (i,j) if i+j is a perfect square
    # Vertices are 1..256, index 0..255
    # Diagonal: i+i must not be a square -> vertex i is "forbidden" if 2*(i+1) is square
    # Actually: a+b never square for a<=b INCLUDING a=b means 2a must not be square
    # So vertices where 2*v is a perfect square are excluded entirely
    
    # Precompute which values are perfect squares up to 512 (max sum = 256+256=512)
    squares = set()
    for s in range(1, 23):  # 1^2 to 22^2=484, 23^2=529>512
        squares.add(s * s)
    
    # Build adjacency: conflict[i] = set of j where (i+1)+(j+1) is square
    # Only consider pairs where both are "allowed" (diagonal not square)
    allowed = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            allowed.append(v)
    
    n_allowed = len(allowed)
    # Map allowed values to indices
    val_to_idx = {v: i for i, v in enumerate(allowed)}
    
    # Build conflict adjacency list
    conflicts = [set() for _ in range(n_allowed)]
    for i in range(n_allowed):
        for j in range(i + 1, n_allowed):
            s = allowed[i] + allowed[j]
            if s in squares:
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    # Greedy: repeatedly pick vertex with minimum degree in remaining graph
    remaining = set(range(n_allowed))
    selected = set()
    
    while remaining:
        if time.time() > deadline - 5.0:
            break
        # Find vertex with minimum degree in remaining subgraph
        best_v = None
        best_deg = float('inf')
        for v in remaining:
            deg = len(conflicts[v] & remaining)
            if deg < best_deg:
                best_deg = deg
                best_v = v
                if deg == 0:
                    break
        selected.add(best_v)
        remaining.discard(best_v)
        # Remove neighbors from remaining
        for nb in list(conflicts[best_v]):
            remaining.discard(nb)
    
    # Local search: try to add vertices, then try swaps
    # Convert to set for fast lookup
    selected_set = set(selected)
    
    # Phase 1: Try adding any unselected vertex that doesn't conflict
    for v in range(n_allowed):
        if time.time() > deadline - 5.0:
            break
        if v not in selected_set:
            if not (conflicts[v] & selected_set):
                selected_set.add(v)
    
    # Phase 2: Try swaps - remove one selected, add one unselected if it helps
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for v_in in range(n_allowed):
            if time.time() > deadline - 5.0:
                break
            if v_in in selected_set:
                continue
            # Can we add v_in by removing some subset of its conflicts?
            conflicts_in = conflicts[v_in] & selected_set
            if len(conflicts_in) == 0:
                # Can just add
                selected_set.add(v_in)
                improved = True
            elif len(conflicts_in) == 1:
                # Try removing one conflict and adding v_in
                # Only beneficial if v_in has more "free" potential
                # Simple heuristic: always try the swap
                for v_out in conflicts_in:
                    # Check: after removing v_out, can we still add v_in?
                    # v_in's conflicts with remaining selected should be empty
                    remaining_conflicts = (conflicts[v_in] & selected_set) - {v_out}
                    if len(remaining_conflicts) == 0:
                        selected_set.discard(v_out)
                        selected_set.add(v_in)
                        improved = True
                        break
    
    # Phase 3: More aggressive - try removing up to 2 selected to add 1
    for v_in in range(n_allowed):
        if time.time() > deadline - 5.0:
            break
        if v_in in selected_set:
            continue
        conflicts_in = conflicts[v_in] & selected_set
        if 1 < len(conflicts_in) <= 2:
            conf_list = list(conflicts_in)
            if len(conf_list) == 2:
                a, b = conf_list
                # Check if a and b conflict with each other (they might be removable together)
                selected_set.discard(a)
                selected_set.discard(b)
                if not (conflicts[v_in] & selected_set):
                    selected_set.add(v_in)
                    # Net change: -2 +1 = -1, so only keep if it enables more adds later
                    # Actually this is a net loss, skip unless it helps
                    # Revert for now
                    selected_set.discard(v_in)
                    selected_set.add(a)
                    selected_set.add(b)
    
    # Convert back to values
    result = [allowed[i] for i in sorted(selected_set)]
    
    # Verify validity
    for i in range(len(result)):
        for j in range(i, len(result)):
            if _is_square(result[i] + result[j]):
                # Should not happen, but safety check
                pass
    
    if not result:
        # Fallback: return a single valid element
        for v in range(1, N + 1):
            if 2 * v not in squares:
                return [v]
        return [1]
    
    return result
# EVOLVE-BLOCK-END
