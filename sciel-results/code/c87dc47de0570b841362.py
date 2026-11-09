"""Construct a 4-colouring of K_n with no monochromatic C6, maximizing n."""
import time
import random
import itertools

# Deadline guard: return best result found before this fraction of 150s
_DEADLINE = 145.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _time_left():
    return _DEADLINE - _elapsed()

def _has_c6(edges, n):
    """Check if the set of edges (in a single colour class) contains a 6-cycle.
    edges: set of frozenset or tuple pairs; n: number of vertices.
    Returns True if any 6-cycle exists."""
    # Build adjacency
    adj = [set() for _ in range(n)]
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Count 6-cycles via counting paths of length 5 between vertices
    # A 6-cycle exists iff there is a cycle of length 6.
    # Use the standard count: number of 6-cycles = (1/12) * sum over all 6-tuples...
    # Better: use the formula with adjacency matrix powers.
    # For small n, we can enumerate.
    
    # Count 6-cycles: a 6-cycle is a sequence v0,v1,v2,v3,v4,v5,v0 with all distinct.
    # We can count by: for each ordered pair (a,b) with a<b, count paths of length 5 from a to b
    # that don't revisit vertices. This is complex.
    
    # Alternative: use the known formula. The number of (not necessarily induced) 6-cycles
    # can be computed via:
    # Count = (1/12) * [ (A^6)_{ii} - 6*(A^2)_{ii} + ... ] -- this is for induced.
    # For not necessarily induced, we need to count all closed walks of length 6 that are simple cycles.
    
    # Simplest for n<=20: brute force check all 6-subsets and all permutations.
    # For each 6-subset, check if it contains a 6-cycle in the subgraph.
    # Number of 6-subsets of 20 vertices: C(20,6) = 38760. For each, check if the induced
    # subgraph on those 6 vertices has a Hamiltonian cycle. That's 6! / 2 / 6 = 60 checks per subset.
    # Total: ~2.3M checks, each O(1) with precomputed adj. Should be fast enough.
    
    from itertools import combinations, permutations
    
    # Precompute: for each 6-subset, check if there's a 6-cycle
    # A 6-cycle in the subgraph means there exists a cyclic ordering of the 6 vertices
    # such that consecutive vertices are adjacent.
    
    # For efficiency, we can check: the subgraph on 6 vertices has a Hamiltonian cycle.
    # We can use a quick necessary condition: each vertex must have degree >= 2 in the subgraph.
    
    for subset in combinations(range(n), 6):
        # Check degree condition
        degs = []
        for v in subset:
            d = sum(1 for w in subset if w != v and w in adj[v])
            if d < 2:
                break
            degs.append(d)
        else:
            # All degrees >= 2, now check for Hamiltonian cycle
            # Try all cyclic orderings (fix first vertex, permute rest, consider both directions)
            first = subset[0]
            rest = subset[1:]
            found = False
            for perm in permutations(rest):
                # Check if first, perm[0], perm[1], ..., perm[4], first forms a cycle
                ok = True
                for i in range(5):
                    if perm[i+1] not in adj[perm[i]] if i < 4 else first not in adj[perm[4]]:
                        # Need to check edges: first-perm[0], perm[0]-perm[1], ..., perm[3]-perm[4], perm[4]-first
                        pass
                # Let's do it properly
                verts = [first] + list(perm)
                ok = True
                for i in range(6):
                    u = verts[i]
                    v = verts[(i+1) % 6]
                    if v not in adj[u]:
                        ok = False
                        break
                if ok:
                    return True
    return False

def _has_c6_fast(adj, n):
    """Faster check using adjacency sets."""
    from itertools import combinations, permutations
    
    for subset in combinations(range(n), 6):
        # Quick degree check
        valid = True
        for v in subset:
            d = sum(1 for w in subset if w != v and w in adj[v])
            if d < 2:
                valid = False
                break
        if not valid:
            continue
        
        first = subset[0]
        rest = subset[1:]
        for perm in permutations(rest):
            verts = [first] + list(perm)
            ok = True
            for i in range(6):
                u = verts[i]
                v = verts[(i+1) % 6]
                if v not in adj[u]:
                    ok = False
                    break
            if ok:
                return True
    return False

def _is_valid_coloring(colors, n):
    """Check if a 4-coloring of K_n has no monochromatic C6."""
    # Build edge list per color
    color_edges = [set() for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c = colors[idx]
            color_edges[c].add((i, j))
            idx += 1
    
    for c in range(4):
        adj = [set() for _ in range(n)]
        for (u, v) in color_edges[c]:
            adj[u].add(v)
            adj[v].add(u)
        if _has_c6_fast(adj, n):
            return False
    return True

def _local_search(colors, n, max_iter=5000):
    """Local search: flip colors to try to remove C6s, or try to extend n."""
    # This is a placeholder for now; the main strategy is random + greedy.
    pass

def _try_extend(current_colors, current_n, target_n):
    """Try to extend a coloring of K_{current_n} to K_{target_n} by adding vertices."""
    # Add vertices one at a time
    colors = list(current_colors)
    n = current_n
    for new_v in range(n, target_n):
        # Assign colors to edges (new_v, j) for j in 0..new_v-1
        # Try random assignments and check validity
        best_assignment = None
        for attempt in range(200):
            if _time_left() < 5.0:
                break
            new_edge_colors = [random.randint(0, 3) for _ in range(new_v)]
            trial_colors = colors + new_edge_colors
            # Quick check: only need to check the new color classes for C6
            # Build adj for each color with the new edges
            valid = True
            for c in range(4):
                adj = [set() for _ in range(target_n)]
                idx = 0
                for i in range(target_n):
                    for j in range(i+1, target_n):
                        if idx < len(trial_colors):
                            col = trial_colors[idx]
                        else:
                            col = 0  # shouldn't happen
                        if col == c:
                            adj[i].add(j)
                            adj[j].add(i)
                        idx += 1
                if _has_c6_fast(adj, target_n):
                    valid = False
                    break
            if valid:
                best_assignment = new_edge_colors
                break
        
        if best_assignment is None:
            return None  # Failed to extend
        colors += best_assignment
        n = new_v + 1
    
    return colors

def construct():
    global _START
    _START = time.time()
    
    # Start with the known K9 coloring
    n = 9
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    best_n = n
    best_colors = colors
    
    # Verify it's valid
    if not _is_valid_coloring(colors, n):
        # Shouldn't happen, but fallback
        best_n = 1
        best_colors = []
    
    # Try to extend to larger n using random restarts and local search
    target_max = 17  # Best known
    
    # Strategy: try to extend step by step with randomization
    for target in range(best_n + 1, target_max + 1):
        if _time_left() < 10.0:
            break
        
        success = False
        for restart in range(200):
            if _time_left() < 5.0:
                break
            
            # Try extending from current best
            result = _try_extend(best_colors, best_n, target)
            if result is not None:
                # Verify
                if _is_valid_coloring(result, target):
                    best_colors = result
                    best_n = target
                    success = True
                    break
            else:
                # Try a different approach: random coloring of K_target and local repair
                if _time_left() < 5.0:
                    break
                rand_colors = [random.randint(0, 3) for _ in range(target * (target - 1) // 2)]
                if _is_valid_coloring(rand_colors, target):
                    best_colors = rand_colors
                    best_n = target
                    success = True
                    break
        
        if not success:
            # Try harder with more restarts if time permits
            if _time_left() > 20.0:
                for restart in range(500):
                    if _time_left() < 5.0:
                        break
                    rand_colors = [random.randint(0, 3) for _ in range(target * (target - 1) // 2)]
                    if _is_valid_coloring(rand_colors, target):
                        best_colors = rand_colors
                        best_n = target
                        break
            if best_n < target:
                # Can't extend further, break
                break
    
    # If we have time, try to go beyond 17
    if _time_left() > 10.0:
        for target in range(best_n + 1, 20):
            if _time_left() < 5.0:
                break
            for restart in range(1000):
                if _time_left() < 3.0:
                    break
                rand_colors = [random.randint(0, 3) for _ in range(target * (target - 1) // 2)]
                if _is_valid_coloring(rand_colors, target):
                    best_colors = rand_colors
                    best_n = target
                    break
            if best_n < target:
                break
    
    return {'n': best_n, 'colors': best_colors}
