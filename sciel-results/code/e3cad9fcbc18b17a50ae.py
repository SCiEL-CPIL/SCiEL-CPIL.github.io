"""R_4(C6): 4-colouring of K_n without monochromatic C6.
Strategy: Use a fast C6 detector with bitmasks, then do
simulated-annealing-style local search minimizing total C6 count.
Start from known good small colorings and extend.
"""
import time
import random
import sys

def _edge_index(i, j, n):
    """Return the index in the flat colors array for edge (i,j) with i<j."""
    # Sum of (n-1) + (n-2) + ... + (n-i) for the first i rows, then offset j-i-1
    # Row i has edges to i+1, i+2, ..., n-1
    # Number of edges before row i: sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)//2
    return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

def _build_adj_bitmasks(colors, n, color):
    """Build adjacency bitmasks for a given color."""
    adj = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == color:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
            idx += 1
    return adj

def _count_c6s(adj, n):
    """Count the number of 6-cycles in a graph given by adjacency bitmasks.
    A 6-cycle is v0-v1-v2-v3-v4-v5-v0 with all distinct.
    To avoid overcounting, we fix v0 < v1 and count paths v1-v2-v3-v4-v5-v0
    where v0,v1 are the smallest two vertices in the cycle... actually that's complex.
    
    Simpler: count all directed 6-cycles where v0 is the smallest vertex.
    For each v0, for each v1 in adj[v0] with v1 > v0, count paths of length 4
    from v1 to v0 avoiding v0 and v1 in the middle.
    
    Actually, let's just count: for each pair (v0, v1) with v0 < v1 and edge,
    count the number of paths v1 -> v2 -> v3 -> v4 -> v5 -> v0 where 
    v2,v3,v4,v5 are distinct and not in {v0, v1}.
    
    This is still expensive. Let me use a different approach:
    Count = sum over all ordered 6-tuples (v0..v5) with v0 < v1, all distinct,
    such that all consecutive pairs (including v5-v0) are edges.
    
    For n=18, this is at most C(18,6) * 5! / 2 ≈ 18564 * 60 ≈ 1.1M checks per color.
    That's feasible if we do it smartly.
    
    Better: for each v0, for each v1 > v0 in adj[v0], we need to count
    paths of length 4 from v1 to v0 in the graph with v0,v1 removed from internal nodes.
    
    Let's use: for fixed v0, v1 (v0<v1, edge exists), count the number of
    triples (v2, v3, v4, v5) such that:
    - v2 in adj[v1], v2 not in {v0, v1}
    - v3 in adj[v2], v3 not in {v0, v1, v2}
    - v4 in adj[v3], v4 not in {v0, v1, v2, v3}
    - v5 in adj[v4], v5 not in {v0, v1, v2, v3, v4}
    - v0 in adj[v5]
    
    This is O(n^4) per edge pair, which is too slow.
    
    Alternative: precompute for each pair of vertices the number of common neighbors,
    and use inclusion-exclusion or matrix multiplication approaches.
    
    For now, let's use a simpler but slower count that works for n up to ~18.
    """
    total = 0
    # For each v0, for each v1 > v0 with edge, count paths v1->v2->v3->v4->v5->v0
    for v0 in range(n):
        for v1 in range(v0 + 1, n):
            if not (adj[v0] & (1 << v1)):
                continue
            # Count paths of length 4 from v1 to v0, internal vertices not in {v0, v1}
            # v2 in adj[v1] \ {v0, v1}
            # v3 in adj[v2] \ {v0, v1, v2}
            # v4 in adj[v3] \ {v0, v1, v2, v3}
            # v5 in adj[v4] \ {v0, v1, v2, v3, v4}, and v5 adjacent to v0
            count = 0
            for v2 in range(n):
                if v2 == v0 or v2 == v1:
                    continue
                if not (adj[v1] & (1 << v2)):
                    continue
                for v3 in range(n):
                    if v3 == v0 or v3 == v1 or v3 == v2:
                        continue
                    if not (adj[v2] & (1 << v3)):
                        continue
                    for v4 in range(n):
                        if v4 == v0 or v4 == v1 or v4 == v2 or v4 == v3:
                            continue
                        if not (adj[v3] & (1 << v4)):
                            continue
                        for v5 in range(n):
                            if v5 == v0 or v5 == v1 or v5 == v2 or v5 == v3 or v5 == v4:
                                continue
                            if not (adj[v4] & (1 << v5)):
                                continue
                            if adj[v5] & (1 << v0):
                                count += 1
            total += count
    return total

def _has_c6_fast(colors, n, color):
    """Check if a specific color class contains a 6-cycle using bitmasks."""
    adj = _build_adj_bitmasks(colors, n, color)
    # Early termination: just check if count > 0
    for v0 in range(n):
        for v1 in range(v0 + 1, n):
            if not (adj[v0] & (1 << v1)):
                continue
            found = False
            for v2 in range(n):
                if v2 == v0 or v2 == v1:
                    continue
                if not (adj[v1] & (1 << v2)):
                    continue
                for v3 in range(n):
                    if v3 == v0 or v3 == v1 or v3 == v2:
                        continue
                    if not (adj[v2] & (1 << v3)):
                        continue
                    for v4 in range(n):
                        if v4 == v0 or v4 == v1 or v4 == v2 or v4 == v3:
                            continue
                        if not (adj[v3] & (1 << v4)):
                            continue
                        for v5 in range(n):
                            if v5 == v0 or v5 == v1 or v5 == v2 or v5 == v3 or v5 == v4:
                                continue
                            if not (adj[v4] & (1 << v5)):
                                continue
                            if adj[v5] & (1 << v0):
                                return True
    return False

def _total_c6_count(colors, n):
    """Total number of 6-cycles across all 4 color classes."""
    total = 0
    for c in range(4):
        adj = _build_adj_bitmasks(colors, n, c)
        total += _count_c6s(adj, n)
    return total

def _is_valid(colors, n):
    """Check if coloring is valid (no C6 in any color)."""
    for c in range(4):
        if _has_c6_fast(colors, n, c):
            return False
    return True

def _random_coloring(n):
    return [random.randint(0, 3) for _ in range(n * (n - 1) // 2)]

def _local_search(colors, n, time_limit):
    """Local search to minimize total C6 count. Returns (colors, c6_count)."""
    start = time.time()
    current = colors[:]
    current_c6 = _total_c6_count(current, n)
    
    if current_c6 == 0:
        return current, 0
    
    best = current[:]
    best_c6 = current_c6
    
    num_edges = n * (n - 1) // 2
    
    # Simulated annealing
    temp = 10.0
    cooling = 0.999
    
    while time.time() < time_limit:
        # Pick a random edge
        ei = random.randint(0, num_edges - 1)
        old_color = current[ei]
        new_color = random.choice([c for c in range(4) if c != old_color])
        
        current[ei] = new_color
        new_c6 = _total_c6_count(current, n)
        
        delta = new_c6 - current_c6
        
        if delta <= 0 or random.random() < (2.71828 ** (-delta / max(temp, 0.01))):
            current_c6 = new_c6
            if new_c6 < best_c6:
                best_c6 = new_c6
                best = current[:]
                if best_c6 == 0:
                    break
        else:
            current[ei] = old_color
        
        temp *= cooling
    
    return best, best_c6

def _try_extend(colors_small, n_small, n_target, time_limit):
    """Try to extend a valid coloring of K_n_small to K_n_target."""
    # Start with the small coloring and random colors for new edges
    colors = colors_small[:]
    # Add edges involving new vertices
    for i in range(n_small, n_target):
        for j in range(i + 1, n_target):
            colors.append(random.randint(0, 3))
        for j in range(i):
            colors.append(random.randint(0, 3))
    
    # Reorder: we need edges in lexicographic order for all n_target vertices
    # Let's rebuild properly
    colors = [random.randint(0, 3) for _ in range(n_target * (n_target - 1) // 2)]
    # Copy over the small coloring
    idx_small = 0
    for i in range(n_small):
        for j in range(i + 1, n_small):
            colors[_edge_index(i, j, n_target)] = colors_small[idx_small]
            idx_small += 1
    
    # Local search
    result, c6 = _local_search(colors, n_target, time_limit)
    return result, c6

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Known small valid coloring for n=6 (trivial - C6 needs 6 vertices, 
    # but K6 colored with 4 colors might have C6... let's use a simple one)
    # For n=5, no C6 is possible (need 6 vertices), so any coloring works.
    # For n=6, we need to be careful.
    
    # Let's start with n=5 (trivially valid) and build up.
    best_n = 5
    best_colors = [0] * (5 * 4 // 2)  # All zeros, no C6 possible with < 6 vertices
    
    # Try to find valid colorings for increasing n
    for target_n in range(6, 20):
        if time.time() > deadline - 10:
            break
        
        found = False
        # Multiple attempts with different strategies
        for attempt in range(30):
            if time.time() > deadline - 10:
                break
            
            if target_n <= 12:
                # For small n, use random init + local search
                colors = _random_coloring(target_n)
            else:
                # For larger n, extend from best known
                if best_n >= 5:
                    colors, _ = _try_extend(best_colors, best_n, target_n, 
                                           min(2, deadline - 10 - (time.time() - start_time)))
                else:
                    colors = _random_coloring(target_n)
            
            # Local search
            time_left = deadline - 10 - (time.time() - start_time)
            if time_left < 1:
                break
            
            result, c6 = _local_search(colors, target_n, time_left)
            
            if c6 == 0:
                best_n = target_n
                best_colors = result[:]
                found = True
                break
        
        if not found:
            # If we couldn't find for this n, try next (maybe we can skip)
            continue
    
    # Verify the best result
    if best_n >= 6:
        if not _is_valid(best_colors, best_n):
            # Fallback to a known valid small coloring
            best_n = 5
            best_colors = [0] * (5 * 4 // 2)
    
    return {'n': best_n, 'colors': best_colors}
