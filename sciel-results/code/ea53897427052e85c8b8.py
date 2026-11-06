import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin under 150s

    n = 16
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: idx for idx, e in enumerate(edges)}

    # Precompute K4s and their edge indices
    k4s = list(combinations(range(n), 4))
    k4_edge_indices = []
    for quad in k4s:
        e_indices = []
        for i in range(4):
            for j in range(i + 1, 4):
                e_indices.append(edge_index[(quad[i], quad[j])])
        k4_edge_indices.append(e_indices)
    num_k4s = len(k4s)

    # Map each edge to the K4s it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_indices in enumerate(k4_edge_indices):
        for e_idx in e_indices:
            edge_to_k4s[e_idx].append(k4_idx)

    # Precompute adjacency in conflict graph: two edges conflict if they share a vertex
    # For independent set (color class), we need edges that do NOT share a vertex
    # So conflict[e1][e2] = True if they share a vertex
    # We'll represent non-conflict (compatible) as adjacency for independent set
    # compatible[e1] = set of edges that don't share a vertex with e1
    compatible = [set() for _ in range(num_edges)]
    for i in range(num_edges):
        v1, v2 = edges[i]
        for j in range(i + 1, num_edges):
            v3, v4 = edges[j]
            # They are compatible if no shared vertex
            if v1 != v3 and v1 != v4 and v2 != v3 and v2 != v4:
                compatible[i].add(j)
                compatible[j].add(i)

    # Greedy maximal independent set (color class) construction
    def greedy_color_class(available, rng):
        """
        Build a maximal independent set from 'available' (set of edge indices).
        Use MRV heuristic: pick the edge with fewest compatible neighbors in available.
        Returns the set of edges in this color class.
        """
        if not available:
            return set()
        
        # Start with a random edge
        current = rng.choice(list(available))
        color_class = {current}
        available.discard(current)
        
        # Iteratively add edges
        while True:
            # Find edges in available that are compatible with all in color_class
            candidates = set()
            for e in available:
                # Check if e is compatible with all edges in color_class
                is_compatible = True
                for ce in color_class:
                    if e not in compatible[ce]:
                        is_compatible = False
                        break
                if is_compatible:
                    candidates.add(e)
            
            if not candidates:
                break
            
            # MRV: pick the one with fewest compatible neighbors in available
            best_e = None
            best_degree = float('inf')
            for e in candidates:
                deg = len(compatible[e] & available)
                if deg < best_degree:
                    best_degree = deg
                    best_e = e
            
            color_class.add(best_e)
            available.discard(best_e)
        
        return color_class

    # Verify coloring: check every K4 has >= 5 distinct colors
    def verify(colors):
        for k4_idx in range(num_k4s):
            e_indices = k4_edge_indices[k4_idx]
            distinct = set(colors[e] for e in e_indices)
            if len(distinct) < 5:
                return False
        return True

    # Count distinct colors used
    def count_colors(colors):
        return len(set(colors))

    # Local search to fix violations
    def local_search_fix(colors, K, time_limit, rng):
        """
        Fix violations by reassigning colors to edges in violating K4s.
        Returns (colors, num_violations)
        """
        num_violations = 0
        for k4_idx in range(num_k4s):
            e_indices = k4_edge_indices[k4_idx]
            distinct = set(colors[e] for e in e_indices)
            if len(distinct) < 5:
                num_violations += 1
        
        if num_violations == 0:
            return colors, 0
        
        # Build list of violating K4s
        violating = []
        for k4_idx in range(num_k4s):
            e_indices = k4_edge_indices[k4_idx]
            distinct = set(colors[e] for e in e_indices)
            if len(distinct) < 5:
                violating.append(k4_idx)
        
        no_improve = 0
        max_no_improve = 2000
        
        while time.time() < time_limit and no_improve < max_no_improve:
            if not violating:
                break
            
            # Pick a random violating K4
            k4_idx = rng.choice(violating)
            e_indices = k4_edge_indices[k4_idx]
            
            # Try to fix by changing one edge in this K4
            fixed = False
            for e_idx in e_indices:
                old_color = colors[e_idx]
                # Try all K colors
                for new_color in range(K):
                    if new_color == old_color:
                        continue
                    colors[e_idx] = new_color
                    
                    # Check if this K4 is now fixed
                    distinct = set(colors[e] for e in e_indices)
                    if len(distinct) >= 5:
                        # Check if we broke any other K4
                        broke = False
                        for other_k4 in edge_to_k4s[e_idx]:
                            if other_k4 == k4_idx:
                                continue
                            oe = k4_edge_indices[other_k4]
                            od = set(colors[e] for e in oe)
                            if len(od) < 5:
                                broke = True
                                break
                        
                        if not broke:
                            # Update violating list
                            if k4_idx in violating:
                                violating.remove(k4_idx)
                            fixed = True
                            no_improve = 0
                            break
                    else:
                        colors[e_idx] = old_color
                
                if fixed:
                    break
            
            if not fixed:
                no_improve += 1
                # If stuck, try a random change
                e_idx = rng.randint(0, num_edges - 1)
                colors[e_idx] = rng.randint(0, K - 1)
        
        # Final count
        num_violations = 0
        violating = []
        for k4_idx in range(num_k4s):
            e_indices = k4_edge_indices[k4_idx]
            distinct = set(colors[e] for e in e_indices)
            if len(distinct) < 5:
                num_violations += 1
                violating.append(k4_idx)
        
        return colors, num_violations

    # Main: try greedy construction for K = 13, 14, 15
    best_result = None
    
    for K in [13, 14, 15]:
        if time.time() > deadline - 10:
            break
        
        time_limit_for_this_K = min(40.0, deadline - time.time() - 5)
        
        # Try multiple random seeds for greedy
        for seed in [42, 123, 456, 789, 1024, 2048]:
            if time.time() > deadline - 10:
                break
            
            rng = random.Random(seed)
            available = set(range(num_edges))
            colors = [-1] * num_edges
            color_count = 0
            
            while available and color_count < K:
                color_class = greedy_color_class(available, rng)
                if not color_class:
                    break
                for e in color_class:
                    colors[e] = color_count
                    available.discard(e)
                color_count += 1
            
            # If we still have edges left, assign them to existing colors (may cause violations)
            if available:
                for e in available:
                    colors[e] = rng.randint(0, K - 1)
            
            # Verify
            if verify(colors):
                best_result = colors
                break
            
            # Try local search fix
            if time.time() < deadline - 10:
                fixed_colors, num_viol = local_search_fix(colors[:], K, min(15.0, deadline - time.time() - 5), rng)
                if num_viol == 0:
                    best_result = fixed_colors
                    break
        
        if best_result is not None:
            break
    
    # If greedy+local search didn't work, try a more aggressive approach:
    # Start from a known good structure and optimize
    if best_result is None and time.time() < deadline - 10:
        # Try structured initialization: color (i,j) = (i * a + j * b) % K for various a,b
        for K in [13, 14, 15]:
            if time.time() > deadline - 10:
                break
            for a in range(1, K):
                for b in range(1, K):
                    if a == b:
                        continue
                    colors = [0] * num_edges
                    for idx, (i, j) in enumerate(edges):
                        colors[idx] = (i * a + j * b) % K
                    if verify(colors):
                        best_result = colors
                        break
                if best_result is not None:
                    break
            if best_result is not None:
                break
    
    # Final fallback: use the seed program's approach with a different seed
    if best_result is None and time.time() < deadline - 5:
        # Simple modular coloring that's likely to have many distinct colors per K4
        # (i+j) % 17 uses 17 colors but let's try to compress
        for K in [13, 14, 15]:
            colors = [(i + j) % K for i in range(16) for j in range(i + 1, 16)]
            if verify(colors):
                best_result = colors
                break
    
    # Absolute last resort: use 17 colors (known to work from (i+j)%17)
    if best_result is None:
        best_result = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    return best_result
