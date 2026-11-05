import time
import random
import itertools

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = 16
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {(i, j): k for k, (i, j) in enumerate(edges)}
    
    # Precompute K4s and their edges
    k4s = []
    k4_edges = []
    edge_to_k4s = [[] for _ in range(num_edges)]
    
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        e_idx = [
            edge_index[(a, b)], edge_index[(a, c)], edge_index[(a, d)],
            edge_index[(b, c)], edge_index[(b, d)], edge_index[(c, d)]
        ]
        k4s.append(quad)
        k4_edges.append(e_idx)
        for e in e_idx:
            edge_to_k4s[e].append(len(k4s) - 1)
    
    num_k4s = len(k4s)
    
    def count_distinct_colors(k4_idx, colors):
        """Count distinct colors in a K4, treating -1 as wildcard."""
        e_indices = k4_edges[k4_idx]
        color_set = set()
        for e in e_indices:
            c = colors[e]
            if c != -1:
                color_set.add(c)
        return len(color_set)
    
    def is_fully_colored(k4_idx, colors):
        e_indices = k4_edges[k4_idx]
        for e in e_indices:
            if colors[e] == -1:
                return False
        return True
    
    def check_k4_valid(k4_idx, colors):
        """Check if a fully-colored K4 has at least 5 distinct colors."""
        e_indices = k4_edges[k4_idx]
        color_set = set(colors[e] for e in e_indices)
        return len(color_set) >= 5
    
    def get_violations(colors):
        """Get list of K4 indices that violate the constraint (fully colored with < 5 distinct)."""
        violations = []
        for k4_idx in range(num_k4s):
            if is_fully_colored(k4_idx, colors):
                if not check_k4_valid(k4_idx, colors):
                    violations.append(k4_idx)
        return violations
    
    def try_assign_color(colors, e_idx, c, target_C):
        """Try assigning color c to edge e_idx. Returns True if no immediate violation."""
        for k4_idx in edge_to_k4s[e_idx]:
            e_indices = k4_edges[k4_idx]
            # Check if this K4 would be fully colored after assignment
            all_colored = True
            for e in e_indices:
                if e == e_idx:
                    continue
                if colors[e] == -1:
                    all_colored = False
                    break
            
            if all_colored:
                # Build the set of colors for this K4
                color_set = set()
                for e in e_indices:
                    if e == e_idx:
                        color_set.add(c)
                    else:
                        color_set.add(colors[e])
                if len(color_set) < 5:
                    return False
        return True
    
    def greedy_assign(colors, edge_order, target_C, rng):
        """Greedy assignment with random color ordering."""
        for e_idx in edge_order:
            color_options = list(range(target_C))
            rng.shuffle(color_options)
            
            assigned = False
            for c in color_options:
                if try_assign_color(colors, e_idx, c, target_C):
                    colors[e_idx] = c
                    assigned = True
                    break
            
            if not assigned:
                return False
        return True
    
    def local_search(colors, target_C, rng, max_iters=1000):
        """Local search to fix violations by reassigning colors."""
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                return False
            
            violations = get_violations(colors)
            if not violations:
                return True
            
            # Pick a random violation
            k4_idx = rng.choice(violations)
            e_indices = k4_edges[k4_idx]
            
            # Count distinct colors currently
            current_colors = [colors[e] for e in e_indices]
            current_distinct = len(set(current_colors))
            
            # We need at least 5 distinct. Currently have current_distinct.
            # Need to add (5 - current_distinct) new colors.
            # Strategy: pick edges in this K4 and try to change their color to a new one.
            
            # Find edges that can be changed to add new distinct colors
            # An edge can be changed to color c if:
            # 1. c is not already in the K4 (to add a new distinct color)
            # 2. Changing doesn't create new violations in other K4s containing this edge
            
            # Try to fix by changing one edge at a time
            edges_to_try = list(e_indices)
            rng.shuffle(edges_to_try)
            
            for e_idx in edges_to_try:
                current_c = colors[e_idx]
                # Try colors not in current K4's color set
                current_color_set = set(current_colors)
                new_color_options = [c for c in range(target_C) if c not in current_color_set and c != current_c]
                rng.shuffle(new_color_options)
                
                for c in new_color_options:
                    # Check if this change is safe
                    safe = True
                    for other_k4 in edge_to_k4s[e_idx]:
                        if other_k4 == k4_idx:
                            continue
                        oe_indices = k4_edges[other_k4]
                        # Check if this K4 is fully colored
                        all_c = True
                        for e in oe_indices:
                            if e == e_idx:
                                continue
                            if colors[e] == -1:
                                all_c = False
                                break
                        if all_c:
                            # Build color set with new color
                            cs = set()
                            for e in oe_indices:
                                if e == e_idx:
                                    cs.add(c)
                                else:
                                    cs.add(colors[e])
                            if len(cs) < 5:
                                safe = False
                                break
                    
                    if safe:
                        colors[e_idx] = c
                        # Check if this fixed the violation or created new ones
                        # Just continue the local search loop
                        break
        return False
    
    best_result = None
    best_colors_used = 100
    
    # Baseline fallback: use a known valid construction
    # The modular construction: color(i,j) = (i + j) % p for appropriate p
    # For n=16, using p=17 gives colors 0..16, so 17 colors.
    # We need to verify this is valid.
    baseline = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    # Verify baseline
    valid = True
    for k4_idx in range(num_k4s):
        e_indices = k4_edges[k4_idx]
        color_set = set(baseline[e] for e in e_indices)
        if len(color_set) < 5:
            valid = False
            break
    
    if valid:
        best_result = baseline[:]
        best_colors_used = len(set(baseline))
    
    rng = random.Random(42)
    
    # Try to find better colorings
    for target_C in [13, 14, 15]:
        if time.time() > deadline - 15:
            break
        
        if target_C >= best_colors_used:
            continue
        
        attempts = 0
        max_attempts = 500 if target_C == 13 else 300
        
        while attempts < max_attempts:
            if time.time() > deadline - 10:
                break
            
            attempts += 1
            colors = [-1] * num_edges
            edge_order = list(range(num_edges))
            rng.shuffle(edge_order)
            
            if greedy_assign(colors, edge_order, target_C, rng):
                # Check full validity
                violations = get_violations(colors)
                if not violations:
                    used = set(colors)
                    num_used = len(used)
                    if num_used < best_colors_used:
                        best_colors_used = num_used
                        best_result = colors[:]
                    if num_used <= 13:
                        return best_result
                else:
                    # Try local search to fix
                    if local_search(colors, target_C, rng, max_iters=200):
                        violations = get_violations(colors)
                        if not violations:
                            used = set(colors)
                            num_used = len(used)
                            if num_used < best_colors_used:
                                best_colors_used = num_used
                                best_result = colors[:]
                            if num_used <= 13:
                                return best_result
    
    if best_result is not None:
        return best_result
    else:
        return baseline
