import time

# Precompute all nonconstant 4-term progressions (a, a+d, a+2d, a+3d), 1-indexed a,d
# Convert to 0-indexed internally.
_ALL_APS = []
for a in range(1, 125):  # a+3d <= 128 => a <= 128-3
    max_d = (128 - a) // 3
    for d in range(1, max_d + 1):
        _ALL_APS.append((a-1, a+d-1, a+2*d-1, a+3*d-1))

# Precompute for each position, which progressions it participates in
pos_in_aps = [[] for _ in range(128)]
for idx, ap in enumerate(_ALL_APS):
    for i in ap:
        pos_in_aps[i].append(idx)

def construct():
    start = time.time()
    DEADLINE = 85.0  # Safety margin before 90s limit

    n = 128
    
    # Try increasing number of colors
    for k_colors in range(3, 11):
        if time.time() > DEADLINE:
            break
            
        # colors: -1 for unassigned
        colors = [-1] * n
        # domains: list of sets of allowed colors
        domains = [set(range(k_colors)) for _ in range(n)]
        
        solution_found = False
        node_count = 0

        def is_valid_assignment():
            for ap in _ALL_APS:
                if len({colors[i] for i in ap}) < 3:
                    return False
            return True

        def backtrack_v2():
            nonlocal solution_found, node_count
            if time.time() > DEADLINE:
                return False
            if solution_found:
                return True
            
            node_count += 1
            if node_count % 1000 == 0 and time.time() > DEADLINE:
                return False

            # Find unassigned variable with MRV
            min_size = k_colors + 1
            var_idx = -1
            for i in range(n):
                if colors[i] == -1:
                    size = len(domains[i])
                    if size < min_size:
                        min_size = size
                        var_idx = i
                        if size <= 1:
                            break
            
            if var_idx == -1:
                if is_valid_assignment():
                    solution_found = True
                    return True
                return False
            
            original_domain = domains[var_idx].copy()
            
            for c in sorted(original_domain):
                colors[var_idx] = c
                domains[var_idx] = {c}
                
                consistent = True
                changed = []
                
                for ap_idx in pos_in_aps[var_idx]:
                    ap = _ALL_APS[ap_idx]
                    assigned_colors = set()
                    unassigned_idxs = []
                    for i in ap:
                        if colors[i] != -1:
                            assigned_colors.add(colors[i])
                        else:
                            unassigned_idxs.append(i)
                    
                    if not unassigned_idxs:
                        if len(assigned_colors) < 3:
                            consistent = False
                            break
                    elif len(unassigned_idxs) == 1:
                        u = unassigned_idxs[0]
                        if len(assigned_colors) == 2:
                            new_domain = domains[u] - assigned_colors
                            if not new_domain:
                                consistent = False
                                break
                            if new_domain != domains[u]:
                                changed.append((u, domains[u]))
                                domains[u] = new_domain
                        elif len(assigned_colors) == 1:
                            consistent = False
                            break
                    elif len(unassigned_idxs) == 2:
                        u1, u2 = unassigned_idxs
                        if len(assigned_colors) == 2:
                            if domains[u1].issubset(assigned_colors) and domains[u2].issubset(assigned_colors):
                                consistent = False
                                break
                        elif len(assigned_colors) == 1:
                            new_d1 = domains[u1] - assigned_colors
                            new_d2 = domains[u2] - assigned_colors
                            if not new_d1 or not new_d2:
                                consistent = False
                                break
                            if new_d1 != domains[u1]:
                                changed.append((u1, domains[u1]))
                                domains[u1] = new_d1
                            if new_d2 != domains[u2]:
                                changed.append((u2, domains[u2]))
                                domains[u2] = new_d2
                            if new_d1 == new_d2 and len(new_d1) == 1:
                                consistent = False
                                break

                if consistent:
                    if backtrack_v2():
                        return True
                
                # Rollback
                colors[var_idx] = -1
                domains[var_idx] = original_domain
                for u, old_dom in changed:
                    domains[u] = old_dom
                    
            return False

        if backtrack_v2():
            # Found solution with k_colors
            return colors

    # Fallback: return a coloring that is guaranteed valid (unique colors)
    # This yields score 1.0
    return list(range(128))
