import time
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Leave 10s buffer for Python overhead/evaluator
    
    n = 128
    
    # Precompute all non-constant 4-term arithmetic progressions
    # A 4-AP is defined by start i and difference d > 0 such that i+3d <= n
    # Positions are 1-indexed in problem statement, 0-indexed in Python
    aps = []
    for i in range(n):
        for d in range(1, (n - i) // 3 + 1):
            p1 = i
            p2 = i + d
            p3 = i + 2 * d
            p4 = i + 3 * d
            if p4 < n:
                aps.append((p1, p2, p3, p4))
    
    # For each position, store the indices of APs it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for p in ap:
            pos_to_aps[p].append(idx)

    # We try to find a coloring with k colors, starting from k=3
    # The problem asks to minimize the number of labels actually used.
    # The minimum is at least 3 because a 2-coloring would imply some 4-AP uses <=2 colors
    # (actually, it's known that van der Waerden number W(2,4) is 76, so for n=128, 
    # 2 colors are insufficient to avoid monochromatic 4-APs, but we need to avoid 
    # using <=2 colors in ANY 4-AP. This is a stronger condition.
    # If we use 2 colors, can we avoid having a 4-AP with only 1 or 2 colors?
    # A 4-AP with 2 colors is allowed if it has exactly 2 colors? No, "at least 3 distinct colors".
    # So a 4-AP with colors {0, 1} is INVALID.
    # Thus, every 4-AP must have 3 or 4 distinct colors.
    # This implies we cannot have any 4-AP that is monochromatic or bichromatic.
    # This is a very strong constraint.
    
    # Let's try k=3 first.
    # If k=3 works, we return the solution.
    # If not, try k=4, etc.
    
    def solve_with_k_colors(k):
        # domains[pos] is a set of possible colors
        domains = [set(range(k)) for _ in range(n)]
        assignment = [-1] * n
        
        # MRV Heuristic: pick variable with smallest domain
        # We need an efficient way to get the variable with min domain size.
        # For n=128, a linear scan is fine.
        
        def find_var():
            min_size = k + 1
            min_var = -1
            for i in range(n):
                if assignment[i] == -1:
                    size = len(domains[i])
                    if size < min_size:
                        min_size = size
                        min_var = i
            return min_var

        def propagate(pos, color):
            # Assign color to pos, update domains of neighbors
            assignment[pos] = color
            domains[pos] = {color}
            
            # Check all APs involving pos
            for ap_idx in pos_to_aps[pos]:
                p1, p2, p3, p4 = aps[ap_idx]
                # Count colors used in this AP so far
                # We need to check if this AP becomes invalid.
                # An AP is invalid if it uses <= 2 colors.
                # If the AP is fully assigned and uses <= 2 colors, it's a conflict.
                # If the AP is not fully assigned, we can prune domains of unassigned vars.
                
                assigned_colors = set()
                unassigned_vars = []
                
                for p in (p1, p2, p3, p4):
                    if assignment[p] != -1:
                        assigned_colors.add(assignment[p])
                    else:
                        unassigned_vars.append(p)
                        
                # If we already have > 2 colors, no constraint on unassigned vars from this AP
                if len(assigned_colors) > 2:
                    continue
                    
                # If we have exactly 2 colors assigned, the remaining vars MUST take a 3rd color
                # If we have 1 color assigned, the remaining vars must not all be that color?
                # Wait, the condition is "at least 3 distinct colors".
                # So if assigned colors are {c1}, we need at least 2 more distinct colors from unassigned vars.
                # If assigned colors are {c1, c2}, we need at least 1 more distinct color from unassigned vars.
                
                if len(assigned_colors) == 2:
                    # The unassigned vars must contain at least one color NOT in assigned_colors
                    # This means: It is NOT the case that all unassigned vars are restricted to assigned_colors.
                    # Specifically, for each unassigned var, if we restrict its domain to assigned_colors, 
                    # we might still have a valid solution if another var takes a different color.
                    # However, if there is only 1 unassigned var, it MUST be a color not in assigned_colors.
                    # If there are multiple, it's a global constraint. 
                    # Simple propagation: Remove from domain of each unassigned var any color 
                    # that would make it impossible to satisfy the constraint?
                    # Actually, if len(unassigned_vars) == 1, that var must be in Colors \ assigned_colors.
                    if len(unassigned_vars) == 1:
                        v = unassigned_vars[0]
                        # v must be a color not in assigned_colors
                        new_domain = domains[v] - assigned_colors
                        if not new_domain:
                            return False
                        domains[v] = new_domain
                    else:
                        # We can't prune individual domains easily without complex logic, 
                        # but we can note that if all unassigned vars are forced to be in assigned_colors, 
                        # it's a fail. This is a "nogood" check.
                        # For now, skip complex propagation for len > 1.
                        pass
                        
                elif len(assigned_colors) == 1:
                    c = list(assigned_colors)[0]
                    # We need at least 2 distinct colors from unassigned vars that are != c?
                    # No, we need total distinct colors >= 3.
                    # So we need at least 2 colors from (Universe \ {c}) among the unassigned vars.
                    # If len(unassigned_vars) == 1, impossible (can only add 1 color).
                    if len(unassigned_vars) == 1:
                        return False
                    # If len(unassigned_vars) == 2, they must be distinct and != c.
                    if len(unassigned_vars) == 2:
                        v1, v2 = unassigned_vars
                        # v1 != c, v2 != c, v1 != v2
                        # Prune c from domains
                        if c in domains[v1]:
                            domains[v1].remove(c)
                        if c in domains[v2]:
                            domains[v2].remove(c)
                        if not domains[v1] or not domains[v2]:
                            return False
                        # v1 and v2 must be different. This is a binary constraint.
                        # We can't fully propagate this without arc consistency, 
                        # but we can check if their domains are singletons and equal.
                        if len(domains[v1]) == 1 and len(domains[v2]) == 1:
                            if domains[v1] == domains[v2]:
                                return False
                    else:
                        # More than 2 unassigned. Just remove c? 
                        # No, we can have c in one of them, as long as others provide 2 new colors.
                        # So we cannot simply remove c.
                        pass
                        
                elif len(assigned_colors) == 0:
                    # No assigned, no propagation needed yet.
                    pass

            return True

        def backtrack():
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
                
            var = find_var()
            if var == -1:
                # All assigned
                return True
            
            # Try each color in domain
            # Order colors by least constraining value (LCV) might help, but let's start simple
            for color in sorted(domains[var]):
                # Save state
                saved_domains = [set(d) for d in domains]
                saved_assignment = assignment[:]
                
                if propagate(var, color):
                    if backtrack():
                        return True
                
                # Restore state
                domains[:] = saved_domains
                assignment[:] = saved_assignment
                
            return False

        return backtrack(), assignment

    # Try increasing number of colors
    # Theoretical minimum is 3.
    # Let's start with 3.
    for k in range(3, 10):
        if time.time() > deadline:
            break
        success, assignment = solve_with_k_colors(k)
        if success:
            # Verify solution
            valid = True
            for ap in aps:
                colors_in_ap = set(assignment[p] for p in ap)
                if len(colors_in_ap) < 3:
                    valid = False
                    break
            if valid:
                # Return labels 0..127
                return assignment
            else:
                # Should not happen if propagate is correct, but check anyway
                continue
        
    # Fallback: Greedy (should not be reached if solver is good)
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 for d in range(1, (x-1)//3+1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) for d in range(1, (x-1)//2+1))):
                colors.append(color)
                break
    return colors

# EVOLVE-BLOCK-END
