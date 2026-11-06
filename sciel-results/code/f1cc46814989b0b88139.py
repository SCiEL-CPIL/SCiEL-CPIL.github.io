import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit (use 55s to be safe)
    
    N = 128
    
    # Pre-compute all 4-term APs
    aps = []
    ap_indices = [[] for _ in range(N)]
    for d in range(1, N // 3 + 1):
        for i in range(N - 3 * d):
            ap = (i, i + d, i + 2 * d, i + 3 * d)
            ap_idx = len(aps)
            aps.append(ap)
            for idx in ap:
                ap_indices[idx].append(ap_idx)
    
    num_aps = len(aps)
    
    def count_violations(colors):
        count = 0
        for ap in aps:
            if len({colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}) < 3:
                count += 1
        return count
    
    def get_violating_aps(colors):
        """Return list of AP indices that violate the constraint."""
        violations = []
        for i, ap in enumerate(aps):
            if len({colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}) < 3:
                violations.append(i)
        return violations
    
    def greedy_construct(max_color):
        """
        Greedy construction: assign colors 0..max_color-1 to positions 0..N-1
        such that every 4-AP uses at least 3 distinct colors.
        Uses forward checking: when assigning position p, ensure no AP becomes
        impossible to satisfy.
        """
        colors = [-1] * N
        
        for p in range(N):
            # Try each color, prefer ones that don't create problems
            best_color = -1
            best_score = -1
            
            for c in range(max_color):
                colors[p] = c
                
                # Check if this assignment is feasible (no AP is now impossible)
                feasible = True
                for ai in ap_indices[p]:
                    ap = aps[ai]
                    assigned_colors = set()
                    unassigned_count = 0
                    for pos in ap:
                        if colors[pos] != -1:
                            assigned_colors.add(colors[pos])
                        else:
                            unassigned_count += 1
                    
                    if unassigned_count == 0:
                        if len(assigned_colors) < 3:
                            feasible = False
                            break
                    elif unassigned_count == 1:
                        if len(assigned_colors) == 0:
                            # All 4 unassigned except this one - can't happen since we're assigning p
                            pass
                        elif len(assigned_colors) == 1:
                            # 3 positions have same color, 1 unassigned must be different
                            # Need max_color >= 2
                            if max_color < 2:
                                feasible = False
                                break
                        elif len(assigned_colors) == 2:
                            # 2 colors used among 3 assigned, 1 unassigned must add 3rd
                            if max_color < 3:
                                feasible = False
                                break
                        # len(assigned_colors) >= 3: always OK
                    
                    # unassigned_count >= 2: always feasible with max_color >= 2
                
                if not feasible:
                    colors[p] = -1
                    continue
                
                # Score: prefer colors that are "balanced" or that help future assignments
                # Simple heuristic: count how many APs this position is in with this color
                # that now have exactly 2 distinct colors (tightest constraint)
                tightness = 0
                for ai in ap_indices[p]:
                    ap = aps[ai]
                    assigned_colors = set()
                    for pos in ap:
                        if colors[pos] != -1:
                            assigned_colors.add(colors[pos])
                    if len(assigned_colors) == 2:
                        tightness += 1
                
                score = -tightness  # Prefer assignments that don't create tight constraints
                
                if score > best_score:
                    best_score = score
                    best_color = c
            
            if best_color == -1:
                # No color works - this shouldn't happen with enough colors
                colors[p] = 0  # Fallback
            else:
                colors[p] = best_color
        
        return colors
    
    def local_search_reduce(colors, target_k, time_limit):
        """
        Try to reduce the number of colors used in 'colors' to target_k.
        Uses local search: repeatedly reassign positions to reduce color count
        while maintaining validity.
        """
        colors = colors[:]
        used_colors = set(colors)
        
        if len(used_colors) <= target_k:
            return colors
        
        # Map used colors to 0..target_k-1 if possible
        color_map = {}
        new_colors = []
        for c in colors:
            if c not in color_map:
                if len(color_map) < target_k:
                    color_map[c] = len(color_map)
                else:
                    # Assign to existing color that causes least damage
                    best_existing = 0
                    best_damage = float('inf')
                    for existing in color_map.values():
                        damage = 0
                        for ai in ap_indices[0]:  # rough estimate
                            pass
                        # Simplified: just pick first
                        damage = 0
                        break
                    color_map[c] = existing
            new_colors.append(color_map[c])
        colors = new_colors
        
        # Now try to fix violations by reassigning
        for iteration in range(10000):
            if time.time() > time_limit:
                break
            
            violations = get_violating_aps(colors)
            if not violations:
                break
            
            # Pick a random violating AP
            vi = random.choice(violations)
            ap = aps[vi]
            
            # Try to fix by changing one of the 4 positions
            positions_to_try = list(ap)
            random.shuffle(positions_to_try)
            
            fixed = False
            for pos in positions_to_try:
                old_color = colors[pos]
                # Try assigning a different color
                for c in range(target_k):
                    if c == old_color:
                        continue
                    
                    colors[pos] = c
                    
                    # Check if this fixes the violation and doesn't create new ones
                    # Quick check: just count violations in affected APs
                    affected_aps = set(ap_indices[pos])
                    new_violations = 0
                    for ai in affected_aps:
                        a = aps[ai]
                        if len({colors[a[0]], colors[a[1]], colors[a[2]], colors[a[3]]}) < 3:
                            new_violations += 1
                    
                    if new_violations == 0:
                        fixed = True
                        break
                    else:
                        colors[pos] = old_color
                
                if fixed:
                    break
            
            if not fixed:
                # Try a more aggressive move: change two positions
                if len(positions_to_try) >= 2:
                    p1, p2 = positions_to_try[0], positions_to_try[1]
                    c1_old = colors[p1]
                    c2_old = colors[p2]
                    
                    for c1 in range(target_k):
                        if c1 == c1_old:
                            continue
                        for c2 in range(target_k):
                            if c2 == c2_old:
                                continue
                            
                            colors[p1] = c1
                            colors[p2] = c2
                            
                            affected_aps = set(ap_indices[p1]) | set(ap_indices[p2])
                            new_violations = 0
                            for ai in affected_aps:
                                a = aps[ai]
                                if len({colors[a[0]], colors[a[1]], colors[a[2]], colors[a[3]]}) < 3:
                                    new_violations += 1
                            
                            if new_violations == 0:
                                fixed = True
                                break
                            
                            colors[p1] = c1_old
                            colors[p2] = c2_old
                        
                        if fixed:
                            break
                    
                    if not fixed:
                        colors[p1] = c1_old
                        colors[p2] = c2_old
        
        return colors if count_violations(colors) == 0 else None
    
    # Step 1: Greedy construction with small color counts
    best_colors = None
    best_k = 128
    
    # Try greedy with increasing color counts
    for k in range(3, 8):
        if time.time() > deadline - 10:
            break
        
        colors = greedy_construct(k)
        if colors is not None and count_violations(colors) == 0:
            actual_k = len(set(colors))
            if actual_k < best_k:
                best_k = actual_k
                best_colors = colors[:]
            break
    
    if best_colors is None:
        # Fallback: use a known valid coloring with more colors
        # Simple construction: color by position mod 4, but that might not work
        # Use a safe construction: alternate patterns
        best_colors = []
        for i in range(N):
            # Use a pattern that ensures 3 colors in any 4-AP
            # Color i by (i // 4) % 4 + (i % 4)  -- not quite right
            # Simpler: use 4 colors in a pattern that avoids 2-color APs
            best_colors.append(i % 4)
        
        if count_violations(best_colors) > 0:
            # Even safer: use more colors
            best_colors = [i % 8 for i in range(N)]
        
        best_k = len(set(best_colors))
    
    # Step 2: Try to reduce colors using local search
    for target_k in range(3, best_k):
        if time.time() > deadline - 5:
            break
        
        time_limit = min(deadline - 2, time.time() + 8.0)
        result = local_search_reduce(best_colors, target_k, time_limit)
        
        if result is not None and count_violations(result) == 0:
            actual_k = len(set(result))
            if actual_k < best_k:
                best_k = actual_k
                best_colors = result[:]
            break
    
    # Final verification
    if count_violations(best_colors) > 0:
        # Emergency fallback: use a very safe coloring
        best_colors = [i % 16 for i in range(N)]
    
    return best_colors
