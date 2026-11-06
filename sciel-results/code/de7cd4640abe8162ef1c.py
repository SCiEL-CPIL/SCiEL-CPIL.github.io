import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Try n=26 first with structured initialization, then fall back to lower n
    for n in [26, 25, 24]:
        if time.time() > deadline - 20:
            break
        result = solve_n(n, time_limit=18)
        if result is not None:
            return result
    
    # Final fallback: n=11
    return {'n': 11, 'colors': [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]}

def solve_n(n, time_limit=18):
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
    
    # Structured initialization using algebraic construction
    # For n=26: vertices as (x, y) where x in Z_13, y in {0,1}
    # For other n: use random
    if n == 26:
        colors = structured_init_26()
    elif n == 25:
        colors = structured_init_25()
    else:
        colors = [random.randint(0, 4) for _ in range(num_edges)]
    
    # Build adjacency masks: neighbors[c][v] = bitmask of neighbors of v in color c
    neighbors = [[0] * n for _ in range(5)]
    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        neighbors[c][i] |= (1 << j)
        neighbors[c][j] |= (1 << i)
    
    def calc_violations():
        violations = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = neighbors[c][u] & neighbors[c][v]
                    k = common.bit_count()
                    if k >= 2:
                        violations += k * (k - 1) // 2
        return violations
    
    current_violations = calc_violations()
    if current_violations == 0:
        return {'n': n, 'colors': colors}
    
    best_violations = current_violations
    best_colors = colors[:]
    
    start = time.time()
    temperature = 2.0
    cooling_rate = 0.998
    min_temperature = 0.1
    
    # Fast local search with delta evaluation
    while time.time() - start < time_limit:
        e_idx = random.randint(0, num_edges - 1)
        i, j = edges[e_idx]
        old_color = colors[e_idx]
        
        new_color = random.randint(0, 4)
        if new_color == old_color:
            continue
        
        # Calculate delta: change in violations from this edge swap
        delta = calc_delta(i, j, old_color, new_color, neighbors)
        
        if delta < 0 or random.random() < pow(2.71828, -delta / temperature):
            # Apply change
            colors[e_idx] = new_color
            neighbors[old_color][i] &= ~(1 << j)
            neighbors[old_color][j] &= ~(1 << i)
            neighbors[new_color][i] |= (1 << j)
            neighbors[new_color][j] |= (1 << i)
            current_violations += delta
            
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors[:]
                if best_violations == 0:
                    return {'n': n, 'colors': colors}
        else:
            # Revert (no-op since we didn't apply)
            pass
        
        temperature = max(min_temperature, temperature * cooling_rate)
    
    # Final verification
    if best_violations == 0:
        return {'n': n, 'colors': best_colors}
    return None

def calc_delta(i, j, old_c, new_c, neighbors):
    """Calculate change in violation count when edge (i,j) changes from old_c to new_c."""
    delta = 0
    
    # For each color c, consider pairs (u,v) that might gain/lose a common neighbor
    # A violation exists for (u,v) in color c if |N_c(u) ∩ N_c(v)| >= 2
    # Changing edge (i,j) affects:
    # - In old_c: i and j lose each other as neighbors
    # - In new_c: i and j gain each other as neighbors
    
    # For old_c: removing j from N_old_c(i) and i from N_old_c(j)
    # This can only decrease violations (or keep same)
    # For new_c: adding j to N_new_c(i) and i to N_new_c(j)
    # This can only increase violations (or keep same)
    
    # Check violations that involve i or j in old_c (decrease)
    for c in range(5):
        if c == old_c:
            # Removing edge (i,j) from color c
            # For each vertex v != i,j, check if (v,i) or (v,j) had exactly 2 common neighbors
            # and one of them was the other endpoint
            for v in range(len(neighbors[0])):
                if v == i or v == j:
                    continue
                # Before: common neighbors of v and i in color c included j if j was neighbor of both
                # After: j is no longer neighbor of i in color c
                # So if j was in N_c(v) ∩ N_c(i), it's removed
                # Similarly for j and v
                
                # Count common neighbors of v and i before/after
                # Before: N_c(v) ∩ N_c(i) includes j iff j in N_c(v) and j in N_c(i)
                # j in N_c(i) iff edge (i,j) was color c (it was, old_c=c)
                # j in N_c(v) iff edge (v,j) is color c
                
                vj_in_c = (neighbors[c][v] >> j) & 1
                if vj_in_c:
                    # j was common neighbor of v and i
                    # Count how many common neighbors v and i had
                    common_before = (neighbors[c][v] & neighbors[c][i]).bit_count()
                    common_after = (neighbors[c][v] & (neighbors[c][i] & ~(1 << j))).bit_count()
                    # common_after = common_before - 1 (since j was the only one removed from i's side that matters)
                    # But we need to check if this changes violation status
                    if common_before >= 2 and common_after < 2:
                        delta -= 1  # One violation removed
                    elif common_before >= 3:
                        # Both before and after have >=2, but count of pairs changes
                        # Pairs before: C(common_before, 2), after: C(common_after, 2)
                        # Delta = C(common_after,2) - C(common_before,2)
                        cb = common_before
                        ca = common_after
                        delta += ca*(ca-1)//2 - cb*(cb-1)//2
                
                # Similarly for v and j
                vi_in_c = (neighbors[c][v] >> i) & 1
                if vi_in_c:
                    common_before = (neighbors[c][v] & neighbors[c][j]).bit_count()
                    common_after = (neighbors[c][v] & (neighbors[c][j] & ~(1 << i))).bit_count()
                    if common_before >= 2 and common_after < 2:
                        delta -= 1
                    elif common_before >= 3:
                        cb = common_before
                        ca = common_after
                        delta += ca*(ca-1)//2 - cb*(cb-1)//2
        
        if c == new_c:
            # Adding edge (i,j) to color c
            for v in range(len(neighbors[0])):
                if v == i or v == j:
                    continue
                vj_in_c = (neighbors[c][v] >> j) & 1
                if vj_in_c:
                    # j will be common neighbor of v and i
                    common_before = (neighbors[c][v] & neighbors[c][i]).bit_count()
                    common_after = common_before + 1  # j added to i's neighbors in c
                    if common_before < 2 and common_after >= 2:
                        delta += 1
                    elif common_before >= 2:
                        cb = common_before
                        ca = common_after
                        delta += ca*(ca-1)//2 - cb*(cb-1)//2
                
                vi_in_c = (neighbors[c][v] >> i) & 1
                if vi_in_c:
                    common_before = (neighbors[c][v] & neighbors[c][j]).bit_count()
                    common_after = common_before + 1
                    if common_before < 2 and common_after >= 2:
                        delta += 1
                    elif common_before >= 2:
                        cb = common_before
                        ca = common_after
                        delta += ca*(ca-1)//2 - cb*(cb-1)//2
    
    # Also check if i and j themselves form a violation
    # Before: in old_c, i and j are neighbors. Common neighbors of i and j in old_c?
    # After: in new_c, i and j are neighbors. Common neighbors of i and j in new_c?
    
    # In old_c, i and j were neighbors. Check common neighbors of i and j in old_c
    common_ij_old = (neighbors[old_c][i] & neighbors[old_c][j]).bit_count()
    # After removing each other, common neighbors unchanged (they weren't counting each other in common)
    # Actually, i is not in N(i), j is not in N(j), so common neighbors don't include i or j
    # So removing edge doesn't change common neighbors of i and j
    # But wait: we're removing j from N(i) and i from N(j). Does this affect common neighbors of i and j?
    # Common neighbors of i and j = {k : k in N(i) and k in N(j)}
    # Removing j from N(i): j was not in N(j), so j wasn't a common neighbor
    # Removing i from N(j): i was not in N(i), so i wasn't a common neighbor
    # So common neighbors of i and j unchanged. Good.
    
    # In new_c, i and j will be neighbors. Check common neighbors of i and j in new_c
    # Before adding: common_ij_new_before = (neighbors[new_c][i] & neighbors[new_c][j]).bit_count()
    # After adding: i in N(j) for new_c, j in N(i) for new_c
    # But i not in N(i), j not in N(j), so common neighbors unchanged
    # So no change in violation status for pair (i,j)
    
    return delta

def structured_init_26():
    """Algebraic initialization for n=26 using Z_13 x {0,1} structure."""
    n = 26
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Map vertex v to (x, y) where x = v % 13, y = v // 13
    def vertex_to_xy(v):
        return (v % 13, v // 13)
    
    # Quadratic residues mod 13: {1, 3, 4, 9, 10, 12}
    # Map differences to colors:
    # For same y: use (x1 - x2) mod 13
    #   If diff in {1,12} -> 0, {3,10} -> 1, {4,9} -> 2, {5,8} -> 3, {6,7} -> 4
    # For different y: use a different mapping
    
    qr_groups = {
        0: [1, 12],
        1: [3, 10],
        2: [4, 9],
        3: [5, 8],
        4: [6, 7]
    }
    
    diff_to_color = {}
    for c, diffs in qr_groups.items():
        for d in diffs:
            diff_to_color[d] = c
    
    colors = []
    for i, j in edges:
        x1, y1 = vertex_to_xy(i)
        x2, y2 = vertex_to_xy(j)
        
        diff = (x1 - x2) % 13
        if diff == 0:
            # Same x, different y (must be since i != j)
            color = 0  # Default
        else:
            color = diff_to_color.get(diff, 0)
        
        # Adjust for y-difference to avoid conflicts
        if y1 != y2:
            # Add y-difference effect
            ydiff = abs(y1 - y2)
            color = (color + ydiff) % 5
        
        colors.append(color)
    
    return colors

def structured_init_25():
    """Initialization for n=25 using Z_5 x Z_5 affine plane structure."""
    n = 25
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def vertex_to_xy(v):
        return (v % 5, v // 5)
    
    colors = []
    for i, j in edges:
        x1, y1 = vertex_to_xy(i)
        x2, y2 = vertex_to_xy(j)
        
        dx = (x1 - x2) % 5
        dy = (y1 - y2) % 5
        
        if dx == 0 and dy == 0:
            color = 0
        elif dx == 0:
            color = dy % 5
        elif dy == 0:
            color = (dx + 1) % 5
        else:
            slope = (dy * pow(dx, -1, 5)) % 5 if dx != 0 else 0
            color = slope % 5
        
        colors.append(color)
    
    return colors
