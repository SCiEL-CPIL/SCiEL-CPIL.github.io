import numpy as np
import random
from itertools import product

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401
    
    # Field F_49 = F_7[alpha] where alpha^2 = 3
    # Elements represented as (a, b) meaning a + b*alpha
    
    def field_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def field_mul(a, b):
        return ((a[0]*b[0] + 3*b[1]*a[1]) % p, 
                (a[0]*b[1] + a[1]*b[0]) % p)
    
    # Precompute multiplication table
    mul_table = np.zeros((q, q, 2), dtype=np.int64)
    for i in range(q):
        for j in range(q):
            a = (i % p, i // p)
            b = (j % p, j // p)
            r = field_mul(a, b)
            mul_table[i, j] = [r[0], r[1]]
    
    # Generate all lines
    # Each line is a set of 49 points
    # Point represented as (xa, xb, ya, yb)
    
    all_lines = []
    seen_lines = set()
    
    # Vertical lines: x = (xa, xb), varying y
    for xa in range(p):
        for xb in range(p):
            points_on_line = []
            for ya in range(p):
                for yb in range(p):
                    points_on_line.append((xa, xb, ya, yb))
            line_tuple = tuple(sorted(points_on_line))
            if line_tuple not in seen_lines:
                seen_lines.add(line_tuple)
                all_lines.append(frozenset(points_on_line))
    
    # Non-vertical lines: direction (1, m) where m = (ma, mb)
    for ma in range(p):
        for mb in range(p):
            for x0a in range(p):
                for x0b in range(p):
                    for y0a in range(p):
                        for y0b in range(p):
                            points_on_line = []
                            for ta in range(p):
                                for tb in range(p):
                                    xa = (x0a + ta) % p
                                    xb = (x0b + tb) % p
                                    t_m = field_mul((ta, tb), (ma, mb))
                                    ya = (y0a + t_m[0]) % p
                                    yb = (y0b + t_m[1]) % p
                                    points_on_line.append((xa, xb, ya, yb))
                            
                            line_tuple = tuple(sorted(points_on_line))
                            if line_tuple not in seen_lines:
                                seen_lines.add(line_tuple)
                                all_lines.append(frozenset(points_on_line))
    
    # Build point-to-lines incidence
    point_to_lines = {}
    for point in product(range(p), repeat=4):
        point_to_lines[point] = []
    
    for line in all_lines:
        for point in line:
            point_to_lines[point].append(line)
    
    # Convert lines to lists for faster access
    line_points = [list(line) for line in all_lines]
    
    # Precompute for each point, the indices of lines through it
    point_line_indices = {}
    for i, line in enumerate(all_lines):
        for point in line:
            if point not in point_line_indices:
                point_line_indices[point] = []
            point_line_indices[point].append(i)
    
    # Greedy algorithm with random restarts
    best_C_size = 0
    best_C = set()
    
    all_points_list = list(product(range(p), repeat=4))
    
    for restart in range(20):
        random.seed(restart * 1000 + 42)
        C = set()
        # For each line, track which points are in C
        line_C = [set() for _ in range(len(all_lines))]
        
        # For each point, track which lines through it have C on them
        # A line L through x is "good" for x if L ∩ C ⊆ {x}
        # i.e., |L ∩ C| == 0 (if x not in C) or |L ∩ C| == 1 and x in L ∩ C
        
        # Maintain: for each point x, the count of good lines through x
        # good_count[x] = number of lines L through x such that L ∩ C ⊆ {x}
        
        # Initially C is empty, so every line is good for every point on it
        good_count = {point: len(point_line_indices[point]) for point in all_points_list}
        
        candidates = list(all_points_list)
        random.shuffle(candidates)
        
        for point in candidates:
            if point in C:
                continue
            
            # Check if adding point to C maintains validity
            # After adding point, for each line L through point:
            #   - L ∩ C gains point
            #   - For point itself: L is good if |L ∩ C| == 1 (just point)
            #   - For other points x on L: L is good if |L ∩ C| == 0 (impossible since point in C)
            #     or |L ∩ C| == 1 and x in L ∩ C (impossible since point != x)
            #     So L becomes NOT good for other points on L.
            
            # Check: for point itself, is there at least one line L through point
            #   such that after addition, |L ∩ C| == 1?
            #   i.e., before addition, |L ∩ C| == 0.
            
            # Check: for other points x on lines through point,
            #   if the only good line for x was one of the lines through point,
            #   and that line now has |L ∩ C| >= 2, then x loses that good line.
            #   We need to check if x still has at least one good line.
            
            can_add = True
            
            # First check: does point have a line with no C points?
            point_has_clean_line = False
            for line_idx in point_line_indices[point]:
                if len(line_C[line_idx]) == 0:
                    point_has_clean_line = True
                    break
            
            if not point_has_clean_line:
                can_add = False
            else:
                # Check all other points on lines through point
                affected_points = set()
                for line_idx in point_line_indices[point]:
                    for other in line_points[line_idx]:
                        if other != point:
                            affected_points.add(other)
                
                for other in affected_points:
                    # Check if other still has a good line after addition
                    # A line L is good for other if:
                    #   - other not in C: |L ∩ C| == 0
                    #   - other in C: |L ∩ C| == 1 and other in L ∩ C
                    # Before addition, other had good_count[other] good lines.
                    # After addition, lines through point that had 0 C points now have 1 (point).
                    # For other (not in C): a line L through point with |L ∩ C| == 0 becomes |L ∩ C| == 1,
                    #   which is NOT good for other (since other not in C, need |L ∩ C| == 0).
                    #   So other loses those lines as good lines.
                    # For other (in C): a line L through point with |L ∩ C| == 0 becomes |L ∩ C| == 1,
                    #   but point != other, so other not in L ∩ C. NOT good.
                    #   A line L through point with |L ∩ C| == 1 and other in L ∩ C becomes |L ∩ C| == 2.
                    #   NOT good.
                    
                    # So we need to check: after removing the lines through point that were good for other,
                    # does other still have at least one good line?
                    
                    # Count how many good lines other has that are NOT through point
                    # and how many good lines other has that ARE through point but remain good.
                    
                    # Actually, simpler: just check if other has any good line that doesn't go through point,
                    # or if it has a good line through point that remains good.
                    
                    # A line L through point remains good for other if:
                    #   - other not in C: |L ∩ C| == 0 before AND point not in L (impossible, point in L)
                    #     Wait, point IS in L. So after adding point, |L ∩ C| >= 1.
                    #     For other not in C, need |L ∩ C| == 0. Impossible after addition.
                    #   - other in C: |L ∩ C| == 1 and other in L ∩ C.
                    #     Before: |L ∩ C| == 1 and other in L ∩ C, and point not in L ∩ C (since point not in C yet).
                    #     After: |L ∩ C| == 2 (other and point). NOT good.
                    #     Before: |L ∩ C| == 0. After: |L ∩ C| == 1 (point). other not in L ∩ C. NOT good.
                    
                    # So NO line through point remains good for other after adding point.
                    # Therefore, other must have a good line that does NOT go through point.
                    
                    # Check: does other have any good line not in point_line_indices[point]?
                    other_good_lines_not_through_point = 0
                    for line_idx in point_line_indices[other]:
                        if line_idx not in point_line_indices[point]:
                            # Check if this line is good for other
                            c_on_line = line_C[line_idx]
                            if other in C:
                                if len(c_on_line) == 1 and other in c_on_line:
                                    other_good_lines_not_through_point += 1
                            else:
                                if len(c_on_line) == 0:
                                    other_good_lines_not_through_point += 1
                    
                    if other_good_lines_not_through_point == 0:
                        # Check if other has a good line through point that remains good
                        # As analyzed above, no line through point remains good for other.
                        can_add = False
                        break
            
            if can_add:
                # Add point to C
                C.add(point)
                for line_idx in point_line_indices[point]:
                    line_C[line_idx].add(point)
                
                # Update good_count for affected points
                # For point itself: good lines are those with |L ∩ C| == 1 (just point)
                point_good = 0
                for line_idx in point_line_indices[point]:
                    if len(line_C[line_idx]) == 1 and point in line_C[line_idx]:
                        point_good += 1
                good_count[point] = point_good
                
                # For other points on lines through point
                affected = set()
                for line_idx in point_line_indices[point]:
                    for other in line_points[line_idx]:
                        if other != point:
                            affected.add(other)
                
                for other in affected:
                    # Recompute good count for other
                    other_good = 0
                    for line_idx in point_line_indices[other]:
                        c_on_line = line_C[line_idx]
                        if other in C:
                            if len(c_on_line) == 1 and other in c_on_line:
                                other_good += 1
                        else:
                            if len(c_on_line) == 0:
                                other_good += 1
                    good_count[other] = other_good
        
        if len(C) > best_C_size:
            best_C_size = len(C)
            best_C = C.copy()
    
    # Build N from best_C
    N = set(all_points_list) - best_C
    
    # Convert to required format
    points_list = []
    for (xa, xb, ya, yb) in sorted(N):
        points_list.append([[xa, xb], [ya, yb]])
    
    if not points_list:
        for (xa, xb, ya, yb) in sorted(all_points_list):
            points_list.append([[xa, xb], [ya, yb]])
    
    return np.array(points_list, dtype=np.int64)
