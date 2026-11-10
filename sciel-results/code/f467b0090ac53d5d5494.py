import numpy as np
import random
from itertools import product

def nikodym_set() -> np.ndarray:
    p = 7
    k = 3  # alpha^2 = 3
    
    # Field F_49 = F_7[alpha] where alpha^2 = 3
    # Elements as (a, b) meaning a + b*alpha
    
    def field_mul(a, b):
        return ((a[0]*b[0] + 3*b[1]*a[1]) % p, 
                (a[0]*b[1] + a[1]*b[0]) % p)
    
    # Precompute multiplication table
    mul_table = {}
    for a in range(p):
        for b in range(p):
            for c in range(p):
                for d in range(p):
                    mul_table[((a, b), (c, d))] = field_mul((a, b), (c, d))
    
    # All points in F_49^2: (xa, xb, ya, yb)
    all_points = set(product(range(p), repeat=4))
    
    # Generate all lines
    # Vertical lines: x = (xa, xb), y varies
    # Non-vertical lines: direction (1, m) where m = (ma, mb)
    
    all_lines = []
    seen_lines = set()
    
    # Vertical lines
    for xa in range(p):
        for xb in range(p):
            points_on_line = set()
            for ya in range(p):
                for yb in range(p):
                    points_on_line.add((xa, xb, ya, yb))
            line_tuple = frozenset(points_on_line)
            if line_tuple not in seen_lines:
                seen_lines.add(line_tuple)
                all_lines.append(points_on_line)
    
    # Non-vertical lines with direction (1, m)
    for ma in range(p):
        for mb in range(p):
            for x0a in range(p):
                for x0b in range(p):
                    for y0a in range(p):
                        for y0b in range(p):
                            points_on_line = set()
                            for ta in range(p):
                                for tb in range(p):
                                    t = (ta, tb)
                                    tm = mul_table[(t, (ma, mb))]
                                    xa = (x0a + ta) % p
                                    xb = (x0b + tb) % p
                                    ya = (y0a + tm[0]) % p
                                    yb = (y0b + tm[1]) % p
                                    points_on_line.add((xa, xb, ya, yb))
                            
                            line_tuple = frozenset(points_on_line)
                            if line_tuple not in seen_lines:
                                seen_lines.add(line_tuple)
                                all_lines.append(points_on_line)
    
    # Build point-to-lines incidence
    point_to_lines = {pt: [] for pt in all_points}
    for i, line in enumerate(all_lines):
        for pt in line:
            point_to_lines[pt].append(i)
    
    # Greedy approach to maximize complement C
    # Start with C = empty, N = all_points
    # Try to add points to C (remove from N) while maintaining Nikodym property
    
    # The Nikodym condition: for every point x in F_49^2, there exists a line L through x
    # such that L \ {x} ⊆ N, i.e., L ∩ C ⊆ {x}
    
    # Maintain: for each line, the set of points in C on that line
    line_c_sets = [set() for _ in all_lines]
    C = set()
    
    def is_point_covered(pt):
        """Check if point pt has a line L through it with L ∩ C ⊆ {pt}"""
        for line_idx in point_to_lines[pt]:
            c_on_line = line_c_sets[line_idx]
            if len(c_on_line) == 0:
                return True
            elif len(c_on_line) == 1 and pt in c_on_line:
                return True
        return False
    
    def is_valid():
        """Check if current C gives a valid Nikodym set N = all_points - C"""
        for pt in all_points:
            if not is_point_covered(pt):
                return False
        return True
    
    # Try multiple random orderings to find a large C
    best_c = set()
    best_c_size = 0
    
    for seed in range(20):
        random.seed(seed)
        candidates = list(all_points)
        random.shuffle(candidates)
        
        C = set()
        line_c_sets = [set() for _ in all_lines]
        
        for pt in candidates:
            # Try adding pt to C
            C.add(pt)
            for line_idx in point_to_lines[pt]:
                line_c_sets[line_idx].add(pt)
            
            if is_valid():
                if len(C) > best_c_size:
                    best_c_size = len(C)
                    best_c = C.copy()
            else:
                # Revert
                C.remove(pt)
                for line_idx in point_to_lines[pt]:
                    line_c_sets[line_idx].discard(pt)
    
    # Additional passes with different strategies
    for seed in range(5, 25):
        random.seed(seed * 1000 + 7)
        
        # Start from best_c and try to grow further
        C = best_c.copy()
        line_c_sets = [set() for _ in all_lines]
        for pt in C:
            for line_idx in point_to_lines[pt]:
                line_c_sets[line_idx].add(pt)
        
        candidates = list(all_points - C)
        random.shuffle(candidates)
        
        for pt in candidates:
            C.add(pt)
            for line_idx in point_to_lines[pt]:
                line_c_sets[line_idx].add(pt)
            
            if is_valid():
                if len(C) > best_c_size:
                    best_c_size = len(C)
                    best_c = C.copy()
            else:
                C.remove(pt)
                for line_idx in point_to_lines[pt]:
                    line_c_sets[line_idx].discard(pt)
    
    N_final = all_points - best_c
    
    # Convert to required format
    points_list = []
    for (xa, xb, ya, yb) in sorted(N_final):
        points_list.append([[xa, xb], [ya, yb]])
    
    if not points_list:
        for (xa, xb, ya, yb) in sorted(all_points):
            points_list.append([[xa, xb], [ya, yb]])
    
    return np.array(points_list, dtype=np.int64)
