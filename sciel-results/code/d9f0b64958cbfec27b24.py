"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def count_four_lines_from_dual(lines):
    """
    lines: list of (m, c) representing dual lines Y = m*X - c
    Each dual line corresponds to a point (m, c) in primal.
    Count intersection points where exactly 4 lines meet.
    Return 0 if any intersection has 5+ lines.
    """
    n = len(lines)
    if n < 4:
        return 0
    
    # For each pair of lines, find intersection point
    # Line i: Y = m_i * X - c_i
    # Line j: Y = m_j * X - c_j
    # Intersection: m_i * X - c_i = m_j * X - c_j
    # X = (c_i - c_j) / (m_i - m_j) if m_i != m_j
    # If m_i == m_j, lines are parallel (no intersection) unless c_i == c_j (same line)
    
    # Map each intersection point (as a rational or normalized key) to set of lines
    # To avoid floating point, use exact rational representation
    
    # An intersection point can be represented as (X_num, X_den, Y_num, Y_den) normalized
    # Or better: for lines with integer coefficients, the intersection of 
    # Y = m1*X - c1 and Y = m2*X - c2 is at:
    # X = (c1 - c2)/(m1 - m2), Y = m1*(c1-c2)/(m1-m2) - c1 = (m1*c1 - m1*c2 - c1*m1 + c1*m2)/(m1-m2)
    # = (m2*c1 - m1*c2)/(m1-m2)
    # So X = (c1-c2)/(m1-m2), Y = (m2*c1 - m1*c2)/(m1-m2)
    
    # Normalize: multiply by sign of denominator to make denominator positive
    
    intersection_map = defaultdict(set)
    
    for i in range(n):
        for j in range(i+1, n):
            m1, c1 = lines[i]
            m2, c2 = lines[j]
            
            dm = m1 - m2
            if dm == 0:
                # Parallel lines, no intersection (or same line if c1==c2)
                continue
            
            # X = (c1 - c2) / dm
            # Y = (m2*c1 - m1*c2) / dm
            
            x_num = c1 - c2
            y_num = m2 * c1 - m1 * c2
            x_den = dm
            y_den = dm
            
            # Normalize: make denominator positive
            if x_den < 0:
                x_num = -x_num
                x_den = -x_den
                y_num = -y_num
                y_den = -y_den
            
            # Reduce fractions
            g1 = _gcd(abs(x_num), x_den) if x_num != 0 else x_den
            if g1 > 0 and x_num != 0:
                x_num_r = x_num // g1
                x_den_r = x_den // g1
            elif x_num == 0:
                x_num_r, x_den_r = 0, 1
            else:
                x_num_r, x_den_r = x_num, x_den
            
            g2 = _gcd(abs(y_num), y_den) if y_num != 0 else y_den
            if g2 > 0 and y_num != 0:
                y_num_r = y_num // g2
                y_den_r = y_den // g2
            elif y_num == 0:
                y_num_r, y_den_r = 0, 1
            else:
                y_num_r, y_den_r = y_num, y_den
            
            key = (x_num_r, x_den_r, y_num_r, y_den_r)
            intersection_map[key].add(i)
            intersection_map[key].add(j)
    
    count = 0
    for key, line_set in intersection_map.items():
        if len(line_set) >= 5:
            return 0
        if len(line_set) == 4:
            count += 1
    
    return count

def construct():
    random.seed(42)
    
    # Initialize with random lines in dual space
    # Each line is (m, c) corresponding to primal point (m, c)
    # We need |m| <= 1000 and |c| <= 1000
    
    # Start with a structured initial configuration
    # Use a grid-like pattern in dual space
    points = []
    for m in range(-3, 4):
        for c in range(-3, 4):
            if len(points) < 16:
                points.append((m, c))
    
    best_score = count_four_lines_from_dual(points)
    best_points = [p[:] for p in points]
    
    # Simulated annealing in dual space
    T = 100.0
    T_min = 0.1
    cooling = 0.999
    max_iters = 80000
    
    current_points = [p[:] for p in points]
    current_score = best_score
    
    for it in range(max_iters):
        if T < T_min:
            break
        
        # Pick a random line to modify
        idx = random.randint(0, 15)
        old_m, old_c = current_points[idx]
        
        # Random perturbation
        move_range = random.choice([1, 2, 3, 5, 8, 12, 20])
        new_m = old_m + random.randint(-move_range, move_range)
        new_c = old_c + random.randint(-move_range, move_range)
        
        # Clamp to bounds
        new_m = max(-1000, min(1000, new_m))
        new_c = max(-1000, min(1000, new_c))
        
        # Check if this line is already in the set
        if any(current_points[i] == [new_m, new_c] for i in range(16) if i != idx):
            continue
        
        current_points[idx] = [new_m, new_c]
        new_score = count_four_lines_from_dual(current_points)
        
        delta = new_score - current_score
        if delta > 0 or random.random() < pow(2.718281828, delta / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
        else:
            current_points[idx] = [old_m, old_c]
        
        T *= cooling
    
    # Convert dual lines back to primal points
    # Dual line (m, c) corresponds to primal point (m, c)
    result = [[m, c] for m, c in best_points]
    
    return result

# EVOLVE-BLOCK-END
