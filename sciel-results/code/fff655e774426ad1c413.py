import random
import time
import math
from typing import List, Tuple
from collections import defaultdict

def construct():
    """
    Returns 20 distinct integer points [x, y] with |x|, |y| <= 100
    such that no four are collinear.
    Aims to minimize the largest subset with no three collinear
    by maximizing the number of 3-collinear triples using Simulated Annealing.
    """
    N = 20
    BOUND = 100
    
    # Helper: Normalize line representation
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A, B, C = -A, -B, -C
        from math import gcd
        g = gcd(gcd(abs(A), abs(B)), abs(C))
        if g > 0:
            A //= g
            B //= g
            C //= g
        return (A, B, C)

    def is_valid_no_four_collinear(points):
        # Check if any line has >= 4 points
        # For each point, count other points on lines through it
        for i in range(len(points)):
            p = points[i]
            line_counts = defaultdict(int)
            for j in range(len(points)):
                if i == j: continue
                key = get_line_key(p, points[j])
                line_counts[key] += 1
            for c in line_counts.values():
                if c >= 3: # 3 others + self = 4 total
                    return False
        return True

    def count_three_collinear(points):
        count = 0
        line_counts = defaultdict(int)
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                key = get_line_key(points[i], points[j])
                line_counts[key] += 1
        for count_pairs in line_counts.values():
            # Number of triples is C(count_pairs, 2)
            if count_pairs >= 2:
                count += count_pairs * (count_pairs - 1) // 2
        return count

    def random_valid_point(exclude_indices, points):
        """Pick a random point that doesn't create 4-collinear or duplicates."""
        while True:
            x = random.randint(-BOUND, BOUND)
            y = random.randint(-BOUND, BOUND)
            p = (x, y)
            # Check distinct
            if any(p == points[j] for j in range(N) if j not in exclude_indices):
                continue
            # Check 4-collinear
            valid = True
            line_counts = defaultdict(int)
            for j in range(N):
                if j in exclude_indices: continue
                key = get_line_key(p, points[j])
                line_counts[key] += 1
            for c in line_counts.values():
                if c >= 3:
                    valid = False
                    break
            if valid:
                return p

    # Initialization
    # Start with a random valid set
    points = []
    while len(points) < N:
        p = random_valid_point(set(), points)
        points.append(p)
    
    # Initial Energy (Negative Triple Count)
    current_energy = -count_three_collinear(points)
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.99
    max_iters = 50000
    start_time = time.time()
    time_limit = 100 # Leave buffer
    
    best_points = [p for p in points]
    best_energy = current_energy
    
    for i in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random point to move
        idx = random.randint(0, N - 1)
        old_p = points[idx]
        
        # Try to move to a new random position
        # To make moves more local, we can perturb, but random anywhere is okay for SA
        # However, "random anywhere" might jump too much.
        # Let's try a mix: 50% random anywhere, 50% local perturbation
        if random.random() < 0.5:
            new_x = random.randint(-BOUND, BOUND)
            new_y = random.randint(-BOUND, BOUND)
        else:
            new_x = old_p[0] + random.randint(-10, 10)
            new_y = old_p[1] + random.randint(-10, 10)
            if new_x < -BOUND: new_x = -BOUND
            if new_x > BOUND: new_x = BOUND
            if new_y < -BOUND: new_y = -BOUND
            if new_y > BOUND: new_y = BOUND
            
        new_p = (new_x, new_y)
        
        # Check validity of new point in context of other points
        # Exclude idx
        exclude = {idx}
        
        # Check distinct
        if any(new_p == points[j] for j in range(N) if j != idx):
            continue # Invalid move, skip
        
        # Check 4-collinear
        valid = True
        line_counts = defaultdict(int)
        for j in range(N):
            if j == idx: continue
            key = get_line_key(new_p, points[j])
            line_counts[key] += 1
        for c in line_counts.values():
            if c >= 3:
                valid = False
                break
        
        if not valid:
            continue
            
        # Calculate new energy
        # We can update triple count incrementally or recompute.
        # Recompute is O(N^2) = 400. 50000 * 400 = 20M ops. Should be fine.
        new_points = points.copy()
        new_points[idx] = new_p
        new_energy = -count_three_collinear(new_points)
        
        delta_e = new_energy - current_energy
        
        # Acceptance
        if delta_e < 0 or random.random() < math.exp(-delta_e / T):
            points = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = [p for p in points]
        
        T *= cooling
        if T < T_min:
            T = T_min

    # Return best points found
    return best_points
