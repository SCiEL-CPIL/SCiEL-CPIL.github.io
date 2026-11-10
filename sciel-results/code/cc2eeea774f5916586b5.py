"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations
from collections import Counter

def _canonical_line(p1, p2):
    """Return a canonical tuple (A, B, C) for the line passing through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    # Normalize by GCD
    g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A //= g
    B //= g
    C //= g
    # Ensure a unique sign representation
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def _score(points):
    """
    Calculate the score: number of lines with exactly 4 points.
    Returns -1 if any line has >= 5 points (invalid).
    """
    line_counts = Counter()
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            line = _canonical_line(points[i], points[j])
            line_counts[line] += 1
    
    score = 0
    for count in line_counts.values():
        if count >= 10:  # C(5,2)=10 pairs for 5 points
            return -1
        if count == 6:  # C(4,2)=6 pairs for 4 points
            score += 1
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit
    
    # Initial configuration: 4x4 grid is valid and has score 10
    # Points: (0,0) to (3,3)
    # Let's center it slightly to allow moves in all directions within bounds
    # Or just use 0..3.
    initial_points = [[x, y] for x in range(4) for y in range(4)]
    
    best_points = [p[:] for p in initial_points]
    best_score = _score(initial_points)
    
    current_points = [p[:] for p in initial_points]
    current_score = best_score
    
    # Simulated Annealing
    # Temperature schedule
    T = 10.0
    cooling = 0.999
    
    # Move strategies:
    # 1. Move one point to a random location
    # 2. Swap two points
    
    while time.time() < deadline:
        if T < 0.01:
            # Restart with a slightly perturbed best solution or reheat
            T = 10.0
            # Perturb best solution
            current_points = [p[:] for p in best_points]
            for _ in range(3):
                idx = random.randint(0, 15)
                current_points[idx][0] += random.randint(-2, 2)
                current_points[idx][1] += random.randint(-2, 2)
                # Ensure within bounds
                current_points[idx][0] = max(-1000, min(1000, current_points[idx][0]))
                current_points[idx][1] = max(-1000, min(1000, current_points[idx][1]))
            current_score = _score(current_points)
            if current_score == -1:
                current_score = 0 # Treat invalid as low score for SA acceptance logic? 
                # Actually, if invalid, we should probably revert to best or a valid neighbor.
                # For simplicity, if invalid, set score to -100 to discourage.
                current_score = -100
        
        # Choose move type
        if random.random() < 0.5:
            # 1-point move
            idx = random.randint(0, 15)
            old_x, old_y = current_points[idx]
            # Random jump
            new_x = random.randint(-50, 50)
            new_y = random.randint(-50, 50)
            
            # Check distinctness
            valid_move = True
            for i, p in enumerate(current_points):
                if i != idx and p[0] == new_x and p[1] == new_y:
                    valid_move = False
                    break
            
            if not valid_move:
                continue
                
            current_points[idx][0] = new_x
            current_points[idx][1] = new_y
            new_score = _score(current_points)
            
            # Acceptance criterion
            if new_score == -1:
                # Invalid, revert
                current_points[idx][0] = old_x
                current_points[idx][1] = old_y
                continue
                
            delta = new_score - current_score
            if delta > 0:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
            else:
                # Boltzmann acceptance
                if random.random() < math.exp(delta / T):
                    current_score = new_score
                else:
                    current_points[idx][0] = old_x
                    current_points[idx][1] = old_y
        else:
            # 2-point swap
            i1, i2 = random.sample(range(16), 2)
            p1 = current_points[i1][:]
            p2 = current_points[i2][:]
            
            # Check distinctness after swap
            # The set of points remains the same, so distinctness is preserved
            # unless p1 == p2 which is impossible as they are distinct indices in a valid set
            
            current_points[i1] = p2
            current_points[i2] = p1
            new_score = _score(current_points)
            
            if new_score == -1:
                current_points[i1] = p1
                current_points[i2] = p2
                continue

            delta = new_score - current_score
            if delta > 0:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
            else:
                if random.random() < math.exp(delta / T):
                    current_score = new_score
                else:
                    current_points[i1] = p1
                    current_points[i2] = p2
        
        T *= cooling
        
        # Periodically check time
        if time.time() > deadline:
            break
    
    # Final polish: local search on best_points
    # Try 1-point moves on best_points to see if we can improve
    current_points = [p[:] for p in best_points]
    current_score = best_score
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline - 5.0 and no_improve < max_no_improve:
        idx = random.randint(0, 15)
        old_x, old_y = current_points[idx]
        
        # Try small perturbations
        found_improvement = False
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue
                new_x = old_x + dx
                new_y = old_y + dy
                if new_x < -1000 or new_x > 1000 or new_y < -1000 or new_y > 1000:
                    continue
                
                valid_move = True
                for i, p in enumerate(current_points):
                    if i != idx and p[0] == new_x and p[1] == new_y:
                        valid_move = False
                        break
                if not valid_move:
                    continue
                
                current_points[idx][0] = new_x
                current_points[idx][1] = new_y
                new_score = _score(current_points)
                
                if new_score != -1 and new_score > current_score:
                    current_score = new_score
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
                    no_improve = 0
                    found_improvement = True
                    break
                else:
                    current_points[idx][0] = old_x
                    current_points[idx][1] = old_y
            
            if found_improvement:
                break
        
        if not found_improvement:
            no_improve += 1
    
    return best_points

# EVOLVE-BLOCK-END
