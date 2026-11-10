import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from itertools import combinations

def are_collinear(p1, p2, p3):
    return (p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])) == 0

def has_four_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            # Check if any other point is on the line through i,j
            # And if there's a 4th
            # Simpler: count points on line
            # But O(N^3) is fine for N=20
            pass
    # Better:
    # For each pair, count how many other points are on the line.
    # If count >= 2, then 4 collinear.
    for i in range(n):
        for j in range(i+1, n):
            count = 0
            for k in range(n):
                if k == i or k == j: continue
                if are_collinear(points[i], points[j], points[k]):
                    count += 1
            if count >= 2:
                return True
    return False

def calculate_score(points):
    # Find all collinear triples
    triples = []
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if are_collinear(points[i], points[j], points[k]):
                    triples.append((i, j, k))
    
    if not triples:
        return n
        
    # ILP
    # Maximize sum(z)
    # Subject to z_i + z_j + z_k <= 2 for all triples
    # 0 <= z <= 1, integer
    
    c = -np.ones(n) # Minimize -sum(z)
    
    A = np.zeros((len(triples), n))
    for idx, (i, j, k) in enumerate(triples):
        A[idx, i] = 1
        A[idx, j] = 1
        A[idx, k] = 1
        
    cons = LinearConstraint(A, ub=2)
    bounds = Bounds(0, 1)
    
    res = milp(c=c, constraints=cons, bounds=bounds, integrality=np.ones(n))
    
    if res.success:
        return int(-res.fun)
    else:
        # Fallback: greedy or random?
        # If ILP fails, return a conservative estimate
        return n

def construct():
    start_time = time.time()
    deadline = start_time + 110 # 10s buffer
    
    best_score = 20
    best_points = [[i, i*i] for i in range(-10, 10)] # Seed
    
    # Search space: [-10, 10]
    # Try to find a better configuration
    
    # Initial random tries
    for _ in range(100):
        if time.time() > deadline: break
        points = [[random.randint(-10, 10), random.randint(-10, 10)] for _ in range(20)]
        # Ensure distinct
        if len(set(map(tuple, points))) < 20: continue
        if has_four_collinear(points): continue
        
        score = calculate_score(points)
        if score < best_score:
            best_score = score
            best_points = points
            
    # Local search / Hill Climbing
    # Try to swap points to reduce score
    # We want to INCREASE collinearity to DECREASE score
    # So we look for moves that add more triples or keep them while maintaining validity
    
    current_points = best_points
    current_score = best_score
    
    for _ in range(500):
        if time.time() > deadline: break
        
        # Pick a random point to replace
        idx = random.randint(0, 19)
        old_p = current_points[idx]
        
        # Try a few new random points
        found_improvement = False
        for _ in range(20):
            new_p = [random.randint(-10, 10), random.randint(-10, 10)]
            if new_p == old_p: continue
            if new_p in current_points: continue
            
            trial_points = current_points.copy()
            trial_points[idx] = new_p
            
            if has_four_collinear(trial_points): continue
            
            trial_score = calculate_score(trial_points)
            
            if trial_score < current_score:
                current_points = trial_points
                current_score = trial_score
                found_improvement = True
                break
        
        if not found_improvement:
            # If stuck, try a random restart or larger jump
            # Just break and try a new random set?
            # Or accept a neutral move?
            pass

    return best_points if best_score <= current_score else current_points
