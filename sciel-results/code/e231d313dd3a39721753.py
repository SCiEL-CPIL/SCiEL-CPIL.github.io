import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Initial seed: 6x5 grid
    # Points in [-30, 30]
    # Center it to stay within bounds? The grid 0..5, 0..4 is within [-30, 30].
    def get_initial_grid(w, h):
        return [(x, y) for x in range(w) for y in range(h)]
    
    # We will try a few different initial seeds and pick the best one to start search
    # Or just start with the known good one.
    # The task says the current program is the 6x5 grid.
    
    best_points = get_initial_grid(6, 5)
    
    def score(points):
        # points is a list of 30 tuples
        # Compute doubled areas for all triples
        # Use a dictionary to count frequencies
        n = len(points)
        # Precompute coordinates as numpy array for speed if needed, but list is fine
        # 4060 triples
        counts = {}
        # Optimization: only count non-zero areas
        # We can vectorize the triple generation?
        # Let's use a simple loop first, it's fast enough for 30 points.
        for i in range(n):
            pi = points[i]
            for j in range(i+1, n):
                pj = points[j]
                # Vector pi -> pj
                v1 = (pj[0] - pi[0], pj[1] - pi[1])
                for k in range(j+1, n):
                    pk = points[k]
                    v2 = (pk[0] - pi[0], pk[1] - pi[1])
                    # Determinant: v1.x * v2.y - v1.y * v2.x
                    area2 = abs(v1[0]*v2[1] - v1[1]*v2[0])
                    if area2 > 0:
                        counts[area2] = counts.get(area2, 0) + 1
        if not counts:
            return 0
        return max(counts.values())
    
    # Evaluate seed
    best_score = score(best_points)
    
    # Search
    # We will use a simple random perturbation strategy
    # Move one point to a random location in [-30, 30]
    # Ensure distinctness
    
    for _ in range(20000): # Max iterations, will break on time
        if time.time() > deadline:
            break
        
        # Pick a random point to move
        idx = random.randint(0, 29)
        # Pick a new random position
        # Bias towards nearby points? Or global?
        # Global random in [-30, 30] is good for exploration.
        new_x = random.randint(-30, 30)
        new_y = random.randint(-30, 30)
        new_point = (new_x, new_y)
        
        # Check if new point is already in the set (excluding the one being moved)
        # For efficiency, we can check against the list
        # If collision, skip or try again?
        # Since 30 points in 61x61 area, collision probability is low (~30/3721 ~ 0.8%)
        # But we must ensure distinctness.
        if new_point in best_points[:idx] + best_points[idx+1:]:
            continue
            
        # Create new candidate
        candidate = best_points.copy()
        candidate[idx] = new_point
        
        # Evaluate
        s = score(candidate)
        
        if s > best_score:
            best_score = s
            best_points = candidate
        # Simple hill climbing: only accept if better.
        # If we want to escape local optima, we could use simulated annealing,
        # but hill climbing from a good seed is often sufficient for these grid-like structures.
        # Let's add a small chance to accept worse moves to escape plateaus?
        # Given the discrete nature and the "grid" structure, plateaus are likely.
        # Let's try a few restarts if we get stuck.
        
    return best_points
