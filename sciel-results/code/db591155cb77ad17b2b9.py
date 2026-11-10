import time
import random
import itertools
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    total_deadline = start_time + 55.0
    
    # 1. Precompute all possible doubled areas for pairs of points in the grid
    # Points are in [-30, 30] x [-30, 30]
    # We will work with a subset of candidate points to keep memory/computation reasonable, 
    # but the grid is 61x61 = 3721 points.
    # Pairs: 3721 * 3720 / 2 ~ 6.9M pairs. This is too many to precompute all areas for all pairs 
    # if we want to do it fast in Python loops, but we can be smart.
    
    # Alternative: 
    # Instead of precomputing everything, we will:
    # 1. Pick a target area A.
    # 2. Use a greedy algorithm to build a set of 30 points that maximizes triangles with area A.
    
    # What are good target areas?
    # Small areas like 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30.
    # Let's try a few.
    
    def score_target(pts, target_area):
        """Count triangles in pts with doubled area == target_area."""
        count = 0
        n = len(pts)
        # Vectorized check for a specific set of points
        # For 30 points, C(30,3) = 4060 triples. 
        # Doing this in a loop is fast enough if we do it a few times.
        # But we will call this many times. 
        # Let's use a numpy trick for speed if possible, or just optimize the loop.
        
        # Actually, 4060 iterations is very fast in Python (~1ms). 
        # So we can afford to do this frequently.
        
        for i in range(n - 2):
            xi, yi = pts[i]
            for j in range(i + 1, n - 1):
                xj, yj = pts[j]
                for k in range(j + 1, n):
                    xk, yk = pts[k]
                    d = abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                    if d == target_area:
                        count += 1
        return count

    def score_max(pts):
        """Standard score: max multiplicity of any positive area."""
        counts = Counter()
        n = len(pts)
        for i in range(n - 2):
            xi, yi = pts[i]
            for j in range(i + 1, n - 1):
                xj, yj = pts[j]
                for k in range(j + 1, n):
                    xk, yk = pts[k]
                    d = abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                    if d > 0:
                        counts[d] += 1
        if not counts:
            return 0
        return max(counts.values())

    # Candidate pool: All integer points in [-30, 30]^2
    # We might restrict to a slightly smaller box or specific subsets to speed up greedy, 
    # but let's try the full box first. 3721 points.
    all_points = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    
    # Target areas to try
    target_areas = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30]
    
    best_overall_score = 0
    best_overall_pts = []
    
    # Greedy Construction for a specific target area
    def greedy_construct(target_area, max_time=5.0):
        """
        Greedily select 30 points to maximize triangles with area == target_area.
        """
        start = time.time()
        current_pts = []
        current_set = set()
        
        # Helper to get potential next points
        # We want to pick a point p that maximizes the number of new triangles 
        # with area A formed by p and pairs in current_pts.
        
        # For efficiency, we can precompute for each pair in current_pts, 
        # which points complete a triangle of area A?
        # That's expensive to update. 
        # Instead, for each candidate point p, calculate gain = #triangles(p, pair in current_pts) with area A.
        
        # Since we only have 30 points, current_pts is small.
        # Candidates are ~3700.
        # For each candidate, we check against C(|current_pts|, 2) pairs.
        # Step 1: |current_pts|=0, gain=0 for all. Pick random or structured start.
        # Step 2: |current_pts|=1, gain=0. 
        # Step 3: |current_pts|=2, gain depends on 1 pair.
        
        # Let's start with a structured set of points to have a good foundation.
        # E.g., a small grid or line.
        # Or just random 3 points.
        
        # Let's use a random start to break symmetry, but seeded for reproducibility in this run.
        random.seed(42)
        # Pick 3 random points
        current_pts = [random.choice(all_points) for _ in range(3)]
        current_set = set(current_pts)
        
        # Ensure distinct
        while len(current_set) < 3:
             p = random.choice(all_points)
             if p not in current_set:
                 current_pts.append(p)
                 current_set.add(p)
        
        # Greedy loop
        while len(current_pts) < 30:
            if time.time() > start + max_time:
                break
                
            best_gain = -1
            best_point = None
            best_tie_breaker = 0
            
            # To speed up, we can sample a subset of candidates if it's too slow, 
            # but 3700 * C(k,2) is small for k<30.
            # C(29,2) = 406. 3700 * 406 ~ 1.5M ops per step. 
            # 30 steps * 1.5M = 45M ops. Might be too slow in pure Python.
            # Optimization: Only consider candidates that are "close" or from a pre-filtered list?
            # Or, precompute for each pair in current_pts, the set of points that form area A?
            
            # Let's try a different approach: 
            # For each candidate point p, calculate the gain.
            # We can vectorize the area calculation for a fixed p and all pairs in current_pts?
            # Or fix the pair and vectorize over p?
            
            # Let's stick to Python loops but optimize the inner check.
            # Actually, 1.5M simple integer ops in Python might take ~0.5-1s. 
            # 30 steps * 1s = 30s. Too slow for 5s budget per target.
            
            # We need to restrict the candidate pool or use a faster gain calculation.
            # Idea: Restrict candidates to points that are "likely" to form small areas.
            # Or, just pick a random subset of 500 candidates each step?
            # Or, precompute a "compatibility" matrix? No, memory too high.
            
            # Let's try: For each step, pick the best from a random sample of 200 candidates.
            # This is a heuristic greedy.
            
            candidates_sample = random.sample(all_points, 200)
            
            for p in candidates_sample:
                if p in current_set:
                    continue
                    
                # Calculate gain: # pairs (u, v) in current_pts such that area(p, u, v) == target_area
                gain = 0
                # Use local vars for speed
                pts = current_pts
                n_pts = len(pts)
                for i in range(n_pts - 1):
                    xi, yi = pts[i]
                    for j in range(i + 1, n_pts):
                        xj, yj = pts[j]
                        px, py = p
                        d = abs(px * (yi - yj) + xi * (yj - py) + xj * (py - yi))
                        if d == target_area:
                            gain += 1
                
                # Tie breaker: prefer points that are "central" or have good connectivity to others?
                # Just use random tie breaker.
                if gain > best_gain or (gain == best_gain and best_point is None):
                    best_gain = gain
                    best_point = p
            
            if best_point is None:
                # Fallback: pick random valid point
                valid_pts = [p for p in all_points if p not in current_set]
                if not valid_pts:
                    break
                best_point = random.choice(valid_pts)
            
            current_pts.append(best_point)
            current_set.add(best_point)
            
        return current_pts

    # Local Search: Pair Swap
    def local_search_pair_swap(init_pts, target_area, duration):
        current_pts = list(init_pts)
        current_set = set(current_pts)
        current_target_score = score_target(current_pts, target_area)
        # Also track max score to ensure we don't lose the global best
        current_max_score = score_max(current_pts)
        
        end_time = time.time() + duration
        
        while time.time() < end_time:
            # Pick two indices to swap
            i, j = random.sample(range(30), 2)
            p_i = current_pts[i]
            p_j = current_pts[j]
            
            # Pick two replacement points
            # We want to replace p_i and p_j with new points q_i and q_j
            # To keep it simple, let's try swapping p_i with a random point q, 
            # or swapping p_i and p_j with two random points.
            # Single point swap is easier to evaluate.
            
            # Let's do single point swap first: replace p_i with q
            q = random.choice(all_points)
            if q in current_set:
                continue
            if q == p_i:
                continue
            
            # Apply
            current_pts[i] = q
            current_set.remove(p_i)
            current_set.add(q)
            
            new_target_score = score_target(current_pts, target_area)
            
            # We accept if target score increases, OR if target score is same but max score increases?
            # The goal is to maximize the final max score. 
            # If we fix on target_area, we might miss a configuration where a different area is better.
            # However, if we find a high multiplicity for A, that's a good lower bound.
            # Let's accept if new_target_score > current_target_score.
            # If equal, we might accept with small probability to avoid local minima (simulated annealing style).
            
            if new_target_score > current_target_score:
                current_target_score = new_target_score
                # Update max score check periodically? 
                # score_max is expensive. Let's just track target for now, 
                # and evaluate max score at the end of the search or periodically.
            else:
                # Revert
                current_pts[i] = p_i
                current_set.remove(q)
                current_set.add(p_i)
            
        # Final evaluation
        final_max_score = score_max(current_pts)
        return final_max_score, current_pts

    # Strategy:
    # 1. Try greedy construction for several target areas.
    # 2. Take the best result from greedy and run local search on it.
    
    # To save time, we'll run greedy for a few targets, keep the top 2-3, and refine.
    
    greedy_results = []
    
    for ta in target_areas:
        if time.time() > total_deadline - 10: # Leave time for local search
            break
            
        # Run greedy for this target
        pts = greedy_construct(ta, max_time=2.0)
        
        if len(pts) < 30:
            continue
            
        # Evaluate max score of this greedy result
        s = score_max(pts)
        greedy_results.append((s, pts, ta))
        
        # Print debug? No.
        
    # Sort by max score
    greedy_results.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 2 for local search
    for s, pts, ta in greedy_results[:2]:
        if time.time() > total_deadline - 5:
            break
            
        # Run local search
        new_s, new_pts = local_search_pair_swap(pts, ta, duration=4.0)
        
        if new_s > best_overall_score:
            best_overall_score = new_s
            best_overall_pts = new_pts

    # If we didn't find anything good, fall back to a simple grid or random
    if not best_overall_pts:
        # Simple grid
        best_overall_pts = [(i, j) for i in range(5) for j in range(6)]
        # Scale to fit? 0..4, 0..5 is within [-30,30]
        best_overall_score = score_max(best_overall_pts)

    return best_overall_pts
