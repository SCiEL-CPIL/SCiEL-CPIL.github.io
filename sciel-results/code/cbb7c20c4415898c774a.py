import time
import random
from collections import Counter
import numpy as np

def construct():
    deadline = time.time() + 50.0  # safety margin
    random.seed(42)
    
    N = 30
    # Start with a grid
    coords = [(x, y) for x in range(6) for y in range(5)]
    
    def doubled_area(p1, p2, p3):
        return abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
    
    def full_count(points):
        areas = []
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    a = doubled_area(points[i], points[j], points[k])
                    if a > 0:
                        areas.append(a)
        if not areas:
            return Counter(), 0, 0
        c = Counter(areas)
        max_area, max_count = c.most_common(1)[0]
        return c, max_area, max_count

    # Initial setup
    counter, best_area, best_count = full_count(coords)
    best_overall = best_count
    best_points = list(coords)
    best_target = best_area
    
    target = best_target
    current_count = best_count
    
    # Precompute indices for pairs not involving idx
    # For each idx, we need pairs (j, k) from [0..N-1] \ {idx}
    
    def get_pairs_excluding(idx):
        pairs = []
        for j in range(N):
            if j == idx: continue
            for k in range(j+1, N):
                if k == idx: continue
                pairs.append((j, k))
        return pairs

    # Precompute all pair lists for speed
    all_pairs = [get_pairs_excluding(i) for i in range(N)]
    
    no_improve = 0
    max_no_improve = 100
    iterations = 0
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            # Re-evaluate to find current best area
            counter, ba, bc = full_count(coords)
            if bc > best_overall:
                best_overall = bc
                best_points = list(coords)
                best_target = ba
            # Switch target to the current most common area
            target = ba
            current_count = bc
            no_improve = 0
            max_no_improve += 100
            # Optional: Random restart if stuck for too long? 
            # Let's just continue with new target
            continue
        
        # Pick a random point to move
        idx = random.randint(0, N-1)
        old_pt = coords[idx]
        pairs = all_pairs[idx]
        
        improved = False
        # Try 5 random positions
        for _ in range(5):
            new_pt = (random.randint(-30, 30), random.randint(-30, 30))
            if new_pt == old_pt:
                continue
            # Check duplicate
            if new_pt in coords:
                continue
            
            # Calculate delta
            delta = 0
            # We need to subtract old contributions and add new ones
            # Since we only track the count for `target`, we just count matches.
            old_matches = 0
            new_matches = 0
            for j, k in pairs:
                p_j = coords[j]
                p_k = coords[k]
                if doubled_area(old_pt, p_j, p_k) == target:
                    old_matches += 1
                if doubled_area(new_pt, p_j, p_k) == target:
                    new_matches += 1
            
            delta = new_matches - old_matches
            
            if delta > 0:
                # Accept
                # Update counter
                # We don't maintain the full counter in the loop for speed, 
                # but we need to update `current_count`
                current_count += delta
                coords[idx] = new_pt
                improved = True
                no_improve = 0
                if current_count > best_overall:
                    best_overall = current_count
                    best_points = list(coords)
                    # Note: best_target might change if we switch targets later, 
                    # but for now we are optimizing for `target`
                break
            elif delta == 0 and random.random() < 0.05:
                # Neutral move
                coords[idx] = new_pt
                # current_count unchanged
                no_improve += 1
                break
        
        if not improved:
            no_improve += 1
            
        iterations += 1
        
        # Periodically check if we should switch target
        # To avoid O(N^3) full_count every time, we can do it less frequently
        if iterations % 50 == 0:
             # Do a quick check: is there another area with higher count?
             # This is expensive. Let's just rely on the restart logic above.
             pass

    # Final verification
    if len(best_points) != N or len(set(best_points)) != N:
        return [(x,y) for x in range(6) for y in range(5)]
    for p in best_points:
        if not (-30 <= p[0] <= 30 and -30 <= p[1] <= 30):
            return [(x,y) for x in range(6) for y in range(5)]
            
    return best_points
