import numpy as np
import time
from collections import Counter

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin before 60s limit
    
    # Initial 6x5 grid
    points = [(x, y) for x in range(-2, 4) for y in range(-2, 3)]
    n = len(points)
    
    def get_areas_involving(idx, pts):
        """Returns list of doubled areas for all triples involving idx using vectorization."""
        pi = np.array(pts[idx])
        others = np.array([pts[j] for j in range(n) if j != idx])
        # Shape: (29, 2)
        # Compute vectors from pi to each other point
        v1 = others[:, 0] - pi[0]  # x components
        v2 = others[:, 1] - pi[1]  # y components
        
        # For each pair (j, k), det = v1[j]*v2[k] - v2[j]*v1[k]
        # We need all pairs j < k
        areas = []
        for j in range(len(others) - 1):
            for k in range(j + 1, len(others)):
                det = v1[j] * v2[k] - v2[j] * v1[k]
                a = abs(det)
                if a > 0:
                    areas.append(a)
        return areas

    def build_full_hist(pts):
        """Build histogram of all positive doubled areas."""
        h = Counter()
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    pi, pj, pk = pts[i], pts[j], pts[k]
                    det = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
                    a = abs(det)
                    if a > 0:
                        h[a] += 1
        return h

    hist = build_full_hist(points)
    best_score = max(hist.values()) if hist else 0
    best_points = points[:]
    best_hist = hist.copy()
    
    def try_move(idx, new_pos):
        """Try moving point idx to new_pos, return True if it improves score."""
        nonlocal hist, best_score, best_points, best_hist
        
        # Check if new position is valid
        if new_pos in best_points:
            return False
        if new_pos[0] < -30 or new_pos[0] > 30 or new_pos[1] < -30 or new_pos[1] > 30:
            return False
        
        # 1. Remove areas of triples involving idx
        old_areas = get_areas_involving(idx, best_points)
        for a in old_areas:
            hist[a] -= 1
            if hist[a] == 0:
                del hist[a]
                
        # 2. Update point
        old_pos = best_points[idx]
        best_points[idx] = new_pos
        
        # 3. Add areas of triples involving idx
        new_areas = get_areas_involving(idx, best_points)
        for a in new_areas:
            hist[a] += 1
            
        # 4. Check score
        current_score = max(hist.values()) if hist else 0
        
        if current_score > best_score:
            best_score = current_score
            best_hist = hist.copy()
            return True
        else:
            # Revert
            for a in new_areas:
                hist[a] -= 1
                if hist[a] == 0:
                    del hist[a]
            best_points[idx] = old_pos
            for a in old_areas:
                hist[a] += 1
            return False

    # Hill Climbing Loop with periodic deadline checks
    improved = True
    while improved:
        if time.time() - start_time > deadline:
            break
        improved = False
        for idx in range(n):
            if time.time() - start_time > deadline:
                break
            orig_x, orig_y = best_points[idx]
            # Try moves in a 5x5 neighborhood
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = orig_x + dx
                    new_y = orig_y + dy
                    new_pos = (new_x, new_y)
                    if try_move(idx, new_pos):
                        improved = True
                        
    # Final polish: try larger moves for remaining time
    if time.time() - start_time < deadline - 5:
        polish_improved = True
        while polish_improved and time.time() - start_time < deadline - 5:
            polish_improved = False
            for idx in range(n):
                if time.time() - start_time > deadline - 5:
                    break
                orig_x, orig_y = best_points[idx]
                # Larger neighborhood for polish
                for dx in range(-5, 6):
                    for dy in range(-5, 6):
                        if dx == 0 and dy == 0:
                            continue
                        new_pos = (orig_x + dx, orig_y + dy)
                        if try_move(idx, new_pos):
                            polish_improved = True

    return best_points
