"""Combined: vectorized evaluation + lattice candidates + targeted local search + smooth refinement."""
import time
import random
import numpy as np
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin under 90s

    # Precompute triple indices for n=30
    triple_indices = [(i, j, k) for i in range(30) for j in range(i+1, 30) for k in range(j+1, 30)]
    I = np.array([t[0] for t in triple_indices], dtype=np.int64)
    J = np.array([t[1] for t in triple_indices], dtype=np.int64)
    K = np.array([t[2] for t in triple_indices], dtype=np.int64)
    N_TRIPLES = len(I)

    def evaluate_fast(points):
        """
        Vectorized evaluation of max multiplicity of positive doubled areas.
        points: list of 30 (x, y) tuples.
        """
        pts = np.array(points, dtype=np.int64)
        x = pts[:, 0]
        y = pts[:, 1]
        
        # Compute doubled area (determinant) for all triples
        det = (x[J] - x[I]) * (y[K] - y[I]) - (y[J] - y[I]) * (x[K] - x[I])
        det = np.abs(det)
        
        # Filter positive areas
        pos_mask = det > 0
        if not np.any(pos_mask):
            return 0
        
        pos_dets = det[pos_mask]
        
        # Count multiplicities
        unique, counts = np.unique(pos_dets, return_counts=True)
        return int(counts.max())

    def try_config(points, current_best_score):
        """Try a configuration and return updated best if improved."""
        if len(points) != 30:
            return None, current_best_score
        if len(set(points)) != 30:
            return None, current_best_score
        if any(not (-30 <= px <= 30 and -30 <= py <= 30) for px, py in points):
            return None, current_best_score
        score = evaluate_fast(points)
        if score > current_best_score:
            return points, score
        return None, current_best_score

    # Start with a known good configuration: two parallel lines
    best_points = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
    best_score = evaluate_fast(best_points)

    # Phase 1: Systematic grid search
    factors_30 = [(1,30), (2,15), (3,10), (5,6), (6,5), (10,3), (15,2), (30,1)]
    
    for w, h in factors_30:
        for sx in range(1, min(31//w + 1, 31)):
            for sy in range(1, min(31//h + 1, 31)):
                if time.time() > deadline - 20:
                    break
                
                pts = [(i * sx, j * sy) for i in range(w) for j in range(h)]
                
                max_x = (w - 1) * sx
                max_y = (h - 1) * sy
                if max_x > 30 or max_y > 30:
                    continue
                    
                result, new_score = try_config(pts, best_score)
                if result is not None:
                    best_points = result
                    best_score = new_score

    # Centered grids
    for w, h in factors_30:
        for sx in range(1, 6):
            for sy in range(1, 6):
                if time.time() > deadline - 15:
                    break
                    
                width = (w - 1) * sx
                height = (h - 1) * sy
                
                if width > 60 or height > 60:
                    continue
                    
                cx = -width // 2
                cy = -height // 2
                
                pts = [(cx + i * sx, cy + j * sy) for i in range(w) for j in range(h)]
                
                result, new_score = try_config(pts, best_score)
                if result is not None:
                    best_points = result
                    best_score = new_score

    # Sheared grids
    for w, h in [(6,5), (5,6), (10,3), (3,10)]:
        for sx in range(1, 5):
            for sy in range(1, 5):
                for sxy in range(-3, 4):
                    if time.time() > deadline - 10:
                        break
                        
                    pts = [(i * sx + j * sxy, j * sy) for i in range(w) for j in range(h)]
                    
                    result, new_score = try_config(pts, best_score)
                    if result is not None:
                        best_points = result
                        best_score = new_score

    # Phase 2: Local search with vectorized evaluation
    random.seed(42)
    
    # Hill-climbing with single point moves
    improved = True
    max_iterations = 200
    iteration = 0
    while improved and time.time() < deadline - 10 and iteration < max_iterations:
        improved = False
        iteration += 1
        for idx in range(30):
            if time.time() > deadline - 10:
                break
            old_x, old_y = best_points[idx]
            # Try moves in a neighborhood
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = old_x + dx
                    new_y = old_y + dy
                    if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                        continue
                    new_points = list(best_points)
                    new_points[idx] = (new_x, new_y)
                    if len(set(new_points)) != 30:
                        continue
                    score = evaluate_fast(new_points)
                    if score > best_score:
                        best_score = score
                        best_points = new_points
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    # Pairwise swap local search
    if time.time() < deadline - 8:
        for _ in range(100):
            if time.time() > deadline - 8:
                break
            i, j = random.sample(range(30), 2)
            old_i = best_points[i]
            old_j = best_points[j]
            # Try moving point i to a new location and point j to another
            for dx1 in range(-2, 3):
                for dy1 in range(-2, 3):
                    if dx1 == 0 and dy1 == 0:
                        continue
                    new_i = (old_i[0] + dx1, old_i[1] + dy1)
                    if not (-30 <= new_i[0] <= 30 and -30 <= new_i[1] <= 30):
                        continue
                    for dx2 in range(-2, 3):
                        for dy2 in range(-2, 3):
                            if dx2 == 0 and dy2 == 0:
                                continue
                            new_j = (old_j[0] + dx2, old_j[1] + dy2)
                            if not (-30 <= new_j[0] <= 30 and -30 <= new_j[1] <= 30):
                                continue
                            new_points = list(best_points)
                            new_points[i] = new_i
                            new_points[j] = new_j
                            if len(set(new_points)) != 30:
                                continue
                            score = evaluate_fast(new_points)
                            if score > best_score:
                                best_score = score
                                best_points = new_points
                                break
                        if best_points[i] != old_i or best_points[j] != old_j:
                            break
                    if best_points[i] != old_i or best_points[j] != old_j:
                        break
                if best_points[i] != old_i or best_points[j] != old_j:
                    break

    # Phase 3: Random restarts with perturbation
    if time.time() < deadline - 5:
        random.seed(123)
        for restart in range(200):
            if time.time() > deadline - 3:
                break
            
            if random.random() < 0.5:
                # Small perturbation of best
                new_points = list(best_points)
                for _ in range(random.randint(1, 5)):
                    idx = random.randint(0, 29)
                    dx = random.randint(-3, 3)
                    dy = random.randint(-3, 3)
                    new_x = new_points[idx][0] + dx
                    new_y = new_points[idx][1] + dy
                    if -30 <= new_x <= 30 and -30 <= new_y <= 30:
                        new_points[idx] = (new_x, new_y)
            else:
                # Random grid-like structure
                w, h = random.choice([(6,5), (5,6), (10,3), (3,10), (15,2), (2,15)])
                sx = random.randint(1, max(1, 30 // w))
                sy = random.randint(1, max(1, 30 // h))
                cx = random.randint(-5, 5)
                cy = random.randint(-5, 5)
                new_points = [(cx + i*sx + random.randint(-1,1), cy + j*sy + random.randint(-1,1)) 
                              for i in range(w) for j in range(h)]
            
            result, new_score = try_config(new_points, best_score)
            if result is not None:
                best_points = result
                best_score = new_score

    # Phase 4: Fine-grained polish - try all single-point moves with larger neighborhood
    if time.time() < deadline - 2:
        for idx in range(30):
            if time.time() > deadline - 2:
                break
            old_x, old_y = best_points[idx]
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = old_x + dx
                    new_y = old_y + dy
                    if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                        continue
                    new_points = list(best_points)
                    new_points[idx] = (new_x, new_y)
                    if len(set(new_points)) != 30:
                        continue
                    score = evaluate_fast(new_points)
                    if score > best_score:
                        best_score = score
                        best_points = new_points
                        break
                else:
                    continue
                break

    return best_points
# EVOLVE-BLOCK-END
