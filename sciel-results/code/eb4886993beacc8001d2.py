import time
import numpy as np
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    n = 10
    tol = 1e-7
    deadline = time.time() + 595  # Safety margin
    
    def is_feasible(centers, sides):
        """Check if the configuration is valid."""
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
        if np.any(centers - sides[:, None] / 2 < -tol) or np.any(centers + sides[:, None] / 2 > 1 + tol):
            return False
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                    return False
        return True
    
    def overlap_penalty(centers, sides, penalty_weight=100):
        """Calculate penalty for overlaps and boundary violations."""
        penalty = 0.0
        # Boundary penalty
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            left = cx - s / 2
            right = cx + s / 2
            bottom = cy - s / 2
            top = cy + s / 2
            if left < -tol:
                penalty += penalty_weight * (-left - tol) ** 2
            if right > 1 + tol:
                penalty += penalty_weight * (right - 1 - tol) ** 2
            if bottom < -tol:
                penalty += penalty_weight * (-bottom - tol) ** 2
            if top > 1 + tol:
                penalty += penalty_weight * (top - 1 - tol) ** 2
        
        # Overlap penalty
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                # If both dx < t and dy < t, there's overlap
                overlap_x = max(0, t - dx)
                overlap_y = max(0, t - dy)
                if overlap_x > 0 and overlap_y > 0:
                    penalty += penalty_weight * (overlap_x * overlap_y) ** 0.5
        
        # Small side penalty (encourage larger squares)
        penalty += 0.01 * np.sum(np.maximum(0, 0.1 - sides) ** 2)
        
        return penalty
    
    def objective(params):
        """Objective function: minimize negative sum of sides + penalties."""
        centers = params[:2 * n].reshape(n, 2)
        sides = params[2 * n:]
        
        # Apply bounds
        centers = np.clip(centers, 0, 1)
        sides = np.maximum(sides, 1e-6)
        
        # Penalty for being outside [0, 1]
        boundary_penalty = 0.0
        for i in range(n):
            if centers[i, 0] - sides[i] / 2 < -tol:
                boundary_penalty += 1000 * (-tol - (centers[i, 0] - sides[i] / 2)) ** 2
            if centers[i, 0] + sides[i] / 2 > 1 + tol:
                boundary_penalty += 1000 * (centers[i, 0] + sides[i] / 2 - 1 - tol) ** 2
            if centers[i, 1] - sides[i] / 2 < -tol:
                boundary_penalty += 1000 * (-tol - (centers[i, 1] - sides[i] / 2)) ** 2
            if centers[i, 1] + sides[i] / 2 > 1 + tol:
                boundary_penalty += 1000 * (centers[i, 1] + sides[i] / 2 - 1 - tol) ** 2
        
        overlap_pen = overlap_penalty(centers, sides, penalty_weight=500)
        
        return -np.sum(sides) + boundary_penalty + overlap_pen
    
    def expand_square_i(centers, sides, i, max_iter=100):
        """Try to expand square i as much as possible by adjusting its position."""
        best_centers = centers.copy()
        best_sides = sides.copy()
        current_score = np.sum(sides)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            # Try expanding side
            new_s = sides[i] * 1.01 + 0.001
            if new_s > 1.0:
                break
            
            # Check if we can place square i with side new_s
            # Try several candidate positions
            best_pos = None
            best_overlap = float('inf')
            
            # Candidate positions: corners, edges, random
            candidates = []
            # Corners with the new size
            s2 = new_s / 2
            candidates.append([s2, s2])
            candidates.append([1 - s2, s2])
            candidates.append([s2, 1 - s2])
            candidates.append([1 - s2, 1 - s2])
            # Current position
            candidates.append(centers[i].tolist())
            # Some random positions
            for _ in range(10):
                candidates.append([
                    np.random.uniform(s2, 1 - s2),
                    np.random.uniform(s2, 1 - s2)
                ])
            
            for cx, cy in candidates:
                test_centers = centers.copy()
                test_centers[i] = [cx, cy]
                
                # Check feasibility with new side
                test_sides = sides.copy()
                test_sides[i] = new_s
                
                # Quick check: is it feasible?
                feasible = True
                for j in range(n):
                    if i == j:
                        continue
                    t = (new_s + sides[j]) / 2 - tol
                    if abs(cx - centers[j, 0]) < t and abs(cy - centers[j, 1]) < t:
                        feasible = False
                        break
                
                if feasible:
                    # Check boundary
                    if cx - s2 >= -tol and cx + s2 <= 1 + tol and cy - s2 >= -tol and cy + s2 <= 1 + tol:
                        return test_centers, test_sides
            
            # If we couldn't expand, break
            break
        
        return best_centers, best_sides
    
    # Start with the known good 2-large + 8-small configuration
    centers = np.array([
        [0.25, 0.25],
        [0.75, 0.75],
        [0.125, 0.625],
        [0.375, 0.625],
        [0.125, 0.875],
        [0.375, 0.875],
        [0.625, 0.125],
        [0.875, 0.125],
        [0.625, 0.375],
        [0.875, 0.375],
    ])
    sides = np.array([
        0.5, 0.5,
        0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25
    ])
    
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_score = np.sum(sides)
    
    # Phase 1: Try to expand squares one by one
    for i in range(n):
        if time.time() > deadline:
            break
        new_centers, new_sides = expand_square_i(best_centers, best_sides, i)
        if np.sum(new_sides) > best_score + 1e-9:
            if is_feasible(new_centers, new_sides):
                best_centers = new_centers
                best_sides = new_sides
                best_score = np.sum(new_sides)
    
    # Phase 2: Multiple passes of expansion
    for _ in range(3):
        if time.time() > deadline:
            break
        for i in range(n):
            if time.time() > deadline:
                break
            new_centers, new_sides = expand_square_i(best_centers, best_sides, i)
            if np.sum(new_sides) > best_score + 1e-9:
                if is_feasible(new_centers, new_sides):
                    best_centers = new_centers
                    best_sides = new_sides
                    best_score = np.sum(new_sides)
    
    # Phase 3: Local search - try small perturbations
    np.random.seed(42)
    for _ in range(50):
        if time.time() > deadline:
            break
        
        trial_centers = best_centers.copy()
        trial_sides = best_sides.copy()
        
        # Perturb a random square
        i = np.random.randint(n)
        delta = 0.01 * np.random.randn(2)
        trial_centers[i] += delta
        
        # Try to expand this square
        for _ in range(5):
            new_s = trial_sides[i] * 1.005 + 0.0005
            if new_s > 1.0:
                break
            
            s2 = new_s / 2
            # Try to find a valid position
            valid = False
            for cx, cy in [trial_centers[i].tolist(),
                          [np.clip(trial_centers[i, 0], s2, 1 - s2), trial_centers[i, 1]],
                          [trial_centers[i, 0], np.clip(trial_centers[i, 1], s2, 1 - s2)]]:
                test_feasible = True
                for j in range(n):
                    if i == j:
                        continue
                    t = (new_s + trial_sides[j]) / 2 - tol
                    if abs(cx - trial_centers[j, 0]) < t and abs(cy - trial_centers[j, 1]) < t:
                        test_feasible = False
                        break
                if test_feasible and cx - s2 >= -tol and cx + s2 <= 1 + tol and cy - s2 >= -tol and cy + s2 <= 1 + tol:
                    trial_centers[i] = [cx, cy]
                    trial_sides[i] = new_s
                    valid = True
                    break
            
            if not valid:
                break
        
        if np.sum(trial_sides) > best_score + 1e-9:
            if is_feasible(trial_centers, trial_sides):
                best_centers = trial_centers
                best_sides = trial_sides
                best_score = np.sum(trial_sides)
    
    # Final feasibility check
    if not is_feasible(best_centers, best_sides):
        # Fallback to the original configuration
        return centers, sides
    
    return best_centers, best_sides
