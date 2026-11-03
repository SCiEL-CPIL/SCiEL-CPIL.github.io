"""Simulated Annealing circle packing in perimeter-4 rectangle."""
import numpy as np
import time
import random

N = 21

# EVOLVE-BLOCK-START
def construct():
    """
    Pack N circles in a rectangle of perimeter 4 (i.e., width + height = 2).
    Maximize sum of radii.
    
    Approach: Simulated Annealing with penalty-based objective.
    """
    best_result = None
    best_score = -np.inf
    
    # Try several aspect ratios
    # w + h = 2
    # aspect = w/h
    # w = 2*aspect/(1+aspect), h = 2/(1+aspect)
    aspects = [1.0, 1.2, 0.8, 1.5, 0.66, 2.0, 0.5]
    
    # Time budget per aspect
    time_budget = 10.0
    
    for aspect in aspects:
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        
        # Initialize
        # Grid initialization is often better than random for SA to start in a "good" basin
        side = int(np.ceil(np.sqrt(N)))
        positions = np.zeros((N, 2))
        radii = np.full(N, 0.05)
        
        for i in range(N):
            row = i // side
            col = i % side
            positions[i, 0] = (col + 0.5) * w / side
            positions[i, 1] = (row + 0.5) * h / side
            
        # Run SA
        result = _simulate_annealing(positions, radii, w, h, time_budget)
        
        if result is not None:
            score = np.sum(result[:, 2])
            if score > best_score:
                best_score = score
                best_result = result
                
    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        w, h = 1.0, 1.0
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result

def _simulate_annealing(pos, rad, w, h, time_budget):
    """
    Simulated Annealing to maximize sum of radii.
    """
    x = pos[:, 0].copy()
    y = pos[:, 1].copy()
    r = rad.copy()
    
    C_ov = 10.0
    C_bd = 10.0
    
    def total_cost():
        cost = 0.0
        # Overlaps
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                min_dist = r[i] + r[j]
                if dist_sq < min_dist * min_dist:
                    dist = np.sqrt(dist_sq)
                    overlap = min_dist - dist
                    cost += C_ov * overlap * overlap
        # Boundaries
        for i in range(N):
            if x[i] < r[i]:
                cost += C_bd * (r[i] - x[i])**2
            if x[i] > w - r[i]:
                cost += C_bd * (x[i] - (w - r[i]))**2
            if y[i] < r[i]:
                cost += C_bd * (r[i] - y[i])**2
            if y[i] > h - r[i]:
                cost += C_bd * (y[i] - (h - r[i]))**2
                
        cost -= np.sum(r)
        return cost

    current_cost = total_cost()
    best_cost = current_cost
    best_x = x.copy()
    best_y = y.copy()
    best_r = r.copy()
    
    T = 1.0
    T_min = 0.001
    alpha = 0.999 # Cooling rate
    
    start_time = time.time()
    
    while T > T_min and (time.time() - start_time) < time_budget:
        # Select random circle
        i = random.randint(0, N-1)
        
        # Perturbation
        # Move position
        dx = random.uniform(-0.05, 0.05)
        dy = random.uniform(-0.05, 0.05)
        
        # Adjust radius
        dr = random.uniform(-0.01, 0.01)
        
        old_x = x[i]
        old_y = y[i]
        old_r = r[i]
        
        x[i] += dx
        y[i] += dy
        r[i] += dr
        
        # Clamp radius to positive
        if r[i] < 0.01:
            r[i] = 0.01
            
        # Calculate new cost
        new_cost = total_cost()
        delta_cost = new_cost - current_cost
        
        if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_x = x.copy()
                best_y = y.copy()
                best_r = r.copy()
        else:
            # Revert
            x[i] = old_x
            y[i] = old_y
            r[i] = old_r
            
        T *= alpha
        
    # Final validation of best state
    valid = True
    for i in range(N):
        if best_r[i] <= 0:
            valid = False
            break
        if best_x[i] < best_r[i] - 1e-6 or best_x[i] > w - best_r[i] + 1e-6:
            valid = False
            break
        if best_y[i] < best_r[i] - 1e-6 or best_y[i] > h - best_r[i] + 1e-6:
            valid = False
            break
            
    if valid:
        for i in range(N):
            for j in range(i+1, N):
                dx = best_x[i] - best_x[j]
                dy = best_y[i] - best_y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < best_r[i] + best_r[j] - 1e-6:
                    valid = False
                    break
            if not valid:
                break
                
    if valid:
        result = np.zeros((N, 3))
        result[:, 0] = best_x
        result[:, 1] = best_y
        result[:, 2] = best_r
        return result
    else:
        return None

# EVOLVE-BLOCK-END
