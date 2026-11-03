import numpy as np
import time
import random

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    best_score = -np.inf
    best_circles = None

    def validate(circles, w, h, tol=1e-7):
        if w <= 0 or h <= 0:
            return False
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        if np.any(r < -tol):
            return False
            
        # Boundary check
        if np.any(x < r - tol) or np.any(x > w - r + tol) or np.any(y < r - tol) or np.any(y > h - r + tol):
            return False
            
        # Overlap check
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if dist < r[i] + r[j] - tol:
                    return False
        return True

    def _energy(circles, w, h):
        """
        Calculate energy: negative sum of radii + penalty for overlaps and boundary violations.
        Lower is better.
        """
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Sum of radii (we want to maximize, so negate)
        e = -np.sum(r)
        
        # Overlap penalty
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    e += 10.0 * overlap * overlap
        
        # Boundary penalty
        for i in range(N):
            if x[i] < r[i]:
                e += 10.0 * (r[i] - x[i]) ** 2
            if x[i] > w - r[i]:
                e += 10.0 * (x[i] - (w - r[i])) ** 2
            if y[i] < r[i]:
                e += 10.0 * (r[i] - y[i]) ** 2
            if y[i] > h - r[i]:
                e += 10.0 * (y[i] - (h - r[i])) ** 2
        
        return e

    def _local_search(w_init, max_time=100):
        h_init = 2.0 - w_init
        
        # Initialization: Grid
        cols = 5
        rows = 5
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        count = 0
        # Use a slightly jittered grid to break symmetry
        for i in range(cols):
            for j in range(rows):
                if count < N:
                    x[count] = (i + 0.5) * w_init / cols + np.random.uniform(-0.01, 0.01)
                    y[count] = (j + 0.5) * h_init / rows + np.random.uniform(-0.01, 0.01)
                    cell_w = w_init / cols
                    cell_h = h_init / rows
                    r[count] = 0.3 * min(cell_w, cell_h)
                    count += 1
        
        # Ensure initial validity (roughly)
        # Simple repair: clamp and push
        for _ in range(100):
            # Clamp
            for i in range(N):
                x[i] = np.clip(x[i], r[i], w_init - r[i])
                y[i] = np.clip(y[i], r[i], h_init - r[i])
            
            # Push apart
            moved = False
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[j] - x[i]
                    dy = y[j] - y[i]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = r[i] + r[j]
                    if dist < min_dist:
                        if dist < 1e-9:
                            angle = np.random.uniform(0, 2*np.pi)
                            dx = np.cos(angle)
                            dy = np.sin(angle)
                            dist = 1.0
                        else:
                            dx /= dist
                            dy /= dist
                        overlap = min_dist - dist
                        push = 0.5 * overlap
                        x[i] -= dx * push
                        y[i] -= dy * push
                        x[j] += dx * push
                        y[j] += dy * push
                        moved = True
            
            if not moved:
                break
            
            # Re-clamp
            for i in range(N):
                x[i] = np.clip(x[i], r[i], w_init - r[i])
                y[i] = np.clip(y[i], r[i], h_init - r[i])
        
        circles = np.column_stack((x, y, r))
        current_energy = _energy(circles, w_init, h_init)
        current_score = np.sum(r)
        
        last_improvement_time = time.time()
        temp = 0.05
        min_temp = 0.001
        
        while time.time() < deadline - 5:
            if time.time() - last_improvement_time > 5:
                temp *= 0.9
                if temp < min_temp:
                    break
            
            # Try to improve
            improved = False
            for _ in range(20):
                if time.time() > deadline - 5:
                    break
                
                # Pick a random circle
                i = np.random.randint(N)
                
                # Try to increase radius
                old_r = r[i]
                r[i] += 0.005
                
                # Check if this causes significant overlap
                # We don't fully resolve here, we just evaluate energy
                new_circles = circles.copy()
                new_circles[i, 2] = r[i]
                
                new_energy = _energy(new_circles, w_init, h_init)
                
                # Accept if energy decreases (improvement) or with probability exp(-dE/T)
                delta_e = new_energy - current_energy
                if delta_e < 0:
                    circles = new_circles
                    current_energy = new_energy
                    current_score = np.sum(circles[:, 2])
                    last_improvement_time = time.time()
                    improved = True
                else:
                    # Accept with probability
                    if np.random.rand() < np.exp(-delta_e / max(temp, 1e-9)):
                        circles = new_circles
                        current_energy = new_energy
                        current_score = np.sum(circles[:, 2])
                        # Don't update last_improvement_time for SA acceptances to keep cooling going?
                        # Actually, better to only update on true improvement to avoid stalling.
                
                # Now, try to repair overlaps caused by the radius increase
                # Do a few steps of force-directed relaxation
                for _ in range(10):
                    # Calculate forces
                    fx = np.zeros(N)
                    fy = np.zeros(N)
                    
                    # Overlap forces
                    for k in range(N):
                        for l in range(k + 1, N):
                            dx = circles[l, 0] - circles[k, 0]
                            dy = circles[l, 1] - circles[k, 1]
                            dist = np.sqrt(dx*dx + dy*dy)
                            min_dist = circles[k, 2] + circles[l, 2]
                            if dist < min_dist:
                                if dist < 1e-9:
                                    angle = np.random.uniform(0, 2*np.pi)
                                    dx = np.cos(angle)
                                    dy = np.sin(angle)
                                    dist = 1.0
                                else:
                                    dx /= dist
                                    dy /= dist
                                force = 0.1 * (min_dist - dist)
                                fx[k] -= dx * force
                                fy[k] -= dy * force
                                fx[l] += dx * force
                                fy[l] += dy * force
                    
                    # Boundary forces
                    for k in range(N):
                        if circles[k, 0] < circles[k, 2]:
                            fx[k] += 0.5 * (circles[k, 2] - circles[k, 0])
                        elif circles[k, 0] > w_init - circles[k, 2]:
                            fx[k] -= 0.5 * (circles[k, 0] - (w_init - circles[k, 2]))
                        
                        if circles[k, 1] < circles[k, 2]:
                            fy[k] += 0.5 * (circles[k, 2] - circles[k, 1])
                        elif circles[k, 1] > h_init - circles[k, 2]:
                            fy[k] -= 0.5 * (circles[k, 1] - (h_init - circles[k, 2]))
                    
                    # Apply forces
                    step = 0.01
                    circles[:, 0] += fx * step
                    circles[:, 1] += fy * step
                    
                    # Clamp
                    for k in range(N):
                        circles[k, 0] = np.clip(circles[k, 0], circles[k, 2], w_init - circles[k, 2])
                        circles[k, 1] = np.clip(circles[k, 1], circles[k, 2], h_init - circles[k, 2])
                
                # Check if the relaxation helped or hurt
                relaxed_energy = _energy(circles, w_init, h_init)
                if relaxed_energy < current_energy:
                    current_energy = relaxed_energy
                    current_score = np.sum(circles[:, 2])
                    last_improvement_time = time.time()
                    improved = True
                else:
                    # Revert the radius increase and position changes if they made things worse?
                    # This is tricky in SA. Let's just keep the state if it's not terrible.
                    # Actually, if we increased r and then relaxed, and energy went up, we should probably revert.
                    # But we already overwrote 'circles'.
                    # Let's simplify: only accept the radius increase if the *relaxed* state is better than the *pre-increase* state.
                    pass # The logic above updates current_energy if relaxed is better.
            
            if not improved and time.time() - last_improvement_time > 10:
                # Random restart / perturbation
                for _ in range(5):
                    i = np.random.randint(N)
                    circles[i, 0] = np.random.uniform(circles[i, 2], w_init - circles[i, 2])
                    circles[i, 1] = np.random.uniform(circles[i, 2], h_init - circles[i, 2])
                # Repair
                for _ in range(20):
                    for i in range(N):
                        for j in range(i + 1, N):
                            dx = circles[j, 0] - circles[i, 0]
                            dy = circles[j, 1] - circles[i, 1]
                            dist = np.sqrt(dx*dx + dy*dy)
                            min_dist = circles[i, 2] + circles[j, 2]
                            if dist < min_dist:
                                if dist < 1e-9:
                                    angle = np.random.uniform(0, 2*np.pi)
                                    dx = np.cos(angle)
                                    dy = np.sin(angle)
                                    dist = 1.0
                                else:
                                    dx /= dist
                                    dy /= dist
                                overlap = min_dist - dist
                                push = 0.5 * overlap
                                circles[i, 0] -= dx * push
                                circles[i, 1] -= dy * push
                                circles[j, 0] += dx * push
                                circles[j, 1] += dy * push
                    for i in range(N):
                        circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], w_init - circles[i, 2])
                        circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], h_init - circles[i, 2])
                
                current_energy = _energy(circles, w_init, h_init)
                current_score = np.sum(circles[:, 2])
                last_improvement_time = time.time()
                
        # Final validation and cleanup
        # Ensure strict validity
        for _ in range(50):
            valid = True
            for i in range(N):
                for j in range(i + 1, N):
                    dist = np.sqrt((circles[i, 0] - circles[j, 0])**2 + (circles[i, 1] - circles[j, 1])**2)
                    if dist < circles[i, 2] + circles[j, 2] - 1e-9:
                        valid = False
                        # Push apart
                        dx = circles[j, 0] - circles[i, 0]
                        dy = circles[j, 1] - circles[i, 1]
                        if dist < 1e-9:
                            angle = np.random.uniform(0, 2*np.pi)
                            dx = np.cos(angle)
                            dy = np.sin(angle)
                            dist = 1.0
                        else:
                            dx /= dist
                            dy /= dist
                        overlap = circles[i, 2] + circles[j, 2] - dist
                        push = 0.5 * overlap
                        circles[i, 0] -= dx * push
                        circles[i, 1] -= dy * push
                        circles[j, 0] += dx * push
                        circles[j, 1] += dy * push
            
            for i in range(N):
                circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], w_init - circles[i, 2])
                circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], h_init - circles[i, 2])
            
            if valid:
                break
        
        # If still invalid, shrink radii slightly to make it valid
        if not validate(circles, w_init, h_init, 1e-7):
             # This shouldn't happen often, but if it does, shrink all radii by 10%
             circles[:, 2] *= 0.9
             # Re-repair
             for _ in range(20):
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = circles[j, 0] - circles[i, 0]
                        dy = circles[j, 1] - circles[i, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        min_dist = circles[i, 2] + circles[j, 2]
                        if dist < min_dist:
                            if dist < 1e-9:
                                angle = np.random.uniform(0, 2*np.pi)
                                dx = np.cos(angle)
                                dy = np.sin(angle)
                                dist = 1.0
                            else:
                                dx /= dist
                                dy /= dist
                            overlap = min_dist - dist
                            push = 0.5 * overlap
                            circles[i, 0] -= dx * push
                            circles[i, 1] -= dy * push
                            circles[j, 0] += dx * push
                            circles[j, 1] += dy * push
                for i in range(N):
                    circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], w_init - circles[i, 2])
                    circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], h_init - circles[i, 2])
        
        return circles, np.sum(circles[:, 2])

    # Try several aspect ratios
    # Square is 1.0. Elongated might be better?
    aspect_ratios = [0.9, 1.0, 1.1]
    
    for w in aspect_ratios:
        if time.time() > deadline - 15:
            break
            
        try:
            circles, score = _local_search(w)
            
            # Final validation
            if validate(circles, w, 2.0 - w, 1e-7):
                if score > best_score:
                    best_score = score
                    best_circles = circles
        except Exception:
            continue

    if best_circles is None:
        # Fallback: Simple grid in a square
        w = 1.0
        h = 1.0
        cells = []
        for i in range(5):
            for j in range(5):
                if len(cells) >= N:
                    break
                x = (i + 0.5) * 0.2
                y = (j + 0.5) * 0.2
                r = 0.08
                cells.append([x, y, r])
        best_circles = np.array(cells)
        
    return best_circles
