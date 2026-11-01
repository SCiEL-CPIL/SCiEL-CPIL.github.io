import numpy as np
import time
import random
from scipy.optimize import minimize

N = 32
TIME_LIMIT = 110.0  # Safety margin for 120s limit

def construct():
    start_time = time.time()
    
    # Initializations
    def init_hexagonal():
        pts = []
        # Hexagonal packing in unit square
        # Approximate side length for N circles
        # For N=32, roughly 6x6 grid with some adjustments
        # Let's just use a dense hex grid and pick the best N
        r_est = 0.1
        dx = 2 * r_est
        dy = dx * np.sqrt(3) / 2
        rows = int(1 / dy) + 2
        cols = int(1 / dx) + 2
        candidates = []
        for i in range(rows):
            for j in range(cols):
                x = j * dx + (i % 2) * (dx / 2)
                y = i * dy
                if 0 <= x <= 1 and 0 <= y <= 1:
                    candidates.append((x, y))
        # Shuffle and take N
        random.shuffle(candidates)
        if len(candidates) < N:
            # Fallback to grid if hex is too sparse
            return init_grid()
        pts = candidates[:N]
        radii = [0.05] * N # Small initial radii
        return np.array(pts), np.array(radii)

    def init_grid():
        side = int(np.ceil(np.sqrt(N)))
        pts = []
        for i in range(side):
            for j in range(side):
                if len(pts) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                pts.append((x, y))
        if len(pts) < N:
            # Fill remaining randomly
            while len(pts) < N:
                pts.append((random.random(), random.random()))
        pts = np.array(pts[:N])
        radii = np.array([0.05] * N)
        return pts, radii

    def init_random():
        pts = np.random.rand(N, 2)
        radii = np.random.rand(N) * 0.05 + 0.01
        return pts, radii

    inits = [
        init_hexagonal(),
        init_grid(),
        init_random(),
        init_random(),
        init_random()
    ]

    best_score = -np.inf
    best_result = None

    for pts, radii in inits:
        if time.time() - start_time > TIME_LIMIT:
            break
            
        # Simulated Annealing
        state = np.concatenate([pts.flatten(), radii.flatten()])
        # Bounds for SA: x,y in [0,1], r in [0.01, 0.5]
        # We will handle bounds in the penalty or projection
        
        def energy(state):
            if time.time() - start_time > TIME_LIMIT:
                return np.inf
            p = state[:2*N].reshape(N, 2)
            r = state[2*N:]
            
            # Penalty for overlaps
            overlap_pen = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(p[i] - p[j])
                    if d < r[i] + r[j]:
                        overlap_pen += (r[i] + r[j] - d)**2
            
            # Penalty for boundaries
            bound_pen = 0.0
            for i in range(N):
                if p[i][0] < r[i]:
                    bound_pen += (r[i] - p[i][0])**2
                if p[i][0] > 1 - r[i]:
                    bound_pen += (p[i][0] - (1 - r[i]))**2
                if p[i][1] < r[i]:
                    bound_pen += (r[i] - p[i][1])**2
                if p[i][1] > 1 - r[i]:
                    bound_pen += (p[i][1] - (1 - r[i]))**2
            
            # Objective: Maximize sum of radii
            # We minimize -sum(r) + large_penalty * (overlap + bound)
            penalty = overlap_pen + bound_pen
            return -np.sum(r) + 1000.0 * penalty

        current_energy = energy(state)
        best_state = state.copy()
        best_local_energy = current_energy
        
        T = 0.1
        step_size = 0.05
        for step in range(5000): # Max steps per init
            if time.time() - start_time > TIME_LIMIT:
                break
                
            # Perturb
            new_state = state + np.random.normal(0, step_size, state.shape)
            # Clip positions to [0,1] and radii to [0.01, 0.5]
            new_state[:2*N] = np.clip(new_state[:2*N], 0, 1)
            new_state[2*N:] = np.clip(new_state[2*N:], 0.01, 0.5)
            
            new_energy = energy(new_state)
            
            if new_energy < current_energy or np.random.random() < np.exp((current_energy - new_energy) / T):
                state = new_state
                current_energy = new_energy
                if current_energy < best_local_energy:
                    best_local_energy = current_energy
                    best_state = state.copy()
            
            # Cool down
            T *= 0.995
            if T < 0.001:
                break
        
        # Evaluate best state from this init
        p = best_state[:2*N].reshape(N, 2)
        r = best_state[2*N:]
        
        # Ensure validity by shrinking radii slightly if needed
        # Check for any remaining overlaps or boundary violations
        valid = True
        for i in range(N):
            if p[i][0] < r[i] or p[i][0] > 1 - r[i] or p[i][1] < r[i] or p[i][1] > 1 - r[i]:
                valid = False
        for i in range(N):
            for j in range(i+1, N):
                if np.linalg.norm(p[i] - p[j]) < r[i] + r[j]:
                    valid = False
        
        if not valid:
            # Shrink radii to make it valid
            # Simple approach: shrink all by 5%
            r = r * 0.95
            # Re-check and shrink again if necessary
            for _ in range(10):
                valid = True
                for i in range(N):
                    if p[i][0] < r[i] or p[i][0] > 1 - r[i] or p[i][1] < r[i] or p[i][1] > 1 - r[i]:
                        valid = False
                for i in range(N):
                    for j in range(i+1, N):
                        if np.linalg.norm(p[i] - p[j]) < r[i] + r[j]:
                            valid = False
                if valid:
                    break
                r = r * 0.95
        
        score = np.sum(r)
        if score > best_score:
            best_score = score
            best_result = np.column_stack([p, r])

    if best_result is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_result
