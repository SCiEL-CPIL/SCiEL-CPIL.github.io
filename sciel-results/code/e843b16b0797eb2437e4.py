"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    def evaluate(circles):
        """Check validity and return sum of radii. Returns -inf if invalid."""
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        # Boundary check
        if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
            return -np.inf
        if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
            return -np.inf
        if np.any(r < 1e-12):
            return -np.inf
        # Non-overlap check
        n = len(circles)
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - 1e-9:
                    return -np.inf
        return np.sum(r)
    
    def init_circles():
        """Initialize circles in a grid with small radii."""
        side = int(np.ceil(np.sqrt(N)))
        positions = []
        for i in range(side):
            for j in range(side):
                if len(positions) >= N:
                    break
                positions.append(((i + 0.5) / side, (j + 0.5) / side))
        # Fill remaining if needed
        while len(positions) < N:
            positions.append((rng.random(), rng.random()))
        positions = positions[:N]
        x = np.array([p[0] for p in positions])
        y = np.array([p[1] for p in positions])
        # Initial radii: small, will grow
        r = np.full(N, 0.02)
        return np.column_stack([x, y, r])
    
    def forces(circles, temperature):
        """Compute forces on positions and radii."""
        x = circles[:, 0].copy()
        y = circles[:, 1].copy()
        r = circles[:, 2].copy()
        n = len(circles)
        
        fx = np.zeros(n)
        fy = np.zeros(n)
        fr = np.zeros(n)
        
        # Repulsion between overlapping circles
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-12:
                    dx = rng.random() - 0.5
                    dy = rng.random() - 0.5
                    dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    # Force proportional to overlap, with temperature scaling
                    f = 50.0 * overlap / temperature
                    fx[i] += f * dx / dist
                    fy[i] += f * dy / dist
                    fx[j] -= f * dx / dist
                    fy[j] -= f * dy / dist
                    # Radii should shrink if overlapping
                    fr[i] -= 10.0 * overlap / temperature
                    fr[j] -= 10.0 * overlap / temperature
        
        # Wall repulsion
        for i in range(n):
            # Left wall
            if x[i] - r[i] < 0:
                overlap = -x[i] + r[i]
                f = 50.0 * overlap / temperature
                fx[i] += f
                fr[i] -= 10.0 * overlap / temperature
            # Right wall
            if x[i] + r[i] > 1:
                overlap = x[i] + r[i] - 1
                f = 50.0 * overlap / temperature
                fx[i] -= f
                fr[i] -= 10.0 * overlap / temperature
            # Bottom wall
            if y[i] - r[i] < 0:
                overlap = -y[i] + r[i]
                f = 50.0 * overlap / temperature
                fy[i] += f
                fr[i] -= 10.0 * overlap / temperature
            # Top wall
            if y[i] + r[i] > 1:
                overlap = y[i] + r[i] - 1
                f = 50.0 * overlap / temperature
                fy[i] -= f
                fr[i] -= 10.0 * overlap / temperature
        
        # Growth force: push radii up
        fr += 1.0
        
        return fx, fy, fr
    
    def step(circles, temperature, dt):
        """One optimization step."""
        fx, fy, fr = forces(circles, temperature)
        
        x = circles[:, 0] + dt * fx
        y = circles[:, 1] + dt * fy
        r = circles[:, 2] + dt * fr
        
        # Clamp to valid ranges
        x = np.clip(x, 0.01, 0.99)
        y = np.clip(y, 0.01, 0.99)
        r = np.clip(r, 0.005, 0.45)
        
        return np.column_stack([x, y, r])
    
    # Main optimization loop with simulated annealing
    best_circles = None
    best_score = -np.inf
    
    # Try multiple restarts
    for restart in range(3):
        if time.time() > deadline - 10:
            break
        
        circles = init_circles()
        if restart > 0:
            # Perturb initial positions
            circles[:, 0] += rng.normal(0, 0.05, N)
            circles[:, 1] += rng.normal(0, 0.05, N)
            circles[:, 0] = np.clip(circles[:, 0], 0.05, 0.95)
            circles[:, 1] = np.clip(circles[:, 1], 0.05, 0.95)
        
        temperature = 1.0
        dt = 0.01
        max_iter = 5000
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            circles = step(circles, temperature, dt)
            
            # Anneal temperature
            temperature *= 0.9995
            dt *= 0.9999
            
            # Check score periodically
            if it % 50 == 0:
                score = evaluate(circles)
                if score > best_score:
                    best_score = score
                    best_circles = circles.copy()
        
        # Final evaluation
        score = evaluate(circles)
        if score > best_score:
            best_score = score
            best_circles = circles.copy()
    
    # If no valid solution found, fall back to grid
    if best_circles is None or best_score <= -1e6:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Final polish: try to improve with smaller steps
    if time.time() < deadline - 5:
        circles = best_circles.copy()
        temperature = 0.1
        dt = 0.001
        for it in range(2000):
            if time.time() > deadline - 3:
                break
            circles = step(circles, temperature, dt)
            if it % 100 == 0:
                score = evaluate(circles)
                if score > best_score:
                    best_score = score
                    best_circles = circles.copy()
    
    return best_circles
# EVOLVE-BLOCK-END
