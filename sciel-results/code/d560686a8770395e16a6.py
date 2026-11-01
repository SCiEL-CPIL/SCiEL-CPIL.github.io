"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """Pack 32 circles with disjoint interiors inside [0,1]^2, maximizing sum of radii."""
    best_value = -np.inf
    best_circles = None
    
    # We will use a physics-based optimization (Langevin Dynamics)
    # State vector: [x1, y1, r1, x2, y2, r2, ...]
    # Objective: Minimize E = -sum(r) + penalty_overlaps + penalty_walls
    
    def energy_and_grad(state):
        """
        Calculate energy and gradient for the state vector.
        state: shape (3*N,)
        """
        circles = state.reshape(N, 3)
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # 1. Objective term: -sum(r)
        # Gradient of -sum(r) w.r.t r is -1
        grad = np.zeros_like(state)
        grad[2::3] = -1.0 # d/dr (-r) = -1
        
        # 2. Overlap penalties
        # For each pair i, j:
        # penalty = max(0, ri + rj - dist_ij)^2
        # We use a soft version or just hard quadratic for overlapping parts.
        # dist = sqrt((xi-xj)^2 + (yi-yj)^2)
        # To avoid sqrt(0), add small epsilon.
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq + 1e-12)
                
                sum_r = r[i] + r[j]
                overlap = sum_r - dist
                
                if overlap > 0:
                    # Penalty = overlap^2
                    # d(Penalty)/dx_i = 2 * overlap * d(overlap)/dx_i
                    # d(overlap)/dx_i = -d(dist)/dx_i = -dx/dist
                    # So d(Penalty)/dx_i = 2 * overlap * (-dx/dist)
                    
                    factor = 2.0 * overlap / dist
                    grad[i*3] -= factor * dx
                    grad[i*3+1] -= factor * dy
                    grad[j*3] -= factor * dx  # Wait, d(overlap)/dx_j = -d(dist)/dx_j = -(-dx)/dist = dx/dist
                    # Let's re-verify signs.
                    # overlap = ri + rj - dist
                    # dist = sqrt((xi-xj)^2 + ...)
                    # d(dist)/dx_i = (xi-xj)/dist = dx/dist
                    # d(overlap)/dx_i = -dx/dist
                    # d(Pen)/dx_i = 2*overlap * (-dx/dist)
                    
                    # d(dist)/dx_j = (xj-xi)/dist = -dx/dist
                    # d(overlap)/dx_j = -(-dx/dist) = dx/dist
                    # d(Pen)/dx_j = 2*overlap * (dx/dist)
                    
                    grad[i*3] += -2.0 * overlap * (dx / dist)
                    grad[i*3+1] += -2.0 * overlap * (dy / dist)
                    grad[j*3] += 2.0 * overlap * (dx / dist)
                    grad[j*3+1] += 2.0 * overlap * (dy / dist)
                    
                    # Gradient w.r.t radii
                    # d(overlap)/dr_i = 1
                    # d(Pen)/dr_i = 2 * overlap
                    grad[i*3+2] += 2.0 * overlap
                    grad[j*3+2] += 2.0 * overlap
                    
        # 3. Wall penalties
        # Left: x - r < 0 => pen = (x-r)^2 if x-r<0
        # Right: x + r > 1 => pen = (x+r-1)^2 if x+r>1
        # Bottom: y - r < 0
        # Top: y + r > 1
        
        for i in range(N):
            # Left
            if x[i] - r[i] < 0:
                pen = (x[i] - r[i])**2
                # d(pen)/dx = 2(x-r)*1
                # d(pen)/dr = 2(x-r)*(-1)
                grad[i*3] += 2.0 * (x[i] - r[i])
                grad[i*3+2] += -2.0 * (x[i] - r[i])
            
            # Right
            if x[i] + r[i] > 1:
                pen = (x[i] + r[i] - 1)**2
                # d(pen)/dx = 2(x+r-1)*1
                # d(pen)/dr = 2(x+r-1)*1
                grad[i*3] += 2.0 * (x[i] + r[i] - 1)
                grad[i*3+2] += 2.0 * (x[i] + r[i] - 1)
                
            # Bottom
            if y[i] - r[i] < 0:
                grad[i*3+1] += 2.0 * (y[i] - r[i])
                grad[i*3+2] += -2.0 * (y[i] - r[i])
                
            # Top
            if y[i] + r[i] > 1:
                grad[i*3+1] += 2.0 * (y[i] + r[i] - 1)
                grad[i*3+2] += 2.0 * (y[i] + r[i] - 1)
                
        energy = -np.sum(r)
        # Add penalties to energy for tracking
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    energy += overlap**2
            if x[i] - r[i] < 0: energy += (x[i] - r[i])**2
            if x[i] + r[i] > 1: energy += (x[i] + r[i] - 1)**2
            if y[i] - r[i] < 0: energy += (y[i] - r[i])**2
            if y[i] + r[i] > 1: energy += (y[i] + r[i] - 1)**2
            
        return energy, grad

    def optimize(initial_state, steps=2000, lr=0.01, momentum=0.9):
        state = initial_state.copy()
        velocity = np.zeros_like(state)
        
        for step in range(steps):
            e, g = energy_and_grad(state)
            velocity = momentum * velocity - lr * g
            state = state + velocity
            
            # Clip state to reasonable bounds to prevent divergence
            # x, y in [-0.5, 1.5], r in [0, 0.6]
            state[0::3] = np.clip(state[0::3], -0.5, 1.5)
            state[1::3] = np.clip(state[1::3], -0.5, 1.5)
            state[2::3] = np.clip(state[2::3], 0.0, 0.6)
            
        # Post-process: ensure strict validity by shrinking radii slightly if needed
        circles = state.reshape(N, 3)
        # Simple repair: if invalid, shrink radii
        # We'll check validity and if invalid, return None or a repaired version
        return circles

    def check_valid(circles, eps=1e-7):
        n = len(circles)
        for i in range(n):
            x, y, r = circles[i]
            if x - r < -eps or x + r > 1 + eps or y - r < -eps or y + r > 1 + eps:
                return False
            if r < -eps:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                if dist < ri + rj - eps:
                    return False
        return True

    # Initial configurations
    configs = []
    
    # 1. Grid
    side = int(np.ceil(np.sqrt(N)))
    grid = []
    for i in range(side):
        for j in range(side):
            if len(grid) >= N: break
            grid.append([(i+0.5)/side, (j+0.5)/side, 0.5/side - 0.01])
    configs.append(np.array(grid).flatten())
    
    # 2. Random
    np.random.seed(42)
    for _ in range(5):
        x = np.random.rand(N)
        y = np.random.rand(N)
        r = np.full(N, 0.05)
        configs.append(np.column_stack((x, y, r)).flatten())
        
    # 3. Clustered
    for _ in range(2):
        # Place circles in 4 corners
        centers = [(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)]
        pts = []
        for c in centers:
            for _ in range(N//4):
                x = c[0] + np.random.randn()*0.05
                y = c[1] + np.random.randn()*0.05
                r = 0.05
                pts.append([x, y, r])
        configs.append(np.array(pts).flatten())

    for cfg in configs:
        circles = optimize(cfg, steps=3000, lr=0.005, momentum=0.9)
        
        # Repair: If invalid, shrink radii
        if not check_valid(circles):
            # Try to shrink radii to make it valid
            # Simple bisection on scale factor of radii
            valid = False
            for scale in np.linspace(0.5, 1.0, 10):
                test_circles = circles.copy()
                test_circles[:, 2] *= scale
                if check_valid(test_circles):
                    circles = test_circles
                    valid = True
                    break
            if not valid:
                continue
                
        value = np.sum(circles[:, 2])
        if value > best_value:
            best_value = value
            best_circles = circles.copy()

    if best_circles is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_circles
# EVOLVE-BLOCK-END
