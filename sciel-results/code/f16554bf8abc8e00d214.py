import numpy as np
import random

N = 26

def construct():
    # Parameters for Langevin Dynamics
    num_steps = 20000
    dt = 0.005
    gamma = 0.5  # Friction
    k_rep = 10.0 # Repulsion strength
    k_growth = 0.1 # Growth pressure
    T_start = 0.5
    T_end = 0.01
    
    # Initialize positions and radii
    # Start with small radii to avoid initial overlaps
    x = np.random.uniform(0.1, 0.9, N)
    y = np.random.uniform(0.1, 0.9, N)
    r = np.full(N, 0.05)
    
    vx = np.zeros(N)
    vy = np.zeros(N)
    vr = np.zeros(N)
    
    best_sum = 0.0
    best_state = None

    for step in range(num_steps):
        # Annealing schedule
        alpha = step / num_steps
        T = T_start * (1 - alpha) + T_end * alpha
        
        # Calculate forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        fr = np.zeros(N)
        
        # Growth force (pushes radii to increase)
        fr += k_growth
        
        # Repulsion between circles
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
                
                min_dist = r[i] + r[j]
                
                if dist < min_dist:
                    # Overlap exists
                    overlap = min_dist - dist
                    # Force is proportional to overlap
                    force_mag = k_rep * overlap
                    
                    # Normalize direction
                    if dist > 1e-12:
                        nx = dx / dist
                        ny = dy / dist
                    else:
                        nx = 1.0
                        ny = 0.0
                    
                    # Apply force to centers
                    fx[i] += force_mag * nx
                    fy[i] += force_mag * ny
                    fx[j] -= force_mag * nx
                    fy[j] -= force_mag * ny
                    
                    # Apply force to radii (shrink both slightly to resolve overlap faster)
                    fr[i] -= force_mag * 0.5
                    fr[j] -= force_mag * 0.5

        # Boundary forces (keep circles inside [0,1]^2)
        for i in range(N):
            # Left wall
            if x[i] < r[i]:
                overlap = r[i] - x[i]
                force_mag = k_rep * overlap
                fx[i] += force_mag
                fr[i] -= force_mag * 0.5
            # Right wall
            if x[i] > 1.0 - r[i]:
                overlap = x[i] - (1.0 - r[i])
                force_mag = k_rep * overlap
                fx[i] -= force_mag
                fr[i] -= force_mag * 0.5
                
            # Bottom wall
            if y[i] < r[i]:
                overlap = r[i] - y[i]
                force_mag = k_rep * overlap
                fy[i] += force_mag
                fr[i] -= force_mag * 0.5
            # Top wall
            if y[i] > 1.0 - r[i]:
                overlap = y[i] - (1.0 - r[i])
                force_mag = k_rep * overlap
                fy[i] -= force_mag
                fr[i] -= force_mag * 0.5

        # Update velocities (Langevin dynamics)
        # v_new = (1 - gamma*dt) * v_old + dt * F + sqrt(2*k_B*T*gamma*dt) * noise
        # Simplified: v += dt * F - gamma * v * dt + noise
        noise_factor = np.sqrt(2 * T * gamma * dt)
        
        vx = vx * (1 - gamma * dt) + fx * dt + noise_factor * np.random.randn(N)
        vy = vy * (1 - gamma * dt) + fy * dt + noise_factor * np.random.randn(N)
        vr = vr * (1 - gamma * dt) + fr * dt + noise_factor * 0.5 * np.random.randn(N) # Less noise for radii
        
        # Update positions
        x += vx * dt
        y += vy * dt
        r += vr * dt
        
        # Enforce positivity of radii
        r = np.maximum(r, 1e-6)
        
        # Hard clamp positions to prevent flying out of bounds
        # This is a safety net; forces should handle it
        for i in range(N):
            if x[i] < 0:
                x[i] = 0
                vx[i] = abs(vx[i])
            elif x[i] > 1:
                x[i] = 1
                vx[i] = -abs(vx[i])
                
            if y[i] < 0:
                y[i] = 0
                vy[i] = abs(vy[i])
            elif y[i] > 1:
                y[i] = 1
                vy[i] = -abs(vy[i])
                
            # Ensure radius doesn't exceed half the square
            r[i] = min(r[i], 0.5)

        # Check validity and update best
        valid = True
        for i in range(N):
            if x[i] < r[i] or x[i] > 1 - r[i] or y[i] < r[i] or y[i] > 1 - r[i]:
                valid = False
                break
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                min_dist = r[i] + r[j]
                if dist_sq < min_dist * min_dist:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            current_sum = np.sum(r)
            if current_sum > best_sum:
                best_sum = current_sum
                best_state = (x.copy(), y.copy(), r.copy())

    # If we found a valid state, return it
    if best_state is not None:
        bx, by, br = best_state
        return np.column_stack((bx, by, br))
    else:
        # Fallback: Return a trivial grid if simulation failed completely
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
