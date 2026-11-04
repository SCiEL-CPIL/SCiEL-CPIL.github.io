import numpy as np

N = 21

def construct():
    best_value = -np.inf
    best_result = None
    
    # Try different aspect ratios for the rectangle
    # w + h = 2
    aspect_ratios = np.linspace(0.8, 1.2, 9) # w from 0.8 to 1.2
    
    for w in aspect_ratios:
        h = 2.0 - w
        
        # Initial state: Jittered grid
        aspect = w / h
        c = int(np.sqrt(N * aspect))
        if c < 1: c = 1
        r_rows = int(np.ceil(N / c))
        
        cw = w / c
        ch = h / r_rows
        
        positions = []
        radii = []
        
        for i in range(c):
            for j in range(r_rows):
                if len(positions) >= N:
                    break
                x = (i + 0.5) * cw + np.random.uniform(-0.1, 0.1) * cw
                y = (j + 0.5) * ch + np.random.uniform(-0.1, 0.1) * ch
                x = np.clip(x, 0.1, w - 0.1)
                y = np.clip(y, 0.1, h - 0.1)
                positions.append([x, y])
                r_val = 0.45 * min(cw, ch)
                radii.append(r_val)
        
        while len(positions) < N:
            positions.append([w/2 + np.random.uniform(-0.1, 0.1), h/2 + np.random.uniform(-0.1, 0.1)])
            radii.append(0.1)
            
        state = np.zeros(3 * N)
        for i in range(N):
            state[3*i] = positions[i][0]
            state[3*i+1] = positions[i][1]
            state[3*i+2] = radii[i]
            
        # Simulated Annealing Parameters
        T = 1.0
        T_min = 1e-6
        cooling_rate = 0.995
        max_iters = 15000
        
        def calculate_cost(state, w, h):
            xs = state[0::3]
            ys = state[1::3]
            rs = state[2::3]
            
            sum_r = np.sum(rs)
            
            # Vectorized Overlap Penalty
            dx = xs[:, None] - xs[None, :]
            dy = ys[:, None] - ys[None, :]
            dist = np.sqrt(dx**2 + dy**2)
            min_dist = rs[:, None] + rs[None, :]
            overlap = np.maximum(0, min_dist - dist)
            overlap_penalty = np.sum(overlap**2)
            
            # Wall Penalty
            wall_penalty = 0.0
            # Left
            wall_penalty += np.sum(np.maximum(0, rs - xs)**2)
            # Right
            wall_penalty += np.sum(np.maximum(0, rs - (w - xs))**2)
            # Bottom
            wall_penalty += np.sum(np.maximum(0, rs - ys)**2)
            # Top
            wall_penalty += np.sum(np.maximum(0, rs - (h - ys))**2)
            
            # Negative radius penalty
            neg_r_penalty = np.sum(np.maximum(0, -rs)**2)
            
            cost = -sum_r + 10.0 * overlap_penalty + 10.0 * wall_penalty + 10.0 * neg_r_penalty
            return cost

        current_cost = calculate_cost(state, w, h)
        
        for it in range(max_iters):
            new_state = state.copy()
            idx = np.random.randint(0, N)
            
            step = 0.1 * T + 0.01
            new_state[3*idx] += np.random.uniform(-step, step)
            new_state[3*idx+1] += np.random.uniform(-step, step)
            new_state[3*idx+2] += np.random.uniform(-step, step)
            
            new_state[3*idx] = np.clip(new_state[3*idx], 0, w)
            new_state[3*idx+1] = np.clip(new_state[3*idx+1], 0, h)
            new_state[3*idx+2] = np.clip(new_state[3*idx+2], 0, 1.0)
            
            new_cost = calculate_cost(new_state, w, h)
            
            if new_cost < current_cost:
                state = new_state
                current_cost = new_cost
            else:
                delta = new_cost - current_cost
                if np.random.rand() < np.exp(-delta / T):
                    state = new_state
                    current_cost = new_cost
            
            T *= cooling_rate
            if T < T_min:
                break
        
        # Evaluate final state strictly
        xs = state[0::3]
        ys = state[1::3]
        rs = state[2::3]
        
        valid = True
        if np.any(rs < 0):
            valid = False
        else:
            # Check walls
            if np.any(xs < rs) or np.any(w - xs < rs) or np.any(ys < rs) or np.any(h - ys < rs):
                valid = False
            else:
                # Check overlaps
                dx = xs[:, None] - xs[None, :]
                dy = ys[:, None] - ys[None, :]
                dist = np.sqrt(dx**2 + dy**2)
                min_dist = rs[:, None] + rs[None, :]
                # Check if any pair overlaps (dist < min_dist - epsilon)
                # We need to exclude diagonal (i==j)
                mask = ~np.eye(N, dtype=bool)
                overlaps = (dist < min_dist - 1e-9) & mask
                if np.any(overlaps):
                    valid = False
        
        if valid:
            val = np.sum(rs)
            if val > best_value:
                best_value = val
                best_result = state.copy()
                
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    circles = []
    for i in range(N):
        x = best_result[3*i]
        y = best_result[3*i+1]
        r = best_result[3*i+2]
        circles.append([x, y, r])
        
    return np.array(circles)
