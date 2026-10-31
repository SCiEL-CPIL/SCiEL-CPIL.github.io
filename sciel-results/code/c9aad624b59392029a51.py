import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    
    # 1. Initialize Hexagonal Packing
    # Rows: 4, 5, 4, 4, 4
    rows = [4, 5, 4, 4, 4]
    centers = []
    r0 = 1.0
    h = r0 * np.sqrt(3)
    for i, count in enumerate(rows):
        y = i * h
        x_start = -(count - 1) * r0
        for j in range(count):
            x = x_start + j * 2 * r0
            if i % 2 == 1:
                x += r0
            centers.append([x, y])
            
    centers = np.array(centers)
    radii = np.ones(N)
    
    x = centers[:, 0].copy()
    y = centers[:, 1].copy()
    r = radii.copy()
    
    # Center and Scale
    cx = np.mean(x)
    cy = np.mean(y)
    x -= cx
    y -= cy
    
    W = np.max(x + r) - np.min(x - r)
    H = np.max(y + r) - np.min(y - r)
    
    scale = 2.0 / (W + H)
    x *= scale
    y *= scale
    r *= scale
    W *= scale
    H *= scale
    
    # Gradient Descent
    lr = 0.05
    momentum = 0.9
    vx = np.zeros_like(x)
    vy = np.zeros_like(y)
    vr = np.zeros_like(r)
    
    best_sum_r = np.sum(r)
    best_state = (x.copy(), y.copy(), r.copy(), W, H)
    
    for step in range(20000):
        if time.time() - start_time > 100:
            break
            
        # Compute Gradients
        gx = np.zeros_like(x)
        gy = np.zeros_like(y)
        gr = np.zeros_like(r)
        
        # 1. Overlap Penalty
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(dx**2 + dy**2 + 1e-12)
        min_dist = r[:, np.newaxis] + r[np.newaxis, :]
        
        viol = np.maximum(0, min_dist - dist)
        mask = ~np.eye(N, dtype=bool)
        active = (viol > 1e-6) & mask
        
        factor = np.zeros_like(dist)
        factor[active] = 2 * viol[active] / dist[active]
        
        gx += np.sum(factor * dx, axis=1)
        gy += np.sum(factor * dy, axis=1)
        gr += np.sum(2 * viol, axis=1)
        
        # 2. Boundary Penalties
        viol_l = np.maximum(0, r - x)
        gx -= viol_l
        gr += viol_l
        
        viol_r = np.maximum(0, x + r - W)
        gx += viol_r
        gr += viol_r
        
        viol_b = np.maximum(0, r - y)
        gy -= viol_b
        gr += viol_b
        
        viol_t = np.maximum(0, y + r - H)
        gy += viol_t
        gr += viol_t
        
        # 3. Objective: Maximize sum(r)
        gr -= 1.0
        
        # Update with Momentum
        vx = momentum * vx - lr * gx
        vy = momentum * vy - lr * gy
        vr = momentum * vr - lr * gr
        
        x += vx
        y += vy
        r += vr
        
        # Project to Perimeter Constraint
        cx = (np.min(x - r) + np.max(x + r)) / 2
        cy = (np.min(y - r) + np.max(y + r)) / 2
        x -= cx
        y -= cy
        
        W_new = np.max(x + r) - np.min(x - r)
        H_new = np.max(y + r) - np.min(y - r)
        
        if W_new + H_new > 1e-6:
            scale = 2.0 / (W_new + H_new)
            x *= scale
            y *= scale
            r *= scale
            W = W_new * scale
            H = H_new * scale
        else:
            W = 1.0
            H = 1.0
            
        current_sum_r = np.sum(r)
        if current_sum_r > best_sum_r:
            best_sum_r = current_sum_r
            best_state = (x.copy(), y.copy(), r.copy(), W, H)
            
    x_best, y_best, r_best, W_best, H_best = best_state
    
    x_out = x_best + W_best / 2
    y_out = y_best + H_best / 2
    
    result = np.column_stack((x_out, y_out, r_best))
    
    return result
