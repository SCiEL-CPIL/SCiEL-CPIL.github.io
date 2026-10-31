def _force_directed_pack(w, h, max_iters=5000):
    n_cols = 5
    n_rows = 5
    positions = np.zeros((N, 2))
    radii = np.zeros(N)
    
    idx = 0
    for i in range(n_cols):
        for j in range(n_rows):
            if idx >= N: break
            positions[idx, 0] = (i + 0.5) * w / n_cols
            positions[idx, 1] = (j + 0.5) * h / n_rows
            radii[idx] = 0.5 * min(w / n_cols, h / n_rows) * 0.9
            idx += 1
            
    while idx < N:
        positions[idx, 0] = w * 0.5
        positions[idx, 1] = h * 0.5
        radii[idx] = 0.01
        idx += 1

    lr = 0.05
    growth_lr = 0.005
    decay = 0.999
    
    for it in range(max_iters):
        if time.time() - START_TIME > DEADLINE - 5:
            break
            
        forces = np.zeros((N, 2))
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist_sq = dx*dx + dy*dy
                min_dist = radii[i] + radii[j]
                if dist_sq < min_dist * min_dist:
                    dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-6
                    overlap = min_dist - dist
                    if dist > 1e-9:
                        fx = (dx / dist) * overlap
                        fy = (dy / dist) * overlap
                    else:
                        angle = np.random.rand() * 2 * np.pi
                        fx = np.cos(angle) * overlap
                        fy = np.sin(angle) * overlap
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
                    
        for i in range(N):
            if positions[i, 0] < radii[i]:
                forces[i, 0] += (radii[i] - positions[i, 0])
            if positions[i, 0] > w - radii[i]:
                forces[i, 0] -= (positions[i, 0] - (w - radii[i]))
            if positions[i, 1] < radii[i]:
                forces[i, 1] += (radii[i] - positions[i, 1])
            if positions[i, 1] > h - radii[i]:
                forces[i, 1] -= (positions[i, 1] - (h - radii[i]))
                
        positions += lr * forces
        for i in range(N):
            positions[i, 0] = np.clip(positions[i, 0], radii[i], w - radii[i])
            positions[i, 1] = np.clip(positions[i, 1], radii[i], h - radii[i])
            
        # Validity check
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j]:
                    valid = False
                    break
            if not valid: break
            
        if valid:
            radii += growth_lr
            for i in range(N):
                max_r = min(positions[i, 0], w - positions[i, 0], positions[i, 1], h - positions[i, 1])
                radii[i] = min(radii[i], max_r)
                
        lr *= decay
        growth_lr *= decay

    return positions, radii
