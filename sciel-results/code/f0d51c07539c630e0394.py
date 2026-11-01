def _force_directed_step(c, r_target, steps=100, learning_rate=0.1):
    c = c.copy()
    n = len(c)
    
    for step in range(steps):
        forces = np.zeros_like(c)
        
        # Pairwise repulsion (Vectorized)
        # D[i,j] = distance between i and j
        D = np.sqrt(np.sum((c[:, np.newaxis] - c[np.newaxis, :])**2, axis=2))
        np.fill_diagonal(D, 1e9) # Ignore self
        
        # Overlap mask
        overlap_mask = D < 2 * r_target
        # Force magnitude: proportional to overlap
        # F = (2r - D) / (2r) * direction
        # We only want upper triangle to avoid double counting? 
        # Forces are symmetric. F_ij = -F_ji.
        # If we sum over all j, we get net force.
        
        # Normalize vectors
        # D is distance. Direction is (c_i - c_j) / D
        # We can compute force matrix directly
        # F_ij = (2r - D_ij) * (c_i - c_j) / D_ij^2 ? No, simple repulsion is fine.
        
        # Let's use F = k * (2r - D) / D^2 * (c_i - c_j)
        # To avoid division by zero, add epsilon
        D_safe = np.where(D > 1e-9, D, 1e-9)
        
        # Compute vector differences
        diff = c[:, np.newaxis] - c[np.newaxis, :]
        
        # Force magnitude factor
        # Only apply if overlapping
        mag = np.zeros_like(D)
        mag[overlap_mask] = (2 * r_target - D[overlap_mask]) / D_safe[overlap_mask]
        
        # Force vector for each pair i->j
        # F_ij = mag[i,j] * diff[i,j]
        # Sum over j for each i
        forces += np.sum(mag * diff, axis=1)
        
        # Boundary repulsion
        for i in range(n):
            if c[i][0] < -0.5 + r_target:
                forces[i][0] += (-0.5 + r_target - c[i][0])
            if c[i][0] > 0.5 - r_target:
                forces[i][0] -= (c[i][0] - (0.5 - r_target))
            if c[i][1] < -0.5 + r_target:
                forces[i][1] += (-0.5 + r_target - c[i][1])
            if c[i][1] > 0.5 - r_target:
                forces[i][1] -= (c[i][1] - (0.5 - r_target))
        
        c += learning_rate * forces
        c = np.clip(c, -0.5, 0.5)
        
    return c
