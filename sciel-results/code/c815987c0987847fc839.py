    def _pbd_refine(params, deadline):
        p = params.copy()
        x = p[0::3]
        y = p[1::3]
        r = p[2::3]
        
        # Initialize r to be slightly less than half min distance
        # ...
        
        t0 = time.time()
        growth = 1e-4
        relax_iters = 20
        
        while time.time() < deadline:
            # 1. Grow
            r *= (1 + growth)
            
            # 2. Relax (Gauss-Seidel)
            for _ in range(relax_iters):
                # Wall constraints
                for i in range(N):
                    # Max radius allowed by walls
                    max_r = min(x[i], 1-x[i], y[i], 1-y[i])
                    if r[i] > max_r:
                        r[i] = max_r
                
                # Pair constraints
                for i in range(N):
                    for j in range(i+1, N):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        d = np.sqrt(dx*dx + dy*dy)
                        if d < 1e-8:
                            # Jitter
                            x[i] += 1e-5
                            continue
                        
                        min_d = r[i] + r[j]
                        if d < min_d:
                            # Overlap
                            overlap = min_d - d
                            # Unit vector
                            ux = dx / d
                            uy = dy / d
                            
                            # Try to move apart
                            # Move i by -u * overlap/2, j by u * overlap/2
                            shift = overlap / 2.0
                            
                            # Tentative new positions
                            new_xi = x[i] - ux * shift
                            new_yi = y[i] - uy * shift
                            new_xj = x[j] + ux * shift
                            new_yj = y[j] + uy * shift
                            
                            # Check walls for i
                            # If new position violates wall, we can't move fully.
                            # We need to find the max allowed shift that keeps both in bounds.
                            # This is complex. Simpler: Apply shift, then clamp to walls, then check residual overlap.
                            
                            x[i] = new_xi
                            y[i] = new_yi
                            x[j] = new_xj
                            y[j] = new_yj
                            
                            # Clamp to walls
                            x[i] = np.clip(x[i], 0, 1)
                            y[i] = np.clip(y[i], 0, 1)
                            x[j] = np.clip(x[j], 0, 1)
                            y[j] = np.clip(y[j], 0, 1)
                            
                            # Re-evaluate distance
                            dx = x[i] - x[j]
                            dy = y[i] - y[j]
                            d_new = np.sqrt(dx*dx + dy*dy)
                            
                            # If still overlapping, shrink radii
                            if d_new < r[i] + r[j]:
                                residual = (r[i] + r[j]) - d_new
                                # Shrink both by half residual
                                shrink = residual / 2.0
                                r[i] -= shrink
                                r[j] -= shrink
                                # Ensure r > 0
                                if r[i] < 1e-6: r[i] = 1e-6
                                if r[j] < 1e-6: r[j] = 1e-6
                                # Re-clamp to walls after shrink (radius decreased, so wall constraint might be looser, but we already clamped pos)
                                # Actually, if we shrink r, we don't need to move pos for wall, but we might have moved pos to wall.
                                # The wall clamp on pos ensures pos is in [0,1].
                                # The radius must be <= min(x, 1-x, y, 1-y).
                                max_ri = min(x[i], 1-x[i], y[i], 1-y[i])
                                max_rj = min(x[j], 1-x[j], y[j], 1-y[j])
                                if r[i] > max_ri: r[i] = max_ri
                                if r[j] > max_rj: r[j] = max_rj
            # Update params array
            p[0::3] = x
            p[1::3] = y
            p[2::3] = r
    