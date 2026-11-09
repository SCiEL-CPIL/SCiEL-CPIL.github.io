<<<<<<< SEARCH
    # Construct output
    inner = []
    for i in range(N):
        x = best_params[i*3]
        y = best_params[i*3+1]
        theta_deg = math.degrees(best_params[i*3+2])
        inner.append([x, y, theta_deg])

    return inner, (0.0, 0.0), best_L, 0.0
=======
    # Final Validity Check
    def _check_validity(params):
        pos = params[:N*2].reshape(N, 2)
        th = params[N*2:3*N]
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                d = math.sqrt(dx*dx + dy*dy)
                min_d = _min_sep_lookup(th[i] - th[j])
                # Use a small epsilon to allow touching
                if d < min_d - 1e-9:
                    return False
        return True

    if not _check_validity(best_params):
        # Fallback to Phase 1 aligned solution (which is valid)
        # Re-construct Phase 1 aligned solution
        # We need to re-run Phase 1 logic or store it.
        # Since we didn't store it, we can just use the lattice with theta=0.
        # But we need the specific 12 centers.
        # Easiest: Re-run a simplified Phase 1.
        all_centers = []
        for q in range(-2, 3):
            for r in range(-2, 3):
                if abs(q + r) <= 2:
                    x = 1.5 * q
                    y = SQRT3 * (r + q / 2.0)
                    all_centers.append((x, y))
        
        # Sort by distance from origin to pick central 12?
        # Or just take the first 12?
        # The original code sorted by L_h. With theta=0, L_h depends on position.
        # Centers closer to origin have smaller L_h.
        centers_sorted = sorted(all_centers, key=lambda c: c[0]**2 + c[1]**2)
        selected = centers_sorted[:12]
        
        best_params = np.zeros(N*3)
        best_L = 0.0
        for idx, (x, y) in enumerate(selected):
            best_params[idx*3] = x
            best_params[idx*3+1] = y
            best_params[idx*3+2] = 0.0
            L_h = _required_L(x, y, 0.0)
            if L_h > best_L:
                best_L = L_h

    inner = []
    for i in range(N):
        x = best_params[i*3]
        y = best_params[i*3+1]
        theta_deg = math.degrees(best_params[i*3+2])
        inner.append([x, y, theta_deg])

    return inner, (0.0, 0.0), best_L, 0.0
>>>>>>> REPLACE
