    def relax_radii(centers, radii):
        radii = radii.copy()
        for _ in range(20):
            changed = False
            for i in range(N):
                cx, cy = centers[i]
                r_bound = min(cx, 1-cx, cy, 1-cy)
                
                min_dist = np.inf
                for j in range(N):
                    if i == j:
                        continue
                    dist = np.sqrt((cx - centers[j, 0])**2 + (cy - centers[j, 1])**2)
                    # Space available for r_i is dist - r_j
                    # We want r_i <= dist - r_j
                    # So max r_i is min_j(dist - r_j)
                    min_dist = min(min_dist, dist - radii[j])
                
                new_r = max(0, min(r_bound, min_dist))
                if abs(new_r - radii[i]) > 1e-9:
                    radii[i] = new_r
                    changed = True
            if not changed:
                break
        return radii
