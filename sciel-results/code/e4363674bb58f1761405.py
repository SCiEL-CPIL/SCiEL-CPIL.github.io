    def _fast_max_area(points):
        n = len(points)
        # Generate all triples i < j < k
        # Using itertools is slow. Use numpy indices.
        # But 4060 is small.
        # Let's try a semi-vectorized approach:
        # For each i, compute areas with all j > i, k > j.
        # This is a loop of 30, each doing vector ops on ~400 items.
        areas = []
        pts = np.array(points)
        for i in range(n):
            # points j, k > i
            rest = pts[i+1:]
            m = len(rest)
            if m < 2: continue
            # indices for j, k in rest
            # j_idx < k_idx
            jj, kk = np.triu_indices(m, 1)
            pj = rest[jj]
            pk = rest[kk]
            pi = pts[i]
            # det = | (pj-pi) x (pk-pi) |
            v1 = pj - pi
            v2 = pk - pi
            # cross product z-component
            dets = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
            areas.extend(dets)
        
        if not areas: return 0
        # Count frequencies
        # Areas are integers.
        # Find max count
        # Using np.bincount is fast if range is small.
        # Max possible area: 30x30 box. Max det ~ 1800.
        # Actually max det for points in [-30,30] is 2 * Area.
        # Max area of triangle in 60x60 box is 1800. Det is 2*Area = 3600.
        # So bincount length 4000 is safe.
        hist, _ = np.histogram(areas, bins=4000, range=(0, 4000))
        return int(np.max(hist))
    