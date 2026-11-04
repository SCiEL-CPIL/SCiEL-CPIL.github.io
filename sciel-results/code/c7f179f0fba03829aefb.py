<<<<<<< SEARCH
    def min_distance_sat(v1, v2):
        """
        Compute the minimum signed distance between two convex polygons.
        Positive if separated, negative if overlapping.
        """
        min_gap = float('inf')
        polys = [v1, v2]
        for poly in polys:
            n = len(poly)
            for i in range(n):
                p1 = poly[i]
                p2 = poly[(i + 1) % n]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                norm_len = np.linalg.norm(normal)
                if norm_len == 0:
                    continue
                normal /= norm_len
                
                proj1 = v1 @ normal
                proj2 = v2 @ normal
                
                gap1 = proj1.max() - proj2.min()
                gap2 = proj2.max() - proj1.min()
                
                # The separation along this axis is min(gap1, gap2)
                # If this is positive, they are separated along this axis.
                # If negative, they overlap along this axis.
                # The global minimum distance is the minimum of these gaps.
                min_gap = min(min_gap, min(gap1, gap2))
                
        return min_gap
=======
    def min_distance_sat(v1, v2):
        """
        Compute the minimum signed distance between two convex polygons.
        Positive if separated, negative if overlapping.
        """
        gaps = []
        polys = [v1, v2]
        for poly in polys:
            n = len(poly)
            for i in range(n):
                p1 = poly[i]
                p2 = poly[(i + 1) % n]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                norm_len = np.linalg.norm(normal)
                if norm_len == 0:
                    continue
                normal /= norm_len
                
                proj1 = v1 @ normal
                proj2 = v2 @ normal
                
                # gap = min(proj2) - max(proj1)
                # If gap > 0, v1 is "before" v2 along normal.
                # If gap < 0, they overlap along this normal.
                gap = proj2.min() - proj1.max()
                gaps.append(gap)
                
        if not gaps:
            return 0.0
            
        # If any gap is positive, they are separated.
        # The distance is the minimum positive gap.
        positive_gaps = [g for g in gaps if g > 0]
        if positive_gaps:
            return min(positive_gaps)
        else:
            # All gaps are <= 0, so they overlap.
            # The penetration depth is the maximum overlap (most negative gap).
            # We return a negative distance.
            return min(gaps)
>>>>>>> REPLACE

This logic is correct.

Now, one more thing: The `objective` function uses `min_distance_sat`. If `d < 0`, we penalize.
The penalty `100.0 * (-d)**2` is smooth.

I will also increase the number of multi-starts and iterations to ensure convergence.

