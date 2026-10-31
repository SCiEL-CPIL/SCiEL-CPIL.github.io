    def get_true_score(points):
        min_a = np.inf
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    a = 0.5 * abs(np.cross(points[j]-points[i], points[k]-points[i]))
                    if a < min_a:
                        min_a = a
        # Check validity
        for p in points:
            x, y = p
            h = np.sqrt(3)/2
            w = y/h
            v = x - 0.5*y/h
            u = 1 - x - 0.5*y/h
            if u < -1e-6 or v < -1e-6 or w < -1e-6:
                return -np.inf
        return min_a

    best_min_area = -np.inf
    best_points = None
    
    # ... in loop ...
    current_min_area = get_true_score(res.x.reshape(N, 2))
    if current_min_area > best_min_area:
        best_min_area = current_min_area
        best_points = res.x.reshape(N, 2)
