    def calc_min_area(pts):
        # pts (N,2)
        # Compute all pairs? No, triples.
        # N=11 is small.
        # Let's stick to loops but optimize them slightly.
        min_a = 1e9
        for i in range(N):
            pi = pts[i]
            for j in range(i+1, N):
                v1 = pts[j] - pi
                for k in range(j+1, N):
                    v2 = pts[k] - pi
                    a = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                    if a < min_a:
                        min_a = a
        return min_a
    