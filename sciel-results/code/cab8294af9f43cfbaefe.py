        def halton_sequence(n, d):
            points = []
            for i in range(1, n + 1):
                point = []
                for j in range(d):
                    f = 1.0
                    i_val = i
                    p = j + 2
                    while i_val > 0:
                        f = f / p
                        point.append(f * (i_val % p))
                        i_val = i_val // p
                points.append(point)
            return np.array(points)
