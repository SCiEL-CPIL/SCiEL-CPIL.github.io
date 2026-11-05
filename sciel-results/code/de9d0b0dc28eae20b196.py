    for i in range(n):
        pi = all_points[i]
        for j in range(i + 1, n):
            pj = all_points[j]
            mask = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                pk = all_points[k]
                if _is_isosceles(pi, pj, pk):
                    mask |= (1 << k)
            iso_pair[i][j] = mask
            iso_pair[j][i] = mask
