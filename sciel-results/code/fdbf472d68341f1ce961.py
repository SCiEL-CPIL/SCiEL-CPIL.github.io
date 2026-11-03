# final cleanup: ensure feasibility
for _ in range(100):
    D = pair_hex_norms(best_pts)
    np.fill_diagonal(D, np.inf)
    if np.min(D) >= min_d:
        break
    for i in range(n):
        for j in range(i+1, n):
            if D[i,j] < min_d:
                v = best_pts[i]-best_pts[j]
                k = int(np.argmax(np.abs(v @ NORMS.T)))
                sgn = 1.0 if (v @ NORMS[k]) >= 0 else -1.0
                push = (min_d - D[i,j])*0.5
                best_pts[i] += sgn*NORMS[k]*push
                best_pts[j] -= sgn*NORMS[k]*push
