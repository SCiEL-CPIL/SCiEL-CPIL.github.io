for i in range(n):
    for j in range(m):
        for k in range(p):
            s = 0.0
            for c in range(r):
                s += U_arr[i*m + j, c] * V_arr[j*p + k, c] * W_arr[k*n + i, c]
            if abs(s - 1.0) > tol:
                return False
