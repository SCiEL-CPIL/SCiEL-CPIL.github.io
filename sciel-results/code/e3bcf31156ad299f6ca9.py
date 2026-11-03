for i in range(n):
    for j in range(m):
        for k in range(p):
            s = np.dot(U_arr[i*m+j, :], V_arr[j*p+k, :] * W_arr[k*n+i, :])
            if abs(s - 1.0) > 1e-9:
                return False
