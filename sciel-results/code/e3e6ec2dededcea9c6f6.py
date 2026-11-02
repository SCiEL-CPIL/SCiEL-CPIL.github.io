for i in range(n):
    for j in range(m):
        for k in range(p):
            val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
            if abs(val - 1) > tol:
                return False
