s += U[i*m + j, c] * V[jp*p + k, c] * W[k*n + i, c]
if abs(s - (1.0 if j == jp else 0.0)) > tol:
