# Check off-diagonal entries (should be 0)
for i in range(n):
    for j in range(m):
        for k in range(p):
            for i2 in range(n):
                for j2 in range(m):
                    for k2 in range(p):
                        if i2 == i and j2 == j and k2 == k:
                            continue
                        s2 = 0.0
                        for c_idx in range(r):
                            s2 += U[i2*m+j2, c_idx] * V[j2*p+k2, c_idx] * W[k2*n+i2, c_idx]
                        if abs(s2) > 1e-8:
                            return False
