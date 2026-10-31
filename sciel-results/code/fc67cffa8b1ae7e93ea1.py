import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin

    n, m, p = 4, 4, 4
    rank = 49

    # Initialize factor matrices
    # U shape: (n*m, rank) = (16, 49)
    # V shape: (m*p, rank) = (16, 49)
    # W shape: (p*n, rank) = (16, 49)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Precompute the 2x2 Strassen factor matrices as numpy arrays for efficiency
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)

    # M0 = (A00 + A11) * (B00 + B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1

    # M1 = (A10 + A11) * B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1

    # M2 = A00 * (B01 - B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1

    # M3 = A11 * (B10 - B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1

    # M4 = (A00 + A01) * B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1

    # M5 = (A10 - A00) * (B00 + B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1

    # M6 = (A01 - A11) * (B10 + B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Build the 49-rank decomposition using Kronecker product of Strassen factors
    # Precompute index mappings
    # For U: flat index i*4+j, block (I,J) = (i//2, j//2), inner (ii,jj) = (i%2, j%2)
    # U[i*4+j, s*7+t] = U2[I*2+J, s] * U2[ii*2+jj, t]
    
    U_arr = np.zeros((16, 49), dtype=int)
    V_arr = np.zeros((16, 49), dtype=int)
    W_arr = np.zeros((16, 49), dtype=int)
    
    # Precompute block/inner index mappings
    I_map = [i // 2 for i in range(4)]
    J_map = [j // 2 for j in range(4)]
    ii_map = [i % 2 for i in range(4)]
    jj_map = [j % 2 for j in range(4)]
    
    # For each rank index s*7+t, compute contributions
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            
            # U: U[i*4+j, r] = U2[I*2+J, s] * U2[ii*2+jj, t]
            for i in range(4):
                for j in range(4):
                    coeff = U2[I_map[i]*2 + J_map[j], s] * U2[ii_map[i]*2 + jj_map[j], t]
                    if coeff != 0:
                        U_arr[i*4 + j, r] = coeff
            
            # V: V[k*4+l, r] = V2[K*2+L, s] * V2[kk*2+ll, t]
            for k in range(4):
                for l in range(4):
                    coeff = V2[I_map[k]*2 + J_map[l], s] * V2[ii_map[k]*2 + jj_map[l], t]
                    if coeff != 0:
                        V_arr[k*4 + l, r] = coeff
            
            # W: W[j*4+i, r] = W2[I*2+J, s] * W2[ii*2+jj, t]
            # where I = i//2 (row block of C), J = j//2 (col block of C)
            for j in range(4):
                for i in range(4):
                    coeff = W2[I_map[i]*2 + J_map[j], s] * W2[ii_map[i]*2 + jj_map[j], t]
                    if coeff != 0:
                        W_arr[j*4 + i, r] = coeff
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Convert to list of lists
    U = U_arr.tolist()
    V = V_arr.tolist()
    W = W_arr.tolist()

    return {"U": U, "V": V, "W": W}
