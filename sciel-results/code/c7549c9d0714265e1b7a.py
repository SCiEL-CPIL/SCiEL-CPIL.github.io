import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def check_decomposition_fast(U, V, W, n, m, p):
    """Vectorized check of tensor decomposition."""
    r = U.shape[1]
    # Build the tensor T[a,b,c,d,i,k] = sum_l U[a*m+b,l]*V[c*p+d,l]*W[k*n+i,l]
    # Target: T[a,b,c,d,i,k] = 1 if a==i and c==k and b==d else 0
    # Reshape U, V, W to exploit structure
    U_r = U.reshape(n, m, r)  # [a, b, l]
    V_r = V.reshape(m, p, r)  # [c, d, l]
    W_r = W.reshape(p, n, r)  # [k, i, l]
    
    # T[a,b,c,d,i,k] = sum_l U_r[a,b,l] * V_r[c,d,l] * W_r[k,i,l]
    # We need T[a,b,c,d,i,k] == delta(a,i)*delta(c,k)*delta(b,d)
    
    # Check all (a,b,c,d,i,k) tuples
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for i in range(n):
                        for k in range(p):
                            target = 1.0 if (a == i and c == k and b == d) else 0.0
                            # T[a,b,c,d,i,k] = sum_l U_r[a,b,l] * V_r[c,d,l] * W_r[k,i,l]
                            s = np.sum(U_r[a,b,:] * V_r[c,d,:] * W_r[k,i,:])
                            if abs(s - target) > 1e-9:
                                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Try the rank-34 construction
    U233 = np.zeros((6, 17))
    V233 = np.zeros((9, 17))
    W233 = np.zeros((6, 17))
    
    # Part 1: M_222 (Strassen-like, rank 7)
    U233[0, 0] = 1; U233[4, 0] = 1
    V233[0, 0] = 1; V233[4, 0] = 1
    W233[0, 0] = 1
    W233[3, 0] = 1
    
    U233[3, 1] = 1; U233[4, 1] = 1
    V233[0, 1] = 1
    W233[1, 1] = 1
    W233[3, 1] = -1
    
    U233[0, 2] = 1
    V233[1, 2] = 1; V233[4, 2] = -1
    W233[2, 2] = 1
    W233[3, 2] = 1
    
    U233[4, 3] = 1
    V233[3, 3] = 1; V233[0, 3] = -1
    W233[0, 3] = 1
    W233[1, 3] = 1
    
    U233[0, 4] = 1; U233[1, 4] = 1
    V233[4, 4] = 1
    W233[0, 4] = -1
    W233[2, 4] = 1
    
    U233[3, 5] = 1; U233[0, 5] = -1
    V233[0, 5] = 1; V233[1, 5] = 1
    W233[3, 5] = 1
    
    U233[1, 6] = 1; U233[4, 6] = -1
    V233[3, 6] = 1; V233[4, 6] = 1
    W233[0, 6] = 1
    
    # Part 2: M_221 (Rank 4)
    U233[0, 7] = 1
    V233[2, 7] = 1
    W233[4, 7] = 1
    
    U233[1, 8] = 1
    V233[5, 8] = 1
    W233[4, 8] = 1
    
    U233[3, 9] = 1
    V233[2, 9] = 1
    W233[5, 9] = 1
    
    U233[4, 10] = 1
    V233[5, 10] = 1
    W233[5, 10] = 1
    
    # Part 3: M_213 (Rank 6)
    U233[2, 11] = 1
    V233[6, 11] = 1
    W233[0, 11] = 1
    
    U233[2, 12] = 1
    V233[7, 12] = 1
    W233[2, 12] = 1
    
    U233[2, 13] = 1
    V233[8, 13] = 1
    W233[4, 13] = 1
    
    U233[5, 14] = 1
    V233[6, 14] = 1
    W233[1, 14] = 1
    
    U233[5, 15] = 1
    V233[7, 15] = 1
    W233[3, 15] = 1
    
    U233[5, 16] = 1
    V233[8, 16] = 1
    W233[5, 16] = 1
    
    U_comb = np.zeros((6, 34))
    V_comb = np.zeros((18, 34))
    W_comb = np.zeros((12, 34))
    
    U_comb[:, :17] = U233
    U_comb[:, 17:] = U233
    
    for j in range(3):
        for k in range(3):
            V_comb[j * 6 + k, :17] = V233[j * 3 + k]
            V_comb[j * 6 + 3 + k, 17:] = V233[j * 3 + k]
            
    for k in range(3):
        for i in range(2):
            W_comb[k * 2 + i, :17] = W233[k * 2 + i]
            W_comb[(k + 3) * 2 + i, 17:] = W233[k * 2 + i]
            
    if time.time() < deadline:
        if check_decomposition_fast(U_comb, V_comb, W_comb, 2, 3, 6):
            return {
                "U": U_comb.tolist(),
                "V": V_comb.tolist(),
                "W": W_comb.tolist()
            }
    
    # Fallback to schoolbook rank 36
    U_school = np.zeros((6, 36))
    V_school = np.zeros((18, 36))
    W_school = np.zeros((12, 36))
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(6):
                U_school[i * 3 + j, col] = 1
                V_school[j * 6 + k, col] = 1
                W_school[k * 2 + i, col] = 1
                col += 1
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }

if __name__ == "__main__":
    construct()
