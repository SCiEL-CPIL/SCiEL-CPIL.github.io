import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Estimated rank: 24 + 24 + 18 + 18 = 84
    # Add buffer
    rank = 100
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Helper to add Strassen 2x2x2 terms
    def add_strassen(r1, r2, c1, c2, k1, k2):
        nonlocal col
        # m1 = (A[r1,c1] + A[r2,c2]) * (B[c1,k1] + B[c2,k2])
        U[r1*m + c1, col] += 1
        U[r2*m + c2, col] += 1
        V[c1*p + k1, col] += 1
        V[c2*p + k2, col] += 1
        W[k1*n + r1, col] += 1
        W[k2*n + r2, col] += 1
        W[k1*n + r2, col] += 1
        W[k2*n + r1, col] += 1
        col += 1
        
        # m2 = (A[r2,c1] + A[r2,c2]) * B[c1,k1]
        U[r2*m + c1, col] += 1
        U[r2*m + c2, col] += 1
        V[c1*p + k1, col] += 1
        W[k1*n + r2, col] += 1
        col += 1
        
        # m3 = A[r1,c1] * (B[c1,k2] - B[c2,k2])
        U[r1*m + c1, col] += 1
        V[c1*p + k2, col] += 1
        V[c2*p + k2, col] -= 1
        W[k2*n + r1, col] += 1
        col += 1
        
        # m4 = A[r2,c2] * (B[c2,k1] - B[c1,k1])
        U[r2*m + c2, col] += 1
        V[c2*p + k1, col] += 1
        V[c1*p + k1, col] -= 1
        W[k1*n + r2, col] += 1
        col += 1
        
        # m5 = (A[r1,c1] + A[r1,c2]) * B[c2,k2]
        U[r1*m + c1, col] += 1
        U[r1*m + c2, col] += 1
        V[c2*p + k2, col] += 1
        W[k2*n + r1, col] += 1
        col += 1
        
        # m6 = (A[r2,c1] - A[r1,c1]) * (B[c1,k1] + B[c1,k2])
        U[r2*m + c1, col] += 1
        U[r1*m + c1, col] -= 1
        V[c1*p + k1, col] += 1
        V[c1*p + k2, col] += 1
        W[k1*n + r2, col] += 1
        W[k2*n + r2, col] += 1
        col += 1
        
        # m7 = (A[r1,c2] - A[r2,c2]) * (B[c2,k1] + B[c2,k2])
        U[r1*m + c2, col] += 1
        U[r2*m + c2, col] -= 1
        V[c2*p + k1, col] += 1
        V[c2*p + k2, col] += 1
        W[k1*n + r1, col] += 1
        W[k2*n + r1, col] += 1
        col += 1

    def add_schoolbook(r1, r2, c1, c2, k1, k2):
        nonlocal col
        for r in range(r1, r2):
            for c in range(c1, c2):
                for k in range(k1, k2):
                    U[r*m + c, col] += 1
                    V[c*p + k, col] += 1
                    W[k*n + r, col] += 1
                    col += 1

    # 1. Top-Left 3x3x3 block: A[0:3, 0:3] * B[0:3, 0:3] -> C[0:3, 0:3]
    # Use Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen(0, 1, 0, 1, 0, 1)
    
    # Schoolbook for the remaining parts of this 3x3x3 block
    # C[0:2, 2] = A[0:2, 0:2]*B[0:2, 2] + A[0:2, 2]*B[2, 2]
    add_schoolbook(0, 2, 0, 2, 2, 3)
    add_schoolbook(0, 2, 2, 3, 2, 3)
    
    # C[2, 0:2] = A[2, 0:2]*B[0:2, 0:2] + A[2, 2]*B[2, 0:2]
    add_schoolbook(2, 3, 0, 2, 0, 2)
    add_schoolbook(2, 3, 2, 3, 0, 2)
    
    # C[2, 2] = A[2, 0:2]*B[0:2, 2] + A[2, 2]*B[2, 2]
    add_schoolbook(2, 3, 0, 2, 2, 3)
    add_schoolbook(2, 3, 2, 3, 2, 3)

    # 2. Top-Right 3x3x3 block: A[0:3, 0:3] * B[0:3, 3:6] -> C[0:3, 3:6]
    # Use Strassen for A[0:2, 0:2] * B[0:2, 3:5] -> C[0:2, 3:5]
    add_strassen(0, 1, 0, 1, 3, 4)
    
    # Schoolbook for the remaining parts
    # C[0:2, 5] = A[0:2, 0:2]*B[0:2, 5] + A[0:2, 2]*B[2, 5]
    add_schoolbook(0, 2, 0, 2, 5, 6)
    add_schoolbook(0, 2, 2, 3, 5, 6)
    
    # C[2, 3:5] = A[2, 0:2]*B[0:2, 3:5] + A[2, 2]*B[2, 3:5]
    add_schoolbook(2, 3, 0, 2, 3, 5)
    add_schoolbook(2, 3, 2, 3, 3, 5)
    
    # C[2, 5] = A[2, 0:2]*B[0:2, 5] + A[2, 2]*B[2, 5]
    add_schoolbook(2, 3, 0, 2, 5, 6)
    add_schoolbook(2, 3, 2, 3, 5, 6)

    # 3. Bottom-Left 3x2x3 block: A[0:3, 3:5] * B[3:5, 0:3] -> C[0:3, 0:3]
    add_schoolbook(0, 3, 3, 5, 0, 3)

    # 4. Bottom-Right 3x2x3 block: A[0:3, 3:5] * B[3:5, 3:6] -> C[0:3, 3:6]
    add_schoolbook(0, 3, 3, 5, 3, 6)

    # Trim to actual used rank
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
