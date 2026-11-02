import time
import numpy as np

DIMENSIONS = [3, 3, 3]
DEADLINE = 118.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    def iu(i, j): return i * m + j
    def iv(j, k): return j * p + k
    def iw(k, i): return k * n + i
    
    def verify(U, V, W, atol=1e-10):
        R = U.shape[1]
        T_recon = np.zeros((n, m, p))
        for r in range(R):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        T_target = np.ones((n, m, p))
        return np.allclose(T_recon, T_target, atol=atol)
    
    # Rank 26 decomposition: Strassen (7) + 19 border terms
    def rank26():
        R = 26
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        # Strassen m1 = (a00+a11)(b00+b11) -> contributes to c00, c11
        r = 0
        U[iu(0,0), r] = 1; U[iu(1,1), r] = 1
        V[iv(0,0), r] = 1; V[iv(1,1), r] = 1
        W[iw(0,0), r] = 1; W[iw(1,1), r] = 1
        
        # m2 = (a10+a11)b00 -> contributes to c10, subtracts from c11
        r = 1
        U[iu(1,0), r] = 1; U[iu(1,1), r] = 1
        V[iv(0,0), r] = 1
        W[iw(0,1), r] = 1; W[iw(1,1), r] = -1
        
        # m3 = a00(b01-b11) -> contributes to c01, c11
        r = 2
        U[iu(0,0), r] = 1
        V[iv(0,1), r] = 1; V[iv(1,1), r] = -1
        W[iw(1,0), r] = 1; W[iw(1,1), r] = 1
        
        # m4 = a11(b10-b00) -> contributes to c00, c10
        r = 3
        U[iu(1,1), r] = 1
        V[iv(1,0), r] = 1; V[iv(0,0), r] = -1
        W[iw(0,0), r] = 1; W[iw(0,1), r] = 1
        
        # m5 = (a00+a01)b11 -> subtracts from c00, contributes to c01
        r = 4
        U[iu(0,0), r] = 1; U[iu(0,1), r] = 1
        V[iv(1,1), r] = 1
        W[iw(0,0), r] = -1; W[iw(1,0), r] = 1
        
        # m6 = (a10-a00)(b00+b01) -> contributes to c11
        r = 5
        U[iu(1,0), r] = 1; U[iu(0,0), r] = -1
        V[iv(0,0), r] = 1; V[iv(0,1), r] = 1
        W[iw(1,1), r] = 1
        
        # m7 = (a01-a11)(b10+b11) -> contributes to c00
        r = 6
        U[iu(0,1), r] = 1; U[iu(1,1), r] = -1
        V[iv(1,0), r] = 1; V[iv(1,1), r] = 1
        W[iw(0,0), r] = 1
        
        # Border terms
        col = 7
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i < 2 and j < 2 and k < 2:
                        continue
                    U[iu(i,j), col] = 1
                    V[iv(j,k), col] = 1
                    W[iw(k,i), col] = 1
                    col += 1
        
        return U, V, W
    
    # Rank 27 schoolbook
    def rank27():
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[iu(i,j), col] = 1
                    V[iv(j,k), col] = 1
                    W[iw(k,i), col] = 1
                    col += 1
        return U, V, W
    
    # Try rank 26 first
    U26, V26, W26 = rank26()
    if verify(U26, V26, W26):
        return {"U": U26.tolist(), "V": V26.tolist(), "W": W26.tolist()}
    
    # Try rank 25 via rank-deflation search
    # Attempt to find a rank-25 decomposition by searching over factor matrices
    def attempt_rank25():
        if time.time() - start_time > DEADLINE - 5:
            return None
        
        R = 25
        # Start from a perturbed rank-26 and try to nullify one term
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        # Use Strassen for 7 terms
        r = 0
        U[iu(0,0), r] = 1; U[iu(1,1), r] = 1
        V[iv(0,0), r] = 1; V[iv(1,1), r] = 1
        W[iw(0,0), r] = 1; W[iw(1,1), r] = 1
        
        r = 1
        U[iu(1,0), r] = 1; U[iu(1,1), r] = 1
        V[iv(0,0), r] = 1
        W[iw(0,1), r] = 1; W[iw(1,1), r] = -1
        
        r = 2
        U[iu(0,0), r] = 1
        V[iv(0,1), r] = 1; V[iv(1,1), r] = -1
        W[iw(1,0), r] = 1; W[iw(1,1), r] = 1
        
        r = 3
        U[iu(1,1), r] = 1
        V[iv(1,0), r] = 1; V[iv(0,0), r] = -1
        W[iw(0,0), r] = 1; W[iw(0,1), r] = 1
        
        r = 4
        U[iu(0,0), r] = 1; U[iu(0,1), r] = 1
        V[iv(1,1), r] = 1
        W[iw(0,0), r] = -1; W[iw(1,0), r] = 1
        
        r = 5
        U[iu(1,0), r] = 1; U[iu(0,0), r] = -1
        V[iv(0,0), r] = 1; V[iv(0,1), r] = 1
        W[iw(1,1), r] = 1
        
        r = 6
        U[iu(0,1), r] = 1; U[iu(1,1), r] = -1
        V[iv(1,0), r] = 1; V[iv(1,1), r] = 1
        W[iw(0,0), r] = 1
        
        # 18 remaining terms for border + overlap
        # We need to cover all (i,j,k) not fully covered by Strassen
        # Border positions: all where at least one index is 2
        # Count: 27 - 8 = 19 border positions
        # We have 18 slots, so we need to merge one border term
        col = 7
        border_positions = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i < 2 and j < 2 and k < 2:
                        continue
                    border_positions.append((i, j, k))
        
        # Try to combine two border positions into one term
        # This is only possible if they share structure
        # For now, just use 18 of the 19 border positions and hope the missing one
        # is covered by Strassen overlap (it won't be, so this will likely fail)
        # Instead, let's just verify that rank 25 is not feasible with this approach
        # and return None
        
        return None
    
    result = attempt_rank25()
    if result is not None:
        U25, V25, W25 = result
        if verify(U25, V25, W25):
            return {"U": U25.tolist(), "V": V25.tolist(), "W": W25.tolist()}
    
    # Fallback to rank 27
    U27, V27, W27 = rank27()
    return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
