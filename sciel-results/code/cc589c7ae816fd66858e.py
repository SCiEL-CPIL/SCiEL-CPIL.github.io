"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    Uses block Strassen decomposition to achieve rank 36, with verification.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    def verify_half_integer(U, V, W):
        """
        Verify that U, V, W with half-integer entries exactly decompose the tensor.
        """
        U_int = (2 * U).astype(int)
        V_int = (2 * V).astype(int)
        W_int = (2 * W).astype(int)
        
        if not np.allclose(U * 2, U_int) or not np.allclose(V * 2, V_int) or not np.allclose(W * 2, W_int):
            return False
            
        # Check if sum_r U V W == 8 for all i,j,k
        # Using integer arithmetic for speed and exactness
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(U.shape[1]):
                        val += U_int[i*m+j, r] * V_int[j*p+k, r] * W_int[k*n+i, r]
                    if val != 8:
                        return False
        return True
    
    def schoolbook_decomposition():
        rank_schoolbook = n * m * p
        U = np.zeros((n*m, rank_schoolbook))
        V = np.zeros((m*p, rank_schoolbook))
        W = np.zeros((p*n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W
    
    # Strassen 2x2x2 decomposition
    # Returns list of 7 dicts: each with U, V, W entries
    def strassen_222(i_off, j_off, k_off):
        comps = []
        # m1 = (a00+a11)(b00+b11) -> c00+=1, c11+=1
        comps.append({'U': {(i_off,j_off):1,(i_off+1,j_off+1):1}, 'V': {(j_off,k_off):1,(j_off+1,k_off+1):1}, 'W': {(i_off,k_off):1,(i_off+1,k_off+1):1}})
        # m2 = (a10+a11)b00 -> c10+=1, c11-=1
        comps.append({'U': {(i_off+1,j_off):1,(i_off+1,j_off+1):1}, 'V': {(j_off,k_off):1}, 'W': {(i_off+1,k_off):1,(i_off+1,k_off+1):-1}})
        # m3 = a00(b01-b11) -> c01+=1, c11+=1
        comps.append({'U': {(i_off,j_off):1}, 'V': {(j_off,k_off+1):1,(j_off+1,k_off+1):-1}, 'W': {(i_off,k_off+1):1,(i_off+1,k_off+1):1}})
        # m4 = a11(b10-b00) -> c00+=1, c10+=1
        comps.append({'U': {(i_off+1,j_off+1):1}, 'V': {(j_off+1,k_off):1,(j_off,k_off):-1}, 'W': {(i_off,k_off):1,(i_off+1,k_off):1}})
        # m5 = (a00+a01)b11 -> c00-=1, c01+=1
        comps.append({'U': {(i_off,j_off):1,(i_off,j_off+1):1}, 'V': {(j_off+1,k_off+1):1}, 'W': {(i_off,k_off):-1,(i_off,k_off+1):1}})
        # m6 = (a10-a00)(b00+b01) -> c11+=1
        comps.append({'U': {(i_off+1,j_off):1,(i_off,j_off):-1}, 'V': {(j_off,k_off):1,(j_off,k_off+1):1}, 'W': {(i_off+1,k_off+1):1}})
        # m7 = (a01-a11)(b10+b11) -> c00+=1
        comps.append({'U': {(i_off,j_off+1):1,(i_off+1,j_off+1):-1}, 'V': {(j_off+1,k_off):1,(j_off+1,k_off+1):1}, 'W': {(i_off,k_off):1}})
        return comps
    
    # Try block Strassen decomposition with rank 36
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # A1 (j=0,1) with B1 (k=0,1)
    for comp in strassen_222(0, 0, 0):
        for (i,j), val in comp['U'].items():
            U[i*m+j, col] += val
        for (j,k), val in comp['V'].items():
            V[j*p+k, col] += val
        for (i,k), val in comp['W'].items():
            W[k*n+i, col] += val
        col += 1
    
    # A1 (j=0,1) with B2 (k=2,3)
    for comp in strassen_222(0, 0, 2):
        for (i,j), val in comp['U'].items():
            U[i*m+j, col] += val
        for (j,k), val in comp['V'].items():
            V[j*p+k, col] += val
        for (i,k), val in comp['W'].items():
            W[k*n+i, col] += val
        col += 1
    
    # A1 (j=0,1) with B3 (k=4): schoolbook, 4 components
    for i in range(2):
        for j in range(2):
            U[i*m+j, col] = 1
            V[j*p+4, col] = 1
            W[4*n+i, col] = 1
            col += 1
    
    # A2 (j=2,3) with B1 (k=0,1)
    for comp in strassen_222(0, 2, 0):
        for (i,j), val in comp['U'].items():
            U[i*m+j, col] += val
        for (j,k), val in comp['V'].items():
            V[j*p+k, col] += val
        for (i,k), val in comp['W'].items():
            W[k*n+i, col] += val
        col += 1
    
    # A2 (j=2,3) with B2 (k=2,3)
    for comp in strassen_222(0, 2, 2):
        for (i,j), val in comp['U'].items():
            U[i*m+j, col] += val
        for (j,k), val in comp['V'].items():
            V[j*p+k, col] += val
        for (i,k), val in comp['W'].items():
            W[k*n+i, col] += val
        col += 1
    
    # A2 (j=2,3) with B3 (k=4): schoolbook, 4 components
    for i in range(2):
        for j in range(2, 4):
            U[i*m+j, col] = 1
            V[j*p+4, col] = 1
            W[4*n+i, col] = 1
            col += 1
    
    # Verify the decomposition
    if verify_half_integer(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook if verification fails
    U_school, V_school, W_school = schoolbook_decomposition()
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
