import numpy as np

def construct():
    n, m, p = 2, 5, 6
    rank = 54
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    r = 0
    
    # Helper to add a Strassen block for 2x2 * 2x2
    def strassen_222(i0, j0, k0):
        nonlocal r
        # U indices: i0*m+j0, i0*m+j0+1, (i0+1)*m+j0, (i0+1)*m+j0+1
        # V indices: j0*p+k0, j0*p+k0+1, (j0+1)*p+k0, (j0+1)*p+k0+1
        # W indices: k0*n+i0, k0*n+i0+1, (k0+1)*n+i0, (k0+1)*n+i0+1
        
        # M1 = (A00 + A11)(B00 + B11)
        U[i0*m+j0, r] += 1
        U[(i0+1)*m+(j0+1), r] += 1
        V[j0*p+k0, r] += 1
        V[(j0+1)*p+(k0+1), r] += 1
        W[k0*n+i0, r] += 1
        W[(k0+1)*n+(i0+1), r] += 1
        r += 1
        
        # M2 = (A10 + A11)B00
        U[(i0+1)*m+j0, r] += 1
        U[(i0+1)*m+(j0+1), r] += 1
        V[j0*p+k0, r] += 1
        W[k0*n+(i0+1), r] += 1
        W[(k0+1)*n+(i0+1), r] -= 1
        r += 1
        
        # M3 = A00(B01 - B11)
        U[i0*m+j0, r] += 1
        V[j0*p+(k0+1), r] += 1
        V[(j0+1)*p+(k0+1), r] -= 1
        W[(k0+1)*n+i0, r] += 1
        W[(k0+1)*n+(i0+1), r] += 1
        r += 1
        
        # M4 = A11(B10 - B00)
        U[(i0+1)*m+(j0+1), r] += 1
        V[(j0+1)*p+k0, r] += 1
        V[j0*p+k0, r] -= 1
        W[k0*n+i0, r] += 1
        W[k0*n+(i0+1), r] += 1
        r += 1
        
        # M5 = (A00 + A01)B11
        U[i0*m+j0, r] += 1
        U[i0*m+(j0+1), r] += 1
        V[(j0+1)*p+(k0+1), r] += 1
        W[k0*n+i0, r] -= 1
        W[(k0+1)*n+i0, r] += 1
        r += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[(i0+1)*m+j0, r] += 1
        U[i0*m+j0, r] -= 1
        V[j0*p+k0, r] += 1
        V[j0*p+(k0+1), r] += 1
        W[(k0+1)*n+(i0+1), r] += 1
        r += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[i0*m+(j0+1), r] += 1
        U[(i0+1)*m+(j0+1), r] -= 1
        V[(j0+1)*p+k0, r] += 1
        V[(j0+1)*p+(k0+1), r] += 1
        W[k0*n+i0, r] += 1
        r += 1

    # Helper to add a naive 2x1 * 1x2 block
    def naive_212(i0, j0, k0):
        nonlocal r
        # A is 2x1 at col j0
        # B is 1x2 at row j0, cols k0, k0+1
        # C is 2x2 at rows i0, i0+1, cols k0, k0+1
        
        # 1. A[i0, j0] * B[j0, k0] -> C[i0, k0]
        U[i0*m+j0, r] += 1
        V[j0*p+k0, r] += 1
        W[k0*n+i0, r] += 1
        r += 1
        
        # 2. A[i0, j0] * B[j0, k0+1] -> C[i0, k0+1]
        U[i0*m+j0, r] += 1
        V[j0*p+(k0+1), r] += 1
        W[(k0+1)*n+i0, r] += 1
        r += 1
        
        # 3. A[i0+1, j0] * B[j0, k0] -> C[i0+1, k0]
        U[(i0+1)*m+j0, r] += 1
        V[j0*p+k0, r] += 1
        W[k0*n+(i0+1), r] += 1
        r += 1
        
        # 4. A[i0+1, j0] * B[j0, k0+1] -> C[i0+1, k0+1]
        U[(i0+1)*m+j0, r] += 1
        V[j0*p+(k0+1), r] += 1
        W[(k0+1)*n+(i0+1), r] += 1
        r += 1

    # Apply Strassen for j-pairs (0,1) and (2,3) with k-pairs (0,1), (2,3), (4,5)
    strassen_222(0, 0, 0)
    strassen_222(0, 0, 2)
    strassen_222(0, 0, 4)
    
    strassen_222(0, 2, 0)
    strassen_222(0, 2, 2)
    strassen_222(0, 2, 4)
    
    # Apply Naive for j=4 with k-pairs (0,1), (2,3), (4,5)
    naive_212(0, 4, 0)
    naive_212(0, 4, 2)
    naive_212(0, 4, 4)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
