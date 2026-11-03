"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s limit to be safe
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Partition:
    # i-blocks: {0,1} (size 2), {2} (size 1)
    # j-blocks: {0,1} (size 2), {2,3} (size 2), {4} (size 1)
    # k-blocks: {0,1} (size 2), {2,3} (size 2), {4,5} (size 2), {6} (size 1)
    
    # Strassen blocks (2x2x2): 1 i-block * 2 j-blocks * 3 k-blocks = 6, rank 6*7=42
    # Non-Strassen:
    #   i={0,1}, j={0,1}, k={6}: 2x2x1, rank 2
    #   i={0,1}, j={2,3}, k={6}: 2x2x1, rank 2
    #   i={0,1}, j={4}, k={0..6}: 2x1x7, rank 2
    #   i={2}, j={0,1}, k={0..6}: 1x2x7, rank 2
    #   i={2}, j={2,3}, k={0..6}: 1x2x7, rank 2
    #   i={2}, j={4}, k={0..6}: 1x1x7, rank 1
    # Total non-Strassen: 11
    # Grand total: 53
    
    rank = 53
    U = [[0]*rank for _ in range(n*m)]  # 15 x 53
    V = [[0]*rank for _ in range(m*p)]  # 35 x 53
    W = [[0]*rank for _ in range(p*n)]  # 21 x 53
    
    col = 0
    
    # Helper: add a 2x2x2 Strassen block
    def add_strassen(i0, j0, k0, col):
        # U indices: i in {i0, i0+1}, j in {j0, j0+1} -> U[i*m+j]
        # V indices: j in {j0, j0+1}, k in {k0, k0+1} -> V[j*p+k]
        # W indices: k in {k0, k0+1}, i in {i0, i0+1} -> W[k*n+i]
        
        # M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        U[i0*m+j0][col] += 1
        U[(i0+1)*m+(j0+1)][col] += 1
        V[j0*p+k0][col] += 1
        V[(j0+1)*p+(k0+1)][col] += 1
        W[k0*n+i0][col] += 1
        W[(k0+1)*n+(i0+1)][col] += 1
        col += 1
        
        # M2 = A10(B00+B01) -> C10 += 1, C11 -= 1
        U[(i0+1)*m+j0][col] += 1
        V[j0*p+k0][col] += 1
        V[j0*p+(k0+1)][col] += 1
        W[k0*n+(i0+1)][col] += 1
        W[(k0+1)*n+(i0+1)][col] -= 1
        col += 1
        
        # M3 = B10(B11-B01) -> C01 += 1, C11 += 1
        U[i0*m+j0][col] += 1
        V[(j0+1)*p+(k0+1)][col] += 1
        V[j0*p+(k0+1)][col] -= 1
        W[(k0+1)*n+i0][col] += 1
        W[(k0+1)*n+(i0+1)][col] += 1
        col += 1
        
        # M4 = A11(B10-B00) -> C00 += 1, C10 += 1
        U[(i0+1)*m+(j0+1)][col] += 1
        V[(j0+1)*p+k0][col] += 1
        V[j0*p+k0][col] -= 1
        W[k0*n+i0][col] += 1
        W[k0*n+(i0+1)][col] += 1
        col += 1
        
        # M5 = A00(A01+A10) -> C00 -= 1, C01 += 1
        U[i0*m+j0][col] += 1
        U[i0*m+(j0+1)][col] += 1
        U[(i0+1)*m+j0][col] += 1
        V[(j0+1)*p+(k0+1)][col] += 1
        W[k0*n+i0][col] -= 1
        W[(k0+1)*n+i0][col] += 1
        col += 1
        
        # M6 = (A01+A11)(B10+B11) -> C11 += 1
        U[i0*m+(j0+1)][col] += 1
        U[(i0+1)*m+(j0+1)][col] += 1
        V[(j0+1)*p+k0][col] += 1
        V[(j0+1)*p+(k0+1)][col] += 1
        W[(k0+1)*n+(i0+1)][col] += 1
        col += 1
        
        # M7 = (A00-A10)(B01+B11) -> C00 += 1
        U[i0*m+j0][col] += 1
        U[(i0+1)*m+j0][col] -= 1
        V[j0*p+(k0+1)][col] += 1
        V[(j0+1)*p+(k0+1)][col] += 1
        W[k0*n+i0][col] += 1
        col += 1
        
        return col
    
    # Helper: add rank-2 decomposition for 1x2x7 (i fixed, 2 j's, 7 k's)
    def add_1x2x7(i, j0, col):
        # C[i,k] = A[i,j0]B[j0,k] + A[i,j0+1]B[j0+1,k]
        # Term 0: U[i,j0]=1, V[j0,k]=1 for all k, W[k,i]=1 for all k
        U[i*m+j0][col] = 1
        for k in range(p):
            V[j0*p+k][col] = 1
            W[k*n+i][col] = 1
        col += 1
        # Term 1: U[i,j0+1]=1, V[j0+1,k]=1 for all k, W[k,i]=1 for all k
        U[i*m+(j0+1)][col] = 1
        for k in range(p):
            V[(j0+1)*p+k][col] = 1
            W[k*n+i][col] = 1
        col += 1
        return col
    
    # Helper: add rank-2 decomposition for 2x1x7 (2 i's, j fixed, 7 k's)
    def add_2x1x7(i0, j, col):
        # C[i,k] = A[i,j]B[j,k] for i in {i0, i0+1}
        # Term 0: U[i0,j]=1, V[j,k]=1 for all k, W[k,i0]=1 for all k
        U[i0*m+j][col] = 1
        for k in range(p):
            V[j*p+k][col] = 1
            W[k*n+i0][col] = 1
        col += 1
        # Term 1: U[i0+1,j]=1, V[j,k]=1 for all k, W[k,i0+1]=1 for all k
        U[(i0+1)*m+j][col] = 1
        for k in range(p):
            V[j*p+k][col] = 1
            W[k*n+(i0+1)][col] = 1
        col += 1
        return col
    
    # Helper: add rank-1 decomposition for 1x1x7
    def add_1x1x7(i, j, col):
        U[i*m+j][col] = 1
        for k in range(p):
            V[j*p+k][col] = 1
            W[k*n+i][col] = 1
        col += 1
        return col
    
    # Helper: add rank-2 decomposition for 2x2x1
    def add_2x2x1(i0, j0, k, col):
        # C[i,k] = A[i,j0]B[j0,k] + A[i,j0+1]B[j0+1,k] for i in {i0,i0+1}
        # Term 0: U[i,j0]=1 for i in {i0,i0+1}, V[j0,k]=1, W[k,i]=1 for i in {i0,i0+1}
        for i in range(i0, i0+2):
            U[i*m+j0][col] = 1
        V[j0*p+k][col] = 1
        for i in range(i0, i0+2):
            W[k*n+i][col] = 1
        col += 1
        # Term 1: U[i,j0+1]=1 for i in {i0,i0+1}, V[j0+1,k]=1, W[k,i]=1 for i in {i0,i0+1}
        for i in range(i0, i0+2):
            U[i*m+(j0+1)][col] = 1
        V[(j0+1)*p+k][col] = 1
        for i in range(i0, i0+2):
            W[k*n+i][col] = 1
        col += 1
        return col
    
    # Add 6 Strassen blocks
    for j0 in [0, 2]:
        for k0 in [0, 2, 4]:
            if time.time() > deadline:
                break
            col = add_strassen(0, j0, k0, col)
        if time.time() > deadline:
            break
    
    # Non-Strassen parts
    if time.time() < deadline:
        col = add_2x2x1(0, 0, 6, col)  # i={0,1}, j={0,1}, k={6}
    if time.time() < deadline:
        col = add_2x2x1(0, 2, 6, col)  # i={0,1}, j={2,3}, k={6}
    if time.time() < deadline:
        col = add_2x1x7(0, 4, col)  # i={0,1}, j={4}, k={0..6}
    if time.time() < deadline:
        col = add_1x2x7(2, 0, col)  # i={2}, j={0,1}, k={0..6}
    if time.time() < deadline:
        col = add_1x2x7(2, 2, col)  # i={2}, j={2,3}, k={0..6}
    if time.time() < deadline:
        col = add_1x1x7(2, 4, col)  # i={2}, j={4}, k={0..6}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
