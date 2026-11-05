import time

DIMENSIONS = [3, 4, 4]

def construct():
    start = time.time()
    n, m, p = 3, 4, 4
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(col, i_off, j_off, k_off):
        U[i_off*m + j_off, col] += 1
        U[(i_off+1)*m + (j_off+1), col] += 1
        V[j_off*p + k_off, col] += 1
        V[(j_off+1)*p + (k_off+1), col] += 1
        W[k_off*n + i_off, col] += 1
        W[(k_off+1)*n + (i_off+1), col] += 1
        col += 1
        
        U[(i_off+1)*m + j_off, col] += 1
        U[(i_off+1)*m + (j_off+1), col] += 1
        V[j_off*p + k_off, col] += 1
        W[k_off*n + (i_off+1), col] += 1
        W[(k_off+1)*n + (i_off+1), col] += -1
        col += 1
        
        U[i_off*m + j_off, col] += 1
        V[j_off*p + (k_off+1), col] += 1
        V[(j_off+1)*p + k_off, col] += -1
        W[(k_off+1)*n + i_off, col] += 1
        W[(k_off+1)*n + (i_off+1), col] += 1
        col += 1
        
        U[(i_off+1)*m + (j_off+1), col] += 1
        V[(j_off+1)*p + k_off, col] += 1
        V[j_off*p + k_off, col] += -1
        W[k_off*n + i_off, col] += -1
        W[(k_off+1)*n + i_off, col] += 1
        col += 1
        
        U[i_off*m + j_off, col] += 1
        U[i_off*m + (j_off+1), col] += 1
        V[(j_off+1)*p + (k_off+1), col] += 1
        W[k_off*n + i_off, col] += 1
        W[k_off*n + (i_off+1), col] += 1
        col += 1
        
        U[(i_off+1)*m + j_off, col] += 1
        U[i_off*m + j_off, col] += -1
        V[j_off*p + k_off, col] += 1
        V[j_off*p + (k_off+1), col] += 1
        W[k_off*n + i_off, col] += 1
        col += 1
        
        U[i_off*m + (j_off+1), col] += 1
        U[(i_off+1)*m + (j_off+1), col] += -1
        V[(j_off+1)*p + k_off, col] += 1
        V[(j_off+1)*p + (k_off+1), col] += 1
        W[(k_off+1)*n + (i_off+1), col] += 1
        col += 1
        
        return col
    
    col = add_strassen(col, 0, 0, 0)
    col = add_strassen(col, 0, 0, 2)
    col = add_strassen(col, 0, 2, 0)
    col = add_strassen(col, 0, 2, 2)
    
    for j in range(4):
        for k in range(4):
            U[2*m + j, col] = 1
            V[j*p + k, col] = 1
            W[k*n + 2, col] = 1
            col += 1
    
    elapsed = time.time() - start
    if elapsed > 140:
        # Safety: return what we have
        pass
    
    return {"U": U, "V": V, "W": W}
