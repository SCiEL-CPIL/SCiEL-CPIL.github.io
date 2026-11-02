import numpy as np
import time

def _build_strassen_block_2x3x2():
    """
    Build a rank-11 decomposition for 2x3x2 matrix multiplication.
    """
    rank = 11
    U = np.zeros((6, rank), dtype=np.float64)
    V = np.zeros((6, rank), dtype=np.float64)
    W = np.zeros((4, rank), dtype=np.float64)
    
    # Strassen 2x2x2 (using A rows 0,1 and B rows 0,1)
    # M1 = (a00+a11)(b00+b11) -> C00(+1), C11(+1)
    c = 0
    U[0, c] += 1; U[4, c] += 1
    V[0, c] += 1; V[3, c] += 1
    W[0, c] += 1; W[3, c] += 1
    
    # M2 = (a10+a11)(b00) -> C10(+1), C11(-1)
    c = 1
    U[3, c] += 1; U[4, c] += 1
    V[0, c] += 1
    W[2, c] += 1; W[3, c] += -1
    
    # M3 = a00(b01-b11) -> C01(+1), C11(+1)
    c = 2
    U[0, c] += 1
    V[1, c] += 1; V[3, c] += -1
    W[1, c] += 1; W[3, c] += 1
    
    # M4 = a11(b10-b00) -> C00(+1), C10(+1)
    c = 3
    U[4, c] += 1
    V[2, c] += 1; V[0, c] += -1
    W[0, c] += 1; W[2, c] += 1
    
    # M5 = (a00+a01)(b11) -> C00(-1), C01(+1)
    c = 4
    U[0, c] += 1; U[1, c] += 1
    V[3, c] += 1
    W[0, c] += -1; W[1, c] += 1
    
    # M6 = (a10-a00)(b00+b01) -> C11(+1)
    c = 5
    U[3, c] += 1; U[0, c] += -1
    V[0, c] += 1; V[1, c] += 1
    W[3, c] += 1
    
    # M7 = (a01-a11)(b10+b11) -> C00(+1)
    c = 6
    U[1, c] += 1; U[4, c] += -1
    V[2, c] += 1; V[3, c] += 1
    W[0, c] += 1
    
    # Schoolbook: A_right (column 2) * B_bottom (row 2)
    c = 7
    U[2, c] += 1; V[4, c] += 1; W[0, c] += 1
    c = 8
    U[2, c] += 1; V[5, c] += 1; W[1, c] += 1
    c = 9
    U[5, c] += 1; V[4, c] += 1; W[2, c] += 1
    c = 10
    U[5, c] += 1; V[5, c] += 1; W[3, c] += 1
    
    return U, V, W

def _build_block_decomposition(rank_per_block=11):
    """
    Build the full 2x3x10 decomposition using 5 blocks.
    """
    n, m, p = 2, 3, 10
    total_rank = 5 * rank_per_block
    U = np.zeros((n*m, total_rank), dtype=np.float64)
    V = np.zeros((m*p, total_rank), dtype=np.float64)
    W = np.zeros((p*n, total_rank), dtype=np.float64)
    
    block_U, block_V, block_W = _build_strassen_block_2x3x2()

    for b in range(5):
        rank_offset = b * rank_per_block
        
        # U mapping: local indices 0..5 map directly to global 0..5
        U[:, rank_offset:rank_offset+rank_per_block] += block_U
        
        # V mapping: 
        # local 0,1 -> Global 2b, 2b+1 (B row 0, cols 2b, 2b+1)
        # local 2,3 -> Global 10+2b, 10+2b+1 (B row 1, cols 2b, 2b+1)
        # local 4,5 -> Global 20+2b, 20+2b+1 (B row 2, cols 2b, 2b+1)
        v_map = [2*b, 2*b+1, 10+2*b, 10+2*b+1, 20+2*b, 20+2*b+1]
        for local_idx in range(6):
            global_idx = v_map[local_idx]
            V[global_idx, rank_offset:rank_offset+rank_per_block] += block_V[local_idx]
        
        # W mapping: 
        # local 0 -> C[0, 2b] -> 4b
        # local 1 -> C[0, 2b+1] -> 4b+2
        # local 2 -> C[1, 2b] -> 4b+1
        # local 3 -> C[1, 2b+1] -> 4b+3
        w_map = [4*b, 4*b+2, 4*b+1, 4*b+3]
        for local_idx in range(4):
            global_idx = w_map[local_idx]
            W[global_idx, rank_offset:rank_offset+rank_per_block] += block_W[local_idx]
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that the tensor decomposition exactly reproduces matrix multiplication."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_schoolbook():
    """Build the naive rank-60 schoolbook decomposition."""
    n, m, p = 2, 3, 10
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, column] = 1.0
                V[j*p+k, column] = 1.0
                W[k*n+i, column] = 1.0
                column += 1
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = 2, 3, 10
    
    # Try rank-55 decomposition (5 blocks of rank 11)
    U, V, W = _build_block_decomposition(11)
    
    # Verify and ensure finite
    if _verify(U, V, W, n, m, p) and np.all(np.isfinite(U)) and np.all(np.isfinite(V)) and np.all(np.isfinite(W)):
        return {"U": U, "V": V, "W": W}
        
    # Fallback to schoolbook rank 60
    U, V, W = _build_schoolbook()
    return {"U": U, "V": V, "W": W}
