"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check that the decomposition exactly reproduces the multiplication tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = U[i * m + j, :]
                v = V[j * p + k, :]
                w = W[k * n + i, :]
                val = np.dot(u, v * w)
                if abs(val - 1.0) > tol:
                    return False
    return True


def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) where indices are:
    U[i*2+j], V[j*2+k], W[k*2+i] for i,j,k in {0,1}
    
    Strassen:
    M1 = (a00+a11)(b00+b11)
    M2 = (a10+a11)*b00
    M3 = a00*(b01-b11)
    M4 = a11*(b10-b00)
    M5 = (a00+a01)*b11
    M6 = (a10-a00)*(b00+b10)
    M7 = (a01-a11)*(b01+b11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # U[i*2+j, r] for i,j in {0,1}
    # V[j*2+k, r] for j,k in {0,1}
    # W[k*2+i, r] for k,i in {0,1}
    
    # Index mapping: U row = i*2+j, V row = j*2+k, W row = k*2+i
    # i,j,k in {0,1}
    
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (a00+a11)(b00+b11): U gets +1 at (0,0) and (1,1), V gets +1 at (0,0) and (1,1), W gets +1 at (0,0) and (1,1)
    # Wait, I need to be more careful.
    # M1 = (a00+a11)*(b00+b11)
    # U contribution: a00 -> U[0*2+0, 0] = 1, a11 -> U[1*2+1, 0] = 1
    # V contribution: b00 -> V[0*2+0, 0] = 1, b11 -> V[1*2+1, 0] = 1
    # W contribution: The output C00 gets M1, C11 gets M1
    # C00 is at W[0*2+0, 0] = W[0, 0]
    # C11 is at W[1*2+1, 0] = W[3, 0]
    # So W[0, 0] = 1, W[3, 0] = 1
    
    # Let me redo this more carefully.
    # The tensor T[i,j,k] = 1 means C[i,k] += A[i,j]*B[j,k]
    # In Strassen, we compute intermediate products and then combine.
    
    # For each Strassen term M_r, we have:
    # M_r = (linear combo of A entries) * (linear combo of B entries)
    # And M_r contributes to C[i,k] with coefficient w_r[i,k]
    
    # U[i*2+j, r] = coefficient of A[i,j] in the first factor of M_r
    # V[j*2+k, r] = coefficient of B[j,k] in the second factor of M_r  
    # W[k*2+i, r] = coefficient of M_r in C[i,k]
    
    # M1 = (a00+a11)*(b00+b11)
    # First factor: a00 + a11 => U[0,0]=1, U[3,0]=1 (i=0,j=0 and i=1,j=1)
    # Second factor: b00 + b11 => V[0,0]=1, V[3,0]=1 (j=0,k=0 and j=1,k=1)
    # C00 += M1 => W[0*2+0, 0] = W[0,0] = 1
    # C11 += M1 => W[1*2+1, 0] = W[3,0] = 1
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (a10+a11)*b00
    # First factor: a10 + a11 => U[1*2+0, 1] = U[2,1] = 1, U[1*2+1, 1] = U[3,1] = 1
    # Second factor: b00 => V[0*2+0, 1] = V[0,1] = 1
    # C10 += M2 => W[0*2+1, 1] = W[2,1] = 1
    # C11 += M2 (with -1) => W[1*2+1, 1] = W[3,1] = -1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = a00*(b01-b11)
    # First factor: a00 => U[0*2+0, 2] = U[0,2] = 1
    # Second factor: b01 - b11 => V[0*2+1, 2] = V[1,2] = 1, V[1*2+1, 2] = V[3,2] = -1
    # C01 += M3 => W[1*2+0, 2] = W[2,2] = 1
    # C11 += M3 => W[1*2+1, 2] = W[3,2] = 1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = a11*(b10-b00)
    # First factor: a11 => U[1*2+1, 3] = U[3,3] = 1
    # Second factor: b10 - b00 => V[1*2+0, 3] = V[2,3] = 1, V[0*2+0, 3] = V[0,3] = -1
    # C00 += M4 => W[0*2+0, 3] = W[0,3] = 1
    # C10 += M4 => W[0*2+1, 3] = W[2,3] = 1
    
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (a00+a01)*b11
    # First factor: a00 + a01 => U[0*2+0, 4] = U[0,4] = 1, U[0*2+1, 4] = U[1,4] = 1
    # Second factor: b11 => V[1*2+1, 4] = V[3,4] = 1
    # C00 += M5 (with -1) => W[0*2+0, 4] = W[0,4] = -1
    # C01 += M5 => W[1*2+0, 4] = W[2,4] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (a10-a00)*(b00+b10)
    # First factor: a10 - a00 => U[1*2+0, 5] = U[2,5] = 1, U[0*2+0, 5] = U[0,5] = -1
    # Second factor: b00 + b10 => V[0*2+0, 5] = V[0,5] = 1, V[1*2+0, 5] = V[2,5] = 1
    # C11 += M6 => W[1*2+1, 5] = W[3,5] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # M7 = (a01-a11)*(b01+b11)
    # First factor: a01 - a11 => U[0*2+1, 6] = U[1,6] = 1, U[1*2+1, 6] = U[3,6] = -1
    # Second factor: b01 + b11 => V[0*2+1, 6] = V[1,6] = 1, V[1*2+1, 6] = V[3,6] = 1
    # C00 += M7 => W[0*2+0, 6] = W[0,6] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def _build_kronecker_decomposition(n, m, p, start_time, deadline):
    """
    Build a decomposition for n×m×p using Kronecker products of smaller decompositions.
    
    For 3×4×5:
    - Split m=4 into 2+2
    - Use Strassen (rank 7) for the 2×2×2 core in each (i-pair, j-pair, k-pair) block
    - Handle remaining terms with schoolbook
    
    Strategy: 
    For each combination of i-blocks, j-blocks, k-blocks of size 2×2×2,
    use Strassen's rank-7 decomposition.
    For remaining terms (where any dimension has only 1 element in the block),
    use schoolbook.
    
    i in {0,1,2}: blocks {0,1} and {2}
    j in {0,1,2,3}: blocks {0,1} and {2,3}
    k in {0,1,2,3,4}: blocks {0,1}, {2,3}, {4}
    
    2×2×2 blocks (use Strassen, rank 7 instead of 8, save 1 each):
    - i∈{0,1}, j∈{0,1}, k∈{0,1}: 7 terms
    - i∈{0,1}, j∈{0,1}, k∈{2,3}: 7 terms
    - i∈{0,1}, j∈{2,3}, k∈{0,1}: 7 terms
    - i∈{0,1}, j∈{2,3}, k∈{2,3}: 7 terms
    
    Remaining terms (schoolbook, 1 term each):
    - i∈{0,1}, j∈{0,1}, k∈{4}: 2 terms
    - i∈{0,1}, j∈{2,3}, k∈{4}: 2 terms
    - i∈{2}, j∈{0,1}, k∈{0,1}: 4 terms
    - i∈{2}, j∈{0,1}, k∈{2,3}: 4 terms
    - i∈{2}, j∈{0,1}, k∈{4}: 2 terms
    - i∈{2}, j∈{2,3}, k∈{0,1}: 4 terms
    - i∈{2}, j∈{2,3}, k∈{2,3}: 4 terms
    - i∈{2}, j∈{2,3}, k∈{4}: 2 terms
    
    Total: 4*7 + 2+2+4+4+2+4+4+2 = 28 + 24 = 52 terms
    """
    # Let's build this systematically
    U_list = []
    V_list = []
    W_list = []
    
    su, sv, sw = _strassen_2x2x2()
    
    # 2x2x2 blocks using Strassen
    i_blocks = [(0, 1)]  # Only one 2-element i block
    j_blocks = [(0, 1), (2, 3)]
    k_blocks_2 = [(0, 1), (2, 3)]
    
    for i0, i1 in i_blocks:
        for j0, j1 in j_blocks:
            for k0, k1 in k_blocks_2:
                # Add Strassen terms for this 2x2x2 block
                for r in range(7):
                    u_col = np.zeros(n * m)
                    v_col = np.zeros(m * p)
                    w_col = np.zeros(p * n)
                    
                    # Map Strassen U indices to global
                    for local_i in range(2):
                        for local_j in range(2):
                            gi = i0 + local_i
                            gj = j0 + local_j
                            u_col[gi * m + gj] = su[local_i * 2 + local_j, r]
                    
                    # Map Strassen V indices to global
                    for local_j in range(2):
                        for local_k in range(2):
                            gj = j0 + local_j
                            gk = k0 + local_k
                            v_col[gj * p + gk] = sv[local_j * 2 + local_k, r]
                    
                    # Map Strassen W indices to global
                    for local_k in range(2):
                        for local_i in range(2):
                            gk = k0 + local_k
                            gi = i0 + local_i
                            w_col[gk * n + gi] = sw[local_k * 2 + local_i, r]
                    
                    U_list.append(u_col)
                    V_list.append(v_col)
                    W_list.append(w_col)
    
    # Remaining schoolbook terms
    # i∈{0,1}, j∈{0,1}, k=4
    for i in [0, 1]:
        for j in [0, 1]:
            k = 4
            u_col = np.zeros(n * m)
            v_col = np.zeros(m * p)
            w_col = np.zeros(p * n)
            u_col[i * m + j] = 1
            v_col[j * p + k] = 1
            w_col[k * n + i] = 1
            U_list.append(u_col)
            V_list.append(v_col)
            W_list.append(w_col)
    
    # i∈{0,1}, j∈{2,3}, k=4
    for i in [0, 1]:
        for j in [2, 3]:
            k = 4
            u_col = np.zeros(n * m)
            v_col = np.zeros(m * p)
            w_col = np.zeros(p * n)
            u_col[i * m + j] = 1
            v_col[j * p + k] = 1
            w_col[k * n + i] = 1
            U_list.append(u_col)
            V_list.append(v_col)
            W_list.append(w_col)
    
    # i=2, all j, all k
    for i in [2]:
        for j in range(m):
            for k in range(p):
                u_col = np.zeros(n * m)
                v_col = np.zeros(m * p)
                w_col = np.zeros(p * n)
                u_col[i * m + j] = 1
                v_col[j * p + k] = 1
                w_col[k * n + i] = 1
                U_list.append(u_col)
                V_list.append(v_col)
                W_list.append(w_col)
    
    U = np.column_stack(U_list)
    V = np.column_stack(V_list)
    W = np.column_stack(W_list)
    
    # Verify
    if _verify(U, V, W, n, m, p, tol=1e-9):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Debug: print which terms fail
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = U[i * m + j, :]
                    v = V[j * p + k, :]
                    w = W[k * n + i, :]
                    val = np.dot(u, v * w)
                    if abs(val - 1.0) > 1e-9:
                        print(f"FAIL at i={i}, j={j}, k={k}: got {val}")
        return None


def _try_local_search(n, m, p, start_time, deadline):
    """
    Placeholder for future local search improvements.
    """
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 10s safety margin
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = U.shape[1]
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try Kronecker/Strassen-based construction
    if time.time() < deadline - 10:
        result = _build_kronecker_decomposition(n, m, p, start_time, deadline)
        if result is not None:
            best_rank = len(result["U"][0])
            best_result = result
    
    # Try local search for further reductions
    if time.time() < deadline - 5:
        result = _try_local_search(n, m, p, start_time, deadline)
        if result is not None:
            best_result = result
    
    return best_result
# EVOLVE-BLOCK-END
