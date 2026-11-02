import time
import sys

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # U: n*m x rank
    # V: m*p x rank
    # W: p*n x rank
    
    # We will build the matrices dynamically
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    def get_A_idx(i, j):
        return i * m + j
        
    def get_B_idx(j, k):
        return j * p + k
        
    def get_C_idx(k, i):
        return k * n + i

    def strassen_block(i_off, j_off, k_off):
        # i_off: 0
        # j_off: 0 or 2
        # k_off: 0, 2, 4
        # Local indices 0,1 for i, j, k in the block
        
        # M1 = (A00+A11)(B00+B11)
        u = [0]*(n*m)
        v = [0]*(m*p)
        w = [0]*(p*n)
        
        # A00, A11
        u[get_A_idx(i_off+0, j_off+0)] = 1
        u[get_A_idx(i_off+1, j_off+1)] = 1
        # B00, B11
        v[get_B_idx(j_off+0, k_off+0)] = 1
        v[get_B_idx(j_off+1, k_off+1)] = 1
        # C00, C11
        w[get_C_idx(k_off+0, i_off+0)] = 1
        w[get_C_idx(k_off+1, i_off+1)] = 1
        add_term(u, v, w)
        
        # M2 = (A10+A11)B00
        u = [0]*(n*m)
        v = [0]*(m*p)
        w = [0]*(p*n)
        u[get_A_idx(i_off+1, j_off+0)] = 1
        u[get_A_idx(i_off+1, j_off+1)] = 1
        v[get_B_idx(j_off+0, k_off+0)] = 1
        w[get_C_idx(k_off+0, i_off+1)] = 1
        w[get_C_idx(k_off+1, i_off+1)] = 1
        add_term(u, v, w)
        
        # M3 = A00(B01-B11)
        u = [0]*(n*m)
        v = [0]*(m*p)
        w = [0]*(p*n)
        u[get_A_idx(i_off+0, j_off+0)] = 1
        v[get_B_idx(j_off+0, k_off+1)] = 1
        v[get_B_idx(j_off+1, k_off+1)] = -1
        w[get_C_idx(k_off+1, i_off+0)] = 1
        add_term(u, v, w)
        
        # M4 = A11(B10-B00)
        u = [0]*(n*m)
        v = [0]*(m*p)
        w = [0]*(p*n)
        u[get_A_idx(i_off+1, j_off+1)] = 1
        v[get_B_idx(j_off+1, k_off+0)] = 1
        v[get_B_idx(j_off+0, k_off+0)] = -1
        w[get_C_idx(k_off+0, i_off+1)] = 1
        w[get_C_idx(k_off+1, i_off+1)] = 1
        add_term(u, v, w)
        
        # M5 = (A00+A01)B11
        u = [0]*(n*m)
        v = [0]*(m*p)
        w = [0]*(p*n)
        u[get_A_idx(i_off+0, j_off+0)] = 1
        u[get_A_idx(i_off+0, j_off+1)] = 1
        v[get_B_idx(j_off+1, k_off+1)] = 1
        w[get_C_idx(k_off+1, i_off+0)] = 1
        w[get_C_idx(k_off+1, i_off+1)] = 1
        add_term(u, v, w)
        
        # M6 = (A10-A00)(B00+B01)
        u = [0]*(n*m)
        v = [0]*(m*p)
        w = [0]*(p*n)
        u[get_A_idx(i_off+1, j_off+0)] = 1
        u[get_A_idx(i_off+0, j_off+0)] = -1
        v[get_B_idx(j_off+0, k_off+0)] = 1
        v[get_B_idx(j_off+0, k_off+1)] = 1
        w[get_C_idx(k_off+1, i_off+1)] = 1
        add_term(u, v, w)
        
        # M7 = (A01-A11)(B10+B11)
        u = [0]*(n*m)
        v = [0]*(m*p)
        w = [0]*(p*n)
        u[get_A_idx(i_off+0, j_off+1)] = 1
        u[get_A_idx(i_off+1, j_off+1)] = -1
        v[get_B_idx(j_off+1, k_off+0)] = 1
        v[get_B_idx(j_off+1, k_off+1)] = 1
        w[get_C_idx(k_off+0, i_off+0)] = 1
        w[get_C_idx(k_off+1, i_off+0)] = 1
        add_term(u, v, w)
        
        # We need to add the reconstruction terms to W?
        # No, the tensor decomposition is defined by the factors U, V, W.
        # The "reconstruction" is implicit in the W matrix entries.
        # Wait, the standard decomposition is C = sum_k U[:,k] V[:,k] W[:,k]^T ?
        # No, C_{ik} = sum_r U_{ir, r} V_{jr, r} W_{kr, r} ?
        # The task says: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)".
        # And "exactly decompose the matrix multiplication tensor".
        # The tensor is $T_{i,j,k} = \delta_{ik}$? No.
        # $C_{ik} = \sum_j A_{ij} B_{jk}$.
        # Tensor $T_{i,j,k}$ such that $C_{ik} = \sum_j T_{i,j,k} A_{ij} B_{jk}$? No.
        # The bilinear map is $A, B \to C$.
        # $C_{ik} = \sum_{j} A_{ij} B_{jk}$.
        # A decomposition is a set of triples $(u_r, v_r, w_r)$ such that
        # $C_{ik} = \sum_r w_{ik, r} (u_{ij, r} \cdot A) (v_{jk, r} \cdot B)$?
        # Actually, the standard form is:
        # $C_{ik} = \sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}$? No.
        # The indices of U, V, W are:
        # U: (n*m) x rank. Index (i*m+j, r).
        # V: (m*p) x rank. Index (j*p+k, r).
        # W: (p*n) x rank. Index (k*n+i, r).
        # The product is $\sum_r U_{i m + j, r} V_{j p + k, r} W_{k n + i, r}$.
        # This sum must equal $\delta_{i,j,k}$? No.
        # It must equal the coefficient of $A_{ij} B_{jk}$ in $C_{ik}$.
        # $C_{ik} = \sum_j A_{ij} B_{jk}$.
        # So the term $A_{ij} B_{jk}$ appears in $C_{ik}$ with coefficient 1.
        # So we need $\sum_r U_{i m + j, r} V_{j p + k, r} W_{k n + i, r} = \delta_{i,k}$? No.
        # The tensor is $T_{i,j,k}$.
        # $T_{i,j,k}$ is 1 if the product $A_{ij} B_{jk}$ contributes to $C_{ik}$.
        # Yes, $C_{ik}$ gets contribution from $A_{ij} B_{jk}$.
        # So $T_{i,j,k} = 1$ for all i,j,k? No.
        # The decomposition is of the tensor $T$ where $T_{i,j,k} = 1$ if $A_{ij} B_{jk}$ is part of $C_{ik}$.
        # Actually, the tensor is defined by $C_{ik} = \sum_{j} A_{ij} B_{jk}$.
        # This is a sum of $n \cdot m \cdot p$ terms $A_{ij} B_{jk} \to C_{ik}$.
        # Each term corresponds to a rank 1 tensor with non-zero entries at $(i,j), (j,k), (k,i)$.
        # So the schoolbook decomposition has rank $n \cdot m \cdot p$.
        # The Strassen decomposition groups these terms.
        
        # My `add_term` adds a rank 1 component.
        # $u$ is the linear form in A.
        # $v$ is the linear form in B.
        # $w$ is the linear form in C.
        # The contribution is $w_{ik} \cdot u_{ij} \cdot v_{jk}$.
        # This matches the structure.
        
        pass # The functions above add the terms directly.
        
    def schoolbook_221_block(i_off, j_off, k_off):
        # i_off: 0
        # j_off: 0 or 2
        # k_off: 6
        # B is 2x1 at (j_off, k_off)
        # A is 2x2 at (i_off, j_off)
        # C is 2x1 at (k_off, i_off)
        
        # M1 = A00 * B0
        u = [0]*(n*m)
        v = [0]*(m*p)
        w = [0]*(p*n)
        u[get_A_idx(i_off+0, j_off+0)] = 1
        v[get_B_idx(j_off+0, k_off)] = 1
        w[get_C_idx(k_off, i_off+0)] = 1
        add_term(u, v, w)
        
        # M2 = A01 * B1
        u = [0]*(n*m)
        v = [0]*(m*p)
        w = [0]*(p*n)
        u[get_A_idx(i_off+0, j_off+1)] = 1
        v[get_B_idx(j_off+1, k_off)] = 1
        w[get_C_idx(k_off, i_off+0)] = 1
        add_term(u, v, w)
        
        # M3 = A10 * B0
        u = [0]*(n*m)
        v = [0]*(m*p)
        w = [0]*(p*n)
        u[get_A_idx(i_off+1, j_off+0)] = 1
        v[get_B_idx(j_off+0, k_off)] = 1
        w[get_C_idx(k_off, i_off+1)] = 1
        add_term(u, v, w)
        
        # M4 = A11 * B1
        u = [0]*(n*m)
        v = [0]*(m*p)
        w = [0]*(p*n)
        u[get_A_idx(i_off+1, j_off+1)] = 1
        v[get_B_idx(j_off+1, k_off)] = 1
        w[get_C_idx(k_off, i_off+1)] = 1
        add_term(u, v, w)

    # Call blocks
    # Part 1: k in 0..3
    strassen_block(0, 0, 0)
    strassen_block(0, 0, 2)
    strassen_block(0, 2, 0)
    strassen_block(0, 2, 2)
    
    # Part 2: k in 4..6
    strassen_block(0, 0, 4)
    schoolbook_221_block(0, 0, 6)
    strassen_block(0, 2, 4)
    schoolbook_221_block(0, 2, 6)
    
    rank = len(U_cols)
    
    # Construct matrices
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for r in range(rank):
        u_vec = U_cols[r]
        v_vec = V_cols[r]
        w_vec = W_cols[r]
        for idx, val in enumerate(u_vec):
            if val != 0:
                U[idx][r] = val
        for idx, val in enumerate(v_vec):
            if val != 0:
                V[idx][r] = val
        for idx, val in enumerate(w_vec):
            if val != 0:
                W[idx][r] = val
                
    return {"U": U, "V": V, "W": W}
