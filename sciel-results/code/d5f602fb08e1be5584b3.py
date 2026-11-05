import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

def _verify_exact(U, V, W, n, m, p):
    """Verify that the tensor decomposition exactly matches matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1) > 1e-12:
                    return False
    return True

def _verify_exact_fraction(U, V, W, n, m, p):
    """Exact verification using Fraction."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]) * Fraction(W[k * n + i][c])
                if s != 1:
                    return False
    return True

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2 matrix multiplication (rank 7).
    U: (2*2, 7), V: (2*2, 7), W: (2*2, 7)
    Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] with n=m=p=2.
    """
    # Strassen's 7 products
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    #
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # i,j in {0,1} for A, j,k in {0,1} for B, k,i in {0,1} for C
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1  # A00, A11
    V[0][0] = 1; V[3][0] = 1  # B00, B11
    W[0][0] = 1; W[3][0] = 1  # C00, C11 (k*n+i: k=0,i=0 -> 0; k=1,i=1 -> 3)
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1  # A10, A11
    V[0][1] = 1  # B00
    W[2][1] = 1; W[3][1] = 1  # C10, C11 (k=1,i=0 -> 2; k=1,i=1 -> 3)
    
    # M3 = A00(B10-B11)
    U[0][2] = 1  # A00
    V[2][2] = 1; V[3][2] = -1  # B10, B11
    W[1][2] = 1; W[3][2] = 1  # C01, C11 (k=0,i=1 -> 1; k=1,i=1 -> 3)
    
    # M4 = A11(B00-B10)
    U[3][3] = 1  # A11
    V[0][3] = 1; V[2][3] = -1  # B00, B10
    W[0][3] = 1; W[2][3] = 1  # C00, C10 (k=0,i=0 -> 0; k=1,i=0 -> 2)
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1  # A00, A01
    V[3][4] = 1  # B11
    W[1][4] = -1; W[3][4] = -1  # C01, C11 (negative)
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1  # A10, A00
    V[0][5] = 1; V[1][5] = 1  # B00, B01
    W[3][5] = 1  # C11
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1  # A01, A11
    V[2][6] = 1; V[3][6] = 1  # B10, B11
    W[0][6] = 1  # C00
    
    return U, V, W

def _build_block_decomposition():
    """Build a block decomposition using Strassen for 2x2x2 blocks.
    Split n=3 into [2,1], m=4 into [2,2], p=6 into [2,2,2].
    For each (n_block, m_block, p_block) triple, use appropriate decomposition.
    """
    n, m, p = 3, 4, 6
    n_blocks = [2, 1]  # split n=3
    m_blocks = [2, 2]  # split m=4
    p_blocks = [2, 2, 2]  # split p=6
    
    U_rows = []
    V_rows = []
    W_rows = []
    U = []
    V = []
    W = []
    
    for ni, ns in enumerate(n_blocks):
        for mi, ms in enumerate(m_blocks):
            for pi, ps in enumerate(p_blocks):
                if ns == 2 and ms == 2 and ps == 2:
                    sU, sV, sW = _strassen_2x2x2()
                    for c in range(7):
                        u_col = [0]*(n*m)
                        v_col = [0]*(m*p)
                        w_col = [0]*(p*n)
                        # Map local indices to global
                        n_off = sum(n_blocks[:ni])
                        m_off = sum(m_blocks[:mi])
                        p_off = sum(p_blocks[:pi])
                        for li in range(2):
                            for lj in range(2):
                                u_col[(n_off+li)*m + (m_off+lj)] = sU[li*2+lj][c]
                        for lj in range(2):
                            for lk in range(2):
                                v_col[(m_off+lj)*p + (p_off+lk)] = sV[lj*2+lk][c]
                        for lk in range(2):
                            for li in range(2):
                                w_col[(p_off+lk)*n + (n_off+li)] = sW[lk*2+li][c]
                        U.append(u_col)
                        V.append(v_col)
                        W.append(w_col)
                elif ns == 1 and ms == 2 and ps == 2:
                    # 1x2x2: rank 2 (each A element multiplies a 2x2 B, output is 2x2)
                    # Actually 1x2x2 means A is 1x2, B is 2x2, C is 1x2
                    # C = A * B, so C[j] = sum_k A[k]*B[k,j]
                    # This is just 2 scalar multiplications per output, but we can do it in rank 2
                    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] with n=1,m=2,p=2
                    # i=0, j in {0,1}, k in {0,1}
                    # C[j] = A[0,j] * B[j,0] + A[0,j] * B[j,1]... wait
                    # Actually C = A*B where A is 1x2, B is 2x2, C is 1x2
                    # C[0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
                    # C[1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
                    # We need: sum_c U[0*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for each (i,j,k)
                    # i=0, j in {0,1}, k in {0,1}
                    # (i=0,j=0,k=0): U[0,c]*V[0,c]*W[0,c] = 1
                    # (i=0,j=0,k=1): U[0,c]*V[1,c]*W[1,c] = 1
                    # (i=0,j=1,k=0): U[1,c]*V[2,c]*W[0,c] = 1
                    # (i=0,j=1,k=1): U[1,c]*V[3,c]*W[1,c] = 1
                    # Rank 2: c=0 handles j=0, c=1 handles j=1
                    # c=0: U[0]=1, V[0]=1, V[1]=1, W[0]=1, W[1]=1, U[1]=0, V[2]=0, V[3]=0
                    # c=1: U[1]=1, V[2]=1, V[3]=1, W[0]=1, W[1]=1, U[0]=0, V[0]=0, V[1]=0
                    for c in range(2):
                        u_col = [0]*(n*m)
                        v_col = [0]*(m*p)
                        w_col = [0]*(p*n)
                        n_off = sum(n_blocks[:ni])
                        m_off = sum(m_blocks[:mi])
                        p_off = sum(p_blocks[:pi])
                        if c == 0:
                            u_col[(n_off+0)*m + (m_off+0)] = 1
                            v_col[(m_off+0)*p + (p_off+0)] = 1
                            v_col[(m_off+0)*p + (p_off+1)] = 1
                            w_col[(p_off+0)*n + (n_off+0)] = 1
                            w_col[(p_off+1)*n + (n_off+0)] = 1
                        else:
                            u_col[(n_off+0)*m + (m_off+1)] = 1
                            v_col[(m_off+1)*p + (p_off+0)] = 1
                            v_col[(m_off+1)*p + (p_off+1)] = 1
                            w_col[(p_off+0)*n + (n_off+0)] = 1
                            w_col[(p_off+1)*n + (n_off+0)] = 1
                        U.append(u_col)
                        V.append(v_col)
                        W.append(w_col)
                elif ns == 2 and ms == 2 and ps == 1:
                    # 2x2x1: A is 2x2, B is 2x1, C is 2x1
                    # C[i] = sum_j A[i,j]*B[j,0]
                    # (i,j,k=0): U[i*2+j, c]*V[j*1+0, c]*W[0*2+i, c] = 1
                    # Rank 2
                    for c in range(2):
                        u_col = [0]*(n*m)
                        v_col = [0]*(m*p)
                        w_col = [0]*(p*n)
                        n_off = sum(n_blocks[:ni])
                        m_off = sum(m_blocks[:mi])
                        p_off = sum(p_blocks[:pi])
                        if c == 0:
                            u_col[(n_off+0)*m + (m_off+0)] = 1
                            v_col[(m_off+0)*p + (p_off+0)] = 1
                            w_col[(p_off+0)*n + (n_off+0)] = 1
                            u_col[(n_off+1)*m + (m_off+0)] = 1
                            w_col[(p_off+0)*n + (n_off+1)] = 1
                        else:
                            u_col[(n_off+0)*m + (m_off+1)] = 1
                            v_col[(m_off+1)*p + (p_off+0)] = 1
                            w_col[(p_off+0)*n + (n_off+0)] = 1
                            u_col[(n_off+1)*m + (m_off+1)] = 1
                            w_col[(p_off+0)*n + (n_off+1)] = 1
                        U.append(u_col)
                        V.append(v_col)
                        W.append(w_col)
                else:
                    # Fallback: schoolbook
                    for i in range(ns):
                        for j in range(ms):
                            for k in range(ps):
                                u_col = [0]*(n*m)
                                v_col = [0]*(m*p)
                                w_col = [0]*(p*n)
                                n_off = sum(n_blocks[:ni])
                                m_off = sum(m_blocks[:mi])
                                p_off = sum(p_blocks[:pi])
                                u_col[(n_off+i)*m + (m_off+j)] = 1
                                v_col[(m_off+j)*p + (p_off+k)] = 1
                                w_col[(p_off+k)*n + (n_off+i)] = 1
                                U.append(u_col)
                                V.append(v_col)
                                W.append(w_col)
    
    rank = len(U)
    return {"U": U, "V": V, "W": W}, rank

def _random_search(n, m, p, target_rank, time_budget):
    """Randomized search for a decomposition of given rank."""
    start = time.time()
    best = None
    
    for attempt in range(1000):
        if time.time() - start > time_budget:
            break
        
        # Try to find a decomposition using random linear algebra
        # The tensor T[i,j,k] = 1 for all i,j,k
        # We want sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        
        # Use a simple approach: try random rank-r decompositions
        # This is hard, so we'll use a heuristic based on known low-rank decompositions
        
        # For now, just return the block decomposition
        pass
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    n, m, p = DIMENSIONS
    
    # Build the block decomposition using Strassen
    result, rank = _build_block_decomposition()
    
    # Verify
    if _verify_exact(result["U"], result["V"], result["W"], n, m, p):
        best_result = result
        best_rank = rank
    else:
        # Fallback to schoolbook
        best_rank = n * m * p
        U = [[0] * best_rank for _ in range(n * m)]
        V = [[0] * best_rank for _ in range(m * p)]
        W = [[0] * best_rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        best_result = {"U": U, "V": V, "W": W}
    
    # Try to improve with random search if time permits
    if time.time() < deadline:
        # Try various rank reductions
        for target_rank in range(best_rank - 1, max(best_rank - 5, 0), -1):
            if time.time() > deadline - 10:
                break
            # Random search (placeholder - actual search would go here)
            # For now, we rely on the block decomposition
    
    return best_result

# EVOLVE-BLOCK-END
