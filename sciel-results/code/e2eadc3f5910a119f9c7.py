"""Rewritten program for 4x5x6 matrix multiplication tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130.0
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build target tensor entries
    target_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_entries.append((a, b, c))
    
    # Schoolbook fallback
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(nm)]
    V_sb = [[0] * schoolbook_rank for _ in range(mp)]
    W_sb = [[0] * schoolbook_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a][col] = 1
                V_sb[b][col] = 1
                W_sb[c][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    def strassen2_decomp():
        """2x2x2 Strassen, rank 7"""
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        # M1=(A11+A22)(B11+B22): C11+=M1, C22+=M1
        U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
        # M2=(A21+A22)B21: C21+=M2
        U[2][1]=1; U[3][1]=1; V[2][1]=1; W[2][1]=1
        # M3=A11(B12-B22): C12+=M3
        U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1
        # M4=A22(B21-B11): C11+=M4, C21+=M4
        U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[2][3]=1
        # M5=(A11+A12)B22: C11-=M5, C12+=M5
        U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
        # M6=(A21-A11)(B11+B12): C22+=M6
        U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
        # M7=(A12-A22)(B21+B22): C11+=M7, C22+=M7
        U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1; W[3][6]=1
        return U, V, W
    
    def verify(U, V, W, rank):
        """Verify that the decomposition is correct."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0.0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    def build_strassen_hybrid():
        """Build decomposition using Strassen where beneficial.
        Partition: n=2+2, m=3+2, p=3+3
        2x3x3 blocks: rank 17 each (4 blocks)
        2x2x3 blocks: rank 11 each (4 blocks)
        Total: 112
        """
        total_rank = 4*17 + 4*11  # 112
        U_full = [[0] * total_rank for _ in range(nm)]
        V_full = [[0] * total_rank for _ in range(mp)]
        W_full = [[0] * total_rank for _ in range(pn)]
        
        col = 0
        
        # Helper: add a 2x3x3 block at (ni_offset, mj_offset, pk_offset)
        def add_2x3x3(ni_off, mj_off, pk_off):
            nonlocal col
            # A: ni=2, mj=3; B: mj=3, pk=3; C: pk=3, ni=2
            # Part 1: A[:,:2] (2x2) * B[:,:2] (3x2) -> C[:,:2] (2x2)
            # Use Strassen for 2x2*2x2
            sU, sV, sW = strassen2_decomp()
            for r in range(7):
                for i in range(2):
                    for j in range(2):
                        U_full[(ni_off+i)*m + (mj_off+j)][col] += sU[i*2+j][r]
                for j in range(2):
                    for k in range(2):
                        V_full[(mj_off+j)*p + (pk_off+k)][col] += sV[j*2+k][r]
                for k in range(2):
                    for i in range(2):
                        W_full[(pk_off+k)*n + (ni_off+i)][col] += sW[k*2+i][r]
                col += 1
            # Part 1b: A[:,2] (2x1) * B[2,:2] (1x2) -> C[:,:2] (2x2)
            # C[k,i] += A[i,2] * B[2,k] for k in [0,2), i in [0,2)
            for i in range(2):
                for k in range(2):
                    U_full[(ni_off+i)*m + (mj_off+2)][col] = 1
                    V_full[(mj_off+2)*p + (pk_off+k)][col] = 1
                    W_full[(pk_off+k)*n + (ni_off+i)][col] = 1
                    col += 1
            # Part 2: A (2x3) * B[:,2] (3x1) -> C[:,2] (2x1)
            # C[2,i] = sum_j A[i,j]*B[j,2] for i in [0,2)
            for i in range(2):
                for j in range(3):
                    U_full[(ni_off+i)*m + (mj_off+j)][col] = 1
                    V_full[(mj_off+j)*p + (pk_off+2)][col] = 1
                    W_full[(pk_off+2)*n + (ni_off+i)][col] = 1
                    col += 1
        
        # Helper: add a 2x2x3 block at (ni_offset, mj_offset, pk_offset)
        def add_2x2x3(ni_off, mj_off, pk_off):
            nonlocal col
            # A: ni=2, mj=2; B: mj=2, pk=3; C: pk=3, ni=2
            # Part 1: A (2x2) * B[:,:2] (2x2) -> C[:,:2] (2x2)
            # Use Strassen for 2x2*2x2
            sU, sV, sW = strassen2_decomp()
            for r in range(7):
                for i in range(2):
                    for j in range(2):
                        U_full[(ni_off+i)*m + (mj_off+j)][col] += sU[i*2+j][r]
                for j in range(2):
                    for k in range(2):
                        V_full[(mj_off+j)*p + (pk_off+k)][col] += sV[j*2+k][r]
                for k in range(2):
                    for i in range(2):
                        W_full[(pk_off+k)*n + (ni_off+i)][col] += sW[k*2+i][r]
                col += 1
            # Part 2: A (2x2) * B[:,2] (2x1) -> C[:,2] (2x1)
            # C[2,i] = sum_j A[i,j]*B[j,2] for i in [0,2)
            for i in range(2):
                for j in range(2):
                    U_full[(ni_off+i)*m + (mj_off+j)][col] = 1
                    V_full[(mj_off+j)*p + (pk_off+2)][col] = 1
                    W_full[(pk_off+2)*n + (ni_off+i)][col] = 1
                    col += 1
        
        # Add all 8 blocks
        # (0,0,0): 2x3x3, ni_off=0, mj_off=0, pk_off=0
        add_2x3x3(0, 0, 0)
        # (0,0,1): 2x3x3, ni_off=0, mj_off=0, pk_off=3
        add_2x3x3(0, 0, 3)
        # (0,1,0): 2x2x3, ni_off=0, mj_off=3, pk_off=0
        add_2x2x3(0, 3, 0)
        # (0,1,1): 2x2x3, ni_off=0, mj_off=3, pk_off=3
        add_2x2x3(0, 3, 3)
        # (1,0,0): 2x3x3, ni_off=2, mj_off=0, pk_off=0
        add_2x3x3(2, 0, 0)
        # (1,0,1): 2x3x3, ni_off=2, mj_off=0, pk_off=3
        add_2x3x3(2, 0, 3)
        # (1,1,0): 2x2x3, ni_off=2, mj_off=3, pk_off=0
        add_2x2x3(2, 3, 0)
        # (1,1,1): 2x2x3, ni_off=2, mj_off=3, pk_off=3
        add_2x2x3(2, 3, 3)
        
        # Verify
        if col != total_rank:
            return None
        
        if verify(U_full, V_full, W_full, total_rank):
            return {"U": U_full, "V": V_full, "W": W_full}
        return None
    
    # Try the Strassen hybrid
    result = build_strassen_hybrid()
    if result is not None:
        best_result = result
        best_rank = 112
    
    # Try to find a lower rank decomposition using peeling strategy
    # The peeling strategy: try removing each column from the 112-rank decomposition
    # and check if the remaining 111 terms still produce the correct tensor.
    
    def try_peeling_reduction():
        """Try to reduce rank by removing redundant columns."""
        if time.time() > deadline - 30:
            return None
        
        # Build the 112-rank decomposition
        U_112 = [[0] * 112 for _ in range(nm)]
        V_112 = [[0] * 112 for _ in range(mp)]
        W_112 = [[0] * 112 for _ in range(pn)]
        
        col = 0
        
        def add_2x3x3_112(ni_off, mj_off, pk_off):
            nonlocal col
            sU, sV, sW = strassen2_decomp()
            for r in range(7):
                for i in range(2):
                    for j in range(2):
                        U_112[(ni_off+i)*m + (mj_off+j)][col] += sU[i*2+j][r]
                for j in range(2):
                    for k in range(2):
                        V_112[(mj_off+j)*p + (pk_off+k)][col] += sV[j*2+k][r]
                for k in range(2):
                    for i in range(2):
                        W_112[(pk_off+k)*n + (ni_off+i)][col] += sW[k*2+i][r]
                col += 1
            for i in range(2):
                for k in range(2):
                    U_112[(ni_off+i)*m + (mj_off+2)][col] = 1
                    V_112[(mj_off+2)*p + (pk_off+k)][col] = 1
                    W_112[(pk_off+k)*n + (ni_off+i)][col] = 1
                    col += 1
            for i in range(2):
                for j in range(3):
                    U_112[(ni_off+i)*m + (mj_off+j)][col] = 1
                    V_112[(mj_off+j)*p + (pk_off+2)][col] = 1
                    W_112[(pk_off+2)*n + (ni_off+i)][col] = 1
                    col += 1
        
        def add_2x2x3_112(ni_off, mj_off, pk_off):
            nonlocal col
            sU, sV, sW = strassen2_decomp()
            for r in range(7):
                for i in range(2):
                    for j in range(2):
                        U_112[(ni_off+i)*m + (mj_off+j)][col] += sU[i*2+j][r]
                for j in range(2):
                    for k in range(2):
                        V_112[(mj_off+j)*p + (pk_off+k)][col] += sV[j*2+k][r]
                for k in range(2):
                    for i in range(2):
                        W_112[(pk_off+k)*n + (ni_off+i)][col] += sW[k*2+i][r]
                col += 1
            for i in range(2):
                for j in range(2):
                    U_112[(ni_off+i)*m + (mj_off+j)][col] = 1
                    V_112[(mj_off+j)*p + (pk_off+2)][col] = 1
                    W_112[(pk_off+2)*n + (ni_off+i)][col] = 1
                    col += 1
        
        add_2x3x3_112(0, 0, 0)
        add_2x3x3_112(0, 0, 3)
        add_2x2x3_112(0, 3, 0)
        add_2x2x3_112(0, 3, 3)
        add_2x3x3_112(2, 0, 0)
        add_2x3x3_112(2, 0, 3)
        add_2x2x3_112(2, 3, 0)
        add_2x2x3_112(2, 3, 3)
        
        # Try removing each column and verify
        U_np = np.array(U_112, dtype=float)
        V_np = np.array(V_112, dtype=float)
        W_np = np.array(W_112, dtype=float)
        
        # Precompute the tensor contribution of each column
        # T[r][a,b,c] = U[a][r]*V[b][r]*W[c][r]
        # We need sum_r T[r][a,b,c] = 1 for all target (a,b,c)
        
        # Build target entries
        target_entries = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    target_entries.append((a, b, c))
        
        # For each column, compute its contribution to each target entry
        # contrib[r][t] = U[a_t][r] * V[b_t][r] * W[c_t][r]
        contrib = np.zeros((112, 120), dtype=float)
        for r in range(112):
            for t, (a, b, c) in enumerate(target_entries):
                contrib[r, t] = U_np[a, r] * V_np[b, r] * W_np[c, r]
        
        # The total sum should be 1 for each target entry
        total = contrib.sum(axis=0)
        
        # For each column to remove, check if the remaining sum is still 1
        for elim in range(112):
            if time.time() > deadline - 10:
                break
            
            remaining_sum = total - contrib[elim]
            if np.allclose(remaining_sum, 1.0, atol=1e-10):
                # This column is redundant!
                other_cols = [s for s in range(112) if s != elim]
                new_rank = 111
                U_new = [[0] * new_rank for _ in range(nm)]
                V_new = [[0] * new_rank for _ in range(mp)]
                W_new = [[0] * new_rank for _ in range(pn)]
                
                new_col = 0
                for s in other_cols:
                    for a in range(nm):
                        U_new[a][new_col] = U_112[a][s]
                    for b in range(mp):
                        V_new[b][new_col] = V_112[b][s]
                    for c in range(pn):
                        W_new[c][new_col] = W_112[c][s]
                    new_col += 1
                
                if verify(U_new, V_new, W_new, new_rank):
                    return {"U": U_new, "V": V_new, "W": W_new}
        
        return None
    
    # Try the peeling reduction
    peel_result = try_peeling_reduction()
    if peel_result is not None:
        best_result = peel_result
        best_rank = 111
    
    return best_result
# EVOLVE-BLOCK-END
