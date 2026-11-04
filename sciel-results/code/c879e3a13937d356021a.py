"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 matrix multiplication (Strassen), rank 7."""
        # Strassen's 7 products for 2x2 matrix multiplication
        # C = A * B where A, B, C are 2x2
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
        # 7 products:
        # p1 = (a+d)(e+h)
        # p2 = (c+d)e
        # p3 = a(f-h)
        # p4 = d(g-e)
        # p5 = (a+b)h
        # p6 = (c-a)(e+f)
        # p7 = (b-d)(g+h)
        # C[0,0] = p1 + p4 - p5 + p7
        # C[0,1] = p3 + p5
        # C[1,0] = p2 + p4
        # C[1,1] = p1 - p2 + p3 + p6
        
        # U maps (i,j) -> coefficients for the 7 products
        # V maps (j,k) -> coefficients for the 7 products  
        # W maps (k,i) -> coefficients for the 7 products (output convention: W[k*n+i, r])
        # For 2x2x2: n=2, m=2, p=2
        # U[i*m+j, r] = U[2*i+j, r]
        # V[j*p+k, r] = V[2*j+k, r]
        # W[k*n+i, r] = W[2*k+i, r]
        
        # Products:
        # p1: (a+d)(e+h) -> U: a=1,d=1; V: e=1,h=1; W: contributes to C[0,0] and C[1,1]
        # p2: (c+d)e -> U: c=1,d=1; V: e=1; W: contributes to C[1,0] and C[1,1]
        # p3: a(f-h) -> U: a=1; V: f=1,h=-1; W: contributes to C[0,1] and C[1,1]
        # p4: d(g-e) -> U: d=1; V: g=1,e=-1; W: contributes to C[0,0] and C[1,0]
        # p5: (a+b)h -> U: a=1,b=1; V: h=1; W: contributes to C[0,0] and C[0,1]
        # p6: (c-a)(e+f) -> U: c=1,a=-1; V: e=1,f=1; W: contributes to C[1,1]
        # p7: (b-d)(g+h) -> U: b=1,d=-1; V: g=1,h=1; W: contributes to C[0,0] and C[1,0]
        
        # Let me define more carefully:
        # C[0,0] = p1 + p4 - p5 + p7
        # C[0,1] = p3 + p5
        # C[1,0] = p2 + p4
        # C[1,1] = p1 - p2 + p3 + p6
        
        # U[2*i+j, r]: row index = 2*i+j
        # i=0,j=0: a -> U[0, r]
        # i=0,j=1: b -> U[1, r]
        # i=1,j=0: c -> U[2, r]
        # i=1,j=1: d -> U[3, r]
        
        # V[2*j+k, r]: row index = 2*j+k
        # j=0,k=0: e -> V[0, r]
        # j=0,k=1: f -> V[1, r]
        # j=1,k=0: g -> V[2, r]
        # j=1,k=1: h -> V[3, r]
        
        # W[2*k+i, r]: row index = 2*k+i
        # k=0,i=0: C[0,0] -> W[0, r]
        # k=0,i=1: C[1,0] -> W[2, r]
        # k=1,i=0: C[0,1] -> W[1, r]
        # k=1,i=1: C[1,1] -> W[3, r]
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # p1: (a+d)(e+h)
        U[0][0] = 1; U[3][0] = 1  # a, d
        V[0][0] = 1; V[3][0] = 1  # e, h
        W[0][0] = 1  # C[0,0]
        W[3][0] = 1  # C[1,1]
        
        # p2: (c+d)e
        U[2][1] = 1; U[3][1] = 1  # c, d
        V[0][1] = 1  # e
        W[2][1] = 1  # C[1,0]
        W[3][1] = -1  # C[1,1]
        
        # p3: a(f-h)
        U[0][2] = 1  # a
        V[1][2] = 1; V[3][2] = -1  # f, -h
        W[1][2] = 1  # C[0,1]
        W[3][2] = 1  # C[1,1]
        
        # p4: d(g-e)
        U[3][3] = 1  # d
        V[2][3] = 1; V[0][3] = -1  # g, -e
        W[0][3] = 1  # C[0,0]
        W[2][3] = 1  # C[1,0]
        
        # p5: (a+b)h
        U[0][4] = 1; U[1][4] = 1  # a, b
        V[3][4] = 1  # h
        W[0][4] = -1  # C[0,0] (negative)
        W[1][4] = 1  # C[0,1]
        
        # p6: (c-a)(e+f)
        U[2][5] = 1; U[0][5] = -1  # c, -a
        V[0][5] = 1; V[1][5] = 1  # e, f
        W[3][5] = 1  # C[1,1]
        
        # p7: (b-d)(g+h)
        U[1][6] = 1; U[3][6] = -1  # b, -d
        V[2][6] = 1; V[3][6] = 1  # g, h
        W[0][6] = 1  # C[0,0]
        W[2][6] = 1  # C[1,0]
        
        return U, V, W
    
    # Try a block decomposition
    # Partition:
    # i: [0,1,2,3] -> groups: [0,1], [2,3]
    # j: [0,1,2,3,4] -> groups: [0,1], [2,3], [4]
    # k: [0,1,2,3,4] -> groups: [0,1], [2,3], [4]
    
    i_groups = [[0,1], [2,3]]
    j_groups = [[0,1], [2,3], [4]]
    k_groups = [[0,1], [2,3], [4]]
    
    # For each combination of (i_group, j_group, k_group), we have a sub-problem
    # of size (len(i_group)) x (len(j_group)) x (len(k_group))
    # We'll use Strassen for 2x2x2 and schoolbook for others
    
    all_U = []
    all_V = []
    all_W = []
    
    for ig in i_groups:
        for jg in j_groups:
            for kg in k_groups:
                ni = len(ig)
                nj = len(jg)
                nk = len(kg)
                
                if ni == 2 and nj == 2 and nk == 2:
                    # Use Strassen
                    U_sub, V_sub, W_sub = strassen_2x2x2()
                else:
                    # Schoolbook for this sub-block
                    rank_sub = ni * nj * nk
                    U_sub = [[0]*rank_sub for _ in range(ni*nj)]
                    V_sub = [[0]*rank_sub for _ in range(nj*nk)]
                    W_sub = [[0]*rank_sub for _ in range(nk*ni)]
                    col = 0
                    for ii in range(ni):
                        for jj in range(nj):
                            for kk in range(nk):
                                U_sub[ii*nj+jj][col] = 1
                                V_sub[jj*nk+kk][col] = 1
                                W_sub[kk*ni+ii][col] = 1
                                col += 1
                
                # Now we need to embed these into the full matrices
                # U_full has shape (4*5, rank) = (20, rank)
                # V_full has shape (5*5, rank) = (25, rank)
                # W_full has shape (5*4, rank) = (20, rank)
                
                rank_sub = len(U_sub[0])
                
                # For each row in U_sub, map to the corresponding row in U_full
                # U_sub row index = ii*nj + jj where ii is local index in ig, jj in jg
                # U_full row index = (ig[ii])*5 + (jg[jj])
                
                # We'll accumulate into the full matrices
                # First, determine the total rank needed
                pass
    
    # Let me restructure: first compute total rank, then fill
    total_rank = 0
    sub_decomps = []
    
    for ig in i_groups:
        for jg in j_groups:
            for kg in k_groups:
                ni = len(ig)
                nj = len(jg)
                nk = len(kg)
                
                if ni == 2 and nj == 2 and nk == 2:
                    U_sub, V_sub, W_sub = strassen_2x2x2()
                else:
                    rank_sub = ni * nj * nk
                    U_sub = [[0]*rank_sub for _ in range(ni*nj)]
                    V_sub = [[0]*rank_sub for _ in range(nj*nk)]
                    W_sub = [[0]*rank_sub for _ in range(nk*ni)]
                    col = 0
                    for ii in range(ni):
                        for jj in range(nj):
                            for kk in range(nk):
                                U_sub[ii*nj+jj][col] = 1
                                V_sub[jj*nk+kk][col] = 1
                                W_sub[kk*ni+ii][col] = 1
                                col += 1
                
                rank_sub = len(U_sub[0])
                total_rank += rank_sub
                sub_decomps.append((ig, jg, kg, U_sub, V_sub, W_sub))
    
    # Now build the full matrices
    U_full = [[0]*total_rank for _ in range(20)]  # 4*5
    V_full = [[0]*total_rank for _ in range(25)]  # 5*5
    W_full = [[0]*total_rank for _ in range(20)]  # 5*4
    
    offset = 0
    for ig, jg, kg, U_sub, V_sub, W_sub in sub_decomps:
        ni = len(ig)
        nj = len(jg)
        nk = len(kg)
        rank_sub = len(U_sub[0])
        
        for r in range(rank_sub):
            # U: local row (ii*nj + jj) -> global row (ig[ii]*5 + jg[jj])
            for ii in range(ni):
                for jj in range(nj):
                    local_row = ii*nj + jj
                    global_row = ig[ii]*5 + jg[jj]
                    U_full[global_row][offset + r] = U_sub[local_row][r]
            
            # V: local row (jj*nk + kk) -> global row (jg[jj]*5 + kg[kk])
            for jj in range(nj):
                for kk in range(nk):
                    local_row = jj*nk + kk
                    global_row = jg[jj]*5 + kg[kk]
                    V_full[global_row][offset + r] = V_sub[local_row][r]
            
            # W: local row (kk*ni + ii) -> global row (kg[kk]*4 + ig[ii])
            for kk in range(nk):
                for ii in range(ni):
                    local_row = kk*ni + ii
                    global_row = kg[kk]*4 + ig[ii]
                    W_full[global_row][offset + r] = W_sub[local_row][r]
        
        offset += rank_sub
    
    # Verify correctness
    def verify(U, V, W, n, m, p):
        """Check that the decomposition is correct."""
        # The tensor T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For matrix multiplication: T[i,j,k] = 1 if we're computing C[i,k] += A[i,j]*B[j,k]
        # So T[i,j,k] = 1 for all valid i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(len(U[0])):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(val - 1) > 1e-9:
                        return False, (i, j, k, val)
        return True, None
    
    valid, info = verify(U_full, V_full, W_full, n, m, p)
    if not valid:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U_full, "V": V_full, "W": W_full}
# EVOLVE-BLOCK-END
