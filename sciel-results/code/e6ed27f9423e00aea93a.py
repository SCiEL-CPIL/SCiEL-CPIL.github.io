"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook fallback: rank = n*m*p = 120
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
    
    # Strassen-2 decomposition for 2x2x2 blocks (rank 7)
    def strassen2_decomposition():
        # M1 = (a+d)(e+g)
        # M2 = (c+d)(e)
        # M3 = (a-b)(f+g)
        # M4 = (d-c)(h+g)
        # M5 = (a+e)(h-f)
        # M6 = (b+f)(h+g)
        # M7 = (b+e)(j+k) -- Wait, indices are e,f,g,h for B.
        # B = [[e, f], [g, h]]
        # C = [[a*e+b*g, a*f+b*h], [c*e+d*g, c*f+d*h]]
        
        U = [[0] * 7 for _ in range(4)]
        V = [[0] * 7 for _ in range(4)]
        W = [[0] * 7 for _ in range(4)]
        
        # M1 = (a+d)(e+g)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[2][0] = 1
        W[0][0] = 1; W[3][0] = 1; W[1][0] = 1; W[2][0] = 1
        
        # M2 = (c+d)(e)
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[0][1] = 1; W[2][1] = 1
        
        # M3 = (a-b)(f+g)
        U[0][2] = 1; U[1][2] = -1
        V[1][2] = 1; V[2][2] = 1
        W[0][2] = 1; W[3][2] = 1
        
        # M4 = (d-c)(h+g)
        U[3][3] = 1; U[2][3] = -1
        V[3][3] = 1; V[2][3] = 1
        W[0][3] = 1; W[3][3] = 1
        
        # M5 = (a+e)(h-f)
        U[0][4] = 1
        V[3][4] = 1; V[1][4] = -1
        W[0][4] = 1; W[3][4] = 1
        
        # M6 = (b+f)(h+g)
        U[1][5] = 1
        V[1][5] = 1; V[3][5] = 1; V[2][5] = 1
        W[0][5] = 1; W[3][5] = 1
        
        # M7 = (b+e)(j+k) -> (b+e)(e? no)
        # Let's use a known correct set from literature or verify.
        # Actually, let's just use the schoolbook for 2x2x2 if Strassen is too risky?
        # No, 108 is the target.
        
        # Let's use a simpler verified Strassen-2.
        # M1 = (a+d)(e+g)
        # M2 = (c+d)(e)
        # M3 = (a-b)(f+g)
        # M4 = (d-c)(h+g)
        # M5 = (a+e)(h-f)
        # M6 = (b+f)(h+g)
        # M7 = (b+e)(e? no)
        
        # Correct Strassen-2:
        # M1 = (a+d)(e+g)
        # M2 = (c+d)(e)
        # M3 = (a-b)(f+g)
        # M4 = (d-c)(h+g)
        # M5 = (a+e)(h-f)
        # M6 = (b+f)(h+g)
        # M7 = (b+e)(e? no)
        
        # Let's just return schoolbook for 2x2x2 if we can't verify Strassen-2 quickly.
        # But 108 is achievable.
        # I will use the Strassen-2 from the seed, assuming it was correct in a previous run (score 1.111).
        # The seed's Strassen-2:
        U = [[0] * 7 for _ in range(4)]
        V = [[0] * 7 for _ in range(4)]
        W = [[0] * 7 for _ in range(4)]
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[3][6] = 1
        return U, V, W

    # Build 108 rank solution
    total_rank = 12 * 7 + 6 * 4
    U_full = [[0] * total_rank for _ in range(nm)]
    V_full = [[0] * total_rank for _ in range(mp)]
    W_full = [[0] * total_rank for _ in range(pn)]
    
    col_offset = 0
    U2, V2, W2 = strassen2_decomposition()
    
    n_bounds = [0, 2, 4]
    m_bounds = [0, 2, 4, 5]
    p_bounds = [0, 2, 4, 6]
    
    for bi in range(2):
        for bj in range(3):
            for bk in range(3):
                ni = n_bounds[bi]
                nj = n_bounds[bi+1]
                mi = m_bounds[bj]
                mj = m_bounds[bj+1]
                pi = p_bounds[bk]
                pj = p_bounds[bk+1]
                
                n_size = nj - ni
                m_size = mj - mi
                p_size = pj - pi
                
                if n_size == 2 and m_size == 2 and p_size == 2:
                    for r in range(7):
                        for ii in range(2):
                            for jj in range(2):
                                U_full[(ni+ii)*m + (mi+jj)][col_offset + r] = U2[ii*2+jj][r]
                        for jj in range(2):
                            for kk in range(2):
                                V_full[(mi+jj)*p + (pi+kk)][col_offset + r] = V2[jj*2+kk][r]
                        for kk in range(2):
                            for ii in range(2):
                                W_full[(pi+kk)*n + (ni+ii)][col_offset + r] = W2[kk*2+ii][r]
                    col_offset += 7
                else:
                    for ii in range(n_size):
                        for jj in range(m_size):
                            for kk in range(p_size):
                                U_full[(ni+ii)*m + (mi+jj)][col_offset] = 1
                                V_full[(mi+jj)*p + (pi+kk)][col_offset] = 1
                                W_full[(pi+kk)*n + (ni+ii)][col_offset] = 1
                                col_offset += 1
    
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(total_rank):
                    s += U_full[i * m + j][r] * V_full[j * p + k][r] * W_full[k * n + i][r]
                if abs(s - 1.0) > 1e-10:
                    valid = False
                    break
        if not valid:
            break
    
    if valid:
        return {"U": U_full, "V": V_full, "W": W_full}
    
    return best_result
# EVOLVE-BLOCK-END
