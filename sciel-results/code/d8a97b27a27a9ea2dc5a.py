import numpy as np
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Helper to add a schoolbook term
    def add_schoolbook(i, j, k):
        nonlocal col
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1

    # Helper to add Strassen terms for a 2x2x2 block
    # Indices: i0, i1 for A rows; j0, j1 for A cols; k0, k1 for B cols
    # A block: A[i0:i1+1, j0:j1+1]
    # B block: B[j0:j1+1, k0:k1+1]
    # C block: C[i0:i1+1, k0:k1+1]
    def add_strassen(i0, j0, k0):
        nonlocal col
        # Strassen's 7 products for 2x2x2
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # We need to map these to the global U, V, W indices.
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        # M1
        U[i0*m + j0][col] = 1
        U[(i0+1)*m + (j0+1)][col] = 1
        V[j0*p + k0][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[k0*n + i0][col] = 1
        W[(k0+1)*n + (i0+1)][col] = 1
        # Note: Strassen terms are linear combinations. 
        # The decomposition is sum_r U_r V_r W_r.
        # Each term r corresponds to one multiplication.
        # But Strassen uses intermediate sums.
        # The standard tensor decomposition for Strassen is:
        # The 7 terms are the rank-1 tensors.
        # U, V, W entries are 1, -1, 0.
        
        # Let's define the 7 terms explicitly for the local 2x2x2 block.
        # Local indices 0,1.
        # Term 1: (A00+A11)(B00+B11) -> C00 += M1, C01 += M1, C10 += M1, C11 += M1? No.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # This means we need 7 rank-1 terms in the global sum.
        # Each term r has U_r, V_r, W_r vectors.
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        # U: A00, A11 are 1. V: B00, B11 are 1. W: C00, C01, C10, C11?
        # No, the W part distributes M1 to the correct C positions with signs.
        # W for Term 1: C00=1, C01=1, C10=1, C11=1? 
        # Wait, the product is M1. The contribution to C is determined by W.
        # So W_r must encode the coefficients for C.
        
        # Let's list the 7 Strassen terms (u, v, w) for local indices.
        # u is vector in A space (4), v in B space (4), w in C space (4).
        
        # T1: u=[1,0,0,1], v=[1,0,0,1], w=[1,1,1,1] ? No.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So:
        # Term 1 (M1): u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1] -> contributes to C00, C11?
        # Actually, w is the vector such that sum u_i v_j w_k = ...
        # The term is sum_{a,b} u_a v_b w_{a,b}? No.
        # The tensor is sum_r U_r V_r W_r.
        # U_r is vector over A indices. V_r over B indices. W_r over C indices.
        # Product is sum_{i,j,k} U_{ij,r} V_{jk,r} W_{ki,r} A_{ij} B_{jk}.
        # This equals sum_{k,i} (sum_j V_{jk,r} B_{jk}) (sum_i U_{ij,r} A_{ij}) W_{ki,r}.
        # This is not quite right. The indices must match.
        
        # Correct Strassen decomposition for 2x2x2:
        # r=0: U=[1,0,0,1], V=[1,0,0,1], W=[1,1,1,1] (Mapped to C00,C01,C10,C11)
        # Actually, W is indexed by (k,i).
        # C00 is (0,0). C01 is (0,1). C10 is (1,0). C11 is (1,1).
        # W_r for M1: C00=1, C01=1, C10=1, C11=1?
        # No, M1 contributes to all 4 C's with coefficient 1?
        # C00 = M1 + ...
        # C01 = ... + M1? No.
        
        # Let's use the standard coefficients:
        # M1: C00+=1, C01+=1, C10+=1, C11+=1? No.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So:
        # T1 (M1): U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1] (W indices: C00=1, C01=0, C10=0, C11=1? No)
        # W is a vector of length 4 (C indices).
        # W[0] (C00) = 1
        # W[1] (C01) = 1 ? No, M1 is not in C01.
        # Wait, C01 = M3 + M5. M1 is not there.
        # So W for T1: [1, 0, 0, 1]? No, C11 has M1.
        # W[0]=1, W[1]=0, W[2]=0, W[3]=1.
        
        # T2 (M2): U=[0,1,1,1]? No. M2 = (A10+A11)B00.
        # U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,0] (C10=1, C11=-1? No, C11 has -M2).
        # W[0]=0, W[1]=0, W[2]=1, W[3]=-1.
        
        # T3 (M3): U=[1,0,0,0], V=[0,0,1,-1], W=[0,1,0,1] (C01=1, C11=1? No, C11 has M3).
        # W[0]=0, W[1]=1, W[2]=0, W[3]=1.
        
        # T4 (M4): U=[0,0,0,1], V=[0,1,-1,0], W=[1,0,1,0] (C00=1, C10=1).
        # W[0]=1, W[1]=0, W[2]=1, W[3]=0.
        
        # T5 (M5): U=[1,1,0,0], V=[0,0,0,1], W=[-1,1,0,0] (C00=-1, C01=1).
        # W[0]=-1, W[1]=1, W[2]=0, W[3]=0.
        
        # T6 (M6): U=[0,-1,1,0], V=[1,0,1,0], W=[0,0,0,1] (C11=1).
        # W[0]=0, W[1]=0, W[2]=0, W[3]=1.
        
        # T7 (M7): U=[1,0,-1,1], V=[0,1,0,1], W=[1,0,0,0] (C00=1).
        # W[0]=1, W[1]=0, W[2]=0, W[3]=0.

        # Map local indices to global.
        # Local A: (0,0)->i0,j0; (0,1)->i0,j0+1; (1,0)->i0+1,j0; (1,1)->i0+1,j0+1.
        # Local B: (0,0)->j0,k0; (0,1)->j0,k0+1; (1,0)->j0+1,k0; (1,1)->j0+1,k0+1.
        # Local C: (0,0)->k0,i0; (0,1)->k0,i0+1; (1,0)->k0+1,i0; (1,1)->k0+1,i0+1.
        
        # U vectors (local 4 to global 9)
        # V vectors (local 4 to global 6)
        # W vectors (local 4 to global 6)
        
        # T1
        U[i0*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # T2
        U[(i0+1)*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        W[(k0+1)*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += -1
        col += 1
        
        # T3
        U[i0*m + j0][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += -1
        W[k0*n + (i0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # T4
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + (k0+1)][col] += 1
        V[(j0+1)*p + k0][col] += -1
        W[k0*n + i0][col] += 1
        W[(k0+1)*n + i0][col] += 1
        col += 1
        
        # T5
        U[i0*m + j0][col] += 1
        U[i0*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += -1
        W[k0*n + (i0+1)][col] += 1
        col += 1
        
        # T6
        U[(i0+1)*m + j0][col] += 1
        U[i0*m + j0][col] += -1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + k0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # T7
        U[i0*m + j0][col] += 1
        U[i0*m + (j0+1)][col] += -1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + (k0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        col += 1

    # Apply Strassen to the top-left 2x2x2 block of the 3x3x3 part
    # i0=0, j0=0, k0=0
    add_strassen(0, 0, 0)
    
    # Add schoolbook for all other terms in 3x3x3
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if not (i < 2 and j < 2 and k < 2):
                    add_schoolbook(i, j, k)
    
    # Add schoolbook for 3x3x2 part (k=3,4)
    for i in range(3):
        for j in range(3):
            for k in range(3, 5):
                add_schoolbook(i, j, k)
                
    # Trim unused columns if any (shouldn't be any)
    final_rank = col
    U = [row[:final_rank] for row in U]
    V = [row[:final_rank] for row in V]
    W = [row[:final_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
