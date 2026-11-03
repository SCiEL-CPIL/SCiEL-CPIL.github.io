"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Schoolbook decomposition as baseline (rank 90)
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
    
    # Try to reduce rank using Strassen on 2x2x2 sub-blocks
    terms = []  # (u_idx, v_idx, w_idx, u_val, v_val, w_val)
    
    def add_term(u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
        terms.append((u_idx, v_idx, w_idx, u_val, v_val, w_val))
    
    # Strassen 2x2x2 decomposition: rank 7 instead of rank 8
    # For 2x2x2, the 7 products are:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b22-b12)
    # M4 = a22(b12-b21)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # Then:
    # c11 = M1 - M4 + M5 - M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M3 + M2 - M6
    
    def strassen_block(i0, j0, k0):
        """Add 7 Strassen terms for 2x2x2 block starting at (i0,j0,k0)"""
        # M1 = (a11+a22)(b11+b22)
        # u: (i0,j0) and (i0+1,j0+1) with +1
        # v: (j0,k0) and (j0+1,k0+1) with +1
        # w: (k0,i0) and (k0+1,i0+1) with +1
        add_term(i0*m + j0, j0*p + k0, k0*n + i0, 1, 1, 1)
        add_term((i0+1)*m + (j0+1), (j0+1)*p + (k0+1), (k0+1)*n + (i0+1), 1, 1, 1)
        # But we need this as ONE term with combined U,V,W vectors
        # Actually, Strassen requires that each M_r is a single scalar product
        # M1 = (a11+a22)(b11+b22) means:
        # U vector has 1 at (i0,j0) and 1 at (i0+1,j0+1)
        # V vector has 1 at (j0,k0) and 1 at (j0+1,k0+1)
        # W vector has 1 at (k0,i0) and 1 at (k0+1,i0+1)
        # This is ONE rank-1 term with multiple non-zero entries
        
        # Let me redo this properly. Each Strassen product M_r is one rank-1 term
        # where U, V, W can have multiple non-zero entries.
        
        # Clear the previous incorrect additions - actually let me just rebuild
        pass
    
    # Let me implement Strassen properly by tracking terms as full vectors
    # Each term is a rank-1 decomposition: sum_r U[:,r] * V[:,r] * W[:,r]
    # For Strassen, each M_r contributes one column to U, V, W
    
    # Reset terms
    terms.clear()
    
    def add_strassen_term(u_entries, v_entries, w_entries):
        """Add a term where u_entries, v_entries, w_entries are dicts {index: value}"""
        terms.append((u_entries, v_entries, w_entries))
    
    # For 2x2x2 Strassen, the 7 products:
    # M1: u={(i0,j0):1, (i0+1,j0+1):1}, v={(j0,k0):1, (j0+1,k0+1):1}, w={(k0,i0):1, (k0+1,i0+1):1}
    # M2: u={(i0+1,j0):1, (i0+1,j0+1):1}, v={(j0,k0):1}, w={(k0,i0+1):1}
    # M3: u={(i0,j0):1}, v={(j0+1,k0+1):1, (j0+1,k0):-1}, w={(k0+1,i0):1}
    # M4: u={(i0+1,j0+1):1}, v={(j0+1,k0):1, (j0,k0+1):-1}, w={(k0,i0+1):1}
    # M5: u={(i0,j0):1, (i0,j0+1):1}, v={(j0+1,k0+1):1}, w={(k0+1,i0):1}
    # M6: u={(i0+1,j0):1, (i0,j0):-1}, v={(j0,k0):1, (j0,k0+1):1}, w={(k0,i0+1):1}
    # M7: u={(i0,j0+1):1, (i0+1,j0+1):-1}, v={(j0+1,k0):1, (j0+1,k0+1):1}, w={(k0+1,i0):1}
    
    # Wait, I need to be more careful. The Strassen algorithm for 2x2 matrices:
    # C = A*B where A, B, C are 2x2
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    
    # In our tensor, for a 2x2x2 block:
    # A[i,j] for i in {0,1}, j in {0,1} -> u-index (i, j)
    # B[j,k] for j in {0,1}, k in {0,1} -> v-index (j, k)
    # C[i,k] for i in {0,1}, k in {0,1} -> w-index (k, i)
    
    # M1 = (a11+a22)(b11+b22): 
    #   u: (0,0)->1, (1,1)->1
    #   v: (0,0)->1, (1,1)->1
    #   w: (0,0)->1, (1,1)->1
    # M2 = (a21+a22)b11:
    #   u: (1,0)->1, (1,1)->1
    #   v: (0,0)->1
    #   w: (0,1)->1
    # M3 = a11(b22-b12):
    #   u: (0,0)->1
    #   v: (1,1)->1, (0,1)->-1
    #   w: (1,0)->1
    # M4 = a22(b12-b21):
    #   u: (1,1)->1
    #   v: (0,1)->1, (1,0)->-1
    #   w: (0,1)->1
    # M5 = (a11+a12)b22:
    #   u: (0,0)->1, (0,1)->1
    #   v: (1,1)->1
    #   w: (1,0)->1
    # M6 = (a21-a11)(b11+b12):
    #   u: (1,0)->1, (0,0)->-1
    #   v: (0,0)->1, (0,1)->1
    #   w: (0,1)->1
    # M7 = (a12-a22)(b21+b22):
    #   u: (0,1)->1, (1,1)->-1
    #   v: (1,0)->1, (1,1)->1
    #   w: (1,0)->1
    
    # Verify: 
    # c11 = M1 - M4 + M5 - M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M3 + M2 - M6
    
    # Let me verify c11:
    # M1 contributes to c11: w(0,0)=1, so +1
    # M4 contributes to c11: w(0,1)=1, so -1 (coefficient -1)
    # M5 contributes to c11: w(1,0)=1, so +1 (coefficient +1)
    # M7 contributes to c11: w(1,0)=1, so -1 (coefficient -1)
    # Total for c11: M1*w(0,0) - M4*w(0,1) + M5*w(1,0) - M7*w(1,0)
    # = M1*1 - M4*1 + M5*1 - M7*1
    # Now check if this gives a11*b11 + a12*b21:
    # M1 = (a11+a22)(b11+b22) = a11*b11 + a11*b22 + a22*b11 + a22*b22
    # M4 = a22(b12-b21) = a22*b12 - a22*b21
    # M5 = (a11+a12)b22 = a11*b22 + a12*b22
    # M7 = (a12-a22)(b21+b22) = a12*b21 + a12*b22 - a22*b21 - a22*b22
    # c11 = M1 - M4 + M5 - M7
    # = (a11*b11 + a11*b22 + a22*b11 + a22*b22) - (a22*b12 - a22*b21) + (a11*b22 + a12*b22) - (a12*b21 + a12*b22 - a22*b21 - a22*b22)
    # = a11*b11 + a11*b22 + a22*b11 + a22*b22 - a22*b12 + a22*b21 + a11*b22 + a12*b22 - a12*b21 - a12*b22 + a22*b21 + a22*b22
    # = a11*b11 + 2*a11*b22 + a22*b11 + 2*a22*b22 - a22*b12 + 2*a22*b21 - a12*b21
    # This doesn't look right. Let me re-examine.
    
    # I think I made errors. Let me look up the correct Strassen formulas.
    # Standard Strassen:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b21-b11)  [NOTE: b21-b11, not b22-b12]
    # M4 = a22(b12-b22)  [NOTE: b12-b22, not b12-b21]
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # c11 = M1 - M4 + M5 - M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M3 + M2 - M6
    
    # Let me re-verify with correct formulas:
    # M3 = a11(b21-b11)
    # M4 = a22(b12-b22)
    
    # c11 = M1 - M4 + M5 - M7
    # M1 = a11*b11 + a11*b22 + a22*b11 + a22*b22
    # M4 = a22*b12 - a22*b22
    # M5 = a11*b22 + a12*b22
    # M7 = a12*b21 + a12*b22 - a22*b21 - a22*b22
    # c11 = (a11*b11 + a11*b22 + a22*b11 + a22*b22) - (a22*b12 - a22*b22) + (a11*b22 + a12*b22) - (a12*b21 + a12*b22 - a22*b21 - a22*b22)
    # = a11*b11 + a11*b22 + a22*b11 + a22*b22 - a22*b12 + a22*b22 + a11*b22 + a12*b22 - a12*b21 - a12*b22 + a22*b21 + a22*b22
    # = a11*b11 + 2*a11*b22 + a22*b11 + 3*a22*b22 - a22*b12 - a12*b21 + a22*b21
    # Still wrong. I'm confusing myself. Let me just use the known correct decomposition.
    
    # Actually, the standard Strassen is:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b21-b11)
    # M4 = a22(b12-b22)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # c11 = M1 - M4 + M5 - M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M3 + M2 - M6
    
    # Let me verify c11 step by step more carefully:
    # c11 should equal a11*b11 + a12*b21
    # M1 = (a11+a22)(b11+b22) = a11*b11 + a11*b22 + a22*b11 + a22*b22
    # -M4 = -a22*(b12-b22) = -a22*b12 + a22*b22
    # +M5 = (a11+a12)*b22 = a11*b22 + a12*b22
    # -M7 = -(a12-a22)*(b21+b22) = -a12*b21 - a12*b22 + a22*b21 + a22*b22
    # Sum: a11*b11 + a11*b22 + a22*b11 + a22*b22 - a22*b12 + a22*b22 + a11*b22 + a12*b22 - a12*b21 - a12*b22 + a22*b21 + a22*b22
    # = a11*b11 + 2*a11*b22 + a22*b11 + 3*a22*b22 - a22*b12 - a12*b21 + a22*b21
    # This is NOT equal to a11*b11 + a12*b21. So my formula is wrong.
    
    # I think the issue is that I'm using the wrong Strassen variant. Let me use a different reference.
    # Actually, I recall now - the standard Strassen uses:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b21-b11)
    # M4 = a22(b12-b22)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # And the reconstruction is:
    # c11 = M1 - M4 + M5 - M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M3 + M2 - M6
    
    # Let me try a completely different approach. I'll just use the schoolbook decomposition
    # since getting Strassen right in this tensor framework is error-prone.
    
    # Given the complexity and risk of errors, let's just return the schoolbook decomposition
    # which is guaranteed correct with rank 90.
    
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
# EVOLVE-BLOCK-END
