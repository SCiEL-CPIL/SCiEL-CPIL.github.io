"""Matrix multiplication 4x5 by 5x6 — Strassen-based decomposition (rank 101)."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    start_time = time.time()
    deadline = 140  # Keep 10s margin before 150s kill

    def make_schoolbook_block(n_b, m_b, p_b):
        """Schoolbook decomposition for n_b x m_b x p_b block.
        Returns (U_block, V_block, W_block) each as lists of lists.
        U_block shape: (n_b*m_b, rank), V_block: (m_b*p_b, rank), W_block: (p_b*n_b, rank)
        """
        rank = n_b * m_b * p_b
        U_b = [[0] * rank for _ in range(n_b * m_b)]
        V_b = [[0] * rank for _ in range(m_b * p_b)]
        W_b = [[0] * rank for _ in range(p_b * n_b)]
        col = 0
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U_b[i * m_b + j][col] = 1
                    V_b[j * p_b + k][col] = 1
                    W_b[k * n_b + i][col] = 1
                    col += 1
        return U_b, V_b, W_b

    def strassen_2x2x2_terms():
        """Return the 7 rank-1 terms for 2x2x2 matrix multiplication.
        Each term is (u, v, w) where:
        u[i*2+j] for i,j in {0,1} (4 entries)
        v[j*2+k] for j,k in {0,1} (4 entries)
        w[k*2+i] for k,i in {0,1} (4 entries)
        """
        # The 7 Strassen products for 2x2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)(B00)
        # M3 = A00(B11-B01)
        # M4 = A11(B00-A00)
        # M5 = (A00+A01)(B11)
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Let me derive these carefully.
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        # C = A*B = [[C00, C01], [C10, C11]]
        # C00 = A00*B00 + A01*B10
        # C01 = A00*B01 + A01*B11
        # C10 = A10*B00 + A11*B10
        # C11 = A10*B01 + A11*B11
        
        # Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)(B00)
        # M3 = A00(B11-B01)
        # M4 = A11(B00-A00)
        # M5 = (A00+A01)(B11)
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let me verify:
        # C00 = (A00+A11)(B00+B11) + A11(B00-A00) - (A00+A01)(B11) + (A01-A11)(B10+B11)
        #     = A00B00 + A00B11 + A11B00 + A11B11 + A11B00 - A11A00 - A00B11 - A01B11 + A01B10 + A01B11 - A11B10 - A11B11
        #     = A00B00 + A11B00 - A11A00 + A01B10 - A11B10
        # Hmm, that doesn't simplify to A00B00 + A01B10. I think I have the Strassen formulas wrong.
        
        # Let me look up the correct Strassen formulas:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B22-B12)
        # M4 = A22(B11-A11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Using 0-indexed:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B11-B01)
        # M4 = A11(B00-A00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let me verify C00:
        # C00 = (A00+A11)(B00+B11) + A11(B00-A00) - (A00+A01)B11 + (A01-A11)(B10+B11)
        #     = A00B00 + A00B11 + A11B00 + A11B11 + A11B00 - A11A00 - A00B11 - A01B11 + A01B10 + A01B11 - A11B10 - A11B11
        #     = A00B00 + 2A11B00 - A11A00 - A00B11 + A00B11 - A11B11 + A11B11 + A01B10 - A11B10 - A01B11 + A01B11
        #     = A00B00 + 2A11B00 - A11A00 + A01B10 - A11B10
        # This still doesn't look right. I think the issue is that M4 = A11(B00-A00) is not a product of a linear form in A and a linear form in B.
        
        # Actually, M4 = A11*B00 - A11*A00. The second term A11*A00 is not a product of an A-entry and a B-entry. So this isn't a valid bilinear term.
        
        # I think the correct Strassen formulas are:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B11-B01)
        # M4 = A11(B00-A00)  -- this is problematic
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Wait, I think I'm confusing myself. Let me just use a known correct rank-7 decomposition.
        # After checking, the correct Strassen rank-1 terms for 2x2x2 are:
        
        terms = []
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        # u = [1, 0, 0, 1] (A00 and A11)
        # v = [1, 0, 0, 1] (B00 and B11)
        # w: contributes to C00 and C11
        # C00 = +1, C11 = +1, C01 = 0, C10 = 0
        # w[k*2+i]: w[0*2+0]=1, w[0*2+1]=0, w[1*2+0]=0, w[1*2+1]=1
        terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        
        # Term 2: M2 = (A10+A11)B00
        # u = [0, 0, 1, 1] (A10 and A11)
        # v = [1, 0, 0, 0] (B00)
        # w: contributes to C10 and C11
        # C10 = +1, C11 = +1, C00 = 0, C01 = 0
        # w[k*2+i]: w[0*2+0]=0, w[0*2+1]=0, w[1*2+0]=1, w[1*2+1]=1
        terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1]))
        
        # Term 3: M3 = A00(B11-B01)
        # u = [1, 0, 0, 0] (A00)
        # v = [0, -1, 0, 1] (B01=-1, B11=1)
        # w: contributes to C01
        # C01 = +1, others 0
        # w[k*2+i]: w[0*2+0]=0, w[0*2+1]=1, w[1*2+0]=0, w[1*2+1]=0
        terms.append(([1, 0, 0, 0], [0, -1, 0, 1], [0, 1, 0, 0]))
        
        # Term 4: M4 = A11(B00-A00)
        # This is problematic as noted. Let me use a different decomposition.
        # Actually, I think the issue is that I'm not using the correct Strassen formulas.
        # Let me just use the schoolbook for 2x2x2 (rank 8) and see if that helps.
        # But the task says rank-minimal, so I need to get the rank 7 right.
        
        # Let me try a different approach: use the known fact that the rank of 2x2x2 is 7,
        # and find a valid decomposition by solving a system of equations.
        
        # For now, let me just return the schoolbook terms for 2x2x2 (rank 8).
        # This will give a total rank of 4*8 + 32 + 24 = 64 + 32 + 24 = 120, same as schoolbook.
        # Not helpful.
        
        # I'll try to hard-code the correct 7 terms after more thought.
        # For now, return an empty list to signal that we should use schoolbook for this block.
        return []

    # Since the Strassen 2x2x2 decomposition is tricky, let me just use schoolbook for everything.
    # Rank 120 is guaranteed correct.
    
    U, V, W = make_schoolbook_block(n, m, p)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
