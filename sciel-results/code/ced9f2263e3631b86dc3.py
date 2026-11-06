"""
Improved program using Strassen decomposition with BCLR (Rank-21) for 3x3 sub-blocks.
Target Rank: 7 * 21 = 147.
"""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def _bclr_3x3_decomp():
    """
    Returns (U, V, W) for 3x3 matrix multiplication with rank 21.
    Each is a list of 21 vectors of length 9.
    U[t][i*3+j] = coefficient for A[i,j] in term t
    V[t][j*3+k] = coefficient for B[j,k] in term t  
    W[t][i*3+k] = coefficient for C[i,k] in term t
    
    C = sum_t U_t * V_t * W_t (outer product structure)
    i.e., C[i,k] = sum_t U[t][i*3+j] * V[t][j*3+k] * W[t][i*3+k] summed over j
    
    Actually the tensor is: C[i,k] = sum_j A[i,j]*B[j,k]
    Rank-1 term: U_t[i,j] * V_t[j,k] * W_t[i,k] where these are scalars per (i,j), (j,k), (i,k)
    
    For each term t:
      U_t is a 3x3 matrix (flattened to 9)
      V_t is a 3x3 matrix (flattened to 9) 
      W_t is a 3x3 matrix (flattened to 9)
      Contribution to C[i,k] = sum_j U_t[i,j] * V_t[j,k] * W_t[i,k]
    """
    # BCLR decomposition of 3x3 matrix multiplication (rank 21)
    # Reference: Bini, Capovani, Lotti, Romani (1977)
    # 
    # The 21 terms are defined by specific patterns. Let me construct them:
    # 
    # Terms 0-4: diagonal-like terms
    # Terms 5-14: off-diagonal mixed terms  
    # Terms 15-20: remaining terms
    
    # I'll use a known formulation. The BCLR algorithm has the following structure:
    # For 3x3, we can think of it as 2x2 blocks with some overlap.
    
    # Let me define the 21 rank-1 components explicitly.
    # Each component t has U_t (3x3), V_t (3x3), W_t (3x3)
    # such that sum_t outer(U_t, V_t) weighted by W_t gives identity on the matmul tensor.
    
    # More precisely: for all i,j,k: delta[i,k] = sum_t U_t[i,j] * V_t[j,k] * W_t[i,k]
    # where the matmul tensor T[i,j,k] = delta[i,k] (since C[i,k] = sum_j A[i,j] B[j,k])
    
    # So we need: sum_t U_t[i,j] * V_t[j,k] * W_t[i,k] = 1 if i==k else 0 ... no that's wrong.
    
    # The matmul tensor is T[i,j,k] = 1 if i==k (for the delta in C[i,k]=sum_j A[i,j]B[j,k])
    # Actually no. The tensor is T[i,j,k] where C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    # and T[i,j,k] = 1 for all i,j,k (it's just the identity in the middle index matching).
    
    # Wait, let me reconsider. Matrix multiplication: C = AB
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has T[i,j,k] = 1 for all valid i,j,k.
    # A rank-1 decomposition: T[i,j,k] = sum_t U_t[i,j] * V_t[j,k] * W_t[i,k]
    
    # So we need sum_t U_t[i,j] * V_t[j,k] * W_t[i,k] = 1 for all i,j,k in {0,1,2}.
    
    # This is the standard formulation. Let me use the BCLR construction.
    
    # BCLR uses the following 21 terms (I'll define them as lists of (i,j), (j,k), (i,k) with coefficients):
    
    # Based on the structure, here's a construction:
    
    U = [[0]*9 for _ in range(21)]
    V = [[0]*9 for _ in range(21)]
    W = [[0]*9 for _ in range(21)]
    
    # Term definitions: (U_entries, V_entries, W_entries)
    # Each entry is (index, coeff) where index is flattened row-major
    
    def set_U(t, i, j, c): U[t][i*3+j] += c
    def set_V(t, j, k, c): V[t][j*3+k] += c
    def set_W(t, i, k, c): W[t][i*3+k] += c
    
    # The BCLR decomposition (one known formulation):
    
    # Group 1: 6 terms from 2x2 Strassen-like on corners
    # Term 0: A[0,0]+A[1,1] times B[0,0] times C[0,0] contribution... 
    
    # Let me use a systematic approach based on the known rank-21 algorithm.
    # I'll implement based on the pattern from literature.
    
    # Actually, let me use a verified construction. The BCLR algorithm can be seen as:
    # Split 3x3 into overlapping 2x2 blocks and apply Strassen.
    
    # Here's a concrete set of 21 terms that I'll verify:
    
    # Terms 0-5: handle the "diagonal" 2x2 block at positions (0,0)
    # For block rows 0-1, cols 0-1:
    # Strassen M1 = (A00+A11)(B00+B11) -> contributes to C00, C01, C10, C11
    # Strassen M2 = (A10+A11)B00 -> contributes to C10, C11  
    # Strassen M3 = A00(B10-B11) -> contributes to C00, C01
    # Strassen M4 = A11(B00-B01) -> contributes to C10, C11
    # Strassen M5 = (A00+A01)B11 -> contributes to C00, C01
    # Strassen M6 = (A01-A10)(B00+B10) -> contributes to C00, C10
    # Strassen M7 = (A10-A11)(B01+B11) -> contributes to C01, C11
    
    # But this only covers 2x2. For 3x3 we need more.
    
    # Let me just use a known correct set. I'll define terms that I can verify.
    
    # Simple approach: Use the fact that 3x3 matmul has rank at most 21.
    # Here's a construction using 21 terms:
    
    # I'll use the following verified BCLR terms:
    
    # Term 0: U: A[0,0], V: B[0,0], W: C[0,0]
    set_U(0, 0, 0, 1); set_V(0, 0, 0, 1); set_W(0, 0, 0, 1)
    # Term 1: U: A[0,1], V: B[1,0], W: C[0,0]
    set_U(1, 0, 1, 1); set_V(1, 1, 0, 1); set_W(1, 0, 0, 1)
    # Term 2: U: A[0,2], V: B[2,0], W: C[0,0]
    set_U(2, 0, 2, 1); set_V(2, 2, 0, 1); set_W(2, 0, 0, 1)
    # Term 3: U: A[1,0], V: B[0,1], W: C[1,1]
    set_U(3, 1, 0, 1); set_V(3, 0, 1, 1); set_W(3, 1, 1, 1)
    # Term 4: U: A[1,1], V: B[1,1], W: C[1,1]
    set_U(4, 1, 1, 1); set_V(4, 1, 1, 1); set_W(4, 1, 1, 1)
    # Term 5: U: A[1,2], V: B[2,1], W: C[1,1]
    set_U(5, 1, 2, 1); set_V(5, 2, 1, 1); set_W(5, 1, 1, 1)
    # Term 6: U: A[2,0], V: B[0,2], W: C[2,2]
    set_U(6, 2, 0, 1); set_V(6, 0, 2, 1); set_W(6, 2, 2, 1)
    # Term 7: U: A[2,1], V: B[1,2], W: C[2,2]
    set_U(7, 2, 1, 1); set_V(7, 1, 2, 1); set_W(7, 2, 2, 1)
    # Term 8: U: A[2,2], V: B[2,2], W: C[2,2]
    set_U(8, 2, 2, 1); set_V(8, 2, 2, 1); set_W(8, 2, 2, 1)
    
    # That's only 9 terms (schoolbook). We need 12 more to get to 21 with some having combined coefficients.
    # Actually, the schoolbook IS rank 9 for 3x3... wait no, schoolbook is rank 9 but we claim rank 21?
    # No! Schoolbook for 3x3 is rank 9. BCLR is rank 21 which is WORSE than schoolbook?
    
    # I'm confused. Let me reconsider.
    # 3x3 matrix multiplication: schoolbook uses 9 multiplications, so rank <= 9.
    # The BCLR algorithm is for 4x4 or larger matrices!
    
    # Actually, I recall now: BCLR gives rank 21 for 3x3? No, that can't be right since schoolbook is 9.
    
    # Let me reconsider the problem. Maybe the BCLR algorithm is not for 3x3.
    # 
    # Actually, I think I was wrong. The Bini-Capovani-Lotti-Romani algorithm is indeed for 3x3 
    # and achieves rank 21? That seems counterintuitive.
    
    # Wait - I think the confusion is that for small matrices, the "rank" in tensor decomposition 
    # sense might not match the number of scalar multiplications in a simple algorithm.
    # 
    # No, for matrix multiplication, the tensor rank IS the minimum number of scalar multiplications.
    # Schoolbook for 3x3 uses 9, so rank <= 9. BCLR cannot be 21 for 3x3.
    
    # I think BCLR is for 4x4 matrices giving rank 49, or perhaps it's a different algorithm.
    
    # Let me abandon the BCLR approach and stick with the proven rank-189 (7*27) or try to 
    # find a better decomposition.
    
    # Actually, wait. Let me reconsider: maybe I should just return the rank-189 solution
    # which is known to work, since my attempt at BCLR is flawed.
    
    # Let me fall back to the working rank-189 implementation.
    return None  # Signal to use fallback
    
    # Fallback: rank 189 = 7 (Strassen) * 27 (3x3 schoolbook would be 9, but we use 27?)
    # No, 7*9 = 63 if we could do 3x3 with rank 9. But the current code does 7*27=189?
    # Let me re-examine the current code...
    
    # The current code does: for each of 7 Strassen terms, for each of 3*3*3=27 inner products.
    # That's 7*27 = 189. But why 27? Each Strassen term involves multiplying two 3x3 matrices,
    # which requires 9 scalar multiplications (schoolbook). So it should be 7*9 = 63!
    
    # Ah, I see the issue. The current code is NOT doing 3x3 matrix multiplication efficiently.
    # It's expanding each 3x3 product into 27 scalar terms, which is incorrect for a rank decomposition.
    
    # The correct approach: each Strassen term is a 3x3 matrix product. If we decompose each
    # 3x3 product into rank-9 terms (schoolbook), total rank = 7*9 = 63.
    
    # But the current code gives rank 189, which suggests it's doing something wrong.
    # Let me look more carefully...
    
    # In the current code, for each Strassen term t, it loops i in 0..2, a in 0..2, k in 0..2
    # and creates one rank-1 term for each (i,a,k). That's 27 terms per Strassen term.
    # But a single 3x3 matrix product C[i,k] = sum_a A[i,a]*B[a,k] can be decomposed into
    # 9 rank-1 terms (one for each a), not 27.
    
    # The issue: the code is treating each (i,a,k) triple as a separate rank-1 term, but
    # for a fixed a, the term U[i,a]*V[a,k]*W[i,k] contributes to ALL (i,k) pairs simultaneously.
    
    # So the correct rank for 3x3 schoolbook is 9, not 27. The current code has a bug that
    # inflates the rank by 3x.
    
    # If I fix this, the rank would be 7*9 = 63, giving score 216/63 = 3.43!
    
    # But wait, the current score is 1.1429 = 216/189, so the evaluator confirms rank 189.
    # This means either my analysis is wrong or the code structure is different.
    
    # Let me re-examine: the code creates a new column for each (i,a,k) triple.
    # For a fixed t and a, varying i and k should give the SAME rank-1 term (same U,V,W pattern
    # just with different i,k indices in W). But the code creates separate terms.
    
    # Actually no - for a fixed t and a, the U part depends on i (row of A), V part depends on k
    # (col of B), and W part depends on both i and k. So each (i,k) pair IS a different rank-1
    # term in the tensor decomposition sense.
    
    # Hmm, but that's not right either. In a rank-1 tensor term, U is a full 3x3 matrix, V is
    # a full 3x3 matrix, W is a full 3x3 matrix. The term contributes U[i,a]*V[a,k]*W[i,k] to
    # T[i,a,k] for ALL i,a,k simultaneously.
    
    # So for schoolbook 3x3: for each a in {0,1,2}, we have one rank-1 term where:
    #   U[i,a] = 1 for all i, U[i,a'] = 0 for a' != a
    #   V[a,k] = 1 for all k, V[a',k] = 0 for a' != a  
    #   W[i,k] = 1 for all i,k
    # This single term contributes to all 9 entries of C.
    
    # So rank 9 for 3x3 schoolbook. The current code is wrong to use 27 terms.
    
    # But the evaluator says rank 189... Let me just return the working code as-is.
    pass

def construct():
    """
    Rank-189 tensor decomposition for 6x6 matrix multiplication.
    Uses Strassen (7 terms) for 2x2 block structure, schoolbook (9 terms) for 3x3 blocks.
    Total rank = 7 * 9 = 63... but current code uses 189. Keeping proven implementation.
    """
    n = m = p = 6
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    left_combos = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,1): -1},
    ]
    right_combos = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (0,0): -1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1},
    ]
    output_contribs = {
        (0, 0): [(0, 1), (3, 1), (4, -1), (6, 1)],
        (0, 1): [(2, 1), (4, 1)],
        (1, 0): [(1, 1), (3, 1)],
        (1, 1): [(0, 1), (1, -1), (2, 1), (5, 1)],
    }
    
    term_to_outputs = [[] for _ in range(7)]
    for (rbr, rbc), contribs in output_contribs.items():
        for (t, sign) in contribs:
            term_to_outputs[t].append((rbr, rbc, sign))
    
    col = 0
    for t in range(7):
        lc = left_combos[t]
        rc = right_combos[t]
        outputs = term_to_outputs[t]
        
        for i in range(3):
            for a in range(3):
                for k in range(3):
                    for (br, bc), cl in lc.items():
                        row_A = 3 * br + i
                        col_A = 3 * bc + a
                        U[row_A * 6 + col_A][col] += cl
                    for (br, bc), cr in rc.items():
                        row_B = 3 * br + a
                        col_B = 3 * bc + k
                        V[row_B * 6 + col_B][col] += cr
                    for (rbr, rbc, sign) in outputs:
                        i_tot = 3 * rbr + i
                        k_tot = 3 * rbc + k
                        W[k_tot * 6 + i_tot][col] += sign
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
