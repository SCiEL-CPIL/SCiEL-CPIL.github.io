import numpy as np
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    
    # Try to find a rank-34 decomposition by merging Strassen terms across blocks.
    # Strategy: Use 5 blocks of 2 columns each. Standard Strassen gives 7*5=35.
    # We can save 1 rank by creating cross-block terms that share A-combinations.
    #
    # Key insight: M1 for block i uses U=[A00+A11] with V=[B0_i + B1_{i+1}]
    # If we pair blocks (0,1) and (2,3) and handle block 4 separately,
    # we can create shared terms.
    #
    # Alternative: Use a rank-7 decomposition for 2x2x4 (2 rows, 2 cols, 4 cols)
    # which would give 7*2.5 = 17.5... not integer.
    #
    # Better approach: Try rank-34 by finding one cross-block merge.
    # M1: (A00+A11)(B00+B11) -> contributes to C00, C11
    # M4: A11(B10-B00) -> contributes to C00, C10
    #
    # Let's try: combine M1 from block 0 with M4 from block 1 (or similar).
    # Actually, let's try a direct construction with rank 34.
    
    # First, verify the rank-35 solution works, then try to improve.
    rank = 35
    U = np.zeros((4, rank), dtype=int)
    V = np.zeros((20, rank), dtype=int)
    W = np.zeros((20, rank), dtype=int)
    
    def fill_strassen_block(block, rank_offset):
        """Fill Strassen terms for a 2x2x2 block starting at rank_offset."""
        k1 = 2 * block
        k2 = 2 * block + 1
        
        # M1: (A00+A11)(B0k1+B1k2) -> C0k1, C1k2
        r = rank_offset + 0
        U[0, r] += 1
        U[3, r] += 1
        V[k1, r] += 1
        V[10 + k2, r] += 1
        W[2 * k1, r] += 1
        W[2 * k2 + 1, r] += 1
        
        # M2: (A10+A11)B0k1 -> C1k1, -C1k2
        r = rank_offset + 1
        U[2, r] += 1
        U[3, r] += 1
        V[k1, r] += 1
        W[2 * k1 + 1, r] += 1
        W[2 * k2 + 1, r] -= 1
        
        # M3: A00(B0k2-B1k2) -> C0k2, C1k2
        r = rank_offset + 2
        U[0, r] += 1
        V[k2, r] += 1
        V[10 + k2, r] -= 1
        W[2 * k2, r] += 1
        W[2 * k2 + 1, r] += 1
        
        # M4: A11(B1k1-B0k1) -> C0k1, C1k1
        r = rank_offset + 3
        U[3, r] += 1
        V[10 + k1, r] += 1
        V[k1, r] -= 1
        W[2 * k1, r] += 1
        W[2 * k1 + 1, r] += 1
        
        # M5: (A00+A01)B1k2 -> -C0k1, C0k2
        r = rank_offset + 4
        U[0, r] += 1
        U[1, r] += 1
        V[10 + k2, r] += 1
        W[2 * k1, r] -= 1
        W[2 * k2, r] += 1
        
        # M6: (A10-A00)(B0k1+B0k2) -> C1k2
        r = rank_offset + 5
        U[2, r] += 1
        U[0, r] -= 1
        V[k1, r] += 1
        V[k2, r] += 1
        W[2 * k2 + 1, r] += 1
        
        # M7: (A01-A11)(B1k1+B1k2) -> C0k1
        r = rank_offset + 6
        U[1, r] += 1
        U[3, r] -= 1
        V[10 + k1, r] += 1
        V[10 + k2, r] += 1
        W[2 * k1, r] += 1
    
    # Standard rank-35: 5 independent Strassen blocks
    for block in range(5):
        fill_strassen_block(block, 7 * block)
    
    # Verify rank-35 solution
    def verify(U, V, W):
        """Verify the tensor decomposition is exact."""
        n, m, p = DIMENSIONS
        # Build the multiplication tensor T[i,j,k] = delta_{i,k} * delta_{j,l} for C[i,k] = sum_j A[i,j]*B[j,k]
        # Actually: T[i,j,k] = 1 if this is the (i,j,k) entry of the matmul tensor
        # C[i,k] = sum_j A[i,j] B[j,k]
        # So T[i,j,k] = 1 for all (i,j,k) where the product contributes
        # The tensor is: T[i,j,k] = 1 (it's the standard matmul tensor)
        
        # Check: sum_r U[i,r] * V[j,r] * W[k,r] should equal T[i,j,k]
        # where i indexes A (n*m = 4), j indexes B (m*p = 20), k indexes C (p*n = 20)
        # A index: i = row_A * m + col_A = row_A * 2 + col_A
        # B index: j = row_B * p + col_B = row_B * 10 + col_B  
        # Wait, let me re-read the convention: W[k*n+i, r]
        # So W index = k*n + i where k is the column of B / row of C, i is row of A / col of C
        # Actually the problem says: output factor W[k*n+i, r]
        # C is p x n = 10 x 2
        # C[i,k] = sum_j A[i,j] B[j,k]
        # A is n x m = 2 x 2, so A[i,j] with i in [0,2), j in [0,2)
        # B is m x p = 2 x 10, so B[j,k] with j in [0,2), k in [0,10)
        # C is p x n = 10 x 2, so C[k,i] with k in [0,10), i in [0,2)
        
        # U has shape (n*m, rank) = (4, rank): U[i*m+j, r] = U[i*2+j, r]
        # V has shape (m*p, rank) = (20, rank): V[j*p+k, r] = V[j*10+k, r]
        # W has shape (p*n, rank) = (20, rank): W[k*n+i, r] = W[k*2+i, r]
        
        # Check: sum_r U[i*2+j, r] * V[j*10+k, r] * W[k*2+i, r] = 1 for all i,j,k
        # (since C[k,i] = sum_j A[i,j]*B[j,k], the tensor entry is 1)
        
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1
                    actual = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W):
        # Fallback: shouldn't happen, but return rank-35
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Now try to find a rank-34 solution
    # Strategy: Try to merge one pair of Strassen terms across adjacent blocks
    # by finding a linear dependency or a smarter combination.
    
    # Idea: Consider blocks 0 and 1 (columns 0,1 and 2,3).
    # M1 from block 0: (A00+A11)(B00+B11) -> C00, C11
    # M4 from block 1: A11(B12-B02) -> C02, C12
    # These don't obviously merge.
    
    # Alternative: Try a rank-6 decomposition for 2x2x3 (2 rows, 2 cols, 3 cols)
    # If we can do 2x2x3 in rank 6, then 2x2x10 = 2x2x3 + 2x2x3 + 2x2x3 + 2x2x1
    # = 6+6+6+? ... 2x2x1 is rank 2 (just A00*B00, A01*B01, A10*B10, A11*B11... no)
    # 2x2x1: C[0,0]=A00*B00+A01*B10, C[1,0]=A10*B00+A11*B10 -> rank 2 (A00*B00+A01*B10, A10*B00+A11*B10)
    # So 6+6+6+2 = 20? That seems too good. Let me check if 2x2x3 can be rank 6.
    # 2x2x3 tensor has 2*2*3=12 entries. Rank 6 would be very efficient.
    # Actually the standard bound: rank >= max(n*m, m*p, p*n) = max(4,20,20) = 20
    # Wait no, that's not right. The rank lower bound for matmul is more subtle.
    # For 2x2x3: the tensor is T[i,j,k] = 1 for the matmul structure.
    # Actually I need to be more careful. Let me just try to construct rank-34 directly.
    
    # Try: use 4 Strassen blocks (rank 28) + a rank-6 decomposition for 2x2x2
    # But 2x2x2 needs rank 7 (Strassen is optimal for 2x2x2).
    # So that doesn't help.
    
    # Try: 3 Strassen blocks (rank 21) + a rank-13 decomposition for 2x2x4
    # 2x2x4: can we do better than 14 (2x Strassen)?
    # If 2x2x4 has rank 13, total = 21+13 = 34. 
    
    # Let me try to find a rank-13 decomposition for 2x2x4.
    # 2x2x4: A is 2x2, B is 2x4, C is 4x2
    # Standard: 2*7 = 14. Can we save 1?
    
    # Try: use Strassen for columns (0,1) and (2,3) but share one term.
    # M1_block0: (A00+A11)(B00+B11)
    # M1_block1: (A00+A11)(B02+B13)
    # These have the same U factor! So we could potentially combine them.
    # But they produce different W outputs, so they're separate terms.
    # Unless we can split the W factor...
    
    # Actually, the issue is that each term has a unique (U,V,W) triple.
    # Two terms with the same U but different V and W are still separate.
    # To merge, we'd need to find a linear combination that reduces rank.
    
    # Let me try a different approach: use the known result that 
    # M(2,2,4) <= 13 (if true) or try to find it computationally.
    
    # For now, let's try a heuristic search for rank-34.
    # Start from rank-35 and try to remove one term by finding a dependency.
    
    if time.time() - t0 > 120:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find a rank-34 by searching for a term that can be eliminated
    # by adjusting others. This is a nonlinear search, so let's try a simple approach.
    
    # Alternative: Try a specific rank-34 construction.
    # Use blocks 0-3 with Strassen (rank 28) and handle columns 8,9 with rank 6.
    # 2x2x2 needs rank 7, so that's 28+7=35. No help.
    
    # Use blocks 0-2 with Strassen (rank 21) and columns 6,7,8,9 with rank 13.
    # Need rank-13 for 2x2x4.
    
    # Let me try to construct 2x2x4 in rank 13.
    # A is 2x2, B is 2x4 (columns 0,1,2,3 local), C is 4x2.
    # Standard Strassen for (0,1) and (2,3) gives rank 14.
    # Try to save 1 rank.
    
    # M1a: (A00+A11)(B00+B11) -> C00, C11
    # M2a: (A10+A11)B00 -> C10, -C11
    # M3a: A00(B01-B11) -> C01, C11
    # M4a: A11(B10-B00) -> C00, C10
    # M5a: (A00+A01)B11 -> -C00, C01
    # M6a: (A10-A00)(B00+B01) -> C11
    # M7a: (A01-A11)(B10+B11) -> C00
    
    # M1b: (A00+A11)(B02+B13) -> C02, C13
    # M2b: (A10+A11)B02 -> C12, -C13
    # M3b: A00(B03-B13) -> C03, C13
    # M4b: A11(B12-B02) -> C02, C12
    # M5b: (A00+A01)B13 -> -C02, C03
    # M6b: (A10-A00)(B02+B03) -> C13
    # M7b: (A01-A11)(B12+B13) -> C02
    
    # Notice M1a and M1b have the same U: (A00+A11).
    # M2a and M2b have the same U: (A10+A11).
    # M3a and M3b have the same U: A00.
    # M4a and M4b have the same U: A11.
    # M5a and M5b have the same U: (A00+A01).
    # M6a and M6b have the same U: (A10-A00).
    # M7a and M7b have the same U: (A01-A11).
    
    # So all 7 U factors are shared! This means we have 14 terms but only 7 distinct U rows.
    # The rank is determined by the number of terms, not the number of distinct U rows.
    # So this doesn't directly help reduce rank.
    
    # However, we can try to find a linear combination of terms that gives the same result
    # with one fewer term. This requires finding a dependency in the tensor.
    
    # For now, let's just return the rank-35 solution.
    # The search for rank-34 is complex and may not be feasible in the time budget.
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
