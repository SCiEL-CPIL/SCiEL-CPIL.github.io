import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 65
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper to add a 2x2 Strassen multiplication
    # X_coeffs: list of 4 (row, col, coeff) for A
    # Y_coeffs: list of 4 (row, col, coeff) for B
    # P_contribs: list of 4 (row, col, coeff) for C
    # base_col: starting column index in U, V, W
    def add_strassen_2x2(X_coeffs, Y_coeffs, P_contribs, base_col):
        # S1 = (X10 + X11) (Y00 + Y01)
        # S2 = X11 (Y10 - Y00)
        # S3 = (X00 + X11) (Y00 + Y11)
        # S4 = X00 (Y11 - Y01)
        # S5 = (X00 + X01) Y11
        # S6 = (X10 - X00) (Y00 + Y01)
        # S7 = (X01 - X11) (Y10 + Y11)
        
        # We need to expand each S into rank-1 terms
        # S1 involves 4 terms in X and 4 in Y? No.
        # S1 = (X10 + X11) * (Y00 + Y01) = X10*Y00 + X10*Y01 + X11*Y00 + X11*Y01
        # This is 4 rank-1 terms.
        # Total terms for 2x2 Strassen is 7 * 4 = 28?
        # No, the standard Strassen 2x2 decomposition has rank 7.
        # The 7 terms are scalar multiplications.
        # My expansion above is wrong. The 7 terms ARE the scalar multiplications.
        # S1 is ONE scalar multiplication of the SUMS.
        # But the sums are not single elements of A and B.
        # So S1 = (sum of A elems) * (sum of B elems).
        # This expands to multiple rank-1 terms.
        
        # Wait. The rank of 2x2x2 is 7.
        # The 7 terms are:
        # T1 = (A10+A11)(B00+B01)
        # T2 = A11(B10-B00)
        # ...
        # Each Tk is a product of two linear forms.
        # A linear form in A is a sum of elements.
        # A linear form in B is a sum of elements.
        # The product of two linear forms is a sum of rank-1 terms.
        # So each Tk contributes multiple rank-1 terms to the global decomposition.
        
        # For T1 = (A10+A11)(B00+B01):
        # Terms: A10*B00, A10*B01, A11*B00, A11*B01.
        # 4 terms.
        
        # For T2 = A11(B10-B00):
        # Terms: A11*B10, -A11*B00.
        # 2 terms.
        
        # For T3 = (A00+A11)(B00+B11):
        # Terms: A00*B00, A00*B11, A11*B00, A11*B11.
        # 4 terms.
        
        # For T4 = A00(B11-B01):
        # Terms: A00*B11, -A00*B01.
        # 2 terms.
        
        # For T5 = (A00+A01)B11:
        # Terms: A00*B11, A01*B11.
        # 2 terms.
        
        # For T6 = (A10-A00)(B00+B01):
        # Terms: A10*B00, A10*B01, -A00*B00, -A00*B01.
        # 4 terms.
        
        # For T7 = (A01-A11)(B10+B11):
        # Terms: A01*B10, A01*B11, -A11*B10, -A11*B11.
        # 4 terms.
        
        # Total terms: 4+2+4+2+2+4+4 = 22 terms per 2x2 block?
        # No, the rank is 7. The 7 terms are the "multiplications".
        # But in the tensor decomposition, each "multiplication" of linear forms
        # expands into elementary rank-1 tensors.
        # The rank of the tensor is the number of elementary rank-1 tensors.
        # So yes, 22 terms for one 2x2 block?
        # That would make 4x4x4 rank 7 * 22 = 154.
        # This is worse than schoolbook 64.
        
        # I made a mistake. The Strassen algorithm reduces the number of scalar multiplications.
        # But the "scalar multiplications" are of the linear forms.
        # The tensor rank is the number of rank-1 terms in the decomposition.
        # If we use Strassen, we are NOT reducing the tensor rank of the multiplication itself
        # unless we are considering the arithmetic complexity.
        # Tensor rank of M_2,2,2 is 7.
        # This means it can be decomposed into 7 rank-1 tensors.
        # The 7 rank-1 tensors are NOT the Strassen terms T1..T7.
        # The Strassen terms T1..T7 are the 7 scalar multiplications.
        # Each Ti is a product of two linear forms.
        # The decomposition into rank-1 tensors is different.
        
        # The rank-1 decomposition of M_2,2,2 uses 7 terms.
        # These terms are specific combinations.
        # For example, one term might be A00*B00 -> C00.
        # But Strassen says we can do it with 7 multiplications.
        # The tensor rank is 7.
        # So we need the explicit rank-1 decomposition of M_2,2,2.
        
        # The 7 rank-1 terms for 2x2x2 are:
        # 1. A00*B00 -> C00
        # 2. A00*B11 -> C01
        # 3. A11*B00 -> C10
        # 4. A11*B11 -> C11
        # 5. (A00+A11)*(B00+B11) -> C00, C01, C10, C11 (with coeffs)
        # ...
        
        # Actually, the standard Strassen decomposition for 2x2 is:
        # M1 = (A10+A11)(B00+B01)
        # M2 = A11(B10-B00)
        # M3 = (A00+A11)(B00+B11)
        # M4 = A00(B11-B01)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M3 + M2 - M5 + M7
        # C01 = M4 + M5
        # C10 = M1 + M3
        # C11 = M1 - M2 + M4 - M6
        
        # This is an arithmetic decomposition.
        # To convert to tensor rank-1 decomposition, we expand each Mi.
        # Mi = Li(A) * Li(B).
        # This is a sum of rank-1 terms.
        # The total number of rank-1 terms is the sum of the number of terms in each Li(A)*Li(B).
        # As calculated before, it is 22.
        # So the tensor rank of 2x2x2 is NOT 7 in this sense?
        # No, the tensor rank IS 7.
        # The 7 rank-1 tensors are NOT the Strassen Mi.
        # The Strassen Mi are 7 scalar multiplications.
        # The tensor rank is the minimum number of rank-1 tensors to sum to M.
        # It is known that R(M_2,2,2) = 7.
        # So there exist 7 rank-1 tensors that sum to M.
        # They are not the Strassen terms.
        
        # However, for the purpose of this task, we just need ANY decomposition.
        # If we use the expanded Strassen form, we get rank 22 for 2x2.
        # Then 4x4 would be 7 * 22 = 154.
        # This is worse than schoolbook 64.
        
        # So we should NOT use Strassen for the tensor decomposition if it increases the rank.
        # The schoolbook rank for 4x4x4 is 64.
        # The schoolbook rank for 4x4x1 is 16.
        # Total 80.
        
        # Is there a rank < 80 decomposition?
        # R(M_4,4,4) is 49?
        # If R(M_4,4,4) = 49, then we need a rank-1 decomposition of M_4,4,4 with 49 terms.
        # This is a known result (Strassen).
        # But how to construct it?
        # The 49 terms come from applying Strassen recursively.
        # But as we saw, the "Strassen terms" are not rank-1.
        
        # Wait. The rank of M_2,2,2 is 7.
        # This means it CAN be decomposed into 7 rank-1 terms.
        # The Strassen algorithm provides a way to compute the product with 7 multiplications.
        # But the tensor decomposition is different.
        
        # Let's check the rank of M_2,2,2.
        # It is 7.
        # The 7 rank-1 terms are:
        # 1. A00 B00 E00
        # 2. A00 B11 E01
        # 3. A11 B00 E10
        # 4. A11 B11 E11
        # 5. (A00+A11)(B00+B11) E00 ... No.
        
        # I will stick to the schoolbook decomposition for now, as it is guaranteed to be correct.
        # Rank 80.
        
        # If I can find a rank 65 decomposition, it would be better.
        # But I am not sure about the exact rank-1 terms for Strassen.
        
        # Let's try to implement the schoolbook decomposition.
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
