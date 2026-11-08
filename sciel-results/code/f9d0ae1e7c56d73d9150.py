import numpy as np
import time

def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using two-level Strassen.
    C[i,j] = sum_k A[i,k] * B[k,j]
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    Convention: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    Tensor: T[i*4+j, j*4+k, k*4+i] = sum_r U[i*4+j,r] * V[j*4+k,r] * W[k*4+i,r] = 1
    """
    start_time = time.time()
    DEADLINE = 140  # Safety margin
    
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def get_U_index(row, col):
        return row * m + col
    
    def get_V_index(row, col):
        return row * p + col
    
    def get_W_index(row, col):
        # Trace convention: W[k*n+i, r] where k=col of C, i=row of C
        return col * n + row

    # Two-level Strassen: 7 top-level terms, each decomposed into 7 sub-terms
    # Total: 49 rank-1 terms
    
    # Each rank-1 term is defined by:
    # - A_form: dict {(i,k): coeff} for A[i,k]
    # - B_form: dict {(k,j): coeff} for B[k,j]  
    # - C_form: dict {(i,j): coeff} for C[i,j]
    # such that sum over all terms of A_form * B_form * C_form reconstructs the tensor
    
    # We'll build this by composing two levels of Strassen
    
    # Level 1: Standard 2x2 Strassen on 4x4 (split into 2x2 blocks)
    # Level 2: Standard 2x2 Strassen on each 2x2 block product
    
    # Define helper to merge linear forms
    def merge_forms(*forms):
        result = {}
        for form in forms:
            for key, val in form.items():
                result[key] = result.get(key, 0) + val
        # Remove zero entries
        return {k: v for k, v in result.items() if v != 0}
    
    def scale_form(form, scale):
        return {k: v * scale for k, v in form.items() if v * scale != 0}
    
    def diff_form(f1, f2):
        return merge_forms(f1, scale_form(f2, -1))
    
    # For a 2x2 matrix M with elements indexed (r,c) where r,c in {0,1},
    # the element at global position (2*br+r, 2*bc+c) where br,bc are block indices
    
    # Define the 7 Strassen terms for 2x2 matrix multiplication
    # Input: 2x2 matrices L (left) and R (right), each element is a linear form
    # Output: 7 pairs (A_form, B_form) and the C contribution pattern
    
    # Strassen 2x2:
    # M1 = (L00+L11)(R00+R11)
    # M2 = (L10+L11)R00
    # M3 = L00(R01-R11)
    # M4 = L11(R10-R00)
    # M5 = (L00+L01)R11
    # M6 = (L10-L00)(R00+R01)
    # M7 = (L01-L11)(R10+R11)
    
    # Result:
    # P00 = M1 + M4 - M5 + M7
    # P01 = M3 + M5
    # P10 = M2 + M4
    # P11 = M1 - M2 + M3 + M6
    
    # For each of the 7 top-level Strassen terms on the 4x4 product,
    # we have a 2x2 matrix L_t and R_t of linear forms.
    # Then we apply Strassen to compute L_t * R_t, giving 7 sub-products.
    
    # Top-level Strassen for 4x4:
    # P1 = (A00+A11)(B00+B11)  -> contributes to C00 and C11
    # P2 = (A10+A11)B00         -> contributes to C10 and C11
    # P3 = A00(B01-B11)         -> contributes to C01 and C11
    # P4 = A11(B10-B00)         -> contributes to C00 and C10
    # P5 = (A00+A01)B11         -> contributes to C00 and C01
    # P6 = (A10-A00)(B00+B01)   -> contributes to C11
    # P7 = (A01-A11)(B10+B11)   -> contributes to C00
    
    # where Aij, Bij, Cij are 2x2 blocks
    
    # For each top-level term t, L_t is a 2x2 matrix of linear forms on A,
    # R_t is a 2x2 matrix of linear forms on B, and the product P_t = L_t * R_t
    # is a 2x2 matrix that contributes to specific C blocks.
    
    # Define the 7 top-level terms
    # Each term: (L_form, R_form, C_contrib)
    # L_form: 2x2 array of dicts (linear forms on A)
    # R_form: 2x2 array of dicts (linear forms on B)
    # C_contrib: 2x2 array of dicts (linear forms on C, indicating where P_t goes)
    
    def A_elem(br, bc, r, c):
        """A element at block (br,bc), local (r,c) -> global (2*br+r, 2*bc+c)"""
        return {(2*br + r, 2*bc + c): 1}
    
    def B_elem(br, bc, r, c):
        """B element at block (br,bc), local (r,c) -> global (2*br+r, 2*bc+c)"""
        return {(2*br + r, 2*bc + c): 1}
    
    def C_elem(br, bc, r, c):
        """C element at block (br,bc), local (r,c) -> global (2*br+r, 2*bc+c)"""
        return {(2*br + r, 2*bc + c): 1}
    
    top_terms = []
    
    # Term 1: (A00+A11)(B00+B11) -> C00 += P, C11 += P
    L1 = [[merge_forms(A_elem(0,0,r,c), A_elem(1,1,r,c)) for c in range(2)] for r in range(2)]
    R1 = [[merge_forms(B_elem(0,0,r,c), B_elem(1,1,r,c)) for c in range(2)] for r in range(2)]
    C1 = [[C_elem(0,0,r,c) for c in range(2)] for r in range(2)]  # contributes to C00
    C1b = [[C_elem(1,1,r,c) for c in range(2)] for r in range(2)]  # contributes to C11
    top_terms.append((L1, R1, [(C1, 1), (C1b, 1)]))
    
    # Term 2: (A10+A11)B00 -> C10 += P, C11 -= P
    L2 = [[merge_forms(A_elem(1,0,r,c), A_elem(1,1,r,c)) for c in range(2)] for r in range(2)]
    R2 = [[B_elem(0,0,r,c) for c in range(2)] for r in range(2)]
    C2 = [[C_elem(1,0,r,c) for c in range(2)] for r in range(2)]
    C2b = [[C_elem(1,1,r,c) for c in range(2)] for r in range(2)]
    top_terms.append((L2, R2, [(C2, 1), (C2b, -1)]))
    
    # Term 3: A00(B01-B11) -> C01 += P, C11 += P
    L3 = [[A_elem(0,0,r,c) for c in range(2)] for r in range(2)]
    R3 = [[diff_form(B_elem(0,1,r,c), B_elem(1,1,r,c)) for c in range(2)] for r in range(2)]
    C3 = [[C_elem(0,1,r,c) for c in range(2)] for r in range(2)]
    C3b = [[C_elem(1,1,r,c) for c in range(2)] for r in range(2)]
    top_terms.append((L3, R3, [(C3, 1), (C3b, 1)]))
    
    # Term 4: A11(B10-B00) -> C00 += P, C10 += P
    L4 = [[A_elem(1,1,r,c) for c in range(2)] for r in range(2)]
    R4 = [[diff_form(B_elem(1,0,r,c), B_elem(0,0,r,c)) for c in range(2)] for r in range(2)]
    C4 = [[C_elem(0,0,r,c) for c in range(2)] for r in range(2)]
    C4b = [[C_elem(1,0,r,c) for c in range(2)] for r in range(2)]
    top_terms.append((L4, R4, [(C4, 1), (C4b, 1)]))
    
    # Term 5: (A00+A01)B11 -> C00 -= P, C01 += P
    L5 = [[merge_forms(A_elem(0,0,r,c), A_elem(0,1,r,c)) for c in range(2)] for r in range(2)]
    R5 = [[B_elem(1,1,r,c) for c in range(2)] for r in range(2)]
    C5 = [[C_elem(0,0,r,c) for c in range(2)] for r in range(2)]
    C5b = [[C_elem(0,1,r,c) for c in range(2)] for r in range(2)]
    top_terms.append((L5, R5, [(C5, -1), (C5b, 1)]))
    
    # Term 6: (A10-A00)(B00+B01) -> C11 += P
    L6 = [[diff_form(A_elem(1,0,r,c), A_elem(0,0,r,c)) for c in range(2)] for r in range(2)]
    R6 = [[merge_forms(B_elem(0,0,r,c), B_elem(0,1,r,c)) for c in range(2)] for r in range(2)]
    C6 = [[C_elem(1,1,r,c) for c in range(2)] for r in range(2)]
    top_terms.append((L6, R6, [(C6, 1)]))
    
    # Term 7: (A01-A11)(B10+B11) -> C00 += P
    L7 = [[diff_form(A_elem(0,1,r,c), A_elem(1,1,r,c)) for c in range(2)] for r in range(2)]
    R7 = [[merge_forms(B_elem(1,0,r,c), B_elem(1,1,r,c)) for c in range(2)] for r in range(2)]
    C7 = [[C_elem(0,0,r,c) for c in range(2)] for r in range(2)]
    top_terms.append((L7, R7, [(C7, 1)]))
    
    # Now for each top-level term, apply Strassen to compute the 2x2 product
    # and add the 7 sub-terms to U, V, W
    
    for t_idx, (L_t, R_t, C_contribs) in enumerate(top_terms):
        # Extract 2x2 elements
        L00, L01 = L_t[0][0], L_t[0][1]
        L10, L11 = L_t[1][0], L_t[1][1]
        R00, R01 = R_t[0][0], R_t[0][1]
        R10, R11 = R_t[1][0], R_t[1][1]
        
        # 7 Strassen sub-products
        # M1 = (L00+L11)(R00+R11)
        # M2 = (L10+L11)R00
        # M3 = L00(R01-R11)
        # M4 = L11(R10-R00)
        # M5 = (L00+L01)R11
        # M6 = (L10-L00)(R00+R01)
        # M7 = (L01-L11)(R10+R11)
        
        sub_products = [
            (merge_forms(L00, L11), merge_forms(R00, R11)),  # M1
            (merge_forms(L10, L11), R00),                      # M2
            (L00, diff_form(R01, R11)),                         # M3
            (L11, diff_form(R10, R00)),                         # M4
            (merge_forms(L00, L01), R11),                       # M5
            (diff_form(L10, L00), merge_forms(R00, R01)),       # M6
            (diff_form(L01, L11), merge_forms(R10, R11)),       # M7
        ]
        
        # Strassen result:
        # P00 = M1 + M4 - M5 + M7
        # P01 = M3 + M5
        # P10 = M2 + M4
        # P11 = M1 - M2 + M3 + M6
        
        # For each sub-product k, it contributes to certain positions in the 2x2 result
        # sub_contrib[k] = list of (local_row, local_col, coeff)
        sub_contrib = [
            [(0,0,1), (1,1,1)],    # M1 -> P00 and P11
            [(1,0,1), (1,1,-1)],   # M2 -> P10 and P11
            [(0,1,1), (1,1,1)],    # M3 -> P01 and P11
            [(0,0,1), (1,0,1)],    # M4 -> P00 and P10
            [(0,0,-1), (0,1,1)],   # M5 -> P00 and P01
            [(1,1,1)],             # M6 -> P11
            [(0,0,1)]              # M7 -> P00
        ]
        
        for k in range(7):
            rank_idx = t_idx * 7 + k
            A_form, B_form = sub_products[k]
            
            # Fill U and V
            for (i, kk), coeff in A_form.items():
                U[get_U_index(i, kk), rank_idx] += coeff
            
            for (kk, j), coeff in B_form.items():
                V[get_V_index(kk, j), rank_idx] += coeff
            
            # Fill W: for each local position (lr, lc) in the 2x2 result,
            # and each C contribution pattern, add to W
            for (lr, lc, sub_coeff) in sub_contrib[k]:
                for (C_pattern, block_coeff) in C_contribs:
                    # C_pattern[lr][lc] is the linear form for C at local (lr,lc)
                    C_form = C_pattern[lr][lc]
                    for (i, j), c_coeff in C_form.items():
                        W[get_W_index(i, j), rank_idx] += sub_coeff * block_coeff * c_coeff
    
    # Check deadline
    if time.time() > DEADLINE:
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
