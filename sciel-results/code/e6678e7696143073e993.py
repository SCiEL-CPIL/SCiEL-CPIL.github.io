import numpy as np
import random

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing a tensor decomposition
    for 4x4 matrix multiplication.
    """
    # We want to decompose the tensor T[i, k, j] = delta_{i,j} ? No, standard matmul tensor is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor is T[i,j,k] such that contraction over j gives C.
    # Rank-1 term: u[i,j] * v[j,k] * w[i,k]
    # We need sum_r u_r[i,j] v_r[j,k] w_r[i,k] = delta_{j?} no, it's 1 for all i,j,k? 
    # Wait, the tensor of matrix multiplication is defined by T_{i,j,k} = 1 if we consider the operation 
    # C_{i,k} = \sum_j A_{i,j} B_{j,k}. 
    # So the tensor is T[i,j,k] = 1.
    # We want to find rank r such that T = \sum_{s=1}^r u_s \otimes v_s \otimes w_s.
    # i.e., 1 = \sum_{s=1}^r u_s[i,j] v_s[j,k] w_s[i,k] for all i,j,k.
    
    # This is equivalent to finding a bilinear algorithm.
    # For 2x2x2, Strassen gives rank 7.
    # For 4x4x4, we can use recursive Strassen (rank 49) or other algorithms.
    # The task asks for 4x4x5.
    # We can treat 4x4x5 as 4x4x4 + 4x4x1.
    # Or just compute 4x4x5 directly.
    
    # Let's implement a generic Strassen-like recursive decomposition for n x n x n where n is power of 2.
    # But here we have 4x4x5. 
    # We can decompose the 5 columns into 4+1.
    # C (4x5) = C_left (4x4) + C_right (4x1).
    # C_left = A @ B_left (4x4).
    # C_right = A @ B_right (4x1).
    
    # We need to return U, V, W.
    # U shape: (n*m, rank) = (16, rank)
    # V shape: (m*p, rank) = (20, rank)
    # W shape: (p*n, rank) = (20, rank)
    # Indices:
    # U[i*m + j] corresponds to A[i,j]
    # V[j*p + k] corresponds to B[j,k]
    # W[k*n + i] corresponds to C[k,i]  <-- Note: "official trace convention W[k*n+i,r]"
    # Wait, the prompt says: "output factor follows the official trace convention W[k*n+i,r]"
    # Usually C[i,k] is row i, col k.
    # If W[k*n + i], then index is k*n + i. This implies W is indexed by (col, row) of C?
    # Let's check standard conventions.
    # If C = A @ B, C[i,k] = sum_j A[i,j] B[j,k].
    # Term s contributes: (sum_{i,j} U_s[i,j] A[i,j]) * (sum_{j,k} V_s[j,k] B[j,k]) * W_s[k,i]?
    # Let's verify the contraction.
    # Let scalar_s = (U_s \circ A) : I = sum_{i,j} U_s[i,j] A[i,j].
    # Let scalar'_s = (V_s \circ B) : I = sum_{j,k} V_s[j,k] B[j,k].
    # Contribution to C[i,k] is scalar_s * scalar'_s * W_s[i,k]?
    # If W is stored as W[k*n + i], then W_s[k*n + i] is the coefficient for C[k,i]? Or C[i,k]?
    # If the index is k*n + i, and k is col, i is row, then it's C[i,k] but stored column-major?
    # Or is it C[k,i]?
    # Let's look at the dimensions.
    # C is 4x5. n=4, p=5.
    # W shape is (p*n, rank) = (20, rank).
    # If W[k*n + i], k in 0..4, i in 0..3.
    # If this maps to C[i,k], then for fixed i, k varies.
    # Let's assume the standard tensor decomposition definition:
    # C[i,k] = sum_s (sum_{i',j'} U_s[i',j'] A[i',j']) (sum_{j',k'} V_s[j',k'] B[j',k']) W_s[i,k]
    # But the prompt specifies the storage format of W.
    # If W_s is a vector of length p*n, and W_s[k*n + i] is the entry corresponding to C[i,k],
    # then we just need to fill W_s[k*n + i] with the coefficient for C[i,k].
    
    # Let's build the Strassen decomposition for 4x4x4 first.
    # Then add the 4x4x1 part.
    
    # Recursive Strassen for n x n x n.
    # Base case n=1: rank 1.
    # U[0,0]=1, V[0,0]=1, W[0,0]=1.
    # Step n -> 2n:
    # Split A, B, C into 4 blocks of size n x n.
    # A00 A01
    # A10 A11
    # 
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 + M7
    
    # We will generate the terms (U, V, W) for the 4x4x4 case.
    # Each term is a tuple of three dicts:
    # U: {(i,j): coeff}
    # V: {(j,k): coeff}
    # W: {(i,k): coeff}
    # Note: In the recursive step, we generate terms for M1..M7.
    # M1 corresponds to a product of two 4x4 matrices (in the 4x4 context, they are 2x2 blocks).
    # Wait, if we are doing 4x4x4, we split into 2x2 blocks.
    # The recursive call is for 2x2x2.
    # The terms from the recursive call give decompositions for M1, etc.
    # Let's say recursive call returns a list of terms for a 2x2x2 multiplication.
    # For M1 = X @ Y, where X = A00+A11 and Y = B00+B11.
    # X is a 2x2 matrix (in the block sense, but actually 2x2 entries).
    # The recursive terms for 2x2x2 give:
    # X @ Y = sum_s (u_s \circ X) (v_s \circ Y) w_s
    # Here u_s, v_s, w_s are 2x2 matrices (conceptually).
    # We need to map these back to the 4x4 indices.
    
    def strassen_terms(n):
        """
        Returns a list of terms for n x n x n matrix multiplication.
        Each term is (U_dict, V_dict, W_dict) where keys are (i,j), (j,k), (i,k).
        """
        if n == 1:
            return [({(0,0): 1}, {(0,0): 1}, {(0,0): 1})]
        
        half = n // 2
        base_terms = strassen_terms(half)
        
        # Define block offsets
        # A00: (0,0), A01: (0, half), A10: (half, 0), A11: (half, half)
        
        def offset_dict(d, row_off, col_off):
            return {(i+row_off, j+col_off): c for (i,j), c in d.items()}
            
        def add_dict(d1, d2):
            res = dict(d1)
            for k, v in d2.items():
                res[k] = res.get(k, 0) + v
            return res
            
        def sub_dict(d1, d2):
            res = dict(d1)
            for k, v in d2.items():
                res[k] = res.get(k, 0) - v
            return res
        
        # We need to construct the terms for M1..M7.
        # For each M_s, we take the base terms (for half x half x half) and adjust the indices and coefficients.
        # The base terms compute X @ Y = sum_s (u_s \circ X) (v_s \circ Y) w_s.
        # Here X and Y are half x half matrices.
        # In the global n x n context, X is formed by combining blocks.
        # For example, for M1, X = A00 + A11.
        # The term (u_s \circ X) is a scalar.
        # u_s \circ X = sum_{i,j} u_s[i,j] X[i,j].
        # X[i,j] = A00[i,j] + A11[i,j].
        # A00[i,j] is at global index (i, j).
        # A11[i,j] is at global index (i+half, j+half).
        # So the coefficient for A[i,j] in this term is u_s[i,j].
        # The coefficient for A[i+half, j+half] in this term is u_s[i,j].
        
        # Similarly for Y = B00 + B11.
        # v_s \circ Y = sum_{i,j} v_s[i,j] Y[i,j].
        # Y[i,j] = B00[i,j] + B11[i,j].
        # Coeff for B[i,j] is v_s[i,j].
        # Coeff for B[i+half, j+half] is v_s[i,j].
        
        # The result is added to C blocks according to the Strassen formula.
        # For M1, C00 += M1, C11 += M1.
        # The w_s from base term is a half x half matrix.
        # It contributes to C00 at (i,j) with coeff w_s[i,j].
        # It contributes to C11 at (i+half, j+half) with coeff w_s[i,j].
        
        # Let's define the structure of the 7 multiplications.
        # Each mult s has:
        # X_s: combination of A blocks
        # Y_s: combination of B blocks
        # C_s: combination of C blocks (with signs)
        
        # We represent block combinations as a list of (block_id, sign).
        # Block ids: 00, 01, 10, 11.
        
        muls = [
            # (X_blocks, Y_blocks, C_blocks)
            # X_blocks: list of (block_id, sign)
            # Y_blocks: list of (block_id, sign)
            # C_blocks: list of (block_id, sign)
            (
                [('00', 1), ('11', 1)],
                [('00', 1), ('11', 1)],
                [('00', 1), ('11', 1)]
            ),
            (
                [('10', 1), ('11', 1)],
                [('00', 1)],
                [('10', 1), ('11', 1)]
            ),
            (
                [('00', 1)],
                [('01', 1), ('11', -1)],
                [('01', 1), ('11', -1)]
            ),
            (
                [('11', 1)],
                [('10', 1), ('00', -1)],
                [('00', -1), ('10', 1)]
            ),
            (
                [('00', 1), ('01', 1)],
                [('11', 1)],
                [('01', 1)]
            ),
            (
                [('10', 1), ('00', -1)],
                [('00', 1), ('01', 1)],
                [('00', 1)]
            ),
            (
                [('01', 1), ('11', -1)],
                [('10', 1), ('11', 1)],
                [('00', -1), ('11', 1)]
            ),
        ]
        
        terms = []
        
        for X_def, Y_def, C_def in muls:
            for u_base, v_base, w_base in base_terms:
                u_new = {}
                v_new = {}
                w_new = {}
                
                # Expand X
                for (bid, sign) in X_def:
                    r_off = half if bid[0] == '1' else 0
                    c_off = half if bid[1] == '1' else 0
                    for (i, j), coeff in u_base.items():
                        key = (i + r_off, j + c_off)
                        u_new[key] = u_new.get(key, 0) + sign * coeff
                        
                # Expand Y
                for (bid, sign) in Y_def:
                    r_off = half if bid[0] == '1' else 0
                    c_off = half if bid[1] == '1' else 0
                    for (i, j), coeff in v_base.items():
                        key = (i + r_off, j + c_off)
                        v_new[key] = v_new.get(key, 0) + sign * coeff
                        
                # Expand C
                for (bid, sign) in C_def:
                    r_off = half if bid[0] == '1' else 0
                    c_off = half if bid[1] == '1' else 0
                    for (i, j), coeff in w_base.items():
                        key = (i + r_off, j + c_off)
                        w_new[key] = w_new.get(key, 0) + sign * coeff
                        
                # Clean up zeros
                u_new = {k: v for k, v in u_new.items() if v != 0}
                v_new = {k: v for k, v in v_new.items() if v != 0}
                w_new = {k: v for k, v in w_new.items() if v != 0}
                
                terms.append((u_new, v_new, w_new))
                
        return terms

    # Get terms for 4x4x4
    terms_4x4 = strassen_terms(4)
    
    # Now handle the 4x4x1 part.
    # C_right (4x1) = A @ B_right (4x1).
    # B_right has columns 4 (since p=5, indices 0..4).
    # We can just use schoolbook for this part, which is rank 4*4*1 = 16.
    # Or we can try to be smarter, but 16 is small.
    # Schoolbook terms for 4x4x1:
    # For i in 0..3, j in 0..3, k in 4 (only col 4):
    # Term: U[(i,j)]=1, V[(j,4)]=1, W[(i,4)]=1?
    # Wait, W index is (i,k) in our local representation.
    # In the final output, W is stored as W[k*n + i].
    # So for C[i,4], the index in W vector is 4*4 + i = 16 + i.
    
    terms_4x1 = []
    for i in range(4):
        for j in range(4):
            u = {(i,j): 1}
            v = {(j,4): 1}
            w = {(i,4): 1} # Local (i,k) where k=4
            terms_4x1.append((u,v,w))
            
    all_terms = terms_4x4 + terms_4x1
    
    rank = len(all_terms)
    
    # Initialize U, V, W matrices
    # U: (16, rank)
    # V: (20, rank)
    # W: (20, rank)
    
    U = np.zeros((16, rank))
    V = np.zeros((20, rank))
    W = np.zeros((20, rank))
    
    for idx, (u, v, w) in enumerate(all_terms):
        for (i, j), coeff in u.items():
            U[i*4 + j, idx] = coeff
            
        for (j, k), coeff in v.items():
            V[j*5 + k, idx] = coeff
            
        for (i, k), coeff in w.items():
            # W index is k*n + i = k*4 + i
            W[k*4 + i, idx] = coeff
            
    return {"U": U, "V": V, "W": W}

# Verification code (commented out for final submission)
# def verify():
#     res = construct()
#     U, V, W = res['U'], res['V'], res['W']
#     n, m, p = 4, 4, 5
#     rank = U.shape[1]
#     
#     # Random test
#     for _ in range(10):
#         A = np.random.randn(n, m)
#         B = np.random.randn(m, p)
#         C = A @ B
#         
#         C_calc = np.zeros((n, p))
#         for s in range(rank):
#             us = U[:, s]
#             vs = V[:, s]
#             ws = W[:, s]
#             
#             # Reshape U_s to (n, m)
#             U_s = us.reshape(n, m)
#             # Reshape V_s to (m, p)
#             V_s = vs.reshape(m, p)
#             # Reshape W_s to (p, n) because W is stored as k*n+i
#             W_s = ws.reshape(p, n)
#             
#             # Calculate scalar products
#             # Note: The decomposition is C = sum_s (U_s \circ A) (V_s \circ B) W_s ?
#             # Let's check the contraction.
#             # Term s contributes:
#             # scalar_u = sum_{i,j} U_s[i,j] A[i,j]
#             # scalar_v = sum_{j,k} V_s[j,k] B[j,k]
#             # Contribution to C[i,k] is scalar_u * scalar_v * W_s[k,i] (since W_s is p x n, W_s[k,i] is coeff for C[i,k])
#             
#             scalar_u = np.sum(U_s * A)
#             scalar_v = np.sum(V_s * B)
#             
#             # W_s is (p, n). We need to add scalar_u * scalar_v * W_s.T to C?
#             # If W_s[k, i] is the coeff for C[i, k], then C += scalar_u * scalar_v * W_s.T
#             C_calc += scalar_u * scalar_v * W_s.T
#             
#         if not np.allclose(C, C_calc):
#             print("Verification failed")
#             print(C)
#             print(C_calc)
#             return False
#     print("Verification passed")
#     return True

# if __name__ == "__main__":
#     verify()
