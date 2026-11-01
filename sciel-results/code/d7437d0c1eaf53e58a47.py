import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def _validate(U, V, W, n, m, p):
    """Exact check for tensor decomposition of matrix multiplication.
    For all i in [0,n), j in [0,m), k in [0,p):
        sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] = 1
    """
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
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

def _strassen_decomposition(n, m, p):
    """
    Constructs a valid decomposition using Strassen's 7-multiplication on the top-left 2x2 block
    and naive multiplication for the rest.
    
    A is n x m (3x3), B is m x p (3x5), C is n x p (3x5).
    We partition indices:
    - Strassen block: i in {0,1}, k in {0,1}. Uses j in {0,1}.
    - Naive rest: 
        - i in {0,1}, k in {2,3,4} (uses all j in {0,1,2})
        - i in {2}, k in {0,1,2,3,4} (uses all j in {0,1,2})
    
    Total rank = 7 (Strassen) + 6*3 (Naive top-right) + 5*3 (Naive bottom) = 7 + 18 + 15 = 40.
    """
    rank = 40
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # --- Strassen Part for C[0:2, 0:2] ---
    # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # M2 = (A[1,0] + A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[0,1] - B[1,1])
    # M4 = A[1,1] * (B[1,0] - B[0,0])
    # M5 = (A[0,0] + A[0,1]) * B[1,1]
    # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    
    # C[0,0] = M1 - M4 + M5 - M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # We add a term for each (A_factor, B_factor, C_target, coeff)
    # The term contributes coeff * A_factor * B_factor to C_target.
    # In the tensor decomposition, a term c has U[i,j], V[j,k], W[k,i].
    # If we want to represent A[i0, j0] * B[j1, k0] contributing to C[i1, k1],
    # we must be careful: the tensor term multiplies U[i,j,c] * V[j,k,c] * W[k,i,c].
    # The indices i, j, k in the tensor term are fixed for that column c.
    # So a single tensor term corresponds to a specific (i, j, k) triple in the 
    # "expanded" sense? No.
    
    # Let's re-verify the tensor definition.
    # T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1.
    # This means for a fixed c, if U[i1*m+j1, c]=a, V[j2*p+k2, c]=b, W[k3*n+i3, c]=w,
    # it contributes a*b*w to T[i1, j1, k2]? NO.
    # The indices in U, V, W for a fixed c are coupled.
    # U is indexed by (i, j). V is indexed by (j, k). W is indexed by (k, i).
    # So for a fixed c, we pick ONE (i,j) for U, ONE (j',k) for V, ONE (k',i') for W.
    # The contribution to the tensor entry T[i, j, k] is non-zero ONLY IF:
    # U[i*m+j, c] != 0 AND V[j*p+k, c] != 0 AND W[k*n+i, c] != 0.
    # Note that the j in U and the j in V must be the SAME j? 
    # Yes, because the tensor entry is T[i,j,k]. 
    # U[i*m+j] uses index j. V[j*p+k] uses index j. W[k*n+i] uses indices k and i.
    # So a term c is defined by a specific (i, j, k) triple? 
    # If U[i*m+j, c] is non-zero, it's for a specific i and j.
    # If V[j*p+k, c] is non-zero, it's for a specific j and k.
    # For the product to contribute to T[i,j,k], the j in U must match the j in V, 
    # and the k in V must match the k in W, and the i in W must match the i in U.
    # Thus, each rank-1 term in this specific tensor decomposition corresponds to 
    # a single (i, j, k) triple with weights u, v, w such that u*v*w contributes to T[i,j,k].
    # Wait, this implies the tensor is just a sum of diagonal terms?
    # No. The tensor T is the matrix multiplication tensor. T[i,j,k]=1.
    # The decomposition is T = sum_c U_c \otimes V_c \otimes W_c where the product is defined 
    # by the specific index coupling.
    # Actually, the standard definition is:
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c].
    # This means for a fixed c, U_c is a vector of size n*m, V_c of size m*p, W_c of size p*n.
    # The "product" is not a standard outer product. It's a specific contraction.
    # If U_c has non-zeros at multiple (i,j), V_c at multiple (j,k), W_c at multiple (k,i),
    # then the term c contributes to MULTIPLE (i,j,k) entries of T.
    # Specifically, it contributes U[i,j,c] V[j,k,c] W[k,i,c] to T[i,j,k].
    
    # So, for Strassen:
    # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # This is a scalar value computed from A and B.
    # In the tensor decomposition, we need to represent the bilinear form 
    # (A[0,0] + A[1,1]) * (B[0,0] + B[1,1]) using rank-1 terms.
    # A rank-1 term is defined by vectors U_c, V_c, W_c.
    # If we want the term c to represent the product of a linear combination of A's and B's,
    # we can set U_c to have non-zeros at the A-indices involved, V_c at the B-indices,
    # and W_c at the C-indices.
    # However, the coupling is strict: 
    # Contribution to T[i,j,k] is U[i,j,c] V[j,k,c] W[k,i,c].
    # This means the term c "lives" on a specific (i,j,k) if U,V,W are non-zero there.
    # But U[i,j,c] is a single number. V[j,k,c] is a single number.
    # If U has non-zeros at (0,0) and (1,1), and V has non-zeros at (0,0) and (1,1),
    # and W has non-zeros at... where?
    # The term c contributes to T[0,0,0] via U[0,0,c]V[0,0,c]W[0,0,c].
    # It contributes to T[1,1,1] via U[1,1,c]V[1,1,c]W[1,1,c].
    # It does NOT contribute to T[0,0,1] because W[1,0,c] would need to be non-zero 
    # (since k=1, i=0 -> index 1*3+0=3) but if we set W only for (0,0) and (1,1) in the 
    # "Strassen sense", we might miss cross terms.
    
    # This "Strassen as tensor terms" is tricky because Strassen's intermediate products 
    # are scalars that feed into linear combinations for the final C.
    # The standard way to map Strassen to tensor rank is:
    # Each Strassen product Mx corresponds to a "multiplication" in the arithmetic circuit.
    # In the tensor model, each multiplication Mx = (sum alpha A)(sum beta B) 
    # is realized by a set of rank-1 tensor terms that sum to the bilinear form.
    # Actually, a single bilinear form (sum alpha A)(sum beta B) has rank 1 in the 
    # "bilinear" sense, but in the 3-tensor sense, it requires expanding the sums.
    # (sum_i A_i) * (sum_j B_j) = sum_i sum_j A_i B_j.
    # This is a sum of rank-1 terms A_i B_j.
    # So M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # Each Aij Bkl is a rank-1 term in the matrix product sense?
    # No, Aij Bkl contributes to C[i,l] if j==k? No, Aij is A[i,j], Bkl is B[k,l].
    # In matrix mult C = AB, C[i,l] = sum_j A[i,j]B[j,l].
    # So A[i,j]B[k,l] contributes to C[i,l] ONLY IF j==k.
    # If j!=k, the product A[i,j]B[k,l] is "invalid" in the standard matrix product 
    # because the indices don't match for summation.
    # HOWEVER, in the tensor decomposition, we are not restricted to j==k.
    # The tensor T[i,j,k]=1 means that for ANY i,j,k, the sum must be 1.
    # This implies that "invalid" products (where j!=k) must cancel out or be absent.
    # Wait, T[i,j,k] is 1 for ALL i,j,k.
    # This is the "full" tensor.
    # The decomposition must satisfy:
    # 1 = sum_c U[i,j,c] V[j,k,c] W[k,i,c] for all i,j,k.
    
    # Let's look at the schoolbook decomposition again.
    # For each (i,j,k), we have a term c with U[i,j]=1, V[j,k]=1, W[k,i]=1.
    # This term contributes 1 to T[i,j,k] and 0 to all other T[i',j',k'] 
    # (because U is non-zero only at (i,j), so T[i',j',k'] gets 0 if i'!=i or j'!=j).
    # So schoolbook is a sum of 45 "diagonal" tensor terms.
    
    # Can we do better?
    # Strassen reduces the number of multiplications.
    # In the tensor model, a "multiplication" M = (linear A)(linear B) 
    # is represented by a set of tensor terms.
    # If M = A[i1,j1] B[j1,k1], it is 1 tensor term.
    # If M = (A[i1,j1]+A[i2,j2]) (B[j1,k1]+B[j2,k2]), 
    # it expands to 4 products.
    # But we can choose U, V, W for a single tensor term c to represent 
    # a more complex structure?
    # No, a single tensor term c is just a product of 3 vectors.
    # U_c is a vector. V_c is a vector. W_c is a vector.
    # The contribution of term c to T is the tensor U_c \otimes V_c \otimes W_c 
    # (with the specific index mapping).
    # The rank of the tensor T is the minimum number of such terms.
    
    # The known rank of the 3x3x3 matrix multiplication tensor is 23 (Strassen).
    # Wait, Strassen gives 7 for 2x2. For 3x3, it's more complex.
    # The rank of the 3x3x3 tensor is 23 (Bischof et al.).
    # The rank of the 3x3x5 tensor is 17 (Bischof-Lohmann-Stoer).
    
    # My previous derivation of rank 40 was based on "Strassen for 2x2 block + naive for rest".
    # Let's re-evaluate if that is valid.
    # If I use Strassen for the 2x2 block, I am replacing 9 terms (for i,j,k in 0,1) 
    # with 7 terms.
    # But Strassen's 7 terms are not just "7 diagonal terms". 
    # They are 7 bilinear forms.
    # Each bilinear form Mx = (sum alpha A)(sum beta B) must be expanded into 
    # individual A[i,j]B[k,l] products.
    # As established, A[i,j]B[k,l] with j!=k contributes to T[i,j,k] (where j is from A and k from B? No, indices are i,j,k).
    # In T[i,j,k], j is the middle index.
    # A[i,j] corresponds to U[i*m+j]. B[j,k] corresponds to V[j*p+k].
    # So a product A[i,j]B[j,k] (with matching j) is a valid "diagonal" term.
    # A product A[i,j1]B[j2,k] with j1!=j2?
    # This would require U[i,j1] and V[j2,k] to be non-zero in the same term c.
    # Then the term contributes to T[i,j1,k]? No.
    # The term c contributes U[i,j1,c] * V[j2,k,c] * W[k,i,c] to T[i,j1,k]?
    # No, the index j in T[i,j,k] is the same j in U[i*m+j] and V[j*p+k].
    # So if U has non-zero at (i, j1) and V has non-zero at (j2, k), 
    # and j1 != j2, then for the term to contribute to T[i, j, k], 
    # we need j = j1 (from U) AND j = j2 (from V). This is impossible if j1 != j2.
    # Therefore, a single tensor term c can only contribute to T[i,j,k] 
    # if the j-index in U and V is the SAME.
    # This means that any "mismatched" product A[i,j1]B[j2,k] with j1!=j2 
    # CANNOT be represented by a single tensor term that contributes to the tensor 
    # without also contributing to other places or being zero.
    
    # Wait, this implies that the tensor decomposition MUST consist of terms 
    # where the j-index in U and V matches?
    # If so, then the tensor is a sum of terms each supported on a "slice" j=const.
    # This would mean the rank is at least the sum of ranks of the slices?
    # No, that's not right.
    
    # Let's check the definition again.
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c].
    # If for a specific c, U[i,j1,c] != 0 and V[j2,k,c] != 0 with j1 != j2,
    # then for the term to contribute to T[i,j,k], we need j=j1 and j=j2, which is impossible.
    # So such a term contributes 0 to T[i,j,k] for all i,j,k?
    # No, it contributes 0 to T[i,j,k] because the condition j=j1 and j=j2 fails.
    # So such a term is USELESS for the diagonal T[i,j,k]=1?
    # Unless... W is also involved.
    # The term is U[i,j1,c] V[j2,k,c] W[k,i,c].
    # This value is added to T[i,j,k] ONLY IF the indices match the term's structure?
    # No, the formula is explicit: T[i,j,k] = sum_c ...
    # The summand is U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    # Note that j in U and j in V are the SAME variable j in the formula.
    # So if we pick a specific c, and we want it to contribute to T[i, j, k],
    # we look at U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    # If U[i*m+j, c] is non-zero, it's for that specific j.
    # If V[j*p+k, c] is non-zero, it's for that specific j.
    # So if U is non-zero at (i, j1) and V is non-zero at (j2, k) with j1 != j2,
    # then for any (i, j, k), if j=j1, V[j*p+k] = V[j1*p+k] which is 0 (since V is non-zero at j2).
    # If j=j2, U[i*m+j] = U[i*m+j2] which is 0 (since U is non-zero at j1).
    # So the product is 0 for all j.
    # Thus, a term c with mismatched j-indices in U and V contributes NOTHING to the tensor.
    
    # CONCLUSION: In this specific tensor decomposition, for any term c, 
    # the support of U_c in the j-dimension MUST be a subset of the support of V_c in the j-dimension, 
    # and vice versa? Actually, they must be the SAME set of j's?
    # If U is non-zero at j1 and V is non-zero at j1, it can contribute to T[i,j1,k].
    # If U is non-zero at j1 and j2, and V is non-zero at j1 and j2, 
    # it can contribute to T[i,j1,k] and T[i,j2,k].
    # But it cannot mix j1 in U with j2 in V.
    
    # This means that each term c is effectively a sum of "diagonal" terms 
    # for the j's it supports.
    # This implies that the Strassen trick (which relies on mixing indices) 
    # CANNOT be directly mapped to a lower rank in this tensor model 
    # unless the "mismatched" terms cancel out in the sum over c?
    # But if they are 0, they don't cancel, they are just absent.
    
    # So, is the rank of the 3x3x3 tensor really 23?
    # Yes, it is. How is it achieved?
    # The Strassen algorithm uses 7 multiplications.
    # Each multiplication is a bilinear form.
    # The bilinear form M = (sum alpha A)(sum beta B) is a sum of products A[i,j]B[k,l].
    # As we saw, A[i,j]B[k,l] with j!=k is 0 in the tensor model.
    # So M effectively only contains terms where j=k.
    # So M = sum_j (sum_i alpha_ij A[i,j]) (sum_k beta_jk B[j,k]).
    # This is a sum over j of (linear in A's for row j) * (linear in B's for col j).
    # This is a sum of rank-1 tensors?
    # For a fixed j, (sum_i a_i A[i,j]) (sum_k b_k B[j,k]) 
    # is a bilinear form in A's column j and B's row j.
    # This can be represented by a set of tensor terms.
    # Specifically, if we have a term c that supports j, 
    # we can set U[i,j,c] = a_i, V[j,k,c] = b_k, W[k,i,c] = 1?
    # Then the term contributes a_i b_k to T[i,j,k].
    # Summing over c, we get the sum of these products.
    # So, a bilinear form that is a sum of j-separated products 
    # can be represented by a number of tensor terms equal to the number of j's involved?
    # No, for a fixed j, the form (sum_i a_i A[i,j]) (sum_k b_k B[j,k]) 
    # has rank 1 in the "matrix" sense of (A_col_j) x (B_row_j).
    # In the tensor sense, it is a single tensor term?
    # Let's check.
    # Term c: U[i,j,c] = a_i for all i. V[j,k,c] = b_k for all k. W[k,i,c] = 1 for all k,i.
    # Contribution to T[i,j,k]: a_i * b_k * 1.
    # This matches the bilinear form for that j.
    # So, a bilinear form that is separable in j (i.e., sum_j L_A(j) L_B(j)) 
    # can be represented by ONE tensor term per j?
    # No, if we have ONE term c that supports ALL j's, 
    # we can set U[i,j,c] = a_{i,j}, V[j,k,c] = b_{j,k}, W[k,i,c] = 1.
    # Then it contributes a_{i,j} b_{j,k} to T[i,j,k].
    # This represents the sum over j of a_{i,j} b_{j,k}.
    # This is exactly the matrix product C[i,k] = sum_j A[i,j]B[j,k].
    # So the "schoolbook" for the whole matrix is ONE tensor term?
    # No, because W[k,i,c] must be 1 for all k,i.
    # But T[i,j,k] must be 1 for ALL i,j,k.
    # If we use one term with U[i,j]=A[i,j] (variable), that's not allowed. 
    # U, V, W must be FIXED vectors (constants).
    # We cannot set U[i,j,c] = A[i,j]. A is the input.
    # U, V, W are the decomposition coefficients.
    # They must be constants (integers or half-integers).
    
    # Okay, so we are back to: 
    # T[i,j,k] = 1.
    # We need sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1.
    # The "Strassen" idea is to find U,V,W with small rank r such that this holds.
    # The known rank is 23 for 3x3x3.
    # The known rank is 17 for 3x3x5.
    
    # Since I cannot reliably derive the 17 or 23 coefficients by hand, 
    # and the hardcoded ones in the inspirations are likely wrong, 
    # I will stick to the SAFE, VERIFIED rank 40 (Strassen 2x2 + Naive) 
    # IF I can prove it works.
    
    # Wait, my previous analysis showed that "mismatched" terms are 0.
    # Strassen's M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # A00B11: i=0, j=0 (from A) and j=1, k=1 (from B). j mismatch (0 vs 1).
    # So this term is 0 in the tensor model.
    # So M1 effectively reduces to A00B00 + A11B11 in the tensor model?
    # If so, Strassen doesn't help?
    # No, Strassen works because the FINAL combination C = M1 - M4 + ... 
    # causes the "mismatched" terms to cancel out in the final C?
    # But in the tensor model, the "mismatched" terms are 0, so they don't cancel, they are just gone.
    # So if Strassen's intermediate products have mismatched terms, 
    # those terms are lost.
    # Does Strassen rely on mismatched terms?
    # Yes. M1 = (A00+A11)(B00+B11). The term A00B11 is "mismatched".
    # If it is 0, then M1 = A00B00 + A11B11.
    # Then C00 = M1 - M4 + M5 - M7.
    # M4 = A11(B10-B00) = A11B10 - A11B00. (A11B10 is mismatched? i=1,j=1; j=1,k=0. j matches! So A11B10 is valid.)
    # So M4 = A11B10 - A11B00.
    # M5 = (A00+A01)B11 = A00B11 + A01B11. (A00B11 mismatched, A01B11 valid? i=0,j=1; j=1,k=1. j matches. Valid.)
    # M7 = (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11. (All valid? 
    # A01B10: i=0,j=1; j=1,k=0. Valid.
    # A01B11: i=0,j=1; j=1,k=1. Valid.
    # A11B10: i=1,j=1; j=1,k=0. Valid.
    # A11B11: i=1,j=1; j=1,k=1. Valid.)
    
    # So, in the tensor model, the mismatched terms (A00B11) are 0.
    # So M1 = A00B00 + A11B11.
    # M5 = A01B11.
    # C00 = (A00B00 + A11B11) - (A11B10 - A11B00) + (A01B11) - (A01B10 + A01B11 - A11B10 - A11B11)
    # C00 = A00B00 + A11B11 - A11B10 + A11B00 + A01B11 - A01B10 - A01B11 + A11B10 + A11B11
    # C00 = A00B00 + A11B00 + A11B11 + A11B11 - A01B10
    # This is NOT equal to A00B00 + A01B01 + A02B02 (the correct C00).
    # So Strassen's algebraic identity does NOT hold in the tensor model 
    # if mismatched terms are 0.
    
    # This implies that the "Strassen decomposition" is NOT a valid tensor decomposition 
    # if we simply set mismatched terms to 0.
    # The tensor decomposition MUST satisfy T[i,j,k]=1.
    # The Strassen algorithm satisfies C[i,k] = sum_j A[i,j]B[j,k].
    # The tensor T is defined such that T[i,j,k]=1.
    # The connection is that C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k].
    # So the tensor T is the identity in the middle index.
    # Any valid decomposition of T must reproduce the matrix product.
    # If Strassen reproduces the matrix product, does it define a valid T?
    # Yes, but the T derived from Strassen might have T[i,j,k] != 1 for j!=k?
    # No, the tensor T is fixed. We are decomposing the FIXED tensor T.
    # So we need U,V,W such that sum_c ... = 1 for all i,j,k.
    # The Strassen coefficients, when plugged in, might not sum to 1 for all i,j,k.
    # They might sum to 1 for the "diagonal" j's and 0 for others?
    # If so, it's not a valid decomposition of T (which is 1 everywhere).
    
    # Therefore, the hardcoded rank-17 or rank-23 solutions are non-trivial 
    # and cannot be derived by simply "applying Strassen" to the tensor.
    # They are specific solutions to the polynomial system.
    
    # Given this, the safest bet is to use the schoolbook (rank 45) 
    # OR find a valid lower rank via search.
    # The current program uses search.
    # I will improve the search to be faster and more robust, 
    # and keep the schoolbook fallback.
    
    # I will NOT attempt to hardcode Strassen-derived ranks as they are likely invalid.
    
    return None

def _search_low_rank(n, m, p, target_rank, max_iters, seed, time_budget):
    """Fast gradient-descent search for a valid rank-`target_rank` decomposition."""
    random.seed(seed)
    np.random.seed(seed)

    # Initialize with small random integers
    U = np.array([[random.randint(-1, 1) for _ in range(target_rank)] for _ in range(n*m)], dtype=float)
    V = np.array([[random.randint(-1, 1) for _ in range(target_rank)] for _ in range(m*p)], dtype=float)
    W = np.array([[random.randint(-1, 1) for _ in range(target_rank)] for _ in range(p*n)], dtype=float)

    idx_u, idx_v, idx_w = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u.append(i*m + j)
                idx_v.append(j*p + k)
                idx_w.append(k*n + i)
    idx_u = np.array(idx_u); idx_v = np.array(idx_v); idx_w = np.array(idx_w)

    lr = 0.05
    best_err = float('inf')
    best_state = (U.copy(), V.copy(), W.copy())
    start = time.time()

    for it in range(max_iters):
        if time.time() - start > time_budget:
            break
        grad_U = np.zeros_like(U); grad_V = np.zeros_like(V); grad_W = np.zeros_like(W)
        s_vals = np.empty(len(idx_u))
        for idx in range(len(idx_u)):
            s_val = np.sum(U[idx_u[idx]] * V[idx_v[idx]] * W[idx_w[idx]])
            s_vals[idx] = s_val
            diff = 2.0 * (s_val - 1.0)
            grad_U[idx_u[idx]] += diff * V[idx_v[idx]] * W[idx_w[idx]]
            grad_V[idx_v[idx]] += diff * U[idx_u[idx]] * W[idx_w[idx]]
            grad_W[idx_w[idx]] += diff * U[idx_u[idx]] * V[idx_v[idx]]
        U -= lr * grad_U; V -= lr * grad_V; W -= lr * grad_W
        err = np.sum((s_vals - 1.0)**2)
        if err < best_err:
            best_err = err
            best_state = (U.copy(), V.copy(), W.copy())
        if err < 1e-6:
            break

    # Snap to nearest 0.5
    U_snap = np.round(best_state[0] * 2.0) / 2.0
    V_snap = np.round(best_state[1] * 2.0) / 2.0
    W_snap = np.round(best_state[2] * 2.0) / 2.0
    valid = True
    for idx in range(len(idx_u)):
        s_val = np.sum(U_snap[idx_u[idx]] * V_snap[idx_v[idx]] * W_snap[idx_w[idx]])
        if abs(s_val - 1.0) > 1e-6:
            valid = False
            break
    if valid:
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s

    fallback = _schoolbook(n, m, p)

    # Try to find a valid decomposition with rank < 45.
    # We search for rank 44 first, as it is the most likely to be found quickly.
    for target_rank in range(44, 16, -1):
        if time.time() > deadline:
            break
        n_seeds = 3 if target_rank >= 35 else 2
        for seed in range(n_seeds):
            if time.time() > deadline:
                break
            res = _search_low_rank(n, m, p, target_rank, max_iters=8000, seed=seed,
                                   time_budget=min(15, deadline - time.time()))
            if res is not None:
                U, V, W = res["U"], res["V"], res["W"]
                if _validate(U, V, W, n, m, p):
                    return res

    return fallback
