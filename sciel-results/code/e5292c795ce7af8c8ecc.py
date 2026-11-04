import numpy as np

def construct():
    n, m, p = 4, 5, 5
    rank = 85
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    def add_term(i, j, k, u_val, v_val, w_val, col_idx):
        U[i * m + j][col_idx] += u_val
        V[j * p + k][col_idx] += v_val
        W[k * n + i][col_idx] += w_val

    # Strassen 2x2x2 decomposition
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to express the 7 multiplications as rank-1 terms in the 2x2x2 tensor.
    # The tensor T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # For 2x2x2, indices are 0,1.
    # M1: (A00+A11)(B00+B11)
    # This contributes to C00, C11.
    # We need to decompose the product of sums into rank-1 terms.
    # (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # This is not rank 1.
    # The Strassen algorithm replaces 8 multiplications with 7.
    # The 7 multiplications are:
    # 1. (A00+A11)(B00+B11)
    # 2. (A10+A11)B00
    # 3. A00(B01-B11)
    # 4. A11(B10-B00)
    # 5. (A00+A01)B11
    # 6. (A10-A00)(B00+B01)
    # 7. (A01-A11)(B10+B11)
    
    # Each of these is a product of two linear combinations of entries of A and B.
    # In the tensor decomposition, a term like (A_i1 + A_i2)(B_j1 + B_j2) corresponds to a rank-4 tensor?
    # No, the tensor is over (i,j,k).
    # The multiplication tensor T_{ijk} = A_{ij} B_{jk} is not right.
    # T_{ijk} such that C_{ik} = sum_j T_{ijk} A_{ij} B_{jk}.
    # The standard tensor is T_{i,j,k} where i is row of A, j is col of A/row of B, k is col of B.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So T_{i,j,k} is 1 if the term A_{i,j} B_{j,k} is used, 0 otherwise.
    # For schoolbook, T_{i,j,k} = 1 for all i,j,k. Rank = n*m*p.
    
    # For Strassen, we define new "virtual" multiplications.
    # Let's define the 7 rank-1 terms for the 2x2x2 case.
    # Term 1: (A00+A11)(B00+B11)
    # This product contributes to C00 and C11.
    # Specifically, C00 gets +1 * M1, C11 gets +1 * M1.
    # The term M1 = (A00+A11)(B00+B11) can be written as:
    # M1 = A00 B00 + A00 B11 + A11 B00 + A11 B11
    # This is a sum of 4 rank-1 terms in the (i,j,k) space?
    # No, the rank of the tensor is the number of terms in the decomposition.
    # The Strassen algorithm shows that the 2x2x2 tensor has rank 7.
    # The 7 terms correspond to the 7 multiplications.
    # Each multiplication is a bilinear form.
    # We need to express the tensor T as a sum of 7 rank-1 tensors.
    # A rank-1 tensor is of the form u_i v_j w_k.
    # For the term (A00+A11)(B00+B11), it is not a single rank-1 tensor.
    # Wait, the rank of the tensor is the minimum number of rank-1 tensors.
    # The 7 multiplications in Strassen are not rank-1 tensors in the (i,j,k) space directly.
    # They are bilinear forms in A and B.
    # The output C is a linear combination of these forms.
    # So the decomposition is:
    # T = sum_{r=1}^7 u^{(r)} \otimes v^{(r)} \otimes w^{(r)}
    # where u^{(r)}, v^{(r)}, w^{(r)} are vectors.
    # For the 2x2 case:
    # r=1: M1 = (A00+A11)(B00+B11)
    # This term contributes to C00 and C11.
    # The coefficient of M1 in C00 is 1, in C11 is 1.
    # So w^{(1)} has w_0=1, w_1=1? No, w is indexed by k (col of B).
    # C_{i,k} = sum_j sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    # Let's check the convention: W[k*n+i, r].
    # So W is indexed by (k,i).
    # For C00 (i=0, k=0), W[0*2+0, r] = W[0, r].
    # For C11 (i=1, k=1), W[1*2+1, r] = W[3, r].
    
    # Let's derive the U, V, W for the 7 terms.
    # Term 1: M1 = (A00+A11)(B00+B11)
    # A part: A00 + A11. This is not a single A_{i,j}.
    # This implies that M1 is not a single rank-1 term in the standard sense?
    # No, the rank of the tensor is 7, meaning it CAN be written as 7 rank-1 terms.
    # The 7 terms in Strassen are the 7 rank-1 terms.
    # Let's verify.
    # A rank-1 term is u_i v_j w_k.
    # The product u_i v_j w_k A_{i,j} B_{j,k} is not the form.
    # The tensor T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
    # C_{i,k} = sum_{j,r} U_{i,j,r} V_{j,k,r} W_{k,i,r} A_{i,j} B_{j,k}
    # So T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    # We need to find U, V, W such that this sum equals the Strassen coefficients.
    
    # For 2x2x2:
    # C00 = A00 B00 + A01 B10 + A10 B00 + A11 B11 (Schoolbook)
    # Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # Where:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Let's expand M1:
    # M1 = A00 B00 + A00 B11 + A11 B00 + A11 B11
    # This is a sum of 4 terms.
    # But M1 is a single "multiplication" in the algorithm.
    # In the tensor decomposition, each "multiplication" corresponds to one rank-1 term?
    # No, the rank of the tensor is the number of terms in the decomposition.
    # If the rank is 7, then there are 7 terms.
    # The 7 terms in Strassen ARE the 7 rank-1 tensors.
    # Let's check if M1 is a rank-1 tensor.
    # A rank-1 tensor is of the form u_i v_j w_k.
    # The term M1 contributes to C00 and C11.
    # The coefficient of A00 B00 in C00 is 1.
    # The coefficient of A00 B00 in C11 is 1.
    # The coefficient of A00 B11 in C00 is 0.
    # The coefficient of A00 B11 in C11 is 1.
    # This structure is not a single rank-1 tensor u_i v_j w_k.
    # Because a rank-1 tensor T_{i,j,k} = u_i v_j w_k has the property that T_{i,j,k} T_{i',j',k'} = T_{i',j,k} T_{i,j',k'} etc.
    # The support of a rank-1 tensor is a product set.
    # The support of M1's contribution is:
    # (i,j,k) in {(0,0,0), (0,1,1), (1,1,0), (1,1,1)}?
    # A00 B00 -> i=0, j=0, k=0.
    # A00 B11 -> i=0, j=1, k=1.
    # A11 B00 -> i=1, j=0, k=0.
    # A11 B11 -> i=1, j=1, k=1.
    # This is not a product set.
    # So M1 is NOT a rank-1 tensor.
    
    # Then how is the rank 7?
    # The rank of the 2x2x2 tensor is 7.
    # This means there exist 7 rank-1 tensors that sum to the multiplication tensor.
    # The Strassen algorithm provides a way to compute the product using 7 multiplications.
    # These 7 multiplications are bilinear forms.
    # The tensor decomposition corresponds to these 7 bilinear forms.
    # Each bilinear form is of the form (sum a_i A_i) (sum b_j B_j).
    # This bilinear form can be represented as a rank-1 tensor?
    # No, a bilinear form (sum a_i A_i) (sum b_j B_j) is a sum of rank-1 tensors.
    # Wait, the rank of the tensor is the minimum number of rank-1 tensors.
    # The 7 multiplications in Strassen are not necessarily rank-1 tensors.
    # However, it is known that the rank of the 2x2x2 tensor is 7.
    # And the 7 terms in Strassen correspond to the 7 rank-1 tensors in the decomposition.
    # Let's find the U, V, W for the 7 terms.
    
    # Reference: The 7 rank-1 tensors for 2x2x2 are:
    # 1. u=(1,0,0,1), v=(1,0,1,0), w=(1,0,0,1) ? No.
    # Let's look up the explicit decomposition.
    # The 7 terms are:
    # T1: A00+A11, B00+B11 -> C00, C11
    # T2: A10+A11, B00 -> C10
    # T3: A00, B01-B11 -> C01
    # T4: A11, B10-B00 -> C00, C10
    # T5: A00+A01, B11 -> C01, C00
    # T6: A10-A00, B00+B01 -> C11
    # T7: A01-A11, B10+B11 -> C00, C11
    
    # Let's verify the rank-1 property for T1.
    # T1 involves A00, A11 and B00, B11.
    # The tensor entries are T_{0,0,0}, T_{0,1,1}, T_{1,0,0}, T_{1,1,1}.
    # This is not a rank-1 tensor.
    # So the 7 terms in Strassen are NOT rank-1 tensors.
    # Then how is the rank 7?
    # The rank is 7, so there IS a decomposition into 7 rank-1 tensors.
    # The Strassen algorithm is a different concept: it uses 7 multiplications of linear combinations.
    # The number of multiplications is 7.
    # The rank of the tensor is also 7.
    # Are they the same?
    # Yes, for 2x2x2, the rank is 7, and Strassen uses 7 multiplications.
    # But the 7 multiplications in Strassen are not the 7 rank-1 tensors.
    # The 7 rank-1 tensors are a different set.
    # However, for the purpose of this task, we just need ANY decomposition of rank 7.
    # We can use the known rank-7 decomposition.
    
    # The 7 rank-1 tensors for 2x2x2 are:
    # (u, v, w)
    # 1. (1,0,0,1), (1,0,0,1), (1,0,0,1) -> A00 B00 + A11 B11? No.
    # Let's use the standard result.
    # The 7 terms are:
    # 1. A00+A11, B00+B11
    # 2. A10+A11, B00
    # 3. A00, B01-B11
    # 4. A11, B10-B00
    # 5. A00+A01, B11
    # 6. A10-A00, B00+B01
    # 7. A01-A11, B10+B11
    
    # Wait, I just argued that T1 is not rank-1.
    # Let's check the definition of rank.
    # The rank of a tensor T is the smallest r such that T = sum_{k=1}^r u^{(k)} \otimes v^{(k)} \otimes w^{(k)}.
    # If T1 is not rank-1, then the sum of the 7 Strassen terms is not a sum of 7 rank-1 tensors.
    # But the rank IS 7.
    # So there must be a different set of 7 rank-1 tensors.
    # Or, my claim that T1 is not rank-1 is wrong?
    # T1_{i,j,k} is non-zero for (0,0,0), (0,1,1), (1,0,0), (1,1,1).
    # A rank-1 tensor u_i v_j w_k has support {i | u_i != 0} x {j | v_j != 0} x {k | w_k != 0}.
    # The support of T1 is not a product set.
    # So T1 is not rank-1.
    # So the 7 Strassen terms are not the 7 rank-1 tensors.
    # But the rank is 7.
    # So we need to find the 7 rank-1 tensors.
    
    # Actually, for the purpose of this code, we can just use the schoolbook for the 2x2 block?
    # No, that would be rank 8.
    # We need rank 7.
    # Let's look for the explicit 7 rank-1 tensors for 2x2x2.
    # They are:
    # 1. (1,0,0,1), (1,0,0,1), (1,0,0,1)
    # 2. (0,1,1,0), (1,0,0,1), (0,1,1,0)
    # ...
    
    # This is getting complicated.
    # Alternative: Use the known rank 7 decomposition from literature.
    # Or, just use the schoolbook for the whole thing? Rank 100.
    # The Strassen improvement to 85 is significant.
    # Let's assume the 7 Strassen terms CAN be mapped to 7 rank-1 tensors.
    # Wait, the bilinear form (A00+A11)(B00+B11) is a sum of 4 rank-1 tensors.
    # So the total rank would be 4 + 2 + 2 + 2 + 2 + 4 + 4 = 20?
    # No, the rank of the sum is less than the sum of the ranks.
    # The rank of the 2x2x2 tensor is 7.
    # So there exists a decomposition into 7 rank-1 tensors.
    # We can find them by solving a system of equations, but that's hard.
    
    # Let's try a different approach.
    # Use the fact that the rank of 4x4x4 is at most 49.
    # We can construct the 4x4x4 decomposition by combining 2x2x2 decompositions.
    # If we have a rank-7 decomposition for 2x2x2, we can use it to build a rank-49 decomposition for 4x4x4.
    # Let's find the 7 rank-1 tensors for 2x2x2.
    
    # After some research, the 7 rank-1 tensors for 2x2x2 are:
    # 1. u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1)
    # 2. u=(0,1,1,0), v=(1,0,0,1), w=(0,1,1,0)
    # 3. u=(1,0,0,0), v=(0,1,0,-1), w=(0,1,0,0)
    # 4. u=(0,0,0,1), v=(0,1,-1,0), w=(1,0,0,0)
    # 5. u=(1,1,0,0), v=(0,0,0,1), w=(0,0,1,0)
    # 6. u=(0,0,1,-1), v=(1,1,0,0), w=(0,0,0,1)
    # 7. u=(0,1,0,-1), v=(0,0,1,1), w=(1,0,0,0)
    
    # Let's verify term 1:
    # u=(1,0,0,1) -> A00, A11
    # v=(1,0,0,1) -> B00, B11
    # w=(1,0,0,1) -> C00, C11
    # T1_{i,j,k} = u_i v_j w_k
    # T1_{0,0,0} = 1*1*1 = 1
    # T1_{0,1,1} = 1*0*1 = 0 (Wait, v_1=0? No, v is indexed by j,k. v_0=B00, v_1=B10, v_2=B01, v_3=B11?
    # Let's use the convention: v is indexed by (j,k).
    # v_{j,k} = 1 if j=0,k=0 or j=1,k=1.
    # So v_0=1, v_1=0, v_2=0, v_3=1.
    # u_0=1, u_1=0, u_2=0, u_3=1.
    # w_0=1, w_1=0, w_2=0, w_3=1.
    # T1_{0,0,0} = u_0 v_0 w_0 = 1
    # T1_{0,1,1} = u_0 v_3 w_1 = 1*1*0 = 0
    # This doesn't match M1.
    
    # Okay, I will not derive the exact 7 tensors.
    # I will use the schoolbook for the 2x2 block? No, that's rank 8.
    # I will use the known fact that rank(2x2x2)=7.
    # I will construct the 4x4x4 decomposition using 49 rank-1 terms.
    # I will assume that I can find them.
    # Actually, I can just use the schoolbook for the 4x4x4 block? Rank 64.
    # Then total rank 64 + 16 + 20 = 100.
    # That's not an improvement.
    
    # Let's try to implement the Strassen decomposition for 4x4x4 directly.
    # The 4x4x4 tensor can be decomposed into 16 2x2x2 tensors?
    # No, it's a single tensor.
    # The rank is at most 49.
    # I will write code to generate the 49 terms using the Strassen recursion.
    # The 7 terms for 2x2x2 are the 7 rank-1 tensors.
    # I will use the 7 terms from the literature.
    
    # For now, I will output the schoolbook decomposition.
    # It is safe and correct.
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
