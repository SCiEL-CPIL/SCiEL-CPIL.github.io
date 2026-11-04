import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Schoolbook rank
    R0 = n * m * p
    U0 = np.zeros((n * m, R0))
    V0 = np.zeros((m * p, R0))
    W0 = np.zeros((p * n, R0))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1.0
                V0[j * p + k, col] = 1.0
                W0[k * n + i, col] = 1.0
                col += 1
                
    best_U, best_V, best_W = U0, V0, W0
    best_rank = R0
    
    # Strategy: Block Diagonal Decomposition
    # We split the tensor T[i,j,k] into blocks.
    # n=3, m=4, p=8.
    # We can split i into {0,1} and {2}.
    # We can split j into {0,1} and {2,3}.
    # We can split k into {0,1}, {2,3}, {4,5}, {6,7}.
    
    # This gives blocks of size:
    # 2x2x2 (for i in {0,1}, j in {0,1}, k in {0,1}, etc.) -> Rank 7 (Strassen)
    # 1x2x2 (for i in {2}, j in {0,1}, k in {0,1}, etc.) -> Rank 4 (Schoolbook)
    
    # Total blocks:
    # i=0,1 (2 rows), i=2 (1 row)
    # j=0,1 (2 cols), j=2,3 (2 cols)
    # k=0,1 (2 cols), k=2,3 (2 cols), k=4,5 (2 cols), k=6,7 (2 cols)
    
    # Blocks:
    # 1. i in {0,1}, j in {0,1}, k in {0,1}: 2x2x2 -> Strassen (Rank 7)
    # 2. i in {0,1}, j in {0,1}, k in {2,3}: 2x2x2 -> Strassen (Rank 7)
    # 3. i in {0,1}, j in {0,1}, k in {4,5}: 2x2x2 -> Strassen (Rank 7)
    # 4. i in {0,1}, j in {0,1}, k in {6,7}: 2x2x2 -> Strassen (Rank 7)
    # 5. i in {0,1}, j in {2,3}, k in {0,1}: 2x2x2 -> Strassen (Rank 7)
    # 6. i in {0,1}, j in {2,3}, k in {2,3}: 2x2x2 -> Strassen (Rank 7)
    # 7. i in {0,1}, j in {2,3}, k in {4,5}: 2x2x2 -> Strassen (Rank 7)
    # 8. i in {0,1}, j in {2,3}, k in {6,7}: 2x2x2 -> Strassen (Rank 7)
    # 9. i in {2}, j in {0,1}, k in {0,1}: 1x2x2 -> Schoolbook (Rank 4)
    # 10. i in {2}, j in {0,1}, k in {2,3}: 1x2x2 -> Schoolbook (Rank 4)
    # 11. i in {2}, j in {0,1}, k in {4,5}: 1x2x2 -> Schoolbook (Rank 4)
    # 12. i in {2}, j in {0,1}, k in {6,7}: 1x2x2 -> Schoolbook (Rank 4)
    # 13. i in {2}, j in {2,3}, k in {0,1}: 1x2x2 -> Schoolbook (Rank 4)
    # 14. i in {2}, j in {2,3}, k in {2,3}: 1x2x2 -> Schoolbook (Rank 4)
    # 15. i in {2}, j in {2,3}, k in {4,5}: 1x2x2 -> Schoolbook (Rank 4)
    # 16. i in {2}, j in {2,3}, k in {6,7}: 1x2x2 -> Schoolbook (Rank 4)
    
    # Total Rank = 8 * 7 + 8 * 4 = 56 + 32 = 88.
    
    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        
    # Derive Strassen coefficients for 2x2x2 tensor
    # We want to find U, V, W of shape (4, 7), (4, 7), (4, 7) such that
    # sum_c U[i,j,c] V[j,k,c] W[k,i,c] = delta_{i,j,k} (where delta is 1 if i,j,k match the multiplication tensor)
    # Actually, the tensor T for 2x2 matrix mult is defined by:
    # T[i,j,k] = 1 if i,j,k are indices of A, B, C such that C[i,k] = sum_j A[i,j]B[j,k]
    # So T[i,j,k] is 1 if the indices are valid. Wait, the tensor T is the structure tensor of matrix multiplication.
    # T_{i,j,k} = 1 if we are multiplying A_{i,j} and B_{j,k} to get C_{i,k}.
    # In the decomposition, we sum over c: U_{i,j}^{(c)} V_{j,k}^{(c)} W_{i,k}^{(c)} = T_{i,j,k}.
    # Note: W index is (i,k) in standard notation, but the task uses W[k*n+i, r].
    # Let's stick to the task's indexing:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # For a 2x2x2 block at offset (i0, j0, k0):
    # Local indices i, j, k in {0, 1}
    # Global indices: I = i0+i, J = j0+j, K = k0+k
    # U[I, J] corresponds to U[i, j] in local
    # V[J, K] corresponds to V[j, k] in local
    # W[K, I] corresponds to W[k, i] in local
    
    # We solve for the 2x2x2 decomposition.
    # Variables: U (4x7), V (4x7), W (4x7)
    # Equations: T[i,j,k] = 1 for all 8 (i,j,k) in {0,1}^3.
    # This is a system of 8 nonlinear equations.
    # However, we can use the known Strassen structure or solve it via optimization.
    # Given the small size, we can try to find a solution using a simple iterative method or known coefficients.
    
    # Let's use the known Strassen coefficients derived correctly.
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e+f)
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c+d)f
    # M7 = (a+b)g
    
    # C00 = M1 - M5 + M7 + M4
    # C01 = M3 + M5
    # C10 = M6 - M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's verify this set one more time carefully.
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A @ B
    # C00 = ae + bg
    # C01 = af + bh
    # C10 = ce + dg
    # C11 = cf + dh
    
    # My previous check failed. Let's try to find a valid set online or derive it.
    # Actually, I can just use the schoolbook for the 2x2 blocks if I can't get Strassen right immediately.
    # Rank 88 is the target. If I use schoolbook for 2x2, rank is 8*8 + 8*4 = 64 + 32 = 96. No gain.
    # So I MUST get Strassen right.
    
    # Let's try a different set of Strassen coefficients.
    # From a reliable source (e.g., Wikipedia "Strassen algorithm"):
    # M1 = (a + d)(e + h)
    # M2 = (c + d)(e + f)
    # M3 = a(f - h)
    # M4 = d(g - e)
    # M5 = (a + b)h
    # M6 = (c + d)f
    # M7 = (a + b)g
    
    # C11 = M1 - M2 + M3 + M6
    # C10 = M6 - M4
    # C01 = M5 + M3
    # C00 = M1 - M5 + M7 + M4
    
    # Let's re-test C00 with A=I, B=I.
    # a=1, b=0, c=0, d=1
    # e=1, f=0, g=0, h=1
    # M1 = (1+1)(1+1) = 4
    # M2 = (0+1)(1+0) = 1
    # M3 = 1(0-1) = -1
    # M4 = 1(0-1) = -1
    # M5 = (1+0)1 = 1
    # M6 = (0+1)0 = 0
    # M7 = (1+0)0 = 0
    # C00 = 4 - 1 + 0 - 1 = 2. (Expected 1)
    # C01 = 1 - 1 = 0. (Expected 0)
    # C10 = 0 - (-1) = 1. (Expected 0)
    # C11 = 4 - 1 - 1 + 0 = 2. (Expected 1)
    
    # The result is [[2, 0], [1, 2]] instead of [[1, 0], [0, 1]].
    # It seems this specific set of M's is not the standard Strassen set or the reconstruction is wrong.
    
    # Let's try the set from the original Strassen paper or a different source.
    # M1 = (a + d)(e + h)
    # M2 = (c + d)(e + f)
    # M3 = a(f - h)
    # M4 = d(g - e)
    # M5 = (a + b)h
    # M6 = (c + d)f
    # M7 = (a + b)g
    
    # Maybe the reconstruction is:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 - M4
    # C11 = M1 - M2 + M3 + M6
    
    # Test C00: 4 + (-1) - 1 + 0 = 2. Still 2.
    
    # It seems I am struggling with the specific coefficients.
    # Let's try to solve for the coefficients numerically.
    # We have 7 multiplications. We can represent them as 7 rank-1 tensors.
    # We want to find U, V, W such that sum U V W = T.
    
    # I will implement a simple randomized search for the 2x2x2 decomposition.
    # If it fails, I will fall back to schoolbook for the whole thing (Rank 96).
    
    def find_2x2x2_decomp():
        # Randomized search for U, V, W of shape (4, 7)
        # T is 2x2x2 tensor with 1s at (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)
        # Wait, T[i,j,k] = 1 if it's a valid multiplication term.
        # For 2x2 matrix mult, all 8 combinations (i,j,k) in {0,1}^3 are valid terms.
        # So T is all ones.
        # We need sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k.
        
        # This is equivalent to finding a rank-7 decomposition of the all-ones 2x2x2 tensor?
        # No, the tensor is not all ones.
        # T[i,j,k] is the coefficient of A[i,j] B[j,k] in the product.
        # For matrix multiplication, T[i,j,k] = 1 for all i,j,k.
        # So yes, we are decomposing the all-ones tensor of size 2x2x2.
        
        # But wait, if T is all ones, then U=1, V=1, W=1 gives rank 1.
        # sum 1*1*1 = 1.
        # So the rank of the all-ones tensor is 1.
        # This implies that if we just use U=1, V=1, W=1, we get the sum of all products.
        # But we need the specific products for each (i,k) output.
        # The tensor T is not just "all ones". It encodes which input indices map to which output index.
        
        # Let's re-read the tensor definition.
        # T[i,j,k] = 1 if the term A[i,j]B[j,k] contributes to C[i,k].
        # In the standard matrix multiplication tensor, the indices are (i,j) for A, (j,k) for B, (i,k) for C.
        # The tensor T is defined on the space of A, B, C.
        # T_{i,j,k} is 1 if the indices are consistent.
        # For 2x2x2, all 8 indices (i,j,k) in {0,1}^3 are consistent.
        # So T is indeed the all-ones tensor.
        
        # Why is Strassen rank 7 then?
        # Because we are not decomposing the tensor T itself, but the *multiplication map*.
        # The tensor decomposition of the matrix multiplication tensor M_{n,n,n} has rank r(n).
        # For n=2, r(2) <= 7.
        # The tensor M_{2,2,2} has 8 non-zero entries, all 1.
        # If T is all ones, its rank is 1.
        # There is a confusion here.
        
        # The tensor T is not just "1". It is a trilinear form.
        # T(A,B,C) = sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} C_{i,k}.
        # For matrix multiplication, we want T(A,B,C) = trace(C^T (A B)) = sum_{i,k} C_{i,k} (A B)_{i,k} = sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k}.
        # So T_{i,j,k} = 1 for all i,j,k.
        
        # If T is the all-ones tensor, why is its rank not 1?
        # Because the rank of a tensor is the minimum number of rank-1 tensors that sum to it.
        # Rank-1 tensor is u v w.
        # Sum u v w = 1.
        # If u=v=w=1, sum is 1.
        # So the rank of the all-ones 2x2x2 tensor is 1.
        
        # This contradicts the fact that Strassen uses 7 multiplications.
        
        # Ah, the "multiplications" in Strassen are not just scalar multiplications of 1.
        # They are multiplications of *linear combinations* of the entries.
        # The tensor decomposition represents the multiplication as:
        # (A B)_{i,k} = sum_c (sum_j U_{i,j,c} A_{i,j}) (sum_l V_{j,l,c} B_{j,l}) W_{i,k,c} ?
        # No.
        
        # The standard tensor decomposition of matrix multiplication is:
        # T = sum_{c=1}^R u_c \otimes v_c \otimes w_c
        # where u_c is in A^*, v_c in B^*, w_c in C^*.
        # T(A,B) = sum_c (u_c . A) (v_c . B) w_c.
        # Here u_c, v_c are vectors of coefficients for A and B.
        # w_c is a vector of coefficients for C.
        
        # For the all-ones tensor, can we write it as sum of rank-1 tensors?
        # Yes, 1*1*1 = 1.
        # So why does Strassen use 7?
        
        # Because the "all-ones tensor" is not the correct tensor for matrix multiplication in this context?
        # Or maybe I am confusing the tensor with the map.
        
        # Let's check the rank of the 2x2 matrix multiplication tensor.
        # It is known to be 7.
        # The tensor is T_{i,j,k} = delta_{i,j,k}? No.
        # The tensor is defined by the bilinear map.
        # The entries of the tensor are the coefficients of the monomials in the expansion.
        # T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} appears in the product.
        # For 2x2, all 8 terms appear.
        # So T is all ones.
        
        # If T is all ones, its rank is 1.
        # This implies that 2x2 matrix multiplication can be done with 1 multiplication?
        # Clearly false.
        
        # The error is in the definition of the tensor rank.
        # The tensor rank of the matrix multiplication tensor M_{n,n,n} is r(n).
        # For n=2, r(2) = 7.
        # The tensor M_{2,2,2} has entries T_{i,j,k}.
        # T_{i,j,k} = 1 if the indices (i,j) in A, (j,k) in B, (i,k) in C are consistent.
        # For 2x2, all 8 are consistent.
        # So T is all ones.
        # But the rank of the all-ones 2x2x2 tensor is 1.
        
        # Why the discrepancy?
        # Because the tensor M_{n,n,n} is not the all-ones tensor?
        # Let's check n=1. T_{0,0,0} = 1. Rank 1.
        # Let's check n=2.
        # Maybe the indices are not all 1?
        # A_{0,0} B_{0,0} -> C_{0,0}
        # A_{0,0} B_{0,1} -> C_{0,1}
        # A_{0,1} B_{1,0} -> C_{0,0}
        # A_{0,1} B_{1,1} -> C_{0,1}
        # A_{1,0} B_{0,0} -> C_{1,0}
        # A_{1,0} B_{0,1} -> C_{1,1}
        # A_{1,1} B_{1,0} -> C_{1,0}
        # A_{1,1} B_{1,1} -> C_{1,1}
        
        # The tensor T is a 3-tensor of size 4x4x4? No.
        # A has 4 entries. B has 4 entries. C has 4 entries.
        # T is a 4x4x4 tensor?
        # T_{(i,j), (j,k), (i,k)} = 1.
        # There are 8 non-zero entries.
        # The rest are 0.
        
        # Ah! The tensor is NOT all ones. It is sparse.
        # The indices (i,j), (j,k), (i,k) are not independent.
        # For a given (i,j) and (j,k), the output is (i,k).
        # So T is defined on the space of A, B, C.
        # The non-zero entries are at positions where the indices match.
        # There are 8 such positions.
        # The other 64-8=56 positions are 0.
        
        # So we are decomposing a sparse tensor with 8 non-zero entries.
        # This tensor is NOT the all-ones tensor.
        
        # Okay, now I can solve for the Strassen coefficients.
        # I will use the schoolbook for the 2x2 blocks if I can't find the Strassen coefficients quickly.
        # But I know Strassen exists.
        
        # Let's try to find the coefficients by solving the linear system for the linear forms.
        # This is complex.
        
        # I will stick to the schoolbook decomposition for the whole matrix to ensure correctness.
        # Rank 96.
        
        return None

    # Since I cannot guarantee a correct Strassen implementation in the time remaining,
    # and the current program already returns schoolbook (96),
    # I will return the schoolbook decomposition.
    
    return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
