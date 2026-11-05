"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 72 is the schoolbook rank.
    # For 3x3 * 3x8, the tensor is sum_{i,j,k} e_i^A e_j^B e_k^C e_i^D e_k^E? No.
    # Indices:
    # A is n x m. B is m x p. C is n x p.
    # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i^A e_j^B e_k^C e_i^D e_k^E?
    # Wait, the factor matrices are U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # The tensor is in V_A^* x V_B^* x V_C^*?
    # Usually T = sum_r u_r v_r w_r.
    # U[r] corresponds to A indices (i,j).
    # V[r] corresponds to B indices (j,k).
    # W[r] corresponds to C indices (i,k).
    # The trace convention for W is W[k*n + i, r].
    # So W index is k*n + i.
    # The tensor decomposition is:
    # T_{i,j,k} = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # We need T_{i,j,k} = 1 if the multiplication E_{ij} * E_{jk} = E_{ik} is valid.
    # E_{ij} * E_{jk} = E_{ik}. This is always valid for any i, j, k.
    # So T_{i,j,k} = 1 for all i, j, k.
    # Wait, is that right?
    # Let's check:
    # (A*B)_{ik} = sum_j A_{ij} B_{jk}.
    # If A = E_{ij'} and B = E_{j'k'}, then (A*B)_{ik} = sum_{j''} A_{ij''} B_{j''k} = A_{ij'} B_{j'k} = delta_{j'', j'} delta_{j'', j'}? No.
    # A_{ij''} is 1 if i=i, j''=j'.
    # B_{j''k} is 1 if j''=j', k=k'.
    # So sum_{j''} A_{ij''} B_{j''k} = 1 * 1 if j''=j' and j''=j'.
    # So it is 1 if j'=j' and k=k'.
    # So (E_{ij'} E_{j'k'})_{ik} = delta_{k, k'}.
    # So E_{ij} E_{jk} = E_{ik}.
    # Yes, the tensor T_{i,j,k} is 1 for all i, j, k.
    # So the schoolbook decomposition is:
    # For each (i,j,k), we have a rank-1 term.
    # U[i*m+j, r] = 1
    # V[j*p+k, r] = 1
    # W[k*n+i, r] = 1
    # This gives rank n*m*p = 3*3*8 = 72.
    
    # Is it possible to do better?
    # For 3x3x3, rank is 23 (Strassen).
    # For 3x3x4, rank is ?
    # For 3x3x8, we can use Strassen's algorithm for 3x3x3 and then multiply by the remaining 5 columns?
    # Strassen for 3x3x3:
    # A (3x3) * B (3x3) = C (3x3).
    # Rank 23.
    # We can split B into B1 (3x3) and B2 (3x5).
    # C = A*B1 + A*B2.
    # C1 = A*B1 (3x3) can be done with rank 23.
    # C2 = A*B2 (3x5).
    # B2 is 3x5.
    # We can split B2 into B21 (3x3) and B22 (3x2).
    # C21 = A*B21 (3x3) rank 23.
    # C22 = A*B22 (3x2).
    # B22 is 3x2.
    # C22 = A * [b1, b2].
    # This is just 2 matrix-vector multiplications? No, A is 3x3, b is 3x1.
    # A*b is a vector.
    # Rank of 3x3x1?
    # A (3x3) * b (3x1) = c (3x1).
    # T_{i,j,k} where k is only 0? No, p=1.
    # T_{i,j} = 1.
    # Rank of 3x3x1 tensor?
    # It's just the identity map on A? No.
    # c_i = sum_j A_{ij} b_j.
    # This is a linear map from A (9-dim) to c (3-dim) for each b?
    # No, it's a bilinear map.
    # The tensor is sum_{i,j} e_i^A e_j^B e_i^C.
    # This is the identity tensor on 3x3 matrices? No.
    # It's the tensor that contracts the second index of A and the first index of B.
    # This is the "matrix-vector multiplication" tensor.
    # Rank of M_{3,3,1}?
    # M_{n,m,1} is the tensor for multiplying an n x m matrix by an m x 1 vector.
    # This is just the matrix A times vector v.
    # The rank is n*m?
    # T_{i,j} = 1.
    # sum_r U[i,j,r] V[j,r] W[i,r] = delta_{i,i}? No.
    # c_i = sum_j A_{ij} v_j.
    # So T_{i,j} = 1.
    # We need sum_r U_{ij,r} V_{j,r} W_{i,r} = 1 for all i,j?
    # No, the tensor is T = sum_{i,j} e_i^C e_{ij}^A e_j^B.
    # So T_{i,j} = 1.
    # If we use rank 1 terms:
    # U_{ij,r} V_{j,r} W_{i,r}.
    # If we set U_{ij,r} = 1 for all i,j, V_{j,r} = 1, W_{i,r} = 1?
    # Then sum_r 1*1*1 = rank.
    # We need sum_r ... = 1.
    # So we can use 1 term?
    # U_{ij,0} = 1, V_{j,0} = 1, W_{i,0} = 1.
    # Then sum_0 1*1*1 = 1.
    # So rank of M_{3,3,1} is 1?
    # Wait.
    # c_i = sum_j A_{ij} v_j.
    # This is linear in A and v.
    # The tensor is T = sum_{i,j} e_i^C \otimes e_{ij}^A \otimes e_j^B.
    # This is a sum of 9 terms.
    # Can it be decomposed into fewer?
    # If we use U_{ij,r}, V_{j,r}, W_{i,r}.
    # We need sum_r U_{ij,r} V_{j,r} W_{i,r} = 1 for all i,j.
    # Let U_{ij,r} = a_{i,r} b_{j,r}.
    # Then sum_r a_{i,r} b_{j,r} V_{j,r} W_{i,r} = 1.
    # Let V_{j,r} = c_{j,r}, W_{i,r} = d_{i,r}.
    # sum_r a_{i,r} b_{j,r} c_{j,r} d_{i,r} = 1.
    # This looks like it could be rank 1 if we choose a, b, c, d appropriately.
    # But a_{i,r} d_{i,r} depends on i, and b_{j,r} c_{j,r} depends on j.
    # Let x_{i,r} = a_{i,r} d_{i,r}, y_{j,r} = b_{j,r} c_{j,r}.
    # sum_r x_{i,r} y_{j,r} = 1.
    # This is possible with rank 1 if x_{i,0}=1, y_{j,0}=1.
    # So U_{ij,0} = a_{i,0} b_{j,0}.
    # V_{j,0} = c_{j,0}.
    # W_{i,0} = d_{i,0}.
    # If a_{i,0}=1, b_{j,0}=1, c_{j,0}=1, d_{i,0}=1.
    # Then U_{ij,0}=1, V_{j,0}=1, W_{i,0}=1.
    # So rank 1 is sufficient for M_{3,3,1}?
    # Let's check.
    # c_i = sum_j A_{ij} v_j.
    # If we use the decomposition:
    # c_i = sum_r U_{ij,r} V_{j,r} W_{i,r} A_{ij} v_j?
    # No, the bilinear form is sum_{i,j} T_{i,j} A_{ij} v_j.
    # If T_{i,j} = 1, then sum_{i,j} A_{ij} v_j.
    # This is just the matrix-vector product.
    # The rank of the tensor T = sum_{i,j} e_i^C e_{ij}^A e_j^B.
    # This tensor is the identity map from A to C? No.
    # It's the contraction map.
    # The rank of the contraction tensor for n x m matrix and m x 1 vector is n*m?
    # No, it's 1?
    # Let's verify with n=2, m=2.
    # T = e_0^C e_{00}^A e_0^B + e_0^C e_{01}^A e_1^B + e_1^C e_{10}^A e_0^B + e_1^C e_{11}^A e_1^B.
    # Can we write this as sum_r u_r v_r w_r?
    # If rank 1: u v w.
    # u_{ij} v_j w_i = 1 for all i,j.
    # This requires u_{ij} to be separable u_{ij} = a_i b_j.
    # Then a_i b_j v_j w_i = (a_i w_i) (b_j v_j) = 1.
    # So a_i w_i = 1/k, b_j v_j = k.
    # This is possible.
    # So rank 1 is sufficient for M_{n,m,1}.
    # Similarly for M_{1,m,p}?
    # So for 3x3x8, we can use Strassen for the first 3 columns (rank 23) and then for the remaining 5 columns, we can treat them as 5 separate M_{3,3,1} problems?
    # No, the 5 columns are part of the same B.
    # B is 3x8.
    # C = A * B.
    # C has 8 columns.
    # C_{i,k} = sum_j A_{ij} B_{jk}.
    # For k=0,1,2, we can use Strassen?
    # Strassen computes A * B1 where B1 is 3x3.
    # This gives C1 (3x3).
    # Then for k=3..7, we have C2 = A * B2 where B2 is 3x5.
    # C2_{i,k} = sum_j A_{ij} B_{j,k}.
    # This is 5 separate matrix-vector multiplications.
    # Each has rank 1?
    # If so, total rank = 23 + 5*1 = 28?
    # Wait, the tensor for C2 is sum_{i,j,k=3..7} e_i^C e_{ij}^A e_{j,k}^B.
    # This is sum_{k=3..7} (sum_{i,j} e_i^C e_{ij}^A e_{j,k}^B).
    # Each term is M_{3,3,1} tensor with B index fixed to k.
    # If rank of M_{3,3,1} is 1, then total rank is 23 + 5 = 28.
    # Is rank of M_{3,3,1} really 1?
    # Let's check the definition of rank.
    # The tensor T is in V_A^* x V_B^* x V_C^*.
    # T = sum_{i,j} e_i^C \otimes e_{ij}^A \otimes e_j^B.
    # We want to write T = sum_r u_r \otimes v_r \otimes w_r.
    # u_r in V_A^*, v_r in V_B^*, w_r in V_C^*.
    # u_r = sum_{i,j} a_{ij,r} e_{ij}^A.
    # v_r = sum_j b_{j,r} e_j^B.
    # w_r = sum_i c_{i,r} e_i^C.
    # Then sum_r u_r v_r w_r = sum_r (sum_{i,j} a_{ij,r} e_{ij}^A) (sum_j b_{j,r} e_j^B) (sum_i c_{i,r} e_i^C).
    # The coefficient of e_{ij}^A e_j^B e_i^C is sum_r a_{ij,r} b_{j,r} c_{i,r}.
    # We need this to be 1 for all i,j.
    # As shown before, this is possible with rank 1.
    # So R(3,3,1) = 1.
    # Then R(3,3,8) <= R(3,3,3) + 5*R(3,3,1) = 23 + 5 = 28.
    # Wait, is R(3,3,3) = 23?
    # Strassen's algorithm for 3x3x3 has rank 23.
    # Is it possible to do better for 3x3x8?
    # Maybe we can use a different decomposition.
    # What if we use Strassen for 4x4x4?
    # No, 3x3x8 is not a power of 2.
    # What about using the fact that 8 = 2^3?
    # Maybe we can use a tensor decomposition that exploits the structure.
    # However, 28 is much better than 72.
    # Let's try to construct the rank 28 decomposition.
    # We need the Strassen decomposition for 3x3x3.
    # And then 5 rank-1 terms for the remaining columns.
    
    # Strassen 3x3x3 decomposition:
    # A = [[a, b, c], [d, e, f], [g, h, i]]
    # B = [[j, k, l], [m, n, o], [p, q, r]]
    # C = A*B.
    # Strassen's algorithm for 3x3x3 uses 23 multiplications.
    # The 23 products are:
    # M1 = (a+e+i)(j+n+r)
    # M2 = (d+g)(j)
    # M3 = (a+b)(j+m)
    # M4 = (c+f)(l+o)
    # M5 = (a+d+e+h)(j+n)
    # M6 = (b+c+e+f)(n+o)
    # M7 = (c+f+e+h)(o+r)
    # M8 = (d+g+e+h)(m+n)
    # M9 = (g+i)(m+p)
    # M10 = (h+i)(n+q)
    # M11 = (a+b+e+f)(j+k)
    # M12 = (c+f+e+h)(l+o) -- Wait, M4 is (c+f)(l+o).
    # I need the exact 23 Strassen products for 3x3x3.
    # This is complex to implement correctly from memory.
    # Given the time constraints and the risk of error, and since the task asks for a "stronger algorithm" but also allows for "keep this program's approach", maybe I should stick to a safer improvement or a known simple decomposition.
    # However, 28 is a huge improvement over 72.
    # If I can't implement Strassen 3x3x3 correctly, I might fail.
    # Let's check if there is a simpler decomposition for 3x3x8.
    # What if I use rank 1 for all 8 columns?
    # Then rank = 8 * R(3,3,1) = 8 * 1 = 8?
    # No, the B matrix is shared.
    # The tensor is T = sum_{i,j,k} e_i^C e_{ij}^A e_{j,k}^B.
    # If I use rank 1 terms for each k, I need the A and B parts to be compatible.
    # For a fixed k, T_k = sum_{i,j} e_i^C e_{ij}^A e_{j,k}^B.
    # This is rank 1.
    # But the B part is e_{j,k}^B, which is different for each k.
    # So I can use 8 rank-1 terms, one for each k.
    # Term r=k:
    # U_{ij,r} = 1 for all i,j.
    # V_{j,r} = 1 if j is the row index? No, V_{j,r} corresponds to B_{j,k}.
    # For a fixed k, B_{j,k} is the element.
    # So V_{j,r} = delta_{k,r}.
    # W_{i,r} = 1 for all i.
    # Then sum_r U_{ij,r} V_{j,r} W_{i,r} = sum_r 1 * delta_{k,r} * 1 = 1 if k=r, else 0.
    # But we need the sum to be 1 for all i,j,k?
    # No, the tensor T_{i,j,k} is 1 for all i,j,k.
    # So for each (i,j,k), we need the sum to be 1.
    # If I use 8 terms, one for each k:
    # Term r (0..7):
    # U_{ij,r} = 1.
    # V_{j,r} = 1 if r == k? No, V is indexed by (j, r).
    # V_{j,r} is the coefficient for B_{j, r'}?
    # The factor matrix V has shape (m*p, rank).
    # V[j*p + k, r].
    # So for a fixed r, V[:, r] is a vector of length m*p.
    # If I set V[j*p + r, r] = 1 for all j, and 0 otherwise.
    # Then for a given (i,j,k), the term r contributes:
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    # If r=k, then V[j*p+k, k] = 1.
    # U[i*m+j, k] = 1.
    # W[k*n+i, k] = 1.
    # So the term is 1.
    # For r != k, V[j*p+k, r] = 0.
    # So the sum is 1.
    # This works!
    # So rank 8 is sufficient?
    # Let's check.
    # U has shape (9, 8).
    # V has shape (24, 8).
    # W has shape (24, 8).
    # For r in 0..7:
    # U[:, r] = 1.
    # V[:, r] = 0, except V[j*p+r, r] = 1 for j in 0..2.
    # W[:, r] = 1.
    # Then T_{i,j,k} = sum_{r=0}^7 U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
    # = sum_{r=0}^7 1 * V[j*p+k, r] * 1.
    # V[j*p+k, r] is 1 if r=k, else 0.
    # So sum is 1.
    # This is a valid decomposition with rank 8!
    # Is it that simple?
    # Let's verify the dimensions and indices.
    # U: (n*m, rank) = (9, 8).
    # V: (m*p, rank) = (24, 8).
    # W: (p*n, rank) = (24, 8).
    # W index is k*n + i.
    # For r=k, W[k*n+i, k] = 1.
    # Yes.
    # So the rank is 8.
    # This is much better than 72.
    # Let's implement this.

    rank = p  # 8
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for r in range(rank):
        # U[:, r] = 1
        U[:, r] = 1
        
        # V[j*p + r, r] = 1 for j in 0..m-1
        for j in range(m):
            V[j * p + r, r] = 1
            
        # W[k*n + i, r] = 1 for all i, where k=r
        for i in range(n):
            W[r * n + i, r] = 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
