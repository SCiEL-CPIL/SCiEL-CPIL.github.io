import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Check if we have time to search for a better rank
    # We will try to find a decomposition of rank < rank_schoolbook
    # For 3x5x6, the rank is likely around 80-85.
    # We use a random search with a tight time limit.
    
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s margin
    
    best_rank = rank_schoolbook
    best_decomp = None
    
    # Generate schoolbook decomposition as fallback
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    best_decomp = (U_school, V_school, W_school)
    
    # Try to find a better rank using random projections
    # This is a heuristic. We look for a set of rank-1 tensors that sum to the multiplication tensor.
    # The multiplication tensor T has T[i,j,k] = 1 if the indices are consistent.
    # Actually, T is the tensor of the bilinear form.
    # T_{i,j,k} = 1 for all i,j,k? No.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The tensor is sum_{i,j,k} e_i \otimes e_j \otimes e_k.
    # Wait, if T_{i,j,k} = 1, then the rank is n*m*p.
    # But we know rank(M_{2,2,2}) = 7 < 8.
    # This means the tensor is NOT sum of all e_i \otimes e_j \otimes e_k.
    # Let's re-read the tensor definition.
    # The tensor T is such that <T, A \otimes B> = tr(AB^T) or similar?
    # No, the output is C = AB.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # This is a bilinear map.
    # The tensor T is defined by T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # For a fixed i,k, the terms are A_{i,0}B_{0,k}, ..., A_{i,m-1}B_{m-1,k}.
    # So T_{i,j,k} = 1 if j is the contraction index.
    # But j is always the contraction index in the sum.
    # So T_{i,j,k} = 1 for all i,j,k?
    # If T_{i,j,k} = 1, then C_{i,k} = sum_{j} 1 * A_{i,j} B_{j,k}.
    # Yes, that's correct.
    # So why is the rank of M_{2,2,2} 7?
    # Because the tensor is NOT the sum of all rank-1 tensors e_i \otimes e_j \otimes e_k.
    # The rank of a tensor is the minimum number of rank-1 tensors that sum to it.
    # If T = sum_{i,j,k} e_i \otimes e_j \otimes e_k, then the rank is n*m*p.
    # But for M_{2,2,2}, T = e_0 \otimes e_0 \otimes e_0 + e_0 \otimes e_0 \otimes e_1 + e_0 \otimes e_1 \otimes e_0 + ...
    # Wait, the tensor for M_{2,2,2} is:
    # C_{0,0} = A_{0,0}B_{0,0} + A_{0,1}B_{1,0}
    # C_{0,1} = A_{0,0}B_{0,1} + A_{0,1}B_{1,1}
    # C_{1,0} = A_{1,0}B_{0,0} + A_{1,1}B_{1,0}
    # C_{1,1} = A_{1,0}B_{0,1} + A_{1,1}B_{1,1}
    # The tensor T has components T_{i,j,k} where i is row of C, j is index of A/B, k is col of C.
    # T_{0,0,0} = 1 (A00 B00 -> C00)
    # T_{0,1,0} = 1 (A01 B10 -> C00)
    # T_{0,0,1} = 1 (A00 B01 -> C01)
    # T_{0,1,1} = 1 (A01 B11 -> C01)
    # T_{1,0,0} = 1 (A10 B00 -> C10)
    # T_{1,1,0} = 1 (A11 B10 -> C10)
    # T_{1,0,1} = 1 (A10 B01 -> C11)
    # T_{1,1,1} = 1 (A11 B11 -> C11)
    # All 8 components are 1.
    # So T = sum_{i,j,k} e_i \otimes e_j \otimes e_k.
    # If this is the case, the rank is 8.
    # But Strassen says rank 7.
    # Contradiction.
    # Ah, the tensor is not just the sum of basis tensors.
    # The rank of the tensor T is the minimum r such that T = sum_{c=1}^r u_c \otimes v_c \otimes w_c.
    # If T = sum_{i,j,k} e_i \otimes e_j \otimes e_k, then it is a sum of 8 rank-1 tensors.
    # Can it be written as a sum of 7?
    # Yes, if there is a linear dependence among the rank-1 tensors that sums to 0.
    # But e_i \otimes e_j \otimes e_k are linearly independent.
    # So the rank is 8?
    # No, Strassen's algorithm shows that matrix multiplication can be done with 7 multiplications.
    # This implies the tensor rank is 7.
    # How?
    # The tensor T is NOT sum_{i,j,k} e_i \otimes e_j \otimes e_k.
    # Let's check the definition of tensor rank for matrix multiplication.
    # The multiplication map is \mu: A \otimes B \to C.
    # The associated tensor T is in A^* \otimes B^* \otimes C^*.
    # T(a, b) = c^T (a^T b) ? No.
    # The tensor T is defined by T_{i,j,k} such that C_{i,k} = sum_{j} T_{i,j,k} A_{i,j} B_{j,k}.
    # In this case, T_{i,j,k} = 1 for all i,j,k.
    # So T = sum_{i,j,k} e_i \otimes e_j \otimes e_k.
    # If the rank is 7, then sum_{i,j,k} e_i \otimes e_j \otimes e_k = sum_{c=1}^7 u_c \otimes v_c \otimes w_c.
    # This is possible if the u_c, v_c, w_c are not the basis vectors.
    # Yes, they can be linear combinations.
    # So we need to find 7 vectors u_c, v_c, w_c such that the sum equals T.
    
    # I will use a random search to find such a decomposition for rank < 90.
    # Given the time, I will try a few ranks.
    
    for rank in range(rank_schoolbook - 1, 80, -1):
        if time.time() > deadline:
            break
        found = False
        for _ in range(100): # 100 trials per rank
            if time.time() > deadline:
                break
            # Random initialization
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Check if the decomposition is valid
            # We need sum_c U[:,c] V[:,c]^T W[:,c] to match the tensor structure.
            # Actually, the condition is:
            # For all i, k: C_{i,k} = sum_j A_{i,j} B_{j,k} = sum_c W[k*n+i, c] (sum_j U[i*m+j, c] A_{i,j}) (sum_l V[l*p+k, c] B_{l,k})?
            # No, the bilinear form is:
            # C_{i,k} = sum_c (sum_j U[i*m+j, c] A_{i,j}) (sum_l V[l*p+k, c] B_{l,k}) * W[k*n+i, c]?
            # This is not linear in A and B if W is multiplied outside.
            # The standard form is:
            # C_{i,k} = sum_c U_c(A) V_c(B) W_c
            # Where U_c(A) is linear in A, V_c(B) is linear in B.
            # U_c(A) = sum_j U[i*m+j, c] A_{i,j}
            # V_c(B) = sum_l V[l*p+k, c] B_{l,k}
            # This is not correct because U depends on i and j.
            # The correct form is:
            # C_{i,k} = sum_c (sum_j U_{i,j,c} A_{i,j}) (sum_l V_{j,l,c} B_{j,l}) ? No.
            
            # Let's use the explicit tensor check.
            # T_{i,j,k} = sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c}
            # We want T_{i,j,k} = 1 for all i,j,k.
            
            # We can check this by computing the tensor and comparing to ones.
            # But this is O(n*m*p*rank) which is 90*rank.
            # For rank 89, this is 8010 operations. Very fast.
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(rank):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        if abs(val - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                # Round to integers or half-integers if possible?
                # The problem asks for integer/half-integer entries.
                # Random floats won't satisfy this.
                # So this approach is flawed for the exact integer requirement.
                pass
        
    # Since random floats won't give integer entries, and finding integer decompositions is hard,
    # I will return the schoolbook decomposition which has integer entries.
    
    U = [[int(x) for x in U_school[i]] for i in range(n*m)]
    V = [[int(x) for x in V_school[i]] for i in range(m*p)]
    W = [[int(x) for x in W_school[i]] for i in range(p*n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
