import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # 130s deadline, leaving 20s margin
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find a lower rank decomposition using Strassen-style block decomposition
    # Split dimensions into blocks and use recursive Strassen-like formulas
    
    # For 2x5 * 5x5, we can try various block decompositions
    # Let's try to find a decomposition with rank < 50
    
    # First, let's try a direct approach using known results
    # The rank of matrix multiplication is at most n*m*p and at least max(n*m, m*p, p*n)
    # For 2x5*5x5: lower bound is max(10, 25, 10) = 25
    
    # Try to construct a rank-25 or lower decomposition
    # This is equivalent to finding if the tensor has rank <= 25
    
    # Let's use a randomized search approach with exact arithmetic
    # We'll try to find vectors u, v, w such that sum_c u_c[i]v_c[j]w_c[k] = T[i,j,k]
    
    # The tensor T[i,j,k] for matrix multiplication 2x5*5x5:
    # C[k,i] = sum_j A[i,j] B[j,k]
    # So the tensor is T[i,j,k] = 1 for all valid i,j,k (since every term contributes)
    # Wait, this is not quite right. The tensor is actually:
    # T is a 3rd order tensor where T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[k,i]
    # This coefficient is 1 for all i,j,k.
    
    # So we need: sum_c u_c[i] * v_c[j] * w_c[k] = 1 for all i in [0,2), j in [0,5), k in [0,5)
    
    # This is equivalent to finding a rank-1 decomposition of the all-ones tensor of size 2x5x5
    # But wait, that's not correct either. The issue is that the indices are coupled.
    
    # Actually, the correct formulation is:
    # We have a bilinear map from (A, B) to C where C[k,i] = sum_j A[i,j] B[j,k]
    # This can be written as: C = sum_c (u_c^T A) (v_c^T B) w_c
    # where u_c is a vector of length n*m, v_c is a vector of length m*p, w_c is a vector of length p*n
    
    # More precisely: C[k,i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over j
    # No, that's not right either.
    
    # Let me re-derive this carefully.
    # We want: C[k,i] = sum_j A[i,j] B[j,k]
    # We propose: C[k,i] = sum_c (sum_j U[i*m+j, c] A[i,j]) * (sum_j V[j*p+k, c] B[j,k]) * W[k*n+i, c]
    # Wait, this doesn't make sense dimensionally.
    
    # The correct form is:
    # C[k,i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # But this sums over j implicitly, which is wrong.
    
    # Actually, the standard formulation is:
    # We write the multiplication as: C = sum_c (U_c * A) (V_c * B) W_c
    # where U_c is n x m, V_c is m x p, W_c is p x n
    # But this is not a rank-1 decomposition in the tensor sense.
    
    # The tensor rank decomposition is:
    # T = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c is in R^{n*m}, v_c is in R^{m*p}, w_c is in R^{p*n}
    # and T[a, b, c] = sum_d U[a, d] V[b, d] W[c, d]
    
    # For matrix multiplication, the tensor T is defined as:
    # T[(i,j), (j',k), (k',i')] = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
    # Wait, no. The tensor is:
    # T[(i,j), (j',k), (k',i')] = 1 if j=j' and k=k' and i=i', else 0
    # But this would give T as a diagonal tensor, which has rank n*m*p.
    
    # I think I'm confusing myself. Let me look at this differently.
    
    # The matrix multiplication tensor is:
    # T[i, j, k] where i indexes A (size n*m), j indexes B (size m*p), k indexes C (size p*n)
    # T[a, b, c] = 1 if the indices are consistent with matrix multiplication, else 0
    
    # Specifically, if a = i*m + j (i in [0,n), j in [0,m)),
    # b = j'*p + k (j' in [0,m), k in [0,p)),
    # c = k'*n + i' (k' in [0,p), i' in [0,n)),
    # then T[a, b, c] = 1 if j=j' and k=k' and i=i', else 0.
    
    # So T is a 3rd order tensor of size (n*m) x (m*p) x (p*n)
    # with T[a,b,c] = 1 if the indices match, else 0.
    
    # This is equivalent to a permutation tensor or a structured tensor.
    
    # For the schoolbook decomposition, we have n*m*p terms, each being a rank-1 tensor
    # that picks out one specific (i,j,k) combination.
    
    # To reduce the rank, we need to find fewer rank-1 tensors that sum to T.
    
    # This is a hard problem. Let's try a different approach.
    
    # Let's use the fact that we can group terms.
    # For example, if we fix i and vary j and k, we get m*p terms.
    # We can try to combine these using linearity.
    
    # Actually, let's try a randomized search with exact arithmetic.
    # We'll initialize U, V, W with random small integers and try to optimize.
    
    # But first, let's check if there's a known result for 2x5*5x5.
    # The rank of 2x5*5x5 matrix multiplication is known to be at most 50 (schoolbook).
    # It might be possible to do better.
    
    # Let's try to find a decomposition with rank 40 or so.
    
    # Approach: Use a greedy algorithm to build the decomposition.
    # Start with T, and at each step, find the best rank-1 tensor to subtract.
    
    # This is computationally expensive, but let's try.
    
    # First, represent T as a 3D array.
    T = np.zeros((n*m, m*p, p*n), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Now, try to find a low-rank decomposition.
    # We'll use a simple greedy approach.
    
    def try_decomposition(max_rank):
        """Try to find a decomposition with at most max_rank terms."""
        T_remaining = T.copy()
        U_list = []
        V_list = []
        W_list = []
        
        for _ in range(max_rank):
            if time.time() > deadline:
                return None
            
            # Find the best rank-1 tensor to subtract
            # We'll try random candidates and pick the one that reduces the L1 norm the most
            best_u = None
            best_v = None
            best_w = None
            best_score = -1
            
            # Try a few random candidates
            for _ in range(50):
                # Generate random sparse vectors
                u = np.zeros(n*m, dtype=np.int8)
                v = np.zeros(m*p, dtype=np.int8)
                w = np.zeros(p*n, dtype=np.int8)
                
                # Set a few random entries to 1 or -1
                for _ in range(3):
                    idx = np.random.randint(0, n*m)
                    u[idx] = np.random.choice([-1, 1])
                for _ in range(3):
                    idx = np.random.randint(0, m*p)
                    v[idx] = np.random.choice([-1, 1])
                for _ in range(3):
                    idx = np.random.randint(0, p*n)
                    w[idx] = np.random.choice([-1, 1])
                
                # Compute the rank-1 tensor
                tensor = np.einsum('a,b,c->abc', u, v, w)
                
                # Compute the improvement
                # We want to minimize ||T_remaining - alpha * tensor||_1 for some alpha
                # For simplicity, let's just try alpha = 1 and alpha = -1
                for alpha in [1, -1]:
                    T_new = T_remaining - alpha * tensor
                    score = np.sum(np.abs(T_new)) - np.sum(np.abs(T_remaining))
                    if score < best_score:
                        best_score = score
                        best_u = u
                        best_v = v
                        best_w = w
                        best_alpha = alpha
            
            if best_u is None or best_score >= 0:
                return None
            
            # Add this term
            U_list.append(best_u)
            V_list.append(best_v)
            W_list.append(best_alpha * best_w)
            T_remaining = T_remaining - best_alpha * np.einsum('a,b,c->abc', best_u, best_v, best_w)
            
            # Check if we've reached zero
            if np.all(T_remaining == 0):
                break
        
        if np.all(T_remaining == 0):
            # Construct the result
            rank = len(U_list)
            U = np.zeros((n*m, rank), dtype=np.int8)
            V = np.zeros((m*p, rank), dtype=np.int8)
            W = np.zeros((p*n, rank), dtype=np.int8)
            
            for c in range(rank):
                U[:, c] = U_list[c]
                V[:, c] = V_list[c]
                W[:, c] = W_list[c]
            
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None
    
    # Try different ranks
    for target_rank in [25, 30, 35, 40, 45, 50]:
        if time.time() > deadline:
            break
        
        result = try_decomposition(target_rank)
        if result is not None:
            # Verify the decomposition
            U = np.array(result["U"])
            V = np.array(result["V"])
            W = np.array(result["W"])
            
            # Check: sum_c U[a,c] V[b,c] W[c_idx,c] should equal T[a,b,c_idx]
            # Wait, the indices are: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
            # So we need: sum_c U[a,c] * V[b,c] * W[c_idx,c] = T[a,b,c_idx]
            
            T_reconstructed = np.einsum('ac,bc,dc->abd', U, V, W)
            
            if np.array_equal(T_reconstructed, T):
                return result
    
    # If we couldn't find a better decomposition, return schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
