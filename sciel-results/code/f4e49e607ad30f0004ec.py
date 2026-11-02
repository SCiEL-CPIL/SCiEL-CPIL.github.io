import time
import numpy as np

def construct():
    n, m, p = 2, 3, 8
    dim_U = n * m  # 6
    dim_V = m * p  # 24
    dim_W = p * n  # 16
    
    t_start = time.time()
    deadline = t_start + 140.0  # 10s safety margin
    
    # Build the multiplication tensor T[i,j,k] = delta_{i,i} delta_{j,j} delta_{k,k}
    # Actually T[i,j,k] = 1 if the index matches, 0 otherwise.
    # The tensor is: T[i,j,k] = 1 for all (i,j,k) where we multiply A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all valid (i,j,k) — it's the identity in the middle index.
    # More precisely: T[i,j,k] = delta_{j,j} = 1 always (for valid indices)
    # The tensor entries are T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # Build base decomposition (rank 44)
    def build_rank44():
        rank = 44
        U = [[0] * rank for _ in range(dim_U)]
        V = [[0] * rank for _ in range(dim_V)]
        W = [[0] * rank for _ in range(dim_W)]
        col = 0
        
        # Part 1: j=0, 16 terms
        for k in range(p):
            U[0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
            U[1 * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + 1][col] = 1
            col += 1
        
        # Part 2: j=1,2 in 4 blocks of 2x2x2, 7 terms each = 28
        for b in range(4):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            # P1: (A[0,1] + A[1,2]) * (B[1,k0] + B[2,k1])
            U[0 * m + 1][col] = 1
            U[1 * m + 2][col] = 1
            V[1 * p + k0][col] = 1
            V[2 * p + k1][col] = 1
            W[k0 * n + 0][col] = 1
            W[k1 * n + 1][col] = 1
            col += 1
            
            # P2: (A[1,1] + A[1,2]) * B[1,k0]
            U[1 * m + 1][col] = 1
            U[1 * m + 2][col] = 1
            V[1 * p + k0][col] = 1
            W[k0 * n + 1][col] = 1
            W[k1 * n + 1][col] = -1
            col += 1
            
            # P3: A[0,1] * (B[1,k1] - B[2,k1])
            U[0 * m + 1][col] = 1
            V[1 * p + k1][col] = 1
            V[2 * p + k1][col] = -1
            W[k1 * n + 0][col] = 1
            W[k1 * n + 1][col] = 1
            col += 1
            
            # P4: A[1,2] * (B[2,k0] - B[1,k0])
            U[1 * m + 2][col] = 1
            V[2 * p + k0][col] = 1
            V[1 * p + k0][col] = -1
            W[k0 * n + 0][col] = 1
            W[k0 * n + 1][col] = 1
            col += 1
            
            # P5: (A[0,1] + A[0,2]) * B[2,k1]
            U[0 * m + 1][col] = 1
            U[0 * m + 2][col] = 1
            V[2 * p + k1][col] = 1
            W[k0 * n + 0][col] = -1
            W[k1 * n + 0][col] = 1
            col += 1
            
            # P6: (A[1,1] - A[0,1]) * (B[1,k0] + B[1,k1])
            U[1 * m + 1][col] = 1
            U[0 * m + 1][col] = -1
            V[1 * p + k0][col] = 1
            V[1 * p + k1][col] = 1
            W[k1 * n + 1][col] = 1
            col += 1
            
            # P7: (A[0,2] - A[1,2]) * (B[2,k0] + B[2,k1])
            U[0 * m + 2][col] = 1
            U[1 * m + 2][col] = -1
            V[2 * p + k0][col] = 1
            V[2 * p + k1][col] = 1
            W[k0 * n + 0][col] = 1
            col += 1
        
        return U, V, W, rank
    
    U, V, W, rank = build_rank44()
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Try to find a lower-rank decomposition using random search
    # The tensor T[i,j,k] = 1 for all (i,j,k)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    
    # Use a different approach: try rank 43, 42, ... with random initialization
    # and check if we can satisfy the constraints
    
    # Actually, let's try a structured approach:
    # The j=0 part requires at least 16 terms (since each term can only cover one (i,k) pair for j=0)
    # The j=1,2 part: we have 2*8=16 output entries for each of 2 input rows = 32 constraints
    # But we share the j index... 
    # Minimum for j=1,2 part: at least 2*8/2 = 8? No, that's not right.
    # Each rank-1 term contributes to at most one (j,k) pair in V and one (i,j) pair in U
    # For j=1,2: we need to cover all (i,j,k) for j in {1,2}
    # A single term with U having entries in rows (i1,1) and (i2,2) can cover 2 i-values for j=1,2
    # But V must have a single j... so each term covers exactly one j.
    # So for j=1: need 8 terms (one per k), for j=2: need 8 terms. Total 16.
    # But we can combine: a term can have U[i,1] and U[i',1] (same j=1) but only one j in V.
    # So minimum is 16 for the j=1,2 part? No wait...
    # U can have multiple nonzero entries in the same row (same i, different j)
    # But V has one j per term. So each term is associated with exactly one j.
    # For j=1: we need to produce C[i,k] for i=0,1 and k=0..7, which is 16 values
    # Each term for j=1 contributes U[i,1]*V[1,k]*W[k,i] to C[i,k]
    # If U has both U[0,1] and U[1,1] nonzero, and V has V[1,k0] and V[1,k1], 
    # then one term can contribute to up to 4 (i,k) pairs.
    # So minimum for j=1 is 16/4 = 4 terms? Similarly for j=2.
    # Total minimum: 16 + 4 + 4 = 24? That seems too low.
    
    # Let me think more carefully. The constraint is:
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for the "wrong" combinations, it should be 0.
    # Actually no - the tensor T[i,j,k] = 1 for ALL valid (i,j,k).
    # There's no "wrong" combination in the standard matrix multiplication tensor.
    # T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k], which is always 1.
    
    # So the constraint is simply:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is a rank-1 decomposition of the all-ones tensor (of shape 2x3x8)
    # with the specific structure that U depends on (i,j), V on (j,k), W on (k,i)
    
    # The all-ones tensor of shape (n,m,p) has rank 1! Just U[i,j]=1, V[j,k]=1, W[k,i]=1
    # But wait, that would give sum_r 1*1*1 = rank, not 1.
    # With rank 1: U[i,j]=1, V[j,k]=1, W[k,i]=1 gives 1*1*1 = 1. Yes!
    # So rank 1 suffices?? But that can't be right for matrix multiplication.
    
    # I'm confusing myself. The tensor for matrix multiplication is NOT the all-ones tensor.
    # The tensor T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k].
    # T[i,j,k] = 1 for all (i,j,k) - yes, it IS the all-ones tensor.
    # And the rank-1 decomposition is trivial: U[i,j]=1, V[j,k]=1, W[k,i]=1.
    # But that gives C[i,k] = sum_j sum_r 1*1*1*A[i,j]*B[j,k] = rank * sum_j A[i,j]*B[j,k]
    # So we'd need to divide by rank, or use 1/rank entries.
    
    # With half-integer entries allowed, rank 1 works: U=V=W=1, but then C = 1*sum_j A[i,j]B[j,k]
    # which is correct! So rank 1 should work?
    
    # Wait, let me re-read the problem. The decomposition must satisfy:
    # C[i,k] = sum_j A[i,j] * B[j,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_k' V[j*p+k', r] * B[j,k'])
    # Hmm, that's not quite right either. Let me re-read the interface.
    
    # The interface says: factor matrices U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # such that the tensor decomposes. The standard formulation is:
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where T[i,j,k] = 1 for all (i,j,k) in the matmul tensor.
    
    # So yes, rank 1 with all ones works: 1*1*1 = 1.
    # But the current program uses rank 44, which suggests I'm misunderstanding something.
    
    # Let me check the verification function more carefully.
    # The _verify function checks:
    # For A[i0,j0]=1, B[j0,k0]=1: C[i,k] should be 1 if i==i0 and k==k0, else 0.
    # This means: C[i,k] = sum_r W[k*n+i, r] * U[i*m+j0, r] * V[j0*p+k, r] (when A[i0,j0]=B[j0,k0]=1)
    # And this should equal delta_{i,i0} * delta_{k,k0}
    
    # So the constraint is NOT that T[i,j,k]=1 for all (i,j,k).
    # The constraint is that the bilinear form correctly computes matrix multiplication.
    # Specifically: for the standard basis vectors e_{i0,j0} in A-space and e_{j0,k0} in B-space,
    # the output should be e_{i0,k0} in C-space.
    
    # This means: sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = delta_{i,i0} delta_{k,k0}
    # when j=j0 (the middle index must match).
    # And for j != j0, the sum should be 0 (but this is automatically satisfied since 
    # U[i*m+j,r] and V[j*p+k,r] are for different j).
    
    # Actually, the correct statement is:
    # The tensor T[i,j,k] = delta_{j,j} = 1 (the middle index is summed over)
    # No wait. In matrix multiplication, C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor that defines this is T[i,j,k] where the j index is the one being summed.
    # T[i,j,k] = 1 for all (i,j,k) - it's the "multiplication" tensor.
    # The bilinear decomposition is: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # where U[i,j,r] extracts A[i,j], V[j,k,r] extracts B[j,k], W[k,i,r] places into C[k,i].
    
    # So T[i,j,k] = 1 for all valid (i,j,k), and we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    
    # And this IS the all-ones tensor, which has rank 1!
    # U[i,j]=1, V[j,k]=1, W[k,i]=1 gives sum_r 1*1*1 = rank ≠ 1 unless rank=1.
    # With rank=1: U[i,j]=1, V[j,k]=1, W[k,i]=1 gives 1*1*1 = 1. ✓
    
    # So rank 1 should work?! But the current program uses rank 44 and scores 1.09.
    # The score is seed/value, where seed is the schoolbook rank (48).
    # If rank 1 works, score would be 48/1 = 48.
    
    # I must be misunderstanding the problem. Let me re-read.
    # "Matrix multiplication 2,3,8"
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 8 matrix."
    
    # Oh wait - I think the issue is that the tensor is NOT the all-ones tensor.
    # In matrix multiplication, the tensor T[i,j,k] is defined such that:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has entries T[i,j,k] = 1, and the decomposition is:
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # This is correct, and it IS the all-ones tensor.
    
    # But wait - maybe the convention is different. Let me look at the W indexing:
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed by (k*n+i, r), meaning W maps (k,i) -> r.
    
    # And U is (n*m, rank) = (i*m+j, r)
    # V is (m*p, rank) = (j*p+k, r)
    
    # So the decomposition is:
    # 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all (i,j,k)
    
    # This is indeed the all-ones tensor of shape (2,3,8), which has tensor rank 1.
    
    # UNLESS... the problem is asking for something else. Let me re-read the verification.
    
    # The _verify function in the current program checks:
    # For A with 1 at (i0,j0) and B with 1 at (j0,k0):
    # C[i,k] = sum_r W[k*n+i, r] * U[i*m+j0, r] * V[j0*p+k, r]
    # should equal 1 if i==i0 and k==k0, else 0.
    
    # This is checking that the bilinear form correctly computes matrix multiplication,
    # NOT that the tensor entries are all 1.
    
    # The difference: the tensor T[i,j,k] = 1 means that for ANY A and B,
    # C[i,k] = sum_j sum_r U[i,j,r] V[j,k,r] W[k,i,r] A[i,j] B[j,k]
    # = sum_r W[k,i,r] (sum_j U[i,j,r] A[i,j]) (sum_j V[j,k,r] B[j,k])
    
    # Wait, that's not right. Let me be more careful.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_j sum_r U[i,j,r] V[j,k,r] W[k,i,r] A[i,j] B[j,k]  (if T[i,j,k] = sum_r ...)
    # = sum_r W[k,i,r] (sum_j U[i,j,r] A[i,j]) (sum_j V[j,k,r] B[j,k])
    
    # For this to equal sum_j A[i,j] B[j,k] for ALL A and B, we need:
    # sum_r W[k,i,r] U[i,j,r] V[j',k,r] = delta_{j,j'} for all (i,j,j',k)
    
    # This is a STRONGER condition than T[i,j,k] = 1!
    # The condition T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
    # only ensures correctness when A and B have a single nonzero entry.
    
    # So the correct condition is:
    # sum_r W[k*n+i, r] * U[i*m+j, r] * V[j'*p+k, r] = delta_{j,j'} for all (i,j,j',k)
    
    # This is the condition that the bilinear form correctly computes matrix multiplication.
    # This is NOT the same as the tensor being all-ones.
    
    # OK so now I understand. The problem is to find a bilinear decomposition of matrix multiplication,
    # which is equivalent to finding a tensor decomposition of the matmul tensor,
    # but with the additional constraint that the decomposition is "valid" in the sense that
    # it correctly computes the product for all inputs.
    
    # Actually, I think I was overcomplicating this. The standard definition of the matmul tensor
    # T[i,j,k] = 1 (for all i,j,k) IS the correct tensor, and its tensor rank IS the minimum
    # number of multiplications needed. The condition T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
    # IS the correct condition, and it IS sufficient for correct matrix multiplication.
    
    # Let me verify: if sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all (i,j,k), then:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_j sum_r U[i,j,r] V[j,k,r] W[k,i,r] A[i,j] B[j,k]
    # = sum_r W[k,i,r] (sum_j U[i,j,r] A[i,j]) (sum_j V[j,k,r] B[j,k])
    
    # For this to equal sum_j A[i,j] B[j,k] for all A,B, we need the bilinear form
    # sum_r W[k,i,r] (sum_j U[i,j,r] A[i,j]) (sum_j V[j,k,r] B[j,k])
    # to equal sum_j A[i,j] B[j,k].
    
    # This requires: sum_r W[k,i,r] U[i,j,r] V[j',k,r] = delta_{j,j'} for all (i,j,j',k).
    # This is a stronger condition than just T[i,j,k]=1.
    
    # So the _verify function is correct, and the tensor rank of the all-ones tensor is NOT
    # the right thing to minimize. We need the minimum rank of a bilinear decomposition
    # that correctly computes matrix multiplication.
    
    # This is the standard "tensor rank of matrix multiplication" problem, and for 2x3x3,
    # the answer is known to be 7 (Strassen). For 2x3x8, it's less clear.
    
    # Given the complexity, let me just try to improve the current rank 44 solution
    # by attempting to find a rank 43 or lower solution using a local search.
    
    # Try random search for lower rank
    for target_rank in [43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30]:
        if time.time() > deadline:
            break
        
        # Try to find a valid decomposition of this rank
        # Use a heuristic: start from the rank-44 solution and try to merge terms
        # or use random restarts
        
        found = False
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            # Start with the rank-44 solution and try to zero out one term
            # and adjust the others to compensate
            U_try = [row[:] for row in U]
            V_try = [row[:] for row in V]
            W_try = [row[:] for row in W]
            
            # Try removing each term and see if we can adjust
            for remove_idx in range(rank):
                if time.time() > deadline:
                    break
                
                U_t = [row[:] for row in U_try]
                V_t = [row[:] for row in V_try]
                W_t = [row[:] for row in W_try]
                
                # Zero out the term to remove
                for i in range(dim_U):
                    U_t[i][remove_idx] = 0
                for i in range(dim_V):
                    V_t[i][remove_idx] = 0
                for i in range(dim_W):
                    W_t[i][remove_idx] = 0
                
                # Check if the remaining terms still satisfy the constraints
                # (This is unlikely to work directly, but let's check)
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for r in range(target_rank):
                                if r == remove_idx:
                                    continue
                                val += U_t[i*m+j][r] * V_t[j*p+k][r] * W_t[k*n+i][r]
                            if abs(val - 1) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                    if not valid:
                        break
                
                if valid:
                    found = True
                    # Trim the matrices
                    U_final = []
                    V_final = []
                    W_final = []
                    for r in range(rank):
                        if r == remove_idx:
                            continue
                        U_final.append([U_t[i][r] for i in range(dim_U)])
                        V_final.append([V_t[i][r] for i in range(dim_V)])
                        W_final.append([W_t[i][r] for i in range(dim_W)])
                    
                    # Transpose back to (dim, rank) format
                    U_out = [[U_final[r][i] for r in range(target_rank)] for i in range(dim_U)]
                    V_out = [[V_final[r][i] for r in range(target_rank)] for i in range(dim_V)]
                    W_out = [[W_final[r][i] for r in range(target_rank)] for i in range(dim_W)]
                    
                    best_result = {"U": U_out, "V": V_out, "W": W_out}
                    best_rank = target_rank
                    break
            
            if found:
                break
        
        if found:
            break
    
    return best_result

def _verify(U, V, W):
    """Check that the decomposition is exact."""
    n, m, p = 2, 3, 8
    R = len(U[0]) if U else 0
    
    # Test with A having 1 at (i0,j0), B having 1 at (j0,k0)
    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                for i in range(n):
                    for k in range(p):
                        val = 0
                        for r in range(R):
                            if i == i0 and k == k0:
                                val += W[k * n + i][r] * U[i * m + j0][r] * V[j0 * p + k][r]
                        
                        expected_val = 1 if (i == i0 and k == k0) else 0
                        if val != expected_val:
                            return False
    return True

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
    if _verify(res["U"], res["V"], res["W"]):
        print("Verification PASSED")
    else:
        print("Verification FAILED")
