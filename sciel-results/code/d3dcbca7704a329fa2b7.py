"""Matrix multiplication 3x4x8 rank-minimal exact tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m  # 12
    mp = m * p  # 32
    pn = p * n  # 24
    total_entries = n * m * p  # 96
    
    def schoolbook():
        rank = total_entries
        U = np.zeros((nm, rank), dtype=np.int64)
        V = np.zeros((mp, rank), dtype=np.int64)
        W = np.zeros((pn, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    best_U, best_V, best_W, best_rank = schoolbook()
    
    def compute_entries(U, V, W, rank):
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        return np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
    
    def count_violations(U, V, W, rank):
        entries = compute_entries(U, V, W, rank)
        return np.sum(entries != 1)
    
    def check_valid(U, V, W, rank):
        return count_violations(U, V, W, rank) == 0
    
    # Greedy construction: start with schoolbook, try to merge factors
    # A factor at (i,j,k) with value 1 contributes to entry (i,j,k) only.
    # To merge, we need factors where U[i,j,c], V[j,k,c], W[k,i,c] are nonzero
    # for multiple (i,j,k) pairs.
    
    # Try a structured approach: use the Kronecker product structure
    # 3x4x8 = 3x4x(2*4) = (3x4x4) x 2 in some sense
    
    # Let's try: for each of the 2 blocks of 4 columns, use a 3x4x4 decomposition
    # and see if we can share factors
    
    # First, let's try a simple improvement: random local search from schoolbook
    # by perturbing and seeing if we can reduce rank
    
    # Actually, let's try a different approach: assume a specific structure
    # U[i*m+j, c] = a_i * b_j * f_c  (rank-1 per factor in U)
    # V[j*p+k, c] = b_j * g_k * h_c  (shares b_j)
    # W[k*n+i, c] = g_k * a_i * i_c  (shares g_k, a_i)
    # Then entry (i,j,k) = sum_c (a_i*b_j*f_c)*(b_j*g_k*h_c)*(g_k*a_i*i_c)
    # = sum_c (a_i^2 * b_j^2 * g_k^2 * f_c*h_c*i_c)
    # This is separable! entry(i,j,k) = A_i * B_j * G_k * S_c where S_c = f_c*h_c*i_c
    # But we need entry(i,j,k) = 1 for all i,j,k, so:
    # A_i * B_j * G_k * sum_c S_c = 1 for all i,j,k
    # This requires A_i * B_j * G_k = constant for all i,j,k, which means
    # A_i = a, B_j = b, G_k = g with a*b*g*sum_c S_c = 1
    # So A_i, B_j, G_k are all constant!
    # U[i*m+j, c] = a * b * f_c
    # V[j*p+k, c] = b * g * h_c
    # W[k*n+i, c] = g * a * i_c
    # Entry = sum_c a*b*f_c * b*g*h_c * g*a*i_c = a^2*b^2*g^2 * sum_c f_c*h_c*i_c = 1
    # This works! But it's a very specific structure.
    # U[i*m+j, c] = a*b*f_c for all i,j -> U has rank 1 (all rows identical up to scale)
    # This is a valid decomposition! Let's construct it.
    
    # Choose a, b, g, and f_c, h_c, i_c such that a^2*b^2*g^2 * sum_c f_c*h_c*i_c = 1
    # Let's use integers: a=1, b=1, g=1, then sum_c f_c*h_c*i_c = 1
    # With f_c=h_c=i_c=1 for c=0, and 0 otherwise: rank 1!
    # But wait, that gives entry(i,j,k) = 1*1*1 * 1*1*1 = 1 for all i,j,k.
    # Let's verify: U[i*m+j, 0] = 1*1*1 = 1 for all i,j
    # V[j*p+k, 0] = 1*1*1 = 1 for all j,k
    # W[k*n+i, 0] = 1*1*1 = 1 for all k,i
    # Entry(i,j,k) = U[i*m+j,0]*V[j*p+k,0]*W[k*n+i,0] = 1*1*1 = 1. YES!
    # Rank 1 decomposition!
    
    # But wait, the problem says "integer/half-integer entries" and rank 1 seems too good.
    # Let me double-check the tensor convention: W[k*n+i, r]
    # So W is indexed by (k,i) where k is the column of the second matrix and i is the row of the first.
    # The matrix product (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # The decomposition says: (A*B)[i,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] ??? 
    # No, the standard tensor decomposition for matrix multiplication is:
    # (A*B)[i,k] = sum_j A[i,j]*B[j,k] = sum_r U[i,j,r]*V[j,k,r]*W[i,k,r]
    # But the problem says W[k*n+i, r], so W is indexed by (k,i), not (i,k).
    # Let me re-read: "The output factor follows the official trace convention W[k*n+i,r]."
    # So W[k*n+i, r] means the index is k*n + i, where k is the column index and i is the row index.
    # So W is indexed by (k,i), not (i,k). This is the transpose of the usual convention.
    # 
    # The decomposition should be: (A*B)[i,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # But this doesn't make sense dimensionally because the sum is over r, and U depends on (i,j),
    # V depends on (j,k), W depends on (k,i). The j index is summed over implicitly?
    # 
    # Actually, in tensor decomposition of matrix multiplication, the tensor T[i,j,k] = delta_{j,j} = 1
    # (the identity in the middle index). The decomposition is:
    # T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # And the matrix product is recovered as: (A*B)[i,k] = sum_j A[i,j]*B[j,k]*T[i,j,k]
    # = sum_j A[i,j]*B[j,k]*sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # = sum_r (sum_j A[i,j]*U[i,j,r]) * (sum_j B[j,k]*V[j,k,r]) * W[k,i,r]
    # 
    # So the condition is: T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k.
    # 
    # With my rank-1 proposal: U[i,j,0] = 1, V[j,k,0] = 1, W[k,i,0] = 1
    # T[i,j,k] = 1*1*1 = 1. YES, this works!
    
    # So the rank is 1?! That can't be right. Let me think again...
    # Actually, I think I'm confusing myself. The matrix multiplication tensor T[i,j,k] is defined as:
    # T[i,j,k] = 1 if the indices are consistent, but actually for matrix multiplication,
    # the tensor is T[i,j,k] = delta_{j,j} which is always 1. No wait...
    # 
    # The matrix multiplication tensor is: T[i,j,k] = 1 for all i,j,k in the appropriate ranges.
    # Because (A*B)[i,k] = sum_j A[i,j]*B[j,k], and the tensor T is such that
    # (A*B)[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k] where T[i,j,k] = 1.
    # 
    # So yes, T[i,j,k] = 1 for all i,j,k. And my rank-1 decomposition gives T[i,j,k] = 1.
    # 
    # But this seems too good to be true. Let me check with a small example.
    # 1x1x1 matrix multiplication: A(1x1) * B(1x1) = C(1x1)
    # U[0,0]=1, V[0,0]=1, W[0,0]=1. C = U*V*W = 1. A*B = A[0]*B[0]. 
    # The decomposition says C = sum_r (A[0]*U[0,0])*(B[0]*V[0,0])*W[0,0] = A[0]*B[0]*1 = A[0]*B[0]. Correct.
    # 
    # 2x2x2: U[i,j,r], V[j,k,r], W[k,i,r] all 1 for r=0.
    # C[i,k] = sum_j A[i,j]*B[j,k]*T[i,j,k] where T[i,j,k]=1.
    # = sum_j A[i,j]*B[j,k]. Correct!
    # 
    # So the rank of matrix multiplication tensor T[i,j,k]=1 is 1?! 
    # That contradicts Strassen's result that rank(2x2x2) = 7.
    # 
    # I must be misunderstanding the tensor convention. Let me re-read the problem.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 8 matrix."
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # 
    # I think the issue is that the tensor T[i,j,k] is NOT all ones. The matrix multiplication
    # tensor is T[i,j,k] = delta_{j,j}... no, that's still 1.
    # 
    # Wait, I think I've been confusing myself. The matrix multiplication tensor T is defined such that:
    # (A*B)[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    # For this to equal sum_j A[i,j]*B[j,k], we need T[i,j,k] = 1 for all i,j,k.
    # 
    # But Strassen's algorithm shows that the rank of this tensor is 7, not 1. How?
    # 
    # Ah, I see my error. The decomposition is:
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # But this is NOT the same as saying the matrix product can be computed with rank r.
    # The matrix product is:
    # (A*B)[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # = sum_j sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] * A[i,j] * B[j,k]
    # = sum_r W[k,i,r] * (sum_j A[i,j]*U[i,j,r]) * (sum_j B[j,k]*V[j,k,r])
    # 
    # So to compute (A*B)[i,k], we need:
    # 1. For each r, compute X[i,r] = sum_j A[i,j]*U[i,j,r] (this is n*r multiplies per i, so n*m*r total)
    # 2. For each r, compute Y[k,r] = sum_j B[j,k]*V[j,k,r] (this is m*p*r total)
    # 3. For each i,k, compute C[i,k] = sum_r X[i,r]*Y[k,r]*W[k,i,r] (this is n*p*r total)
    # 
    # The total number of multiplications is n*m*r + m*p*r + n*p*r = r*(n*m + m*p + n*p).
    # For schoolbook, r = n*m*p, and the "multiplications" are the individual A[i,j]*B[j,k] terms.
    # 
    # So the rank r is the number of "intermediate" products. The condition is indeed
    # T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k.
    # 
    # And my rank-1 decomposition satisfies this! So why is Strassen's rank 7?
    # 
    # I think I'm still confused. Let me look at this more carefully.
    # 
    # Actually, I believe the issue is that U[i,j,r] is NOT a scalar but a function of i and j.
    # In my rank-1 proposal, U[i,j,0] = 1 for all i,j. This means X[i,0] = sum_j A[i,j]*1 = sum_j A[i,j].
    # Similarly Y[k,0] = sum_j B[j,k]*1 = sum_j B[j,k].
    # And C[i,k] = X[i,0]*Y[k,0]*W[k,i,0] = (sum_j A[i,j])*(sum_j B[j,k])*1.
    # But the actual product is sum_j A[i,j]*B[j,k], NOT (sum_j A[i,j])*(sum_j B[j,k]).
    # 
    # So my "decomposition" is WRONG. The condition T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    # is necessary but the U, V, W must be such that the RECONSTRUCTION formula gives the correct product.
    # 
    # Let me re-derive. We want:
    # (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # = sum_r W[k,i,r] * (sum_j A[i,j]*U[i,j,r]) * (sum_j B[j,k]*V[j,k,r])
    # 
    # For this to hold for ALL A and B, we need:
    # sum_j A[i,j]*B[j,k] = sum_r W[k,i,r] * (sum_j A[i,j]*U[i,j,r]) * (sum_j B[j,k]*V[j,k,r])
    # 
    # Expanding the right side:
    # = sum_r W[k,i,r] * sum_j A[i,j]*U[i,j,r] * sum_l B[l,k]*V[l,k,r]
    # = sum_j sum_l A[i,j]*B[l,k] * sum_r W[k,i,r]*U[i,j,r]*V[l,k,r]
    # 
    # For this to equal sum_j A[i,j]*B[j,k] for all A,B, we need:
    # sum_r W[k,i,r]*U[i,j,r]*V[l,k,r] = delta_{j,l}
    # 
    # So the condition is: sum_r U[i,j,r]*V[l,k,r]*W[k,i,r] = delta_{j,l} for all i,j,l,k.
    # NOT T[i,j,k] = 1!
    # 
    # I had the wrong condition! The correct condition is:
    # sum_r U[i,j,r]*V[l,k,r]*W[k,i,r] = delta_{j,l}
    # 
    # This makes much more sense. For schoolbook, U[i,j,r] = 1 if r=(i,j,k) for some k, else 0.
    # V[j,k,r] = 1 if r=(i,j,k) for some i, else 0.
    # W[k,i,r] = 1 if r=(i,j,k) for some j, else 0.
    # Then sum_r U[i,j,r]*V[l,k,r]*W[k,i,r] = 1 if there exists r with U[i,j,r]=V[l,k,r]=W[k,i,r]=1,
    # which requires j=l (since r encodes (i,j,k) and V[l,k,r] requires the j-index to be l).
    # So the sum is delta_{j,l}. Correct!
    # 
    # OK so the correct condition is:
    # For all i in [0,n), j in [0,m), l in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[l*p+k, r] * W[k*n+i, r] = delta_{j,l}
    # 
    # This is a much harder constraint! Let me rewrite the code with the correct condition.
    
    def correct_condition(U, V, W, rank):
        """Check if U,V,W satisfy the correct decomposition condition."""
        # For all i,j,l,k: sum_r U[i*m+j,r]*V[l*p+k,r]*W[k*n+i,r] = delta_{j,l}
        # Use einsum: 'ijr,lkr,kir->ijlk'
        U_r = U.reshape(n, m, rank)  # U[i*m+j, r] -> U_r[i, j, r]
        V_r = V.reshape(m, p, rank)  # V[l*p+k, r] -> V_r[l, k, r]
        W_r = W.reshape(p, n, rank)  # W[k*n+i, r] -> W_r[k, i, r]
        
        # sum_r U[i,j,r]*V[l,k,r]*W[k,i,r]
        # = einsum('ijr,lkr,kir->ijlk', U_r, V_r, W_r)
        result = np.einsum('ijr,lkr,kir->ijlk', U_r, V_r, W_r, optimize=True)
        
        # Expected: delta_{j,l} for all i,k
        # So result[i,j,l,k] should be 1 if j==l, else 0
        expected = np.zeros((n, m, m, p), dtype=np.int64)
        for j in range(m):
            expected[:, j, j, :] = 1
        
        return np.all(result == expected), result
    
    # Verify schoolbook with correct condition
    valid, result = correct_condition(best_U, best_V, best_W, best_rank)
    assert valid, "Schoolbook should satisfy correct condition"
    
    # Now search for lower rank
    def count_violations_correct(U, V, W, rank):
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        result = np.einsum('ijr,lkr,kir->ijlk', U_r, V_r, W_r, optimize=True)
        expected = np.zeros((n, m, m, p), dtype=np.int64)
        for j in range(m):
            expected[:, j, j, :] = 1
        return np.sum(result != expected)
    
    def local_search_correct(U, V, W, rank, max_steps=5000):
        current_viol = count_violations_correct(U, V, W, rank)
        if current_viol == 0:
            return U, V, W, 0
        
        U_curr = U.copy()
        V_curr = V.copy()
        W_curr = W.copy()
        best_viol = current_viol
        best_U = U_curr.copy()
        best_V = V_curr.copy()
        best_W = W_curr.copy()
        
        step = 0
        while step < max_steps and time.time() < deadline:
            step += 1
            mat_idx = random.choice([0, 1, 2])
            if mat_idx == 0:
                rows, cols = nm, rank
            elif mat_idx == 1:
                rows, cols = mp, rank
            else:
                rows, cols = pn, rank
            r = random.randint(0, rows - 1)
            c = random.randint(0, cols - 1)
            old_val = (U_curr[r, c] if mat_idx == 0 else V_curr[r, c] if mat_idx == 1 else W_curr[r, c])
            new_val = random.randint(-2, 3)
            if mat_idx == 0:
                U_curr[r, c] = new_val
            elif mat_idx == 1:
                V_curr[r, c] = new_val
            else:
                W_curr[r, c] = new_val
            new_viol = count_violations_correct(U_curr, V_curr, W_curr, rank)
            if new_viol < best_viol:
                best_viol = new_viol
                best_U = U_curr.copy()
                best_V = V_curr.copy()
                best_W = W_curr.copy()
                if new_viol == 0:
                    break
            elif new_viol <= current_viol:
                current_viol = new_viol
            else:
                if mat_idx == 0:
                    U_curr[r, c] = old_val
                elif mat_idx == 1:
                    V_curr[r, c] = old_val
                else:
                    W_curr[r, c] = old_val
        
        return best_U, best_V, best_W, best_viol
    
    # Try ranks from 48 to 95
    for rank in range(48, 96):
        if time.time() > deadline:
            break
        if rank >= best_rank:
            continue
        
        for attempt in range(20):
            if time.time() > deadline:
                break
            U = np.random.randint(-1, 2, size=(nm, rank))
            V = np.random.randint(-1, 2, size=(mp, rank))
            W = np.random.randint(-1, 2, size=(pn, rank))
            viol = count_violations_correct(U, V, W, rank)
            if viol == 0:
                best_U, best_V, best_W, best_rank = U, V, W, rank
                break
            U_ls, V_ls, W_ls, viol_ls = local_search_correct(U, V, W, rank, max_steps=2000)
            if viol_ls < viol:
                viol = viol_ls
                U, V, W = U_ls, V_ls, W_ls
            if viol == 0:
                best_U, best_V, best_W, best_rank = U, V, W, rank
                break
        if best_rank <= rank:
            break
    
    if best_rank < 96:
        if not correct_condition(best_U, best_V, best_W, best_rank)[0]:
            best_U, best_V, best_W, best_rank = schoolbook()
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
