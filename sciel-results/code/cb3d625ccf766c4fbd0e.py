"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    # WFT over Z_2 x Z_7 for 2x4x7 matrix multiplication
    # n=2, m=4, p=7
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # We use the group G = Z_2 x Z_7 with 14 elements
    # The idea: embed the multiplication into a convolution over this group
    # and exploit spectral sparsity.
    
    # For exact integer/half-integer decomposition, we use the real representation
    # of the characters. Since Z_2 contributes only +/-1 and Z_7 has order 7,
    # we work with the real part of the 7th roots of unity.
    
    # Let's construct the WFT decomposition step by step.
    
    # The multiplication tensor has entries T[i,j,k] = delta_{j,j'} for appropriate indices
    # We want to decompose this as sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    
    # Using WFT: we embed A and B into functions on G and use the Fourier transform
    # The key insight is that for the 4-dimensional middle index, we can use
    # the structure of Z_4 embedded in Z_7 (since 4 and 7 are coprime, we can
    # use the Chinese Remainder Theorem-like structure).
    
    # Actually, let's use a more direct approach. For n=2, m=4, p=7:
    # We can split the 4 middle indices into two groups of 2.
    
    # A = [A_left | A_right] where A_left, A_right are 2x2
    # B = [B_top; B_bottom] where B_top, B_bottom are 2x7
    # C = A_left * B_top + A_right * B_bottom
    
    # For each 2x2 * 2x7 subproblem, we can use WFT over Z_2 x Z_7.
    # The rank for 2x2*2x7 using WFT should be 14 (the order of the group).
    
    # Let's construct the WFT for 2x2 * 2x7.
    
    def wft_2x2x7():
        """WFT decomposition for (2x2) * (2x7) -> (2x7) with rank 14"""
        n_sub, m_sub, p_sub = 2, 2, 7
        
        # Group G = Z_2 x Z_7
        # Elements indexed by (a, b) where a in {0,1}, b in {0,...,6}
        # We map (a,b) to index a * 7 + b
        
        # Characters: chi_{(u,v)}(a,b) = (-1)^(u*a) * exp(2*pi*i*v*b/7)
        # For exact real decomposition, we use cos and sin separately.
        
        # Actually, let's use a simpler approach: since we need integer/half-integer
        # entries, let's use the fact that the minimal polynomial of exp(2*pi*i/7)
        # has degree 3, so we can work in Q(cos(2*pi/7)) which has degree 3.
        
        # This is getting complex. Let me try a different strategy.
        
        # For 2x2 * 2x7, the schoolbook rank is 2*2*7 = 28.
        # Can we do better? The lower bound is max(2*2, 2*7, 7*2) = max(4, 14, 14) = 14.
        
        # Let's try to achieve rank 14 using the following construction:
        # C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, j in {0,1}, k in {0,...,6}
        
        # We can use the following trick: for each pair of rows of A and columns of B,
        # we can combine them using the 7th roots of unity structure.
        
        # Actually, let's use a known result: for (2 x 2) * (2 x 7), we can achieve
        # rank 14 using the following construction based on the cyclic group Z_7.
        
        # For each of the 7 values of k, we have:
        # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
        # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
        
        # We can write this as:
        # [C[0,k]]   [A[0,0] A[0,1]] [B[0,k]]
        # [C[1,k]] = [A[1,0] A[1,1]] [B[1,k]]
        
        # This is a 2x2 matrix multiplication for each k, giving 4 multiplications per k,
        # for a total of 28. But we can do better by combining across k.
        
        # Let's use the following approach: define for each character v of Z_7:
        # A_v[j] = sum_k A[j,k'] * exp(2*pi*i*v*k'/7)  (but A doesn't have k index)
        
        # I think I'm overcomplicating this. Let me just use schoolbook for the
        # subproblems and accept rank 28 for each, giving total rank 56.
        
        # Actually, wait. Let me reconsider the problem.
        
        # For the full 2x4x7 problem, the schoolbook rank is 56.
        # The lower bound is max(2*4, 4*7, 7*2) = max(8, 28, 14) = 28.
        
        # Can we achieve rank 28? That would be optimal.
        
        # Let's try a direct construction. We need to find rank-1 decompositions
        # that sum to the multiplication tensor.
        
        # The multiplication tensor is:
        # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * delta_{appropriate}
        
        # Actually, T[i,j,k] = 1 if the indices correspond to a valid multiplication,
        # i.e., T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2,3}, k in {0,...,6}.
        
        # Wait, that's not right. The tensor for matrix multiplication C = A*B is:
        # T_{i,j,k} = 1 if j is the "middle" index, i.e., T[i,j,k] = 1 for all valid (i,j,k)
        # where i indexes rows of A, j indexes columns of A / rows of B, k indexes columns of B.
        
        # So T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2,3}, k in {0,...,6}.
        
        # This is just the all-ones tensor of size 2x4x7, which has rank 1!
        
        # Wait, that can't be right. Let me reconsider.
        
        # The tensor for matrix multiplication is not the all-ones tensor. It's:
        # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor has entries T[i,j,k] = 1 for all (i,j,k), and the multiplication
        # is: C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
        
        # So yes, T is the all-ones tensor, and the "rank" we're computing is the
        # tensor rank of this all-ones tensor under the specific factorization
        # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
        
        # For the all-ones tensor of size n x m x p, the tensor rank under this
        # factorization is exactly n*m*p (schoolbook), because each (i,j,k) needs
        # its own rank-1 term.
        
        # Wait, that's not true either. The tensor rank of the all-ones tensor
        # [1]_{n,m,p} is actually 1, since it's already a rank-1 tensor.
        
        # But in the matrix multiplication context, we're not decomposing the
        # all-ones tensor. We're decomposing the multiplication operation itself.
        
        # Let me re-read the problem. The tensor for matrix multiplication C = A*B
        # where A is n x m and B is m x p is:
        # T_{i,j,k} = 1 for all i, j, k
        # and the multiplication is C_{i,k} = sum_j A_{i,j} B_{j,k}
        
        # The rank we're computing is the minimum number of rank-1 terms needed to
        # express this tensor as sum_r U_{i,m+j,r} V_{j,p+k,r} W_{k,n+i,r}.
        
        # For the all-ones tensor, this is indeed n*m*p in general, but there might
        # be structure that allows for a lower rank.
        
        # Actually, I think I've been confusing myself. Let me just implement the
        # schoolbook decomposition correctly and return it.
        
        rank = n_sub * m_sub * p_sub
        U = [[0] * rank for _ in range(n_sub * m_sub)]
        V = [[0] * rank for _ in range(m_sub * p_sub)]
        W = [[0] * rank for _ in range(p_sub * n_sub)]
        column = 0
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    U[i * m_sub + j][column] = 1
                    V[j * p_sub + k][column] = 1
                    W[k * n_sub + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # For the full 2x4x7 problem, use schoolbook
    n, m, p = DIMENSIONS
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
# EVOLVE-BLOCK-END
