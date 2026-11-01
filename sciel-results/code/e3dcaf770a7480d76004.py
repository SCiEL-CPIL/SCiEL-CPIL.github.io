"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 10]
_DEADLINE = time.time() + 140  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    if time.time() > _DEADLINE:
        return _construct_rank35()
    
    # Try to find a lower-rank decomposition via exact linear algebra
    # The tensor T_{(i,j),(j,k),(k,i)} = 1 iff indices are consistent
    # Flattening: fix output index (k,i), get matrix M_{(k,i),(i',j')x(j,k')} 
    # Actually, let's compute the rank of the "output flattening":
    # For each output position (k,i), the slice is a 4x20 matrix
    # M_{(k,i)}[(i',j'),(j,k')] = T_{(i',j'),(j,k'),(k,i)} = δ_{i',i}·δ_{j',j}·δ_{k',k}
    # So M_{(k,i)} is a 4x20 matrix that has a 1 at position [(i,j),(j,k)]
    
    # The tensor rank >= rank of any flattening.
    # Let's compute the rank of the flattening where we group input A indices (4) 
    # against input B indices (20), for each fixed output (20).
    # This gives a 20 x (4*20) = 20x80 matrix.
    
    # Actually, a better approach: the tensor can be seen as a 20x4x20 tensor.
    # Flatten to (20*4) x 20 = 80x20 matrix. Its rank is a lower bound.
    # Or flatten to 20 x (4*20) = 20x80.
    
    # Let me construct the flattening matrix F of shape (p*n) x (n*m * m*p) = 20 x 80
    # F[k*n+i, (i'*m+j')*(m*p) + (j'*p+k')] = T[(i',j'),(j',k'),(k,i)]
    # = 1 iff i'=i and j'=j and k'=k... wait, the tensor is T_{(i,j),(j,k),(k,i)}
    # where the first index pair is (i,j) for A, second is (j,k) for B, third is (k,i) for C.
    # So T[(i,j),(j,k),(k,i)] = 1 for all valid i,j,k.
    
    # The flattening: for fixed output (k_out, i_out), the slice over (i_in,j_in) and (j_in2,k_in) is:
    # S[(i_in,j_in),(j_in2,k_in)] = 1 iff i_in=i_out and j_in=j_in2 and k_in=k_out
    # This is a 4x20 matrix with a single 1.
    
    # The full flattening (output vs. combined inputs) is 20 x 80.
    # Each row corresponds to one output (k,i), and has exactly one 1 in the column 
    # corresponding to input (i,j) and (j,k) where j is determined by consistency.
    
    # Wait, for output (k,i), the only non-zero input combination is:
    # A entry (i, j) and B entry (j, k) for any j in {0,1}.
    # So for output (k,i), there are 2 non-zero entries in the 4x20 slice (one for j=0, one for j=1).
    
    # Let me just compute the rank of the 20x80 flattening matrix.
    
    n, m, p = DIMENSIONS
    # Build flattening matrix F: shape (p*n) x (n*m * m*p)
    # Row index: k*n + i for output (k,i)
    # Col index: (i'*m + j')*(m*p) + (j'*p + k') for input A(i',j') and B(j',k')
    # F[row, col] = 1 iff i'=i and j'=j' and k'=k (i.e., consistent indices)
    
    F = np.zeros((p*n, n*m * m*p), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Output index: k*n + i
                out_idx = k * n + i
                # Input A index: i*m + j
                a_idx = i * m + j
                # Input B index: j*p + k
                b_idx = j * p + k
                # Column in flattening: a_idx * (m*p) + b_idx
                col_idx = a_idx * (m*p) + b_idx
                F[out_idx, col_idx] = 1
    
    # Rank of F is a lower bound on tensor rank
    # Use exact SVD or row reduction
    # F is 20x80, so rank <= 20
    # Let's compute rank via exact Gaussian elimination using Fraction
    
    F_frac = [[Fraction(F[i,j]) for j in range(n*m*m*p)] for i in range(p*n)]
    
    # Gaussian elimination
    rank_F = 0
    for col in range(n*m*m*p):
        # Find pivot
        pivot_row = -1
        for row in range(rank_F, p*n):
            if F_frac[row][col] != 0:
                pivot_row = row
                break
        if pivot_row == -1:
            continue
        # Swap
        F_frac[rank_F], F_frac[pivot_row] = F_frac[pivot_row], F_frac[rank_F]
        # Eliminate
        pivot_val = F_frac[rank_F][col]
        for row in range(p*n):
            if row != rank_F and F_frac[row][col] != 0:
                factor = F_frac[row][col] / pivot_val
                for c in range(col, n*m*m*p):
                    F_frac[row][c] -= factor * F_frac[rank_F][c]
        rank_F += 1
        if rank_F == p*n:
            break
    
    # rank_F is a lower bound. If rank_F < 35, we might be able to do better.
    # But the actual tensor rank could be higher than this lower bound.
    # The current decomposition has rank 35. If rank_F <= 20, the lower bound is weak.
    
    # Let's also check the other flattening: (n*m * m*p) x (p*n) which is 80x20
    # Same rank by definition.
    
    # The block-diagonal structure suggests the tensor rank is exactly 5*7=35
    # because the 5 blocks are independent. But let's verify by checking if the
    # 35 vectors are linearly independent in the appropriate sense.
    
    # For now, if we can't find a lower rank, return the rank-35 decomposition.
    # The rank lower bound from flattening is rank_F.
    # If rank_F >= 35, then 35 is optimal.
    # If rank_F < 35, there might be room for improvement, but it's hard to find.
    
    # Given the time constraint, return the known rank-35 decomposition.
    # But first, let's verify it's correct.
    
    result = _construct_rank35()
    
    # Verify correctness with a random test
    if time.time() < _DEADLINE - 5:
        if not _verify_decomposition(result, n, m, p):
            # Fallback: return a trivially correct (but higher rank) schoolbook decomposition
            return _construct_schoolbook(n, m, p)
    
    return result


def _construct_rank35():
    """
    Constructs a rank-35 exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
    
    Strategy:
    The tensor for $C = A B$ where $A$ is $2 \times 2$ and $B$ is $2 \times 10$ can be viewed as
    5 independent $2 \times 2 \times 2$ multiplication problems, one for each pair of columns in $B$.
    Specifically, let $B = [B_0, B_1, B_2, B_3, B_4]$ where each $B_i$ is a $2 \times 2$ matrix.
    Then $C = A B = [A B_0, A B_1, A B_2, A B_3, A B_4]$.
    
    The rank of the $2 \times 2 \times 2$ matrix multiplication tensor is 7 (Strassen's algorithm).
    Since the 5 blocks are independent (the input $A$ is the same, but the outputs and the specific
    $B$ slices are disjoint in the tensor space), the total rank is the sum of the ranks of the blocks.
    Total Rank = 5 * 7 = 35.
    
    The interface requires:
    U: shape (n*m, rank) = (4, 35)
    V: shape (m*p, rank) = (20, 35)
    W: shape (p*n, rank) = (20, 35)
    
    Indices:
    U index: i*m + j  (i in 0..1, j in 0..1) -> 0,1,2,3
    V index: j*p + k  (j in 0..1, k in 0..9) -> 0..19
    W index: k*n + i  (k in 0..9, i in 0..1) -> 0..19 (Official trace convention)
    
    Strassen's 7 products for multiplying 2x2 matrices $M_1$ and $M_2$:
    $M_1 = [[a, b], [c, d]]$, $M_2 = [[e, f], [g, h]]$
    $P_1 = (a+d)(e+h)$
    $P_2 = (c+d)e$
    $P_3 = a(f-h)$
    $P_4 = d(g-e)$
    $P_5 = (a+b)h$
    $P_6 = (c-a)(e+f)$
    $P_7 = (b-d)(g+h)$
    
    Result $M = M_1 M_2$:
    $M_{0,0} = P_1 + P_4 - P_5 + P_7$
    $M_{0,1} = P_3 + P_5$
    $M_{1,0} = P_2 + P_4$
    $M_{1,1} = P_1 - P_2 + P_3 + P_6$
    
    We will construct the 35 terms corresponding to these 7 products for each of the 5 column pairs.
    """
    n, m, p = DIMENSIONS
    # rank = 35
    rank = 35
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 35
    V = [[0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * rank for _ in range(p * n)]  # 20 x 35
    
    # ... rest of the original Strassen-based construction ...
    
    # Strassen coefficients for the 7 products
    # Each product r is defined by coefficients for the 4 inputs of A and 4 inputs of B
    # A inputs: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # B inputs: e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    
    # U coefficients (for A) and V coefficients (for B) for each of the 7 Strassen products
    # Product 1: (a+d)(e+h) -> A: [1, 0, 0, 1], B: [1, 0, 0, 1] (wait, (a+d) is [1,0,0,1] dot A? No.
    # Let's map A indices: 0:a, 1:b, 2:c, 3:d.
    # P1: (a+d)(e+h). A_vec = [1, 0, 0, 1]. B_vec = [1, 0, 0, 1] (for e,h? No. e=B00, f=B01, g=B10, h=B11).
    # e+h corresponds to B indices 0 and 3. So B_vec = [1, 0, 0, 1].
    
    # P2: (c+d)e. A_vec = [0, 0, 1, 1]. B_vec = [1, 0, 0, 0].
    # P3: a(f-h). A_vec = [1, 0, 0, 0]. B_vec = [0, 1, 0, -1].
    # P4: d(g-e). A_vec = [0, 0, 0, 1]. B_vec = [-1, 0, 1, 0].
    # P5: (a+b)h. A_vec = [1, 1, 0, 0]. B_vec = [0, 0, 0, 1].
    # P6: (c-a)(e+f). A_vec = [-1, 0, 1, 0]. B_vec = [1, 1, 0, 0].
    # P7: (b-d)(g+h). A_vec = [0, 1, 0, -1]. B_vec = [0, 0, 1, 1].
    
    strassen_A = [
        [1, 0, 0, 1],  # P1
        [0, 0, 1, 1],  # P2
        [1, 0, 0, 0],  # P3
        [0, 0, 0, 1],  # P4
        [1, 1, 0, 0],  # P5
        [-1, 0, 1, 0], # P6
        [0, 1, 0, -1]  # P7
    ]
    
    strassen_B = [
        [1, 0, 0, 1],  # P1 (e+h)
        [1, 0, 0, 0],  # P2 (e)
        [0, 1, 0, -1], # P3 (f-h)
        [-1, 0, 1, 0], # P4 (g-e)
        [0, 0, 0, 1],  # P5 (h)
        [1, 1, 0, 0],  # P6 (e+f)
        [0, 0, 1, 1]   # P7 (g+h)
    ]
    
    # W coefficients (for C)
    # C indices in tensor W: k*n + i.
    # For a 2x2 block, k is local 0 or 1, i is 0 or 1.
    # Local C indices:
    # C00: i=0, k_local=0
    # C01: i=0, k_local=1
    # C10: i=1, k_local=0
    # C11: i=1, k_local=1
    
    # M00 = P1 + P4 - P5 + P7
    # M01 = P3 + P5
    # M10 = P2 + P4
    # M11 = P1 - P2 + P3 + P6
    
    # We need to map these to the global W indices.
    # Global W index = k_global * n + i = k_global * 2 + i.
    # For block b (0 to 4), the columns are 2*b and 2*b+1.
    # So k_global = 2*b + k_local.
    # W_index = (2*b + k_local) * 2 + i.
    
    # Let's define the W contribution for each Strassen product P_r to the local C matrix.
    # w_contrib[r][local_c_idx] = coefficient
    # local_c_idx: 0->C00, 1->C01, 2->C10, 3->C11
    
    w_contrib = [
        [0] * 4 for _ in range(7)
    ]
    
    # P1 contributes to C00 (+1) and C11 (+1)
    w_contrib[0][0] = 1
    w_contrib[0][3] = 1
    
    # P2 contributes to C10 (+1) and C11 (-1)
    w_contrib[1][2] = 1
    w_contrib[1][3] = -1
    
    # P3 contributes to C01 (+1) and C11 (+1)
    w_contrib[2][1] = 1
    w_contrib[2][3] = 1
    
    # P4 contributes to C00 (+1) and C10 (+1)
    w_contrib[3][0] = 1
    w_contrib[3][2] = 1
    
    # P5 contributes to C00 (-1) and C01 (+1)
    w_contrib[4][0] = -1
    w_contrib[4][1] = 1
    
    # P6 contributes to C11 (+1)
    w_contrib[5][3] = 1
    
    # P7 contributes to C00 (+1)
    w_contrib[6][0] = 1
    
    # Now fill U, V, W
    # Total rank is 35.
    # For each block b in 0..4:
    #   For each Strassen product r in 0..6:
    #     Global rank index col = b * 7 + r
    
    for b in range(5):
        for r in range(7):
            col = b * 7 + r
            
            # Fill U
            # U index: i*m + j.
            # strassen_A[r] is [a, b, c, d] coefficients.
            # a -> i=0, j=0 -> idx 0
            # b -> i=0, j=1 -> idx 1
            # c -> i=1, j=0 -> idx 2
            # d -> i=1, j=1 -> idx 3
            U[0][col] = strassen_A[r][0]
            U[1][col] = strassen_A[r][1]
            U[2][col] = strassen_A[r][2]
            U[3][col] = strassen_A[r][3]
            
            # Fill V
            # V index: j*p + k.
            # strassen_B[r] is [e, f, g, h] coefficients.
            # e -> j=0, k_local=0 -> k_global=2*b -> idx 0*p + 2*b = 2*b
            # f -> j=0, k_local=1 -> k_global=2*b+1 -> idx 0*p + 2*b+1 = 2*b+1
            # g -> j=1, k_local=0 -> k_global=2*b -> idx 1*p + 2*b = 10 + 2*b
            # h -> j=1, k_local=1 -> k_global=2*b+1 -> idx 1*p + 2*b+1 = 10 + 2*b+1
            
            k0 = 2 * b
            k1 = 2 * b + 1
            
            V[0 * p + k0][col] = strassen_B[r][0]
            V[0 * p + k1][col] = strassen_B[r][1]
            V[1 * p + k0][col] = strassen_B[r][2]
            V[1 * p + k1][col] = strassen_B[r][3]
            
            # Fill W
            # W index: k_global * n + i.
            # w_contrib[r] is [C00, C01, C10, C11] coefficients.
            # C00: i=0, k_local=0 -> k_global=2*b -> idx (2*b)*2 + 0
            # C01: i=0, k_local=1 -> k_global=2*b+1 -> idx (2*b+1)*2 + 0
            # C10: i=1, k_local=0 -> k_global=2*b -> idx (2*b)*2 + 1
            # C11: i=1, k_local=1 -> k_global=2*b+1 -> idx (2*b+1)*2 + 1
            
            W[(2 * b) * 2 + 0][col] = w_contrib[r][0]
            W[(2 * b + 1) * 2 + 0][col] = w_contrib[r][1]
            W[(2 * b) * 2 + 1][col] = w_contrib[r][2]
            W[(2 * b + 1) * 2 + 1][col] = w_contrib[r][3]
            
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(decomp, n, m, p, num_tests=100, seed=42):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    import random
    random.seed(seed)
    
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    
    rank = len(U[0])
    
    for _ in range(num_tests):
        # Generate random integer matrices
        A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        
        # Compute C = A * B
        C = [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]
        
        # Compute using decomposition: C[k*n+i] = sum_r U[i*m+j] * V[j*p+k] * W[k*n+i]
        # Wait, the decomposition is: T_{(i,j),(j,k),(k,i)} = sum_r U_r[i,j] * V_r[j,k] * W_r[k,i]
        # So C[i,k] = sum_j A[i,j] * B[j,k] = sum_j sum_r U_r[i,j] * V_r[j,k] * A[i,j] * B[j,k]... 
        # No. The tensor decomposition means:
        # The bilinear form for each output C[i,k] is: sum_j A[i,j]*B[j,k]
        # = sum_r (sum_j U_r[i,j]*A[i,j]) * (sum_j V_r[j,k]*B[j,k])... no that's not right either.
        
        # Correct interpretation: 
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The rank-r decomposition gives:
        # C[i,k] = sum_r W_r[k*n+i] * (sum_j U_r[i*m+j] * A[i,j]) * (sum_j V_r[j*p+k] * B[j,k])
        # Wait, that's not standard. Let me think again.
        
        # Standard matrix multiplication tensor decomposition:
        # T_{(i,j),(j,k),(k,i)} = sum_r U_r[i,j] * V_r[j,k] * W_r[k,i]
        # where T_{(i,j),(j,k),(k,i)} = 1 (it's the identity tensor for matmul)
        # Then: C[i,k] = sum_j A[i,j]*B[j,k] = sum_{i,j,k} A[i,j]*B[j,k]*T_{(i,j),(j,k),(k,i)}
        # = sum_{i,j,k} A[i,j]*B[j,k]*sum_r U_r[i,j]*V_r[j,k]*W_r[k,i]
        # = sum_r W_r[k,i]*sum_j U_r[i,j]*V_r[j,k]*A[i,j]*B[j,k]... 
        
        # Hmm, let me be more careful. The tensor T has entries indexed by (input_A, input_B, output_C).
        # T[a_idx, b_idx, c_idx] where a_idx=(i,j), b_idx=(j,k), c_idx=(k,i)
        # T[a_idx, b_idx, c_idx] = 1 if the indices are consistent (i.e., the j in a matches j in b, 
        # and k in b matches k in c, and i in a matches i in c), else 0.
        
        # Actually for matrix multiplication, T_{(i,j),(j',k),(k',i')} = δ_{j,j'}·δ_{k,k'}·δ_{i,i'}
        # And C[i',k'] = sum_{i,j,k,j'} A[i,j]*B[j',k]*T_{(i,j),(j',k),(k',i')}
        # = sum_{i,j,k,j'} A[i,j]*B[j',k]*δ_{j,j'}·δ_{k,k'}·δ_{i,i'}
        # = A[i',k']... no wait.
        # = sum_{j} A[i',j]*B[j,k']  (after summing over i=j', k=k', i=i')
        
        # OK so the decomposition is:
        # T_{(i,j),(j',k),(k',i')} = sum_r U_r[i,j]*V_r[j',k]*W_r[k',i']
        # Then C[i',k'] = sum_{i,j,j',k} A[i,j]*B[j',k]*T_{(i,j),(j',k),(k',i')}
        # = sum_r sum_{i,j,j',k} A[i,j]*B[j',k]*U_r[i,j]*V_r[j',k]*W_r[k',i']
        # = sum_r W_r[k',i'] * (sum_{i,j} A[i,j]*U_r[i,j]) * (sum_{j',k} B[j',k]*V_r[j',k])
        
        # Wait, that doesn't look right dimensionally. Let me reconsider.
        
        # Actually: sum_{i,j} A[i,j]*U_r[i,j] is a scalar (call it alpha_r)
        # sum_{j',k} B[j',k]*V_r[j',k] is a scalar (call it beta_r)
        # Then C[i',k'] = sum_r W_r[k',i'] * alpha_r * beta_r
        # But this would mean C is a rank-1 combination, which is wrong.
        
        # I think I'm confusing myself. Let me look at the standard formulation.
        
        # The matrix multiplication tensor M_{(i,j),(j,k),(k,i)} = δ_{i,i}δ_{j,j}δ_{k,k} (trivially 1 when indices match)
        # A rank-r decomposition: M = sum_r u_r ⊗ v_r ⊗ w_r
        # where u_r ∈ R^{n*m}, v_r ∈ R^{m*p}, w_r ∈ R^{p*n}
        # M_{(i,j),(j,k),(k,i)} = sum_r u_r[i,j]*v_r[j,k]*w_r[k,i]
        
        # Then for matrices A and B:
        # C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
        # = sum_{i,j,k} A[i,j]*B[j,k]*M_{(i,j),(j,k),(k,i)}  [since M=1 when indices consistent]
        # = sum_{i,j,k} A[i,j]*B[j,k]*sum_r u_r[i,j]*v_r[j,k]*w_r[k,i]
        # = sum_r w_r[k,i]*sum_j u_r[i,j]*v_r[j,k]*A[i,j]*B[j,k]
        
        # Hmm, this still has i and k in the inner sum. Let me expand differently:
        # = sum_r sum_{i,j,k} A[i,j]*B[j,k]*u_r[i,j]*v_r[j,k]*w_r[k,i]
        # = sum_r sum_k w_r[k,i] * sum_j A[i,j]*B[j,k]*u_r[i,j]*v_r[j,k]
        
        # This is getting complicated. Let me just do the direct computation:
        # C[i,k] = sum_r (sum_j A[i,j]*U_r[i*m+j]) * (sum_j B[j,k]*V_r[j*p+k]) * W_r[k*n+i]
        # Wait, that's not right either because i and k appear in multiple places.
        
        # Let me just compute it directly:
        # C[i,k] should equal sum_r sum_{j1,j2} U_r[i*m+j1]*V_r[j2*p+k]*W_r[k*n+i]*A[i,j1]*B[j2,k]
        # But this only works if j1=j2 (from the tensor structure). 
        # Actually no, the tensor decomposition already encodes the j1=j2 constraint.
        
        # Let me just verify numerically:
        # C_decomp[i,k] = sum_r sum_{j1} sum_{j2} U_r[i*m+j1] * V_r[j2*p+k] * W_r[k*n+i] * A[i,j1] * B[j2,k]
        # This should equal C[i,k] = sum_j A[i,j]*B[j,k]
        
        # For the rank-35 Strassen construction, let's verify this.
        
        C_decomp = [[0]*p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                val = 0
                for r in range(rank):
                    w_val = W[k*n+i][r]
                    if w_val == 0:
                        continue
                    # Compute sum over j1 of U_r[i*m+j1]*A[i,j1]
                    u_dot = 0
                    for j1 in range(m):
                        u_dot += U[i*m+j1][r] * A[i][j1]
                    # Compute sum over j2 of V_r[j2*p+k]*B[j2,k]
                    v_dot = 0
                    for j2 in range(m):
                        v_dot += V[j2*p+k][r] * B[j2][k]
                    val += w_val * u_dot * v_dot
                C_decomp[i][k] = val
        
        # Check if C_decomp equals C
        for i in range(n):
            for k in range(p):
                if C_decomp[i][k] != C[i][k]:
                    return False
    
    return True


def _construct_schoolbook(n, m, p):
    """Trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = (i*m + j)*p + k
                U[i*m+j][r] = 1
                V[j*p+k][r] = 1
                W[k*n+i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
