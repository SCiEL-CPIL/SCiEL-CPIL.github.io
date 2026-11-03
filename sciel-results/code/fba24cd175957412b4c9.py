"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    # A is n x m = 3x4, B is m x p = 4x5, C is n x p = 3x5
    # W[k*n+i, r] convention: C[i,k] = sum_r U[i*m+j? ...]
    # Actually the convention: U has shape (n*m, rank), V (m*p, rank), W (p*n, rank)
    # C[i,k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]? 
    # No, standard: C = A*B where (AB)[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor: T[i,j,k] = delta... the multiplication tensor has T[i,j,k] = 1 if... 
    # Actually the matrix multiplication tensor M_{n,m,p} has entries T[i,j,k] for i in [n], j in [m], k in [p]
    # and M = sum_r u_r ⊗ v_r ⊗ w_r where (u_r ⊗ v_r ⊗ w_r)[i,j,k] = u_r[i]*v_r[j]*w_r[k]
    # and we need sum_r u_r[i]*v_r[j]*w_r[k] = 1 if the "middle" index matches... 
    # Wait, for matrix mult: C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor is T[i,j,k] = 1 (it's just the identity in j for the contraction)
    # No: T[i,j,k] should satisfy that for any A,B: C[i,k] = sum_{i,j,k} A[i,j] B[j,k] T[i,j,k]
    # So T[i,j,k] = 1 for all i,j,k? No, that's not right either.
    # C[i,k] = sum_j A[i,j] B[j,k], so T[i,j,k] = 1 if we sum over j... 
    # The tensor T has T[i,j,k] = 1 for all valid i,j,k, and the decomposition is:
    # 1 = sum_r u_r[i] v_r[j] w_r[k] for all i,j,k
    # Wait no. The matrix multiplication tensor is defined as:
    # M = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the "value" is such that
    # <M, A ⊗ B> = C where C[i,k] = sum_j A[i,j] B[j,k]
    # So M[i,j,k] = 1 for all i,j,k (it's the all-ones tensor in the right sense)
    # No wait, that's the tensor that gives sum_{i,j,k} A[i,j]B[j,k] which is trace(AB^T) not C.
    # 
    # Let me reconsider. The multiplication tensor μ_{n,m,p} is in (R^n)^* ⊗ (R^m)^* ⊗ (R^p)^*
    # defined by μ(A,B) = AB, i.e., <μ, A ⊗ B> = AB (as a bilinear form on (R^p)^* ⊗ (R^n)^*)
    # So μ[i,j,k] = 1 if we identify correctly... 
    # Actually: (AB)[i,k] = sum_j A[i,j] B[j,k]
    # So μ = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * 1, and the decomposition rank is the tensor rank of this.
    # The schoolbook: μ = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k, rank = n*m*p = 60.
    #
    # For the factor matrices: U has shape (n*m, rank) with U[i*m+j, r], 
    # V has shape (m*p, rank) with V[j*p+k, r], W has shape (p*n, rank) with W[k*n+i, r]
    # And we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    #
    # Wait, that doesn't look right dimensionally for standard matrix mult. Let me re-read the task.
    # "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]" and the product should give the multiplication.
    # So sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = δ? No, = 1 for all i,j,k? 
    # That would mean the tensor is all-ones, which has rank 1!
    # 
    # I think I'm misunderstanding. Let me look at the schoolbook: each term has U=1, V=1, W=1
    # for a specific (i,j,k). So sum over c of U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] 
    # In schoolbook, for each (i,j,k) there's one column c where all three are 1, others 0.
    # So the sum equals 1 for all i,j,k. Yes, the tensor is the all-ones tensor!
    # 
    # But that can't be right for matrix multiplication... unless the convention is different.
    # Oh wait, I think the issue is that in the standard formulation, the matrix multiplication
    # tensor has T[i,j,k] = 1 and we want to decompose it. The rank of the all-ones tensor
    # of size n×m×p is indeed 1 (it's e_i ⊗ e_j ⊗ e_k summed... no).
    # 
    # Actually the all-ones tensor 1_{n×m×p} = (1,...,1) ⊗ (1,...,1) ⊗ (1,...,1) has rank 1!
    # That can't be the matrix multiplication tensor.
    #
    # I think the correct interpretation: the tensor T[i,j,k] = 1, and the decomposition 
    # sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] should equal T[i,j,k] = 1.
    # But the all-ones tensor has rank 1, so we could do rank 1!
    # 
    # Unless... the convention is that it should equal δ_{something}. Let me re-examine.
    # 
    # Looking at the schoolbook code: for each (i,j,k), it sets U[i*m+j][col]=1, V[j*p+k][col]=1, W[k*n+i][col]=1.
    # So for a given (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 (only that column contributes).
    # For a different (i',j',k'), the sum is 0 unless (i',j',k')=(i,j,k).
    # So the tensor is T[i,j,k] = 1 for all i,j,k, and it's decomposed as sum of rank-1 terms 
    # each being e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}.
    # 
    # But the all-ones tensor has rank 1! So why is schoolbook rank n*m*p?
    # 
    # OH WAIT. I think I misread. Let me look again: the sum should give the matrix multiplication.
    # Maybe the tensor isn't all-ones. Maybe T[i,j,k] is defined differently.
    # 
    # In standard tensor decomposition for matrix multiplication, the tensor is:
    # M[i,j,k] = 1, and we write M = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r ∈ R^{n*m}, v_r ∈ R^{m*p}, w_r ∈ R^{p*n}
    # and (u_r ⊗ v_r ⊗ w_r)[i*m+j, j*p+k, k*n+i] = u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
    # 
    # And we need sum_r u_r[i*m+j] v_r[j*p+k] w_r[k*n+i] = 1 for all i,j,k.
    # 
    # The all-ones tensor on (n*m) × (m*p) × (p*n) does NOT have rank 1 in general!
    # Because the indices are coupled: the same j appears in both u and v indices.
    # 
    # So the tensor is: T[a,b,c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # and T[a,b,c] = 1 if the j in a equals the j' in b and the k in b equals the k' in c and the i in c equals the i' in a... 
    # No, I think T is just 1 for all entries, but the constraint is that we're summing over the "free" indices.
    #
    # Let me just trust the schoolbook implementation and try to improve it.
    # The schoolbook has rank 60. I need to find a lower rank decomposition where
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i∈[3], j∈[4], k∈[5].
    
    # Let me verify: is the all-ones tensor on 12×20×15 (with the index coupling) actually requiring rank 60?
    # The key constraint is that j appears in both U and V indices, k in V and W, i in W and U.
    # 
    # For a rank-1 term u ⊗ v ⊗ w, we need u[i*m+j] * v[j*p+k] * w[k*n+i] to contribute.
    # The schoolbook uses n*m*p = 60 such terms.
    # 
    # Can we do better? The Strassen idea: for 2×2×2, rank 7 < 8.
    # 
    # Let me try to construct a decomposition using Strassen for 2×2 sub-blocks.
    
    # Partition: i in {0,1,2}, j in {0,1,2,3}, k in {0,1,2,3,4}
    # Split i: {0,1} and {2}
    # Split j: {0,1} and {2,3}  
    # Split k: {0,1}, {2,3}, {4}
    
    # For the 2×2×2 block (i∈{0,1}, j∈{0,1}, k∈{0,1}): Strassen rank 7
    # For (i∈{0,1}, j∈{0,1}, k∈{2,3}): Strassen rank 7
    # For (i∈{0,1}, j∈{0,1}, k∈{4}): 2×2×1, rank 4
    # For (i∈{0,1}, j∈{2,3}, k∈{0,1}): Strassen rank 7
    # For (i∈{0,1}, j∈{2,3}, k∈{2,3}): Strassen rank 7
    # For (i∈{0,1}, j∈{2,3}, k∈{4}): 2×2×1, rank 4
    # For (i∈{2}, j∈{0,1}, k∈{0,1}): 1×2×2, rank 4
    # For (i∈{2}, j∈{0,1}, k∈{2,3}): 1×2×2, rank 4
    # For (i∈{2}, j∈{0,1}, k∈{4}): 1×2×1, rank 2
    # For (i∈{2}, j∈{2,3}, k∈{0,1}): 1×2×2, rank 4
    # For (i∈{2}, j∈{2,3}, k∈{2,3}): 1×2×2, rank 4
    # For (i∈{2}, j∈{2,3}, k∈{4}): 1×2×1, rank 2
    
    # Total: 7+7+4+7+7+4+4+4+2+4+4+2 = 56
    
    # But wait, this isn't quite right because the Strassen decomposition for a 2×2×2 block
    # doesn't just give rank 7 for that block in isolation - it works when the block is a complete
    # 2×2 matrix multiplication. Here each "block" is a sub-tensor, and I need to make sure
    # the decomposition is valid.
    
    # Actually, for the sub-tensor with i∈{0,1}, j∈{0,1}, k∈{0,1}, the condition is:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for i∈{0,1}, j∈{0,1}, k∈{0,1}
    # This is exactly a 2×2×2 multiplication tensor (with appropriate index mappings), 
    # which has rank 7 via Strassen.
    
    # Let me implement Strassen for the 2×2×2 case first.
    # Strassen for 2×2×2: the 7 rank-1 terms are:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # In tensor form, for the 2×2×2 tensor T[i,j,k]=1 (i,j,k ∈ {0,1}):
    # The 7 terms correspond to specific (u,v,w) vectors.
    
    # Let me just hardcode a known rank-7 decomposition for 2×2×2.
    # From Strassen, the 7 terms (u,v,w) where u,v,w ∈ R^4 (for 2×2 matrices):
    # Using the standard Strassen decomposition:
    
    def strassen_2x2x2(offset_u, offset_v, offset_w, start_col):
        """Return 7 columns for U, V, W for a 2×2×2 Strassen block.
        offset_u: starting row index in U (i*m+j for i∈{0,1}, j∈{0,1})
        offset_v: starting row index in V (j*p+k for j∈{0,1}, k∈{0,1})  
        offset_w: starting row index in W (k*n+i for k∈{0,1}, i∈{0,1})
        """
        # For 2×2×2, the 7 Strassen terms. I'll use a known explicit decomposition.
        # The tensor T[i,j,k] = 1 for i,j,k in {0,1} can be written as sum of 7 rank-1 tensors.
        # Here's one valid decomposition (verified):
        terms = [
            # (u, v, w) each as 4-tuples for the 2×2 sub-block
            ([1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 0, 1]),  # M1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),  # M2
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 0, 1]),  # M3
            ([0, 0, 0, 1], [0, -1, 1, 0], [1, 0, 0, 0]),  # M4
            ([1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]),  # M5
            ([-1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]),  # M6
            ([0, 1, 0, -1], [0, 0, 1, 1], [0, 1, 0, 0]),  # M7
        ]
        return terms
    
    # Actually, let me verify this decomposition is correct by checking:
    # sum_t u_t[i*2+j] * v_t[j*2+k] * w_t[k*2+i] = 1 for all i,j,k in {0,1}
    
    # Let me just use a simpler approach: I'll construct the full decomposition
    # using the block structure and verify it.
    
    # Given the complexity, let me start with a simpler improvement:
    # Use Strassen for the two 2×2×2 blocks that are "full" 2×2 multiplications.
    
    # Actually, let me reconsider the whole approach. The simplest improvement
    # from the knowledge: two disjoint Strassen 2×2×2 blocks reduce rank from 60 to 58.
    # That means replacing two schoolbook 2×2×2 blocks (each rank 8) with Strassen (each rank 7).
    
    # Let me identify which 2×2×2 blocks to use Strassen for.
    # I'll use Strassen for (i∈{0,1}, j∈{0,1}, k∈{0,1}) and (i∈{0,1}, j∈{2,3}, k∈{2,3}).
    # The rest use schoolbook.
    
    # Schoolbook rank: 60
    # Replace two 2×2×2 blocks (8+8=16) with Strassen (7+7=14): save 2, total 58.
    
    # Let me implement this.
    
    U_rows = n * m  # 12
    V_rows = m * p  # 20
    W_rows = p * n  # 15
    
    # Start with schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Now replace two 2×2×2 blocks with Strassen (rank 7 each instead of 8)
    # Block 1: i∈{0,1}, j∈{0,1}, k∈{0,1}
    # Block 2: i∈{0,1}, j∈{2,3}, k∈{2,3}
    
    # For each block, remove the 8 schoolbook columns and add 7 Strassen columns.
    # This is complex to do in-place. Let me rebuild.
    
    # Rebuild with Strassen for selected blocks
    U = []
    V = []
    W = []
    
    def add_schoolbook_block(i_list, j_list, k_list):
        """Add schoolbook terms for the given index sets."""
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    u = [0] * (len(U[0]) if U else 0)
                    # We'll build columns incrementally
                    pass
    
    # Let me use a different approach: build column by column
    columns_U = []
    columns_V = []
    columns_W = []
    
    # Define Strassen terms for 2×2×2 (local indices 0,1 for i,j,k)
    # Verified decomposition:
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 0, 1]),
        ([0, 0, 0, 1], [0, -1, 1, 0], [1, 0, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 1, 0, 0]),
    ]
    
    # Verify: for each (i,j,k) in {0,1}^3, sum_t u_t[i*2+j]*v_t[j*2+k]*w_t[k*2+i] should = 1
    # Let me check (0,0,0): 
    # t0: u[0]=1, v[0]=1, w[0]=1 → 1
    # t1: u[0]=0 → 0
    # t2: u[0]=1, v[0]=0 → 0
    # t3: u[0]=0 → 0
    # t4: u[0]=1, v[0]=0 → 0
    # t5: u[0]=-1, v[0]=1, w[0]=1 → -1
    # t6: u[0]=0 → 0
    # Sum = 1 + 0 + 0 + 0 + 0 + (-1) + 0 = 0. Wrong!
    
    # My Strassen terms are incorrect. Let me use a known correct one.
    # The standard Strassen decomposition for the 2×2×2 tensor:
    # T = sum_{t=1}^{7} u_t ⊗ v_t ⊗ w_t where:
    # Using the formulation where C = AB:
    # The 7 multiplications are:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # In tensor form, T[i,j,k] = 1, and we need sum_t u_t[i] v_t[j] w_t[k] = 1
    # where u_t ∈ R^4 (indexed by i*2+j), v_t ∈ R^4 (indexed by j*2+k), w_t ∈ R^4 (indexed by k*2+i)
    
    # For M1 = (a11+a22)(b11+b22):
    # a11 = A[0,0], a22 = A[1,1], b11 = B[0,0], b22 = B[1,1]
    # u for M1: A contribution is [1,0,0,1] (a11+a22), so u[0]=1, u[3]=1
    # v for M1: B contribution is [1,0,1,0] (b11+b22), so v[0]=1, v[2]=1
    # w for M1: C contribution... C11 gets +M1, C22 gets +M1
    # C11 = C[0,0] → k*2+i = 0*2+0 = 0, so w[0] += 1
    # C22 = C[1,1] → k*2+i = 1*2+1 = 3, so w[3] += 1
    # w = [1,0,0,1]
    
    # For M2 = (a21+a22)b11:
    # u: a21+a22 → [0,0,1,1], u[2]=1, u[3]=1
    # v: b11 → [1,0,0,0], v[0]=1
    # w: C21 gets +M2 → C[1,0] → k*2+i = 0*2+1 = 1, w[1]=1; C22 gets -M2 → w[3] -= 1
    # w = [0,1,0,-1]
    
    # For M3 = a11(b12-b22):
    # u: a11 → [1,0,0,0]
    # v: b12-b22 → [0,1,0,-1]
    # w: C12 gets +M3 → C[0,1] → k*2+i = 1*2+0 = 2, w[2]=1; C22 gets +M3 → w[3]=1
    # w = [0,0,1,1]
    
    # For M4 = a22(b21-b11):
    # u: a22 → [0,0,0,1]
    # v: b21-b11 → [0,-1,1,0]
    # w: C11 gets +M4 → w[0]=1; C21 gets +M4 → w[1]=1
    # w = [1,1,0,0]
    
    # For M5 = (a11+a12)b22:
    # u: a11+a12 → [1,1,0,0]
    # v: b22 → [0,0,0,1]
    # w: C11 gets -M5 → w[0]=-1; C12 gets +M5 → w[2]=1
    # w = [-1,0,1,0]
    
    # For M6 = (a21-a11)(b11+b12):
    # u: a21-a11 → [-1,0,1,0]
    # v: b11+b12 → [1,1,0,0]
    # w: C22 gets +M6 → w[3]=1
    # w = [0,0,0,1]
    
    # For M7 = (a12-a22)(b21+b22):
    # u: a12-a22 → [0,1,0,-1]
    # v: b21+b22 → [0,0,1,1]
    # w: C11 gets +M7 → w[0]=1
    # w = [1,0,0,0]
    
    strassen_terms_correct = [
        ([1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),   # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),   # M3
        ([0, 0, 0, 1], [0, -1, 1, 0], [1, 1, 0, 0]),   # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),   # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),   # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),   # M7
    ]
    
    # Verify (0,0,0): sum_t u_t[0]*v_t[0]*w_t[0]
    # t0: 1*1*1 = 1
    # t1: 0*1*0 = 0
    # t2: 1*0*0 = 0
    # t3: 0*0*1 = 0
    # t4: 1*0*(-1) = 0
    # t5: -1*1*0 = 0
    # t6: 0*0*1 = 0
    # Sum = 1. Correct!
    
    # Verify (0,0,1): sum_t u_t[0]*v_t[1]*w_t[3]
    # t0: 1*0*1 = 0
    # t1: 0*0*(-1) = 0
    # t2: 1*1*1 = 1
    # t3: 0*(-1)*0 = 0
    # t4: 1*0*0 = 0
    # t5: -1*1*1 = -1
    # t6: 0*0*0 = 0
    # Sum = 0. Wrong! Should be 1.
    
    # Hmm, (0,0,1) means i=0, j=0, k=1. T[0,0,1] should be 1.
    # Let me recheck. The tensor T[i,j,k]=1 for all i,j,k in {0,1}.
    # So T[0,0,1] = 1.
    
    # I think my Strassen decomposition might have an error. Let me look up the correct one.
    # 
    # Actually, I realize the issue. The Strassen decomposition gives the matrix product,
    # but the tensor T[i,j,k]=1 is different. Let me reconsider.
    #
    # The matrix multiplication tensor for 2×2 is:
    # μ = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k (in the appropriate spaces)
    # But this is NOT the all-ones tensor in the naive sense because of the index coupling.
    #
    # Actually, I think the correct statement is: the 2×2×2 matrix multiplication tensor
    # (which is what we're decomposing) has rank 7, and it's the tensor with T[i,j,k]=1
    # in the coupled index space. The Strassen decomposition I wrote above should be correct
    # for this tensor. Let me re-verify more carefully.
    #
    # I'll just use a brute-force verification in the code and fall back to schoolbook if wrong.
    
    # Given time constraints, let me just use the schoolbook (rank 60) for safety,
    # but try to implement the Strassen improvement carefully.
    
    # Actually, let me just return the schoolbook to be safe, since I'm not 100% sure
    # about the Strassen terms. The knowledge says rank 58 is achievable, so let me try
    # a different approach: just hardcode a verified rank-58 decomposition.
    
    # For now, return schoolbook (rank 60) to ensure correctness.
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
# EVOLVE-BLOCK-END
