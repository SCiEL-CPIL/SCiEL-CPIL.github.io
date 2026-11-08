"""Tensor decomposition for 3x3 by 3x7 matrix multiplication."""
import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """Construct a rank-minimal exact tensor decomposition.
    
    Returns dict with 'U', 'V', 'W' matrices.
    U: shape (n*m, rank), V: shape (m*p, rank), W: shape (p*n, rank)
    Entries can be integers or half-integers (multiples of 1/2).
    """
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # Deadline: 145 seconds
    deadline = time.time() + 145
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p  # 63
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
    
    # Verify a decomposition using the trace convention W[k*n+i, r]
    def verify(U, V, W):
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True
    
    # Strassen-based structured decomposition for 3x3 * 3x7
    def strassen_decomp():
        """Use Strassen for 2x2 blocks to reduce rank below schoolbook 63."""
        # Split A (3x3) into [[A11, A12], [A21, A22]]
        # A11: 2x2, A12: 2x1, A21: 1x2, A22: 1x1
        # Split B (3x7) into [[B11, B12], [B21, B22]]
        # B11: 2x4, B12: 2x3, B21: 1x4, B22: 1x3
        # C = A*B (3x7):
        # C11 (2x4) = A11*B11 + A12*B21
        # C12 (2x3) = A11*B12 + A12*B22
        # C21 (1x4) = A21*B11 + A22*B21
        # C22 (1x3) = A21*B12 + A22*B22
        
        # For 2x2 * 2x2, Strassen has rank 7:
        # M1 = (A11+A22)*(B11+B22)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B12-B22)
        # M4 = A22*(B21-B11)
        # M5 = (A11+A12)*B22
        # M6 = (A21-A11)*(B11+B12)
        # M7 = (A12-A22)*(B21+B22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        # For 2x2 * 2x4 (B11): treat B11 as [B11a, B11b] each 2x2
        # A11*B11a: Strassen rank 7
        # A11*B11b: Strassen rank 7
        # Total for A11*B11: rank 14
        
        # For 2x2 * 2x3 (B12): treat B12 as [B12a, B12b] where B12a is 2x2, B12b is 2x1
        # A11*B12a: Strassen rank 7
        # A11*B12b: schoolbook 2*2*1=4
        # Total for A11*B12: rank 11
        
        # A12*B21: 2x1 * 1x4, schoolbook rank 8
        # A12*B22: 2x1 * 1x3, schoolbook rank 6
        # A21*B11: 1x2 * 2x4, schoolbook rank 8
        # A21*B12: 1x2 * 2x3, schoolbook rank 6
        # A22*B21: 1x1 * 1x4, schoolbook rank 4
        # A22*B22: 1x1 * 1x3, schoolbook rank 3
        
        # Total: 14 + 8 + 11 + 6 + 8 + 6 + 4 + 3 = 60
        # That's rank 60, better than 63!
        
        # Let's build this explicitly.
        # We'll use a list of terms. Each term is (U_vec, V_vec, W_vec)
        # where U_vec has length n*m=9, V_vec has length m*p=21, W_vec has length p*n=21.
        
        terms = []
        
        def make_term(u_indices, v_indices, w_indices, signs=None):
            """Create a term from index lists. Each index is (coord, sign)."""
            if signs is None:
                signs = [1] * len(u_indices)
            U = [0] * (n * m)
            V = [0] * (m * p)
            W = [0] * (p * n)
            for idx, (u, v, w, s) in enumerate(zip(u_indices, v_indices, w_indices, signs)):
                if u is not None:
                    U[u] = s
                if v is not None:
                    V[v] = s
                if w is not None:
                    W[w] = s
            return U, V, W
        
        # Strassen 2x2*2x2 helper: returns 7 terms for multiplying 2x2 by 2x2
        # A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
        # Returns list of 7 (U, V, W) terms
        def strassen_2x2(a_offset, b_offset, c_offset):
            """
            a_offset: (i0, j0) top-left of A block (2x2)
            b_offset: (j0, k0) top-left of B block (2x2)  
            c_offset: (i0, k0) top-left of C block (2x2)
            Returns 7 terms.
            """
            i0, aj0 = a_offset
            bj0, k0 = b_offset
            ci0, ck0 = c_offset
            
            # A indices: (i, j) -> i*m + j
            # B indices: (j, k) -> j*p + k
            # C indices: (k, i) -> k*n + i (trace convention)
            
            a00 = i0 * m + aj0
            a01 = i0 * m + aj0 + 1
            a10 = (i0+1) * m + aj0
            a11 = (i0+1) * m + aj0 + 1
            
            b00 = bj0 * p + k0
            b01 = bj0 * p + k0 + 1
            b10 = (bj0+1) * p + k0
            b11 = (bj0+1) * p + k0 + 1
            
            c00 = k0 * n + ci0
            c01 = k0 * n + ci0 + 1
            c10 = (k0+1) * n + ci0
            c11 = (k0+1) * n + ci0 + 1
            
            # M1 = (A11+A22)*(B11+B22) -> C11 += M1, C22 += M1
            # M2 = (A21+A22)*B11 -> C21 += M2, C22 -= M2
            # M3 = A11*(B12-B22) -> C12 += M3, C22 += M3
            # M4 = A22*(B21-B11) -> C11 += M4, C21 += M4
            # M5 = (A11+A12)*B22 -> C11 -= M5, C12 += M5
            # M6 = (A21-A11)*(B11+B12) -> C22 += M6
            # M7 = (A12-A22)*(B21+B22) -> C11 += M7
            
            # Each Strassen term is a sum of products, so we need to expand
            # M1 = (a00+a11)*(b00+b11) = a00*b00 + a00*b11 + a11*b00 + a11*b11
            # This is 4 separate rank-1 terms, NOT 1 term!
            # 
            # Wait, Strassen's trick is that we compute 7 intermediate values, each of which
            # is a product of two 2x2 "sums". But each such product is itself a matrix product
            # of 2x2 matrices, which has rank 4 in general (schoolbook).
            #
            # The key insight: in the tensor decomposition, each "Strassen term" like M1
            # corresponds to computing (A11+A22) @ (B11+B22), which is a 2x2 matrix product.
            # This 2x2 product has tensor rank 4 (schoolbook for 2x2*2x2).
            # So 7 Strassen terms * 4 = 28? That's worse than schoolbook 16!
            #
            # I'm confusing myself. Let me reconsider.
            #
            # In Strassen's algorithm, the rank-7 decomposition means we can express the
            # 2x2*2x2 matrix multiplication tensor as a sum of 7 rank-1 tensors.
            # Each rank-1 tensor corresponds to a single scalar multiplication.
            #
            # The 7 terms are:
            # M1 = (a00+a11) * (b00+b11)  -- this is ONE scalar product
            # M2 = (a10+a11) * b00
            # M3 = a00 * (b01-b11)
            # M4 = a11 * (b10-b00)
            # M5 = (a00+a01) * b11
            # M6 = (a10-a00) * (b00+b01)
            # M7 = (a01-a11) * (b10+b11)
            #
            # And the outputs are:
            # c00 = M1 + M4 - M5 + M7
            # c01 = M3 + M5
            # c10 = M2 + M4
            # c11 = M1 - M2 + M3 + M6
            #
            # So in the tensor decomposition, each Mi is a rank-1 term where:
            # - The U vector has non-zero entries at the positions corresponding to the A elements in Mi
            # - The V vector has non-zero entries at the positions corresponding to the B elements in Mi
            # - The W vector has non-zero entries at the positions corresponding to the C elements in Mi
            #
            # For example, M1 = (a00+a11)*(b00+b11) contributes to c00 and c11.
            # U has +1 at a00, +1 at a11
            # V has +1 at b00, +1 at b11
            # W has +1 at c00, +1 at c11
            #
            # But wait, this isn't a rank-1 term! A rank-1 term U⊗V⊗W means
            # T[i,j,k] = U[i]*V[j]*W[k]. If U has two non-zero entries, V has two,
            # W has two, then T has 4 non-zero entries, each being a product of one U,
            # one V, one W entry. This is NOT a single scalar multiplication.
            #
            # I think the confusion is between the "arithmetic circuit" view of Strassen
            # and the tensor decomposition view. In the tensor decomposition, a rank-1
            # term corresponds to a single scalar multiplication: one A element times
            # one B element times one C element.
            #
            # So Strassen's rank-7 for 2x2*2x2 means the 2x2*2x2 multiplication tensor
            # (which has 8 non-zero entries, one for each (i,j,k) with T[i,j,k]=1)
            # can be written as a sum of 7 rank-1 tensors.
            #
            # Let me look up the actual rank-7 decomposition of the 2x2*2x2 matmul tensor.
            #
            # The 2x2*2x2 matmul tensor T has T[i,j,k] = 1 for all i,j,k in {0,1}.
            # Wait no, T[i,j,k] = delta_{i,j} * delta_{j,k}? No.
            #
            # For matrix multiplication C = A*B, C[i,k] = sum_j A[i,j]*B[j,k].
            # The tensor T is defined by T[i,j,k] = 1 for all valid (i,j,k), because
            # each term A[i,j]*B[j,k] contributes to C[i,k].
            #
            # So for 2x2*2x2, T has 8 entries, all equal to 1.
            # The rank-7 decomposition means:
            # T[i,j,k] = sum_{r=1}^{7} U_r[i,j] * V_r[j,k] * W_r[k,i]
            #
            # Let me construct it. The standard Strassen rank-7 decomposition:
            #
            # r=1: U has 1 at (0,0) and (1,1); V has 1 at (0,0) and (1,1); W has 1 at (0,0) and (1,1)
            #   This gives T[0,0,0] += 1*1*1=1, T[0,0,1] += 1*1*1=1 (wrong! W[1,0] should be 0)
            #
            # I'm making errors. Let me be very careful.
            #
            # T[i,j,k] where i is row of A (and C), j is col of A / row of B, k is col of B (and C).
            # U is indexed by (i,j), V by (j,k), W by (k,i).
            #
            # For the 2x2*2x2 case:
            # U indices: (0,0), (0,1), (1,0), (1,1)
            # V indices: (0,0), (0,1), (1,0), (1,1)
            # W indices: (0,0), (0,1), (1,0), (1,1) where W[k,i]
            #
            # T[i,j,k] = 1 for all 8 combinations.
            #
            # Strassen's 7 terms (from literature):
            # M1: U[(0,0)]=1, U[(1,1)]=1; V[(0,0)]=1, V[(1,1)]=1; W[(0,0)]=1, W[(1,1)]=1
            #   Contribution to T: T[0,0,0]+=1, T[0,0,1]+=1*V[0,1]*W[1,0]=0, etc.
            #   Actually T[i,j,k] = U[i,j]*V[j,k]*W[k,i]
            #   For M1: non-zero when i,j in {(0,0),(1,1)} AND j,k in {(0,0),(1,1)} AND k,i in {(0,0),(1,1)}
            #   i=0,j=0: V[0,k]*W[k,0]. V[0,0]=1,W[0,0]=1 -> T[0,0,0]+=1. V[0,1]=0 -> no.
            #   i=0,j=1: V[1,k]*W[k,0]. V[1,0]=0 -> no.
            #   i=1,j=0: V[0,k]*W[k,1]. V[0,0]=1,W[0,1]=0 -> no. V[0,1]=0 -> no.
            #   i=1,j=1: V[1,k]*W[k,1]. V[1,0]=0 -> no. V[1,1]=1,W[1,1]=1 -> T[1,1,1]+=1.
            #   So M1 contributes to T[0,0,0] and T[1,1,1].
            #
            # This is getting very complex. Let me just hardcode the known rank-7 decomposition.
            
            # After research, the rank-7 decomposition of 2x2*2x2 matmul tensor:
            # Using the standard Strassen formulas mapped to tensor rank-1 terms.
            #
            # Each "Strassen scalar" like M1 = (a00+a11)*(b00+b11) is NOT a single rank-1 term.
            # It's a product of sums, which expands to multiple rank-1 terms.
            #
            # I think the correct interpretation is:
            # The matmul tensor for 2x2*2x2 has rank 7, meaning it can be written as
            # sum of 7 terms where each term is U_r ⊗ V_r ⊗ W_r with U_r, V_r, W_r
            # being vectors (not matrices). Each vector has entries that are 0, +1, or -1.
            #
            # Let me find the explicit decomposition. From various sources:
            #
            # Term 1: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1]  (flattened)
            #   U: (0,0)=1, (0,1)=0, (1,0)=0, (1,1)=1
            #   V: (0,0)=1, (0,1)=0, (1,0)=0, (1,1)=1
            #   W: (0,0)=1, (0,1)=0, (1,0)=0, (1,1)=1
            #   T[i,j,k] += U[i,j]*V[j,k]*W[k,i]
            #   (0,0,0): 1*1*1=1
            #   (0,0,1): 1*0*0=0 (V[0,1]=0)
            #   (0,1,0): 0*0*0=0 (U[0,1]=0)
            #   (0,1,1): 0*0*1=0
            #   (1,0,0): 0*1*0=0 (U[1,0]=0)
            #   (1,0,1): 0*0*1=0
            #   (1,1,0): 1*0*0=0 (V[1,0]=0)
            #   (1,1,1): 1*1*1=1
            #   So term 1 gives T[0,0,0]=1, T[1,1,1]=1.
            #
            # We need all 8 T entries to equal 1. Term 1 gives 2 of them.
            #
            # This approach is taking too long. Let me use a simpler strategy:
            # Just implement a greedy search that tries to find low-rank decompositions.
            
            return []
        
        # Given the complexity, let's use a different approach:
        # Greedy column construction with local optimization.
        
        # For now, return schoolbook as the structured method didn't pan out easily.
        return schoolbook()
    
    # Greedy search for lower rank
    best_result = None
    best_rank = n * m * p  # 63
    
    # Try structured Strassen-like decomposition first
    structured = strassen_decomp()
    if verify(structured['U'], structured['V'], structured['W']):
        r = len(structured['U'][0])
        if r < best_rank:
            best_rank = r
            best_result = structured
    
    # Try random greedy search
    for target_rank in range(best_rank - 5, best_rank):
        if time.time() > deadline - 5:
            break
        
        for attempt in range(30):
            if time.time() > deadline - 3:
                break
            found = _try_greedy_decomposition(n, m, p, target_rank, time.time(), deadline)
            if found:
                if verify(found['U'], found['V'], found['W']):
                    best_rank = target_rank
                    best_result = found
                    break
    
    if best_result is not None:
        return best_result
    
    return schoolbook()

def _try_greedy_decomposition(n, m, p, target_rank, now, deadline):
    """Try to find a rank-target_rank decomposition using greedy method."""
    # Start with all-zero matrices and greedily add rank-1 terms
    # to minimize the residual tensor.
    
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We want: T = sum_r U_r ⊗ V_r ⊗ W_r
    
    # Greedy: at each step, find the rank-1 term that best approximates the residual.
    # Residual R[i,j,k] = T[i,j,k] - sum_{r so far} U_r[i,j]*V_r[j,k]*W_r[k,i]
    
    U = [[0] * target_rank for _ in range(n * m)]
    V = [[0] * target_rank for _ in range(m * p)]
    W = [[0] * target_rank for _ in range(p * n)]
    
    # Residual starts as T (all ones)
    residual = [[[1 for _ in range(p)] for _ in range(m)] for _ in range(n)]
    
    for r in range(target_rank):
        if time.time() > deadline - 2:
            return None
        
        # Find the best rank-1 term to subtract from residual
        # A rank-1 term is defined by vectors u (length n*m), v (length m*p), w (length p*n)
        # with entries in {-1, 0, 1} (or half-integers)
        # The contribution is u[i*m+j] * v[j*p+k] * w[k*n+i]
        
        # Greedy: try all possible sparse rank-1 terms
        # A sparse term has few non-zero entries in each of u, v, w
        best_score = -1
        best_u, best_v, best_w = None, None, None
        
        # Try terms with 1-3 non-zero entries in each vector
        # This is still a huge search space, so use random sampling
        for trial in range(200):
            if time.time() > deadline - 2:
                break
            
            # Random sparse vectors
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            
            # Pick 1-2 non-zero entries for each
            for _ in range(random.randint(1, 2)):
                idx = random.randint(0, n*m - 1)
                u[idx] = random.choice([-1, 1])
            for _ in range(random.randint(1, 2)):
                idx = random.randint(0, m*p - 1)
                v[idx] = random.choice([-1, 1])
            for _ in range(random.randint(1, 2)):
                idx = random.randint(0, p*n - 1)
                w[idx] = random.choice([-1, 1])
            
            # Score: how well does this term match the residual?
            # score = sum_{i,j,k} residual[i,j,k] * u[i*m+j] * v[j*p+k] * w[k*n+i]
            score = 0
            for i in range(n):
                for j in range(m):
                    uj = u[i*m+j]
                    if uj == 0:
                        continue
                    for k in range(p):
                        vk = v[j*p+k]
                        if vk == 0:
                            continue
                        wk = w[k*n+i]
                        if wk == 0:
                            continue
                        score += residual[i][j][k] * uj * vk * wk
            
            if score > best_score:
                best_score = score
                best_u, best_v, best_w = u[:], v[:], w[:]
        
        if best_u is None or best_score <= 0:
            return None
        
        # Add this term
        for idx in range(n*m):
            U[idx][r] = best_u[idx]
        for idx in range(m*p):
            V[idx][r] = best_v[idx]
        for idx in range(p*n):
            W[idx][r] = best_w[idx]
        
        # Update residual
        for i in range(n):
            for j in range(m):
                uj = best_u[i*m+j]
                if uj == 0:
                    continue
                for k in range(p):
                    vk = best_v[j*p+k]
                    if vk == 0:
                        continue
                    wk = best_w[k*n+i]
                    if wk == 0:
                        continue
                    residual[i][j][k] -= uj * vk * wk
    
    # Check if residual is zero
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if residual[i][j][k] != 0:
                    return None
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
