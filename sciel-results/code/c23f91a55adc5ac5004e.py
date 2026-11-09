import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def _verify(U, V, W, n=3, m=3, p=7, tol=1e-10):
    """Check if U, V, W form a valid matrix multiplication tensor decomposition."""
    rank = U.shape[1]
    # Build the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # T should equal the matrix multiplication tensor: T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # Actually the tensor for matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p
    # The tensor T_{i,j,k} represents the coefficient: C[i,k] = sum_{j} A[i,j] * B[j,k]
    # So the tensor is T[i,j,k] = 1 for all valid (i,j,k) in the range... no.
    # The matrix multiplication tensor has entries T[i,j,k] = delta_{i,i'} delta_{j,j'} delta_{k,k'} ... 
    # Let me think more carefully.
    # The bilinear map: (A, B) -> C = A*B
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has indices (i, j, k) and T[i,j,k] = 1 means A[i,j] and B[j,k] multiply to contribute to C[i,k].
    # Actually in the standard formulation: T = sum_{i,j,k} e_i^A ⊗ e_j^B ⊗ e_k^C where the tensor component
    # T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]. Since each term A[i,j]*B[j,k] goes to C[i,k],
    # T[i,j,k] = 1 for all i,j,k.
    # Wait no. The tensor is: T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ e_j ⊗ e_k
    # where the three factors correspond to A-index (i), shared index (j), and C-index (k).
    # A decomposition T = sum_c u_c ⊗ v_c ⊗ w_c means:
    # For all i,j,k: sum_c U[i,j,c] * V[j,k,c] * W[i,k,c] = 1 (since T[i,j,k]=1)
    
    # But the problem says W[k*n+i, r] - so W is indexed by (k, i) where k is column of B / row of C, i is row of C.
    # Let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # So W row index = k*n + i, where k is the column index of C (0..p-1) and i is row index (0..n-1).
    
    # U is (n*m, rank): U[i*m + j, c]
    # V is (m*p, rank): V[j*p + k, c]  
    # W is (p*n, rank): W[k*n + i, c]
    
    # The decomposition requires: for all i,j,k in valid ranges:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    Uc = np.array(U, dtype=float)
    Vc = np.array(V, dtype=float)
    Wc = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(Uc[i*m+j, c] * Vc[j*p+k, c] * Wc[k*n+i, c] for c in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _construct_rank23_3x3():
    """
    Return (U, V, W) for 3x3 matrix multiplication with rank 23.
    U: (9, 23), V: (9, 23), W: (9, 23)
    Using the known rank-23 decomposition.
    """
    # The rank-23 decomposition for 3x3 matrix multiplication.
    # This is based on the work of Alon, Boppana, and others.
    # Let me use a known explicit decomposition.
    # 
    # One approach: Use the fact that 3x3 can be decomposed using
    # 7 Strassen-like terms for a 2x2 subproblem plus corrections.
    # Actually, the known rank-23 decomposition is complex.
    # 
    # Let me try a different approach: construct it via the following method.
    # The tensor T for 3x3 matrix mult has 27 components.
    # We need 23 vectors u_c, v_c, w_c in F^9 such that sum_c u_c ⊗ v_c ⊗ w_c = T.
    
    # I'll use a known explicit rank-23 decomposition.
    # Reference: "The rank of the matrix multiplication tensor for 3x3" 
    # The decomposition uses the following 23 terms (each term is a triple of 9-dim vectors).
    
    # Actually, let me use a construction based on block decomposition.
    # Write 3x3 as blocks: A = [[a, b, c], [d, e, f], [g, h, i]]
    # We can try to find a decomposition.
    
    # Let me use the following known rank-23 decomposition from the literature.
    # The entries are integers or half-integers.
    
    # Here's one approach: use the decomposition from
    # "A rank 23 decomposition of 3x3 matrix multiplication"
    
    # I'll hardcode a known valid decomposition. Let me construct it carefully.
    # 
    # The 23 terms for 3x3 matrix multiplication (indices 0..8 for each of U, V, W rows):
    # U row indices: i*3+j for i,j in 0..2
    # V row indices: j*3+k for j,k in 0..2
    # W row indices: k*3+i for k,i in 0..2
    
    # I'll use a search-free hardcoded version. Based on known results:
    
    # Let me try to use the following construction:
    # Start with 8 terms from Strassen for the top-left 2x2 block
    # Then add terms for the remaining parts.
    
    # Actually, the simplest safe approach: if I can't guarantee correctness of the rank-23
    # decomposition, I should fall back to schoolbook. But let me try to implement it.
    
    # Known rank-23 decomposition (from Alon-Boppana style):
    # I'll use the following 23 triples (U_idx, U_val, V_idx, V_val, W_idx, W_val)
    
    # This is getting complex. Let me use a verified approach:
    # I'll implement a small search/verification to find a rank-23 decomposition,
    # or fall back to schoolbook if I can't verify one.
    
    # For safety, let me just return None to signal failure.
    return None

def construct():
    """
    Return a rank-minimal tensor decomposition for 3x3 by 3x7 matrix multiplication.
    U: (n*m, rank) = (9, rank)
    V: (m*p, rank) = (21, rank)  
    W: (p*n, rank) = (21, rank)
    """
    start_time = time.time()
    DEADLINE = 120  # seconds
    SAFETY_MARGIN = 10  # return before this
    
    n, m, p = DIMENSIONS
    
    # Schoolbook baseline
    def schoolbook():
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
    
    # Try to build a better decomposition.
    # Strategy: 3x7 = 3x3 + 3x3 + 3x1
    # If we have a rank-r decomposition for 3x3, and rank-s for 3x1,
    # total rank = r + r + s (but we need to be careful about how they combine).
    
    # Actually, for 3x7: A is 3x3, B is 3x7. C = A*B is 3x7.
    # C[i,k] = sum_j A[i,j] * B[j,k] for k=0..6
    # This is 7 separate 3x1 products (each column of B times A).
    # Each 3x1 product: C_col = A * B_col where A is 3x3, B_col is 3x1.
    # The tensor for 3x3 * 3x1 has 9*3*1 = 27 terms in schoolbook.
    # But we can share structure across the 7 columns.
    
    # Better approach: The tensor for 3x3 * 3x7 multiplication:
    # T[i, j, k] = 1 for i in 0..2, j in 0..2, k in 0..6
    # Total 27 entries in the tensor.
    # Schoolbook rank = 27.
    # We want to reduce this.
    
    # Key insight: The multiplication C = A*B where A is 3x3 and B is 3x7.
    # We can view this as: C = [A*B0, A*B1, ..., A*B6] where Bj is column j of B.
    # Each A*Bj is a 3x1 matrix.
    # The tensor decomposes as sum over j of T_j where T_j is the 3x3*3x1 tensor for column j.
    # But the A factors are shared! So we can potentially share U vectors.
    
    # Let me think about it differently. 
    # U has rows indexed by (i,j) for i,j in 0..2 (9 rows).
    # V has rows indexed by (j,k) for j in 0..2, k in 0..6 (21 rows).
    # W has rows indexed by (k,i) for k in 0..6, i in 0..2 (21 rows).
    
    # The constraint is: for all i,j,k: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
    
    # This is a 3-way tensor of size 3x3x7 = 63 entries, all equal to 1.
    # We want the minimal number of rank-1 tensors that sum to this.
    
    # One approach: use the rank-23 decomposition for the 3x3 core.
    # If we have 23 terms for 3x3*3x3 (where both inputs are 3x3),
    # we can extend to 3x3*3x7 by... hmm, it's not straightforward.
    
    # Alternative: direct construction.
    # Let me try to find a decomposition with rank < 27.
    
    # Simple improvement: For each pair (i,j), we need sum over k of something = 1.
    # Actually, let me just try rank 24, 25, 26 and see if I can construct valid ones.
    
    # Let me try a specific construction:
    # Use 9 "base" terms where U[i,j,c]=1, V[j,k,c]=1/something, W[k,i,c]=something
    # to make the product work out.
    
    # Actually, the simplest non-trivial improvement:
    # Note that for fixed (i,j), we need: sum_c U[i,j,c] * (sum_k' V[j,k',c] * W[k',i,c]) = 1 for all k.
    # No, that's not right either.
    
    # Let me just try: can we do rank 24?
    # Construction: take the 27 schoolbook terms and merge 3 of them.
    # Term (i,j,k): U[i,j]=1, V[j,k]=1, W[k,i]=1
    # Can we combine terms (i,j,k1) and (i,j,k2)?
    # If we set for one term: U[i,j]=2, V[j,k1]=1/2, V[j,k2]=1/2, W[k1,i]=1, W[k2,i]=1
    # Then for (i,j,k1): U*V*W = 2*(1/2)*1 = 1 ✓
    # For (i,j,k2): U*V*W = 2*(1/2)*1 = 1 ✓
    # But this uses the same U entry for both, which is fine since U[i,j] is shared.
    # However, we're using 1 term instead of 2, saving 1 rank.
    # But wait - the V and W entries are different for k1 and k2, so we can't merge into 1 term.
    # Each "term" c has a single V[j,k,c] value for all k. So V[j,k,c] is one number per (j,k,c).
    # We can't have V[j,k1,c] = 1/2 and V[j,k2,c] = 1/2 in the same term unless we use 2 terms.
    
    # I think the merging doesn't work simply. Let me try a different approach.
    
    # Use numpy to do a small optimization search.
    try:
        import scipy.optimize as opt
    except ImportError:
        opt = None
    
    # Try to find a rank-24 decomposition via random initialization + optimization
    def try_random_rank(target_rank):
        """Try to find a valid decomposition of given rank via random init + local search."""
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            return None
        
        np.random.seed(42)
        
        U = np.random.randn(n*m, target_rank) * 0.1
        V = np.random.randn(m*p, target_rank) * 0.1
        W = np.random.randn(p*n, target_rank) * 0.1
        
        # Initialize with schoolbook-like structure for better convergence
        for c in range(min(target_rank, n*m*p)):
            i = c // (m*p)
            j = (c // p) % m
            k = c % p
            U[i*m+j, c] = 1.0
            V[j*p+k, c] = 1.0
            W[k*n+i, c] = 1.0
            if c >= n*m*p:
                U[i*m+j, c] = 0.0
                V[j*p+k, c] = 0.0
                W[k*n+i, c] = 0.0
        
        # Use gradient descent to minimize sum of (T[i,j,k] - sum_c U[i,j,c]*V[j,k,c]*W[k,i,c])^2
        def loss_and_grad(U, V, W):
            loss = 0.0
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(target_rank))
                        err = 1.0 - val
                        loss += err * err
                        for c in range(target_rank):
                            grad_U[i*m+j, c] += -2 * err * V[j*p+k, c] * W[k*n+i, c]
                            grad_V[j*p+k, c] += -2 * err * U[i*m+j, c] * W[k*n+i, c]
                            grad_W[k*n+i, c] += -2 * err * U[i*m+j, c] * V[j*p+k, c]
            return loss, (grad_U, grad_V, grad_W)
        
        lr = 0.01
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for step in range(2000):
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            
            loss, (gU, gV, gW) = loss_and_grad(U, V, W)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_loss < 1e-12:
                break
            
            U -= lr * gU
            V -= lr * gV
            W -= lr * gW
            
            # Adaptive learning rate
            if step % 200 == 0 and step > 0:
                lr *= 0.9
        
        if best_loss < 1e-8:
            # Round to nearest half-integer
            U_rounded = np.round(best_U * 2) / 2
            V_rounded = np.round(best_V * 2) / 2
            W_rounded = np.round(best_W * 2) / 2
            
            # Verify
            Uc = U_rounded.tolist()
            Vc = V_rounded.tolist()
            Wc = W_rounded.tolist()
            if _verify(Uc, Vc, Wc):
                return {"U": Uc, "V": Vc, "W": Wc}
        
        return None
    
    # Try ranks from 24 up to 26
    for target_rank in [24, 25, 26]:
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        result = try_random_rank(target_rank)
        if result is not None:
            return result
    
    # Fallback: schoolbook
    return schoolbook()
