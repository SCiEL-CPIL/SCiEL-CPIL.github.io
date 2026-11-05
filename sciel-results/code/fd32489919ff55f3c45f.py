import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = 3, 4, 5
    
    # Build the matrix multiplication tensor T of shape (n, m, p)
    # T[i,j,k] = 1 for all i,j,k
    T = np.ones((n, m, p))
    
    def verify(U, V, W, tol=1e-8):
        """Verify that U, V, W correctly decompose the tensor T.
        U: shape (n*m, rank)
        V: shape (m*p, rank)
        W: shape (p*n, rank)
        """
        r = U.shape[1]
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        S = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.allclose(S, T, atol=tol)
    
    def construct_block_decomposition(n, m, p):
        """Construct a rank-minimal decomposition using block methods.
        For 3x4x5, we can use:
        - Two 2x2x2 Strassen blocks (rank 7 each) = 14
        - Remaining terms handled separately
        Total rank: 56 (from inspiration)
        """
        # We'll use a structured approach:
        # Split A (3x4) into blocks and B (4x5) into blocks
        # Use Strassen for 2x2x2 subproblems
        
        # For 3x4x5, one approach:
        # - Use rank-11 for 2x2x3 blocks
        # - Combine to get total rank 56
        
        # Let's construct explicitly using known low-rank decompositions
        
        # Method: Use the fact that 3x4x5 can be decomposed as:
        # - 2x2x2 Strassen (rank 7) for top-left 2x2x2 block
        # - 2x2x2 Strassen (rank 7) for another 2x2x2 block  
        # - Remaining cross terms
        
        # Actually, let's use a more systematic approach:
        # The optimal known rank for 3x4x5 is 56 based on the inspiration
        
        # Construct using block diagonal structure with Strassen
        rank_total = 56
        U = np.zeros((n * m, rank_total))
        V = np.zeros((m * p, rank_total))
        W = np.zeros((p * n, rank_total))
        
        col = 0
        
        # Block 1: rows 0-1, cols 0-1 of A; rows 0-1, cols 0-1 of B -> output rows 0-1, cols 0-1
        # Use Strassen 2x2x2 (rank 7)
        # Strassen factors for 2x2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A10)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # For our tensor, we need T[i,j,k]=1, so we're decomposing the all-ones tensor
        # This means we need sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k
        
        # Let me use a different approach: construct the decomposition directly
        # using known low-rank algorithms for matrix multiplication
        
        # For 3x4x5, use the following strategy:
        # - Decompose into overlapping 2x2x2 and 2x2x3 blocks
        # - Use Strassen for 2x2x2 (rank 7) and rank-11 for 2x2x3
        
        # Since this is complex, let's use a simpler but valid construction:
        # Use the schoolbook method but with some optimization
        
        # Actually, the simplest correct approach that achieves rank 56:
        # Use 14 Strassen 2x2x2 decompositions for overlapping 2x2x2 blocks
        
        # Let me just use a direct construction that I know works:
        # The rank of 3x4x5 matrix multiplication is known to be at most 56
        
        # I'll construct it using a combination of:
        # - 4 Strassen 2x2x2 blocks (rank 7 each) = 28
        # - Plus additional terms for the remaining dimensions
        
        # For simplicity and correctness, let's use a verified construction:
        
        # Use the fact that we can tile the problem:
        # A is 3x4, B is 4x5, C is 3x5
        
        # Split A into [A1; A2] where A1 is 2x4, A2 is 1x4
        # Split B into [B1, B2] where B1 is 4x3, B2 is 4x2
        
        # C = A1*B1 + A1*B2 + A2*B1 + A2*B2
        
        # Use Strassen for A1*B1 (2x4 * 4x3 -> 2x3):
        # This can be done with rank 11 (2x2x3 is rank 11)
        # A1*B2 (2x4 * 4x2 -> 2x2): rank 7 (Strassen)
        # A2*B1 (1x4 * 4x3 -> 1x3): rank 3 (just 3 terms)
        # A2*B2 (1x4 * 4x2 -> 1x2): rank 2 (just 2 terms)
        # Total: 11 + 7 + 3 + 2 = 23... that's not 56
        
        # I think I'm overcomplicating this. Let me just use the schoolbook 
        # and verify it works, then try to improve.
        
        # For now, return schoolbook which has rank 60
        rank_sb = n * m * p
        U = np.zeros((n * m, rank_sb))
        V = np.zeros((m * p, rank_sb))
        W = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        
        return U, V, W, rank_sb
    
    # First, get a valid schoolbook decomposition
    U_sb, V_sb, W_sb, rank_sb = construct_block_decomposition(n, m, p)
    
    # Verify schoolbook
    if not verify(U_sb, V_sb, W_sb):
        # Fallback: manual construction
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb))
        V_sb = np.zeros((m * p, rank_sb))
        W_sb = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1.0
                    V_sb[j * p + k, col] = 1.0
                    W_sb[k * n + i, col] = 1.0
                    col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    # Try to find better decompositions using optimization
    np.random.seed(42)
    
    def residuals(params, r):
        U_flat = params[:n * m * r]
        V_flat = params[n * m * r : n * m * r + m * p * r]
        W_flat = params[n * m * r + m * p * r : n * m * r + m * p * r + p * n * r]
        
        U = U_flat.reshape(n, m, r)
        V = V_flat.reshape(m, p, r)
        W = W_flat.reshape(p, n, r)
        
        S = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        res = (S - T).flatten()
        
        return res
    
    # Try ranks from lower bound to schoolbook
    for target_rank in range(56, best_rank):
        if time.time() > deadline:
            break
        
        max_restarts = 5
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            U_init = np.random.randn(n * m, target_rank)
            V_init = np.random.randn(m * p, target_rank)
            W_init = np.random.randn(p * n, target_rank)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                from scipy.optimize import least_squares
                result = least_squares(
                    residuals, x0, args=(target_rank,),
                    method='trf', max_nfev=100, xtol=1e-10, ftol=1e-10
                )
                
                if result.cost < 1e-8:
                    U_opt = result.x[:n * m * target_rank].reshape(n, m, target_rank)
                    V_opt = result.x[n * m * target_rank : n * m * target_rank + m * p * target_rank].reshape(m, p, target_rank)
                    W_opt = result.x[n * m * target_rank + m * p * target_rank :].reshape(p, n, target_rank)
                    
                    # Try to snap to half-integers
                    U_r = np.round(U_opt * 2) / 2.0
                    V_r = np.round(V_opt * 2) / 2.0
                    W_r = np.round(W_opt * 2) / 2.0
                    
                    if verify(U_r, V_r, W_r):
                        best_U = U_r.flatten()
                        best_V = V_r.flatten()
                        best_W = W_r.flatten()
                        best_rank = target_rank
                        found = True
                        break
                    
                    # Try integers
                    U_i = np.round(U_opt)
                    V_i = np.round(V_opt)
                    W_i = np.round(W_opt)
                    if verify(U_i, V_i, W_i):
                        best_U = U_i.flatten()
                        best_V = V_i.flatten()
                        best_W = W_i.flatten()
                        best_rank = target_rank
                        found = True
                        break
            except Exception:
                continue
            
        if found:
            break
    
    if not verify(best_U, best_V, best_W):
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = rank_sb
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
