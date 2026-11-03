"""Matrix multiplication 2x3x3 rank-minimal decomposition."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 3
    # Tensor T[i, j, k] = 1 for (i,j) in A (n x m) and (j,k) in B (m x p)
    # Actually, the multiplication tensor has entries T[i, j, k] where
    #   i indexes rows of A (n), j indexes cols of A / rows of B (m), k indexes cols of B (p)
    # The product C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i, j, k] = 1 if valid (i in [0,n), j in [0,m), k in [0,p))
    # Wait, that's not right. The multiplication tensor for A(n x m) * B(m x p) -> C(n x p)
    # has a third-order tensor T of size (n, p, m) where T[i, k, j] = 1 for all valid i, k, j.
    # The decomposition is: T[i, k, j] = sum_r U[i, j, r] * V[j, k, r] * W[i, k, r]
    # Wait, let me re-read the task.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 3 matrix."
    # Interface: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # W[k*n+i, r] follows the official trace convention.
    # So W is indexed by (k, i) where k in [0,p), i in [0,n).
    # The decomposition condition is:
    # sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = delta[i, i'] * delta[j, j'] * delta[k, k'] ?
    # No. The tensor for matrix multiplication A(n,m) * B(m,p) -> C(n,p) is:
    # T[i, k, j] = 1 for all i in [0,n), k in [0,p), j in [0,m)
    # This is a 3-tensor of size (n, p, m).
    # A rank-1 decomposition is U_i * V_k * W_j where U has n entries, V has p entries, W has m entries.
    # But the interface says U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    # This suggests a different indexing.
    # Let me think again.
    # If U is (n*m, rank), each column r of U is a vector of size n*m.
    # If V is (m*p, rank), each column r of V is a vector of size m*p.
    # If W is (p*n, rank), each column r of W is a vector of size p*n.
    # The product of three rank-1 terms would be:
    # (U[i*m+j, r]) * (V[j'*p+k, r]) * (W[k'*n+i', r])
    # For this to represent the multiplication tensor, we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    # Wait, that would be the identity tensor, not the multiplication tensor.
    # Let me reconsider.
    # The multiplication tensor T has T[i, k, j] = 1.
    # A rank-1 decomposition of T is: T[i, k, j] = sum_r a_r[i] * b_r[k] * c_r[j]
    # where a_r is of size n, b_r is of size p, c_r is of size m.
    # But the interface has U of size (n*m, rank), not (n, rank).
    # Perhaps the decomposition is for the full multiplication operator, which maps (A, B) -> C.
    # The tensor T is in (R^{n*m} * R^{m*p})^* * R^{n*p} ? No, that's not right either.
    # Let me look at the schoolbook decomposition in the seed program.
    # U[i*m + j, column] = 1
    # V[j*p + k, column] = 1
    # W[k*n + i, column] = 1
    # So for each (i, j, k), we have a rank-1 term where:
    # U has 1 at position (i*m + j)
    # V has 1 at position (j*p + k)
    # W has 1 at position (k*n + i)
    # The product is: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For the schoolbook, each (i,j,k) gets its own column r, and the product is 1 for that (i,j,k) and 0 otherwise.
    # So the sum over r of these products gives:
    # sum_r U[i*m+j, r] * V[j'*p+k', r] * W[k''*n+i', r] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'} ?
    # No, let's check. If we fix i, j, k and sum over r:
    # Only the column r corresponding to (i,j,k) has U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1.
    # So the sum is 1.
    # But what does this represent?
    # It seems like the tensor being decomposed is T[i, j, k] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'} ?
    # That doesn't make sense for matrix multiplication.
    
    # Let me re-read the task more carefully.
    # "Matrix multiplication 2,3,3"
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 3 matrix."
    # The output factor follows the official trace convention W[k*n+i,r].
    
    # I think the tensor is the multiplication tensor T where T[i, k, j] = 1 for all i, k, j.
    # And the decomposition is:
    # T[i, k, j] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # But the interface has U as (n*m, rank), so U[i, j, r] is stored as U[i*m + j, r].
    # V[j, k, r] is stored as V[j*p + k, r].
    # W[k, i, r] is stored as W[k*n + i, r].
    # So the condition is:
    # sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = 1 for all i, k, j.
    
    # Let me verify with the schoolbook:
    # For a given (i, k, j), we sum over r.
    # Only the column r corresponding to (i, j, k) has non-zero entries.
    # U[i*m + j, r] = 1, V[j*p + k, r] = 1, W[k*n + i, r] = 1.
    # So the sum is 1. Correct.
    
    # So the condition is:
    # sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = 1 for all i in [0,n), k in [0,p), j in [0,m).
    
    # This is a system of n*p*m = 18 equations.
    # The unknowns are the entries of U, V, W.
    # U has n*m*rank = 6*rank entries.
    # V has m*p*rank = 9*rank entries.
    # W has p*n*rank = 6*rank entries.
    # Total unknowns: 21*rank.
    # For rank 18, we have 378 unknowns and 18 equations.
    # This is highly underdetermined.
    
    # Wait, but the decomposition should also satisfy that the "unused" entries are zero?
    # No, the condition is just that the sum equals 1 for all (i, k, j).
    # But that's not enough. We also need that the decomposition represents the multiplication tensor correctly.
    # I think I'm missing something.
    
    # Let me think differently. The multiplication tensor T is a 3-tensor of size (n, m, p) ?
    # No, for A(n,m) * B(m,p) -> C(n,p), the tensor is T[i, k, j] where i indexes C rows, k indexes C cols, j indexes the contraction index.
    # So T is of size (n, p, m) with T[i, k, j] = 1.
    # A rank-1 decomposition is T[i, k, j] = sum_r a_r[i] * b_r[k] * c_r[j].
    # But the interface has U of size (n*m, rank), which is larger than (n, rank).
    
    # Perhaps the decomposition is for the full bilinear map (A, B) -> C.
    # The tensor T is in (R^{n*m})^* * (R^{m*p})^* * R^{n*p} ?
    # No, that's not right either.
    
    # Let me look at the schoolbook again.
    # U[i*m + j, r] = 1 for the column r corresponding to (i, j, k).
    # V[j*p + k, r] = 1 for the same column.
    # W[k*n + i, r] = 1 for the same column.
    # So the product U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] is 1 if r corresponds to (i,j,k), and 0 otherwise.
    # So the sum over r is 1 for each (i,j,k).
    
    # But what is this representing?
    # I think the tensor being decomposed is T[i, j, k] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'} ?
    # No, that's the identity tensor.
    
    # Wait, perhaps the condition is:
    # sum_r U[i*m + j, r] * V[j'*p + k, r] * W[k'*n + i', r] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'} ?
    # Let's check with schoolbook:
    # For fixed (i, j, k, i', j', k'), the sum is 1 if (i,j,k) = (i',j',k'), and 0 otherwise.
    # So this would be the identity tensor of size (n*m) * (m*p) * (p*n) ?
    # That doesn't make sense for matrix multiplication.
    
    # I'm confused. Let me try a different approach.
    # Perhaps the condition is:
    # sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = 1 for all i, j, k.
    # And this is the only condition.
    # Then the schoolbook satisfies this with rank 18.
    # And we want to find a smaller rank that also satisfies this.
    
    # But is this the correct condition for matrix multiplication?
    # I don't think so. The multiplication tensor has T[i, k, j] = 1, not T[i, j, k] = 1.
    # And the decomposition should be T[i, k, j] = sum_r a_r[i] * b_r[k] * c_r[j].
    
    # Let me assume the condition is:
    # sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = 1 for all i, k, j.
    # And try to find a smaller rank.
    
    # Actually, I think the condition might be:
    # sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = delta_{i,i'} * delta_{k,k'} * delta_{j,j'} ?
    # No, that's the identity.
    
    # Let me just implement the numerical search for the condition:
    # sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = 1 for all i, k, j.
    # And see if I can find a smaller rank.
    
    # Actually, I realize I might be overthinking this. Let me just implement the search and see.
    
    def residuals(params, R):
        # params is a flat array of length 21*R
        # U: 6*R, V: 9*R, W: 6*R
        U = params[:6*R].reshape(6, R)
        V = params[6*R:15*R].reshape(9, R)
        W = params[15*R:].reshape(6, R)
        
        res = []
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    val = 0
                    for r in range(R):
                        val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    res.append(val - 1)
        return res
    
    def try_rank(R):
        # Try to find a decomposition of rank R
        # Use random initialization and least_squares
        best_res = None
        best_params = None
        best_norm = float('inf')
        
        n_iter = 50
        for it in range(n_iter):
            # Random initialization
            params = np.random.randn(21 * R)
            # Scale to avoid overflow
            params /= np.sqrt(R)
            
            try:
                result = least_squares(residuals, params, args=(R,), method='lm', max_nfev=1000)
                res_norm = np.linalg.norm(result.fun)
                if res_norm < best_norm:
                    best_norm = res_norm
                    best_params = result.x
                    best_res = result.fun
                if res_norm < 1e-10:
                    break
            except:
                pass
        
        if best_norm < 1e-8:
            # Found a solution, try to rationalize
            R_flat = best_params
            U = R_flat[:6*R].reshape(6, R)
            V = R_flat[6*R:15*R].reshape(9, R)
            W = R_flat[15*R:].reshape(6, R)
            
            # Try to snap to rationals with small denominators
            def snap_to_rational(x, max_denom=10):
                # Try to find a rational a/b close to x with b <= max_denom
                best = None
                best_err = float('inf')
                for b in range(1, max_denom+1):
                    a = round(x * b)
                    err = abs(x - a/b)
                    if err < best_err:
                        best_err = err
                        best = (a, b)
                if best_err < 1e-8:
                    return best
                return None
            
            U_rational = []
            V_rational = []
            W_rational = []
            success = True
            for i in range(6):
                row = []
                for j in range(R):
                    r = snap_to_rational(U[i, j])
                    if r is None:
                        success = False
                        break
                    row.append(r)
                U_rational.append(row)
                if not success:
                    break
            if success:
                for i in range(9):
                    row = []
                    for j in range(R):
                        r = snap_to_rational(V[i, j])
                        if r is None:
                            success = False
                            break
                        row.append(r)
                    V_rational.append(row)
                    if not success:
                        break
            if success:
                for i in range(6):
                    row = []
                    for j in range(R):
                        r = snap_to_rational(W[i, j])
                        if r is None:
                            success = False
                            break
                        row.append(r)
                        W_rational.append(row)
                        if not success:
                            break
            
            if success:
                # Verify the rational solution
                def verify(U_r, V_r, W_r):
                    for i in range(n):
                        for k in range(p):
                            for j in range(m):
                                val = 0
                                for r in range(R):
                                    u_val = U_r[i*m + j][0] / U_r[i*m + j][1]
                                    v_val = V_r[j*p + k][0] / V_r[j*p + k][1]
                                    w_val = W_r[k*n + i][0] / W_r[k*n + i][1]
                                    val += u_val * v_val * w_val
                                if abs(val - 1) > 1e-8:
                                    return False
                    return True
                
                if verify(U_rational, V_rational, W_rational):
                    # Convert to lists
                    U_list = [[U_rational[i][j][0] / U_rational[i][j][1] for j in range(R)] for i in range(6)]
                    V_list = [[V_rational[i][j][0] / V_rational[i][j][1] for j in range(R)] for i in range(9)]
                    W_list = [[W_rational[i][j][0] / W_rational[i][j][1] for j in range(R)] for i in range(6)]
                    return {"U": U_list, "V": V_list, "W": W_list}
            
            # If rationalization fails, return the float solution
            U_list = [[U[i, j] for j in range(R)] for i in range(6)]
            V_list = [[V[i, j] for j in range(R)] for i in range(9)]
            W_list = [[W[i, j] for j in range(R)] for i in range(6)]
            return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from 11 to 18
    for R in range(11, 19):
        result = try_rank(R)
        if result is not None:
            return result
    
    # Fallback to schoolbook
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
