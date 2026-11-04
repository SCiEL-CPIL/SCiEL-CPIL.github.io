"""Program for rank-minimal exact tensor decomposition of 2x3 * 3x4 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns an exact tensor decomposition for 2x3 * 3x4 matrix multiplication.
    Tries to find a rank-15 decomposition via alternating least squares with rational snapping.
    Falls back to schoolbook rank 24 if no valid low-rank solution is found.
    """
    start_time = time.time()
    deadline = start_time + 140  # 140s limit, leaving 10s margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 4
    dim_U = n * m  # 6
    dim_V = m * p  # 12
    dim_W = p * n  # 8
    rank_school = n * m * p  # 24
    
    def schoolbook():
        U = [[0] * rank_school for _ in range(dim_U)]
        V = [[0] * rank_school for _ in range(dim_V)]
        W = [[0] * rank_school for _ in range(dim_W)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_exact(U, V, W, rank):
        """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check using multiple random matrices
        rng = np.random.RandomState(12345)
        for _ in range(5):
            A = rng.rand(n, m)
            B = rng.rand(m, p)
            C_true = A @ B
            
            u_vals = U_np.T @ A.flatten()  # (rank,)
            v_vals = V_np.T @ B.flatten()  # (rank,)
            
            # W_np is (dim_W, rank) = (p*n, rank)
            # Index k*n+i corresponds to row i, col k of C
            # So W_np.T has shape (rank, p*n), reshape to (rank, p, n)
            W_matrices = W_np.T.reshape(rank, p, n)
            terms = u_vals * v_vals
            C_decomp = np.tensordot(terms, W_matrices, axes=([0], [0])).T
            
            if not np.allclose(C_decomp, C_true, atol=1e-9):
                return False
        return True
    
    def to_half_int(arr):
        """Round to nearest half-integer."""
        return np.round(arr * 2) / 2.0
    
    def to_fraction_list(arr):
        """Convert numpy array to list of lists of Fractions (integers or half-integers)."""
        result = []
        for row in arr:
            row_list = []
            for val in row:
                # Check if it's a half-integer
                if abs(val * 2 - round(val * 2)) < 1e-9:
                    row_list.append(Fraction(int(round(val * 2)), 2))
                else:
                    row_list.append(Fraction(val).limit_denominator(1000))
            result.append(row_list)
        return result
    
    def compute_error(U, V, W, rank):
        """Compute the error of the decomposition."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        rng = np.random.RandomState(42)
        A = rng.rand(n, m)
        B = rng.rand(m, p)
        C_true = A @ B
        
        u_vals = U_np.T @ A.flatten()
        v_vals = V_np.T @ B.flatten()
        W_matrices = W_np.T.reshape(rank, p, n)
        terms = u_vals * v_vals
        C_decomp = np.tensordot(terms, W_matrices, axes=([0], [0])).T
        
        return np.linalg.norm(C_decomp - C_true)
    
    # Try to find a rank-15 decomposition
    target_rank = 15
    
    def solve_for_U(V, W, rank):
        """Given V and W, solve for U such that the tensor equation holds."""
        # We need: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = delta_{i,k} * 1 (for all i,j,k)
        # Actually, the condition is that for all i,j,k:
        # The coefficient of A_{i,j} B_{j,k} in the output should be 1 if it contributes to C_{i,k}
        # 
        # Let's think about this differently. The tensor T has components T_{i,j,k} = 1.
        # The decomposition is: T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{i,k,r}
        # Wait, this isn't quite right either. Let me use the bilinear map formulation.
        #
        # C_{i,k} = sum_r (sum_{a,b} U_{a,b,r} A_{a,b}) (sum_{c,d} V_{c,d,r} B_{c,d}) W_{i,k,r}
        #
        # For this to equal sum_j A_{i,j} B_{j,k} for all A, B:
        # We need: sum_r U_{a,b,r} V_{c,d,r} W_{i,k,r} = delta_{a,i} delta_{b,j} delta_{c,j} delta_{d,k} * 1
        # where j is summed over.
        #
        # Actually: sum_r U_{a,b,r} V_{c,d,r} W_{i,k,r} should equal 1 if a=i, c=d... no.
        #
        # Let's use the property that the bilinear map must match:
        # For basis vectors e_{a,b} in A-space and f_{c,d} in B-space:
        # The output should be the matrix E_{a,c} if b==d, else 0.
        # Wait, no. A is n x m, B is m x p.
        # A_{i,j} * B_{j,k} contributes to C_{i,k}.
        #
        # So for basis A = E_{a,b} (1 at position (a,b)) and B = F_{c,d} (1 at position (c,d)):
        # The product (A*B)_{i,k} = sum_j A_{i,j} B_{j,k} = A_{a,b} B_{b,k} if i=a, else 0.
        # = delta_{i,a} * B_{b,k} = delta_{i,a} * delta_{k,d} * delta_{b,c}
        #
        # So the output matrix is: delta_{a,b==c} * E_{a,d} (matrix with 1 at (a,d) if b==c, else 0)
        #
        # In the decomposition:
        # u_val = sum_{x,y} U_{x,y,r} A_{x,y} = U_{a,b,r} (for basis A=E_{a,b})
        # v_val = sum_{x,y} V_{x,y,r} B_{x,y} = V_{c,d,r} (for basis B=F_{c,d})
        # Output = sum_r U_{a,b,r} V_{c,d,r} W_{i,k,r} * E_{i,k}
        #
        # This must equal: delta_{b,c} * E_{a,d}
        #
        # So: sum_r U_{a,b,r} V_{c,d,r} W_{i,k,r} = delta_{b,c} * delta_{i,a} * delta_{k,d}
        #
        # This is a system of equations. For fixed V and W, we can solve for U.
        # The unknowns are U_{a,b,r} for a in [0,n), b in [0,m), r in [0,rank).
        # Total unknowns: n*m*rank = 6*15 = 90
        #
        # Equations: for each (a,b,c,d,i,k) such that the LHS involves U_{a,b,r}:
        # sum_r U_{a,b,r} V_{c,d,r} W_{i,k,r} = delta_{b,c} delta_{i,a} delta_{k,d}
        #
        # For fixed (a,b), the equation is:
        # sum_r U_{a,b,r} [V_{c,d,r} W_{i,k,r}] = delta_{b,c} delta_{i,a} delta_{k,d}
        #
        # This is a linear system in U_{a,b,r}. But the RHS depends on (c,d,i,k).
        # We need this to hold for ALL (c,d,i,k).
        #
        # Let's define M_{(c,d,i,k), r} = V_{c,d,r} W_{i,k,r}
        # Then: sum_r U_{a,b,r} M_{(c,d,i,k), r} = delta_{b,c} delta_{i,a} delta_{k,d}
        #
        # For fixed (a,b), this is: U_{a,b}^T M_{(c,d,i,k)} = delta_{b,c} delta_{i,a} delta_{k,d}
        #
        # The RHS is nonzero only when b=c, i=a, k=d. So for each (a,b), we have:
        # U_{a,b}^T M_{(b,d,a,k)} = delta_{k,d} for all d,k
        # i.e., U_{a,b}^T M_{(b,d,a,k)} = delta_{k,d}
        #
        # So for fixed (a,b), we need:
        # sum_r U_{a,b,r} V_{b,d,r} W_{a,k,r} = delta_{d,k} for all d,k
        #
        # This is: (U_{a,b}^T * diag-like) = I
        # Let's define a matrix B_{a,b} of size (m*p, rank) where B_{a,b}[d*p+k, r] = V_{b,d,r} * W_{a,k,r}
        # Then we need: U_{a,b}^T B_{a,b} = I_{m*p}^T? No.
        #
        # Actually: sum_r U_{a,b,r} V_{b,d,r} W_{a,k,r} = delta_{d,k}
        # Let's define for fixed (a,b): a vector u = U[:, a*m+b] (length rank)
        # And a matrix M_{d,k,r} = V[b*p+d, r] * W[k*n+a, r]
        # Then: sum_r u_r M_{d,k,r} = delta_{d,k}
        # This is: u^T M_{d,k} = delta_{d,k} for all d,k
        #
        # Stacking over all (d,k): u^T M = vec(I_{m*p})^T? No, M has entries M_{d,k,r}.
        # Let's form a matrix N of size (m*p, rank) where N[(d,k), r] = V[b*p+d, r] * W[k*n+a, r]
        # Then we need: u^T N = e where e is a vector of length m*p with e[(d,k)] = delta_{d,k}
        # i.e., u^T N = vec(I) (flattened identity)
        #
        # So: N^T u = vec(I)
        # This is a linear system: N^T u = vec(I)
        # N is (m*p, rank) = (12, 15). N^T is (15, 12).
        # We're solving for u (length 15) from 12 equations. Underdetermined.
        # Use least squares: u = (N^T N)^{-1} N^T vec(I)
        
        U = np.zeros((dim_U, rank))
        for a in range(n):
            for b in range(m):
                # Build N: (m*p, rank)
                N = np.zeros((m * p, rank))
                for d in range(m):
                    for k in range(p):
                        idx = d * p + k
                        for r in range(rank):
                            N[idx, r] = V[b * p + d, r] * W[k * n + a, r]
                
                # Target: vec(I) of length m*p
                target = np.zeros(m * p)
                for d in range(m):
                    target[d * p + d] = 1.0
                
                # Solve N^T u = target
                # N^T is (rank, m*p)
                NT = N.T  # (rank, m*p)
                # Least squares: u = pinv(NT) @ target
                u = np.linalg.lstsq(NT, target, rcond=None)[0]
                U[a * m + b] = u
        
        return U
    
    def solve_for_V(U, W, rank):
        """Given U and W, solve for V."""
        # By symmetry with solve_for_U, but now solving for V.
        # From: sum_r U_{a,b,r} V_{c,d,r} W_{i,k,r} = delta_{b,c} delta_{i,a} delta_{k,d}
        # For fixed (c,d): sum_r V_{c,d,r} U_{a,b,r} W_{i,k,r} = delta_{b,c} delta_{i,a} delta_{k,d}
        # Nonzero when b=c, i=a, k=d:
        # sum_r V_{c,d,r} U_{a,c,r} W_{d,n+a,r} ... wait, let me be careful.
        #
        # For fixed (c,d), the equation is:
        # sum_r V_{c,d,r} [U_{a,b,r} W_{i,k,r}] = delta_{b,c} delta_{i,a} delta_{k,d}
        # Nonzero when b=c, i=a, k=d:
        # sum_r V_{c,d,r} U_{a,c,r} W_{d*n+a,r} = 1 for all a,d? No.
        #
        # Let's fix (c,d). The RHS is delta_{b,c} delta_{i,a} delta_{k,d}.
        # So for each (a,b,i,k), the equation holds.
        # For fixed (c,d), varying (a,b,i,k):
        # The equation is: sum_r V_{c,d,r} U_{a,b,r} W_{i,k,r} = delta_{b,c} delta_{i,a} delta_{k,d}
        #
        # For fixed (c,d), let v = V[:, c*p+d] (length rank).
        # Then: sum_r v_r U_{a,b,r} W_{i,k,r} = delta_{b,c} delta_{i,a} delta_{k,d}
        #
        # This must hold for all (a,b,i,k).
        # Define M_{(a,b,i,k), r} = U_{a,b,r} W_{i,k,r}
        # Then: v^T M_{(a,b,i,k)} = delta_{b,c} delta_{i,a} delta_{k,d}
        #
        # The RHS is nonzero only when b=c, i=a, k=d.
        # So for fixed (c,d): v^T M_{(a,c,a,d)} = 1 for all a.
        # i.e., sum_r v_r U_{a,c,r} W_{d*n+a,r} = 1 for all a in [0,n).
        #
        # Let's form a matrix N of size (n, rank) where N[a, r] = U[a*m+c, r] * W[d*n+a, r]
        # Then: v^T N = ones(n)
        # N is (n, rank) = (2, 15). N^T is (15, 2).
        # Solve: N^T v = ones(2)
        
        V = np.zeros((dim_V, rank))
        for c in range(m):
            for d in range(p):
                # Build N: (n, rank)
                N = np.zeros((n, rank))
                for a in range(n):
                    for r in range(rank):
                        N[a, r] = U[a * m + c, r] * W[d * n + a, r]
                
                target = np.ones(n)
                NT = N.T  # (rank, n)
                v = np.linalg.lstsq(NT, target, rcond=None)[0]
                V[c * p + d] = v
        
        return V
    
    def solve_for_W(U, V, rank):
        """Given U and V, solve for W."""
        # From: sum_r U_{a,b,r} V_{c,d,r} W_{i,k,r} = delta_{b,c} delta_{i,a} delta_{k,d}
        # For fixed (i,k): sum_r W_{i,k,r} U_{a,b,r} V_{c,d,r} = delta_{b,c} delta_{i,a} delta_{k,d}
        # Nonzero when b=c, i=a, k=d:
        # For fixed (i,k), let w = W[:, k*n+i] (length rank).
        # sum_r w_r U_{a,b,r} V_{c,d,r} = delta_{b,c} delta_{i,a} delta_{k,d}
        #
        # For fixed (i,k), varying (a,b,c,d):
        # The RHS is nonzero only when b=c, a=i, d=k.
        # So: sum_r w_r U_{i,b,r} V_{b,k,r} = 1 for all b.
        #
        # Form N of size (m, rank) where N[b, r] = U[i*m+b, r] * V[b*p+k, r]
        # Then: w^T N = ones(m)
        # N is (m, rank) = (3, 15). N^T is (15, 3).
        # Solve: N^T w = ones(3)
        
        W = np.zeros((dim_W, rank))
        for i in range(n):
            for k in range(p):
                N = np.zeros((m, rank))
                for b in range(m):
                    for r in range(rank):
                        N[b, r] = U[i * m + b, r] * V[b * p + k, r]
                
                target = np.ones(m)
                NT = N.T  # (rank, m)
                w = np.linalg.lstsq(NT, target, rcond=None)[0]
                W[k * n + i] = w
        
        return W
    
    # Try alternating least squares for rank 15
    best_err = float('inf')
    best_solution = None
    
    num_trials = 50
    max_iters = 200
    
    for trial in range(num_trials):
        if time.time() > deadline - 20:
            break
        
        np.random.seed(trial * 123 + 456)
        U = np.random.randn(dim_U, target_rank) * 0.3
        V = np.random.randn(dim_V, target_rank) * 0.3
        W = np.random.randn(dim_W, target_rank) * 0.3
        
        for it in range(max_iters):
            if time.time() > deadline - 20:
                break
            
            # Alternate solving
            U = solve_for_U(V, W, target_rank)
            V = solve_for_V(U, W, target_rank)
            W = solve_for_W(U, V, target_rank)
            
            err = compute_error(U, V, W, target_rank)
            if err < best_err:
                best_err = err
                best_solution = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-10:
                break
        
        if best_err < 1e-10:
            break
    
    # Check if we found a good solution
    if best_solution is not None and best_err < 1e-8:
        U, V, W = best_solution
        
        # Try to snap to half-integers
        U_hi = to_half_int(U)
        V_hi = to_half_int(V)
        W_hi = to_half_int(W)
        
        err_hi = compute_error(U_hi, V_hi, W_hi, target_rank)
        if err_hi < 1e-8:
            # Verify exactly
            U_frac = to_fraction_list(U_hi)
            V_frac = to_fraction_list(V_hi)
            W_frac = to_fraction_list(W_hi)
            if verify_exact(U_frac, V_frac, W_frac, target_rank):
                return {"U": U_frac, "V": V_frac, "W": W_frac}
        
        # If half-integer snapping didn't work, try integer snapping
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        
        err_int = compute_error(U_int, V_int, W_int, target_rank)
        if err_int < 1e-8:
            U_frac = to_fraction_list(U_int)
            V_frac = to_fraction_list(V_int)
            W_frac = to_fraction_list(W_int)
            if verify_exact(U_frac, V_frac, W_frac, target_rank):
                return {"U": U_frac, "V": V_frac, "W": W_frac}
        
        # Try using the float solution directly with Fractions
        # Convert to Fractions with reasonable denominators
        U_frac = to_fraction_list(U)
        V_frac = to_fraction_list(V)
        W_frac = to_fraction_list(W)
        if verify_exact(U_frac, V_frac, W_frac, target_rank):
            return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    # Fall back to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
