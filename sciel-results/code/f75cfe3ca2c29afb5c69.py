import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin before 150s kill

    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1

    # Schoolbook decomposition (rank 30)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_result = schoolbook()
    best_rank = n * m * p

    def verify(U, V, W):
        """Check if U, V, W correctly decompose T"""
        rank = U.shape[1]
        T_recon = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T_recon += np.outer(np.outer(U[:,r], V[:,r]), W[:,r])
        return np.allclose(T_recon, T, atol=1e-8)

    # Strategy: Construct a rank-28 decomposition explicitly.
    # The matrix multiplication 2x3 * 3x5 can be viewed as:
    # Result C (2x5) = A (2x3) * B (3x5)
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # We can split A into two parts: A1 (2x2) using columns 0,1 and A2 (2x1) using column 2.
    # Similarly B into B1 (2x5) using rows 0,1 and B2 (1x5) using row 2.
    # C = A1*B1 + A2*B2
    # Wait, this is not quite right.
    # C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k]
    
    # Let's use the known result that R(2,3,5) <= 28.
    # One way to achieve this is to use Strassen's algorithm for the 2x2 block and schoolbook for the rest.
    # However, the indices are coupled.
    # Let's try to construct it directly using the "merge" idea or known factors.
    
    # Actually, a simpler approach for 2x3x5:
    # We can treat it as two 2x2x5 multiplications? No.
    # Let's use the explicit construction from the literature or derive it.
    
    # Alternative: Use the fact that R(2,3,5) = 28 is known.
    # We can try to find such a decomposition via a solver if we have time, but hardcoding is safer.
    # Since I don't have the hardcoded factors memorized, I will attempt a local search from the schoolbook solution
    # or a random initialization with a strong solver (like scipy L-BFGS-B on the flattened tensor error).
    
    # However, the non-convexity is high.
    # Let's try a specific heuristic:
    # Start with rank 29 or 30, and try to merge terms.
    # Or, simply try to solve for rank 28 using a randomized local search.
    
    # Given the 150s limit, a robust solver is needed.
    # Let's try to use the "Strassen-like" structure.
    # For 2x2x2, Strassen uses 7.
    # For 2x2x5, we can do 7 * (5/2)? No.
    # R(2,2,5) is known to be 17? Or 16?
    # If R(2,2,5) = 17, and we have a 2x1x5 part (rank 5), total 17+5=22?
    # But the coupling makes it tricky.
    
    # Let's stick to the safe schoolbook (rank 30) if we can't find a better one quickly.
    # But the prompt asks for a stronger algorithm.
    # Let's implement a simple local search to try to reduce rank from 30 to 29 or 28.
    
    def try_reduce_rank(start_rank, target_rank, max_iters=500):
        # Initialize with schoolbook factors (first target_rank columns)
        # This is a valid decomposition of a subset? No, it's a subset of terms.
        # The sum of the first target_rank terms is not equal to T.
        # So we need to optimize U, V, W of size (N, target_rank) to minimize ||T - sum_r U_r V_r W_r||^2
        
        # Flatten T
        T_flat = T.flatten()
        
        # Objective function
        def objective(params):
            # Unpack
            size_U = n*m * target_rank
            size_V = m*p * target_rank
            U_flat = params[:size_U]
            V_flat = params[size_U:size_U+size_V]
            W_flat = params[size_U+size_V:]
            
            U = U_flat.reshape(n*m, target_rank)
            V = V_flat.reshape(m*p, target_rank)
            W = W_flat.reshape(p*n, target_rank)
            
            # Compute reconstruction
            T_recon = np.zeros((n*m, m*p, p*n))
            for r in range(target_rank):
                T_recon += np.outer(np.outer(U[:,r], V[:,r]), W[:,r])
                
            return np.sum((T_recon - T)**2)
        
        # Initial guess: Use the first target_rank terms of schoolbook, scaled?
        # Or random.
        # Random is better for escaping local minima.
        np.random.seed(123)
        init_U = np.random.randn(n*m, target_rank) * 0.1
        init_V = np.random.randn(m*p, target_rank) * 0.1
        init_W = np.random.randn(p*n, target_rank) * 0.1
        
        x0 = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
        
        # Use scipy minimize
        from scipy.optimize import minimize
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.success or res.fun < 1e-6:
                size_U = n*m * target_rank
                size_V = m*p * target_rank
                U_flat = res.x[:size_U]
                V_flat = res.x[size_U:size_U+size_V]
                W_flat = res.x[size_U+size_V:]
                
                U = U_flat.reshape(n*m, target_rank)
                V = V_flat.reshape(m*p, target_rank)
                W = W_flat.reshape(p*n, target_rank)
                
                # Round to half-integers if possible?
                # The problem statement says "integer/half-integer entries".
                # We should check if the solution is close to such values.
                # If not, it's still a valid real-valued decomposition?
                # "with integer/half-integer entries that exactly decompose"
                # This implies the final answer MUST have integer/half-integer entries.
                # So we need to round and verify.
                
                U_rounded = np.round(U * 2) / 2.0
                V_rounded = np.round(V * 2) / 2.0
                W_rounded = np.round(W * 2) / 2.0
                
                if verify(U_rounded, V_rounded, W_rounded):
                    return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                else:
                    # Try rounding to integers?
                    U_rounded = np.round(U)
                    V_rounded = np.round(V)
                    W_rounded = np.round(W)
                    if verify(U_rounded, V_rounded, W_rounded):
                        return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
        except Exception as e:
            pass
        return None

    # Try to find rank 28
    if time.time() < deadline - 10:
        result_28 = try_reduce_rank(30, 28, max_iters=100)
        if result_28:
            best_result = result_28
            best_rank = 28
    else:
        # Fallback to schoolbook
        pass

    return best_result
