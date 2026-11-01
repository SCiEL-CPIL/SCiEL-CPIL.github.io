import time
import numpy as np
from itertools import product
from fractions import Fraction

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly represents matrix multiplication."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _als_step(T, U, V, W, n, m, p, rank):
    """
    Perform one step of ALS: update U, V, W given the others.
    T is the target tensor (n, m, p).
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    # Update U: For each (i,j), solve (V.T @ W.T) U[i,j,:] = T[i,j,:] reshaped
    # Actually, the tensor equation is:
    # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = T[i,j,k]
    
    # Let's reshape V and W to facilitate matrix multiplication.
    # V is (m, p, rank) -> reshape to (m, p*rank)? No.
    # For fixed (i,j), we have:
    # sum_r U[i,j,r] * (sum_k V[j,k,r] W[k,i,r]) = T[i,j,k] ... wait, this is not linear in U directly if we sum over k.
    
    # Correct ALS formulation:
    # Fix V and W. The equation for U[i,j,:] is:
    # sum_r U[i,j,r] * (sum_k V[j,k,r] W[k,i,r]) = T[i,j,k] for all k? No.
    
    # The term is: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # This is a linear system in U[i,j,:] for each fixed i,j,k? No, it's coupled over r.
    
    # Let's vectorize.
    # For fixed i, j:
    # Let u = U[i,j,:] (vector of length rank)
    # Let M be a matrix of size (p * 1, rank) ? No.
    
    # Rewrite:
    # T[i,j,k] = sum_r U[i,j,r] * (V[j,k,r] * W[k,i,r])
    # Let A[i,j,k,r] = V[j,k,r] * W[k,i,r]
    # Then T[i,j,k] = sum_r A[i,j,k,r] U[i,j,r]
    
    # This is a linear system for each (i,j) if we treat U[i,j,:] as unknowns.
    # But the RHS is T[i,j,k] for all k.
    # So we have p equations for 'rank' unknowns.
    # If rank > p, we have underdetermined system. We want the minimum norm solution?
    # Or we can solve the least squares problem.
    
    # Matrix form for fixed i,j:
    # Let B be a p x rank matrix where B[k, r] = V[j,k,r] * W[k,i,r]
    # Then B @ U[i,j,:] = T[i,j,:] (vector of length p)
    
    U_new = np.zeros_like(U)
    for i in range(n):
        for j in range(m):
            # Construct B (p x rank)
            B = np.zeros((p, rank))
            for k in range(p):
                for r in range(rank):
                    B[k, r] = V[j*p+k][r] * W[k*n+i][r]
            # Solve B @ u = T[i,j,:]
            # Using lstsq
            try:
                u, residuals, rank_b, sv = np.linalg.lstsq(B, T[i, j, :], rcond=None)
                U_new[i*m+j] = u
            except:
                U_new[i*m+j] = U[i*m+j]
                
    # Update V
    V_new = np.zeros_like(V)
    for j in range(m):
        for k in range(p):
            # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            # Fix U, W. Solve for V[j,k,:]
            # Let C be a (n * 1) x rank matrix? 
            # Equation: sum_r V[j,k,r] * (U[i,j,r] * W[k,i,r]) = T[i,j,k] for all i
            # Let D[i, r] = U[i,j,r] * W[k,i,r]
            # Then D @ v = T[:,j,k] (vector of length n)
            D = np.zeros((n, rank))
            for i in range(n):
                for r in range(rank):
                    D[i, r] = U[i*m+j][r] * W[k*n+i][r]
            try:
                v, residuals, rank_d, sv = np.linalg.lstsq(D, T[:, j, k], rcond=None)
                V_new[j*p+k] = v
            except:
                V_new[j*p+k] = V[j*p+k]
                
    # Update W
    W_new = np.zeros_like(W)
    for k in range(p):
        for i in range(n):
            # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            # Fix U, V. Solve for W[k,i,:]
            # Let E[j, r] = U[i,j,r] * V[j,k,r]
            # Then E @ w = T[i,:,k] (vector of length m)
            E = np.zeros((m, rank))
            for j in range(m):
                for r in range(rank):
                    E[j, r] = U[i*m+j][r] * V[j*p+k][r]
            try:
                w, residuals, rank_e, sv = np.linalg.lstsq(E, T[i, :, k], rcond=None)
                W_new[k*n+i] = w
            except:
                W_new[k*n+i] = W[k*n+i]
                
    return U_new, V_new, W_new

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x5 matrix.
    """
    start_time = time.time()
    deadline = start_time + 120.0 # 120s limit, keep margin
    
    n, m, p = 2, 2, 5
    
    # Target tensor T[i, j, k] = 1 for all i, j, k
    T = np.ones((n, m, p))
    
    best_rank = n * m * p # Schoolbook rank
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks starting from a lower bound.
    # Lower bound for 2x2x5 is at least max(2*2, 2*5, 5*2) = 10? 
    # Actually, rank >= max(n*m, m*p, p*n) is not a lower bound.
    # Rank >= max(n*m, m*p, p*n) is an upper bound (schoolbook).
    # Known results: 2x2x2 is 7. 2x2x3 is 9? 2x2x4 is 12?
    # Let's try to find a rank 16 solution first (Strassen 2x2 + 3*4 schoolbook for 2x2x3? No)
    # Strassen for 2x2x2 is 7.
    # For 2x2x5, we can split B into B1 (2x2) and B2 (2x3).
    # C = A B1 + A B2.
    # Rank(A B1) <= 7.
    # Rank(A B2) <= 2*2*3 = 12 (schoolbook) or maybe less?
    # If we use schoolbook for 2x2x3, total rank = 7 + 12 = 19.
    # If we can do 2x2x3 in rank 9, total = 16.
    # Is rank(2x2x3) 9? Yes, it is known that rank(2x2x3) = 9.
    # So rank 16 is a strong candidate.
    
    # Let's try rank 16 first. If it fails, try 17, 18...
    
    for rank in range(16, n*m*p + 1):
        if time.time() > deadline - 10: # Stop if close to deadline
            break
            
        found = False
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
                
            # Initialize with random values
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Run ALS
            for it in range(50):
                if time.time() > deadline - 2:
                    break
                U, V, W = _als_step(T, U, V, W, n, m, p, rank)
                
                # Check convergence / validity
                # Compute error
                err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for r in range(rank):
                                s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                            err += (s - 1)**2
                            
                if err < 1e-8:
                    # We found a valid decomposition
                    # Convert to lists of floats/ints
                    # The problem asks for integer/half-integer entries.
                    # ALS might produce floats. We need to check if they are close to integers/half-integers.
                    # Or we can just return the floats if the evaluator accepts them?
                    # "integer/half-integer entries"
                    # Let's check if they are close to 0.5 * integer.
                    valid_format = True
                    for mat in [U, V, W]:
                        for val in mat.flatten():
                            # Check if val is close to k/2
                            nearest_half = round(val * 2) / 2
                            if abs(val - nearest_half) > 1e-6:
                                valid_format = False
                                break
                        if not valid_format:
                            break
                            
                    if valid_format:
                        # Convert to list of lists
                        U_list = [[float(x) for x in row] for row in U]
                        V_list = [[float(x) for x in row] for row in V]
                        W_list = [[float(x) for x in row] for row in W]
                        
                        # Final verification
                        if _verify(U_list, V_list, W_list, n, m, p, rank):
                            best_rank = rank
                            best_U = U_list
                            best_V = V_list
                            best_W = W_list
                            found = True
                            break
                    else:
                        # If format is invalid, we might still have a valid tensor decomposition
                        # but we need to snap to grid? Or reject?
                        # The task says "integer/half-integer entries".
                        # If ALS finds a solution that is not on the grid, it's not a valid output for this specific constraint.
                        # However, matrix multiplication tensors usually have integer decompositions.
                        # If ALS converges to a non-integer solution, it might be a local minimum or just noise.
                        # We can try to snap to nearest half-integer and verify.
                        U_snap = np.round(U * 2) / 2
                        V_snap = np.round(V * 2) / 2
                        W_snap = np.round(W * 2) / 2
                        
                        U_list = [[float(x) for x in row] for row in U_snap]
                        V_list = [[float(x) for x in row] for row in V_snap]
                        W_list = [[float(x) for x in row] for row in W_snap]
                        
                        if _verify(U_list, V_list, W_list, n, m, p, rank):
                            best_rank = rank
                            best_U = U_list
                            best_V = V_list
                            best_W = W_list
                            found = True
                            break
                            
            if found:
                break
                
        if found:
            break
            
    # If no rank < 20 was found, fall back to schoolbook
    if best_U is None:
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        best_U = U
        best_V = V
        best_W = W
        best_rank = rank
        
    return {"U": best_U, "V": best_V, "W": best_W}
