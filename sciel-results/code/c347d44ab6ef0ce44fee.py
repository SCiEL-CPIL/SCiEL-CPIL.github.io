"""Structured rank-reduction for 2x4x7 matrix multiplication tensor."""
import numpy as np
import time

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 @ 4x7.
    
    Strategy: Exploit the 2-row structure. For n=2, the W matrix has shape (p*2, rank).
    We parameterize the decomposition using the constraint that each output C[i,k]
    must equal sum_j A[i,j]*B[j,k]. We use a randomized algebraic approach to find
    lower-rank solutions by treating the problem as a system of polynomial equations
    and using structured initialization with local search.
    """
    n, m, p = 2, 4, 7
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s limit
    
    def verify_decomposition(U_m, V_m, W_m, n, m, p, num_tests=20):
        """Fast verification using random integer inputs."""
        np.random.seed(42)
        for _ in range(num_tests):
            A = np.random.randint(-5, 6, size=(n, m))
            B = np.random.randint(-5, 6, size=(m, p))
            C_expected = A @ B
            rank_m = U_m.shape[1]
            C_comp = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        for r in range(rank_m):
                            s += U_m[i*m+j, r] * V_m[j*p+k, r] * W_m[k*n+i, r]
                    C_comp[i, k] = s
            if not np.allclose(C_comp, C_expected, atol=1e-9):
                return False
        return True
    
    # Start with the known rank-50 decomposition as baseline
    def build_rank50():
        rank = 50
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        
        def add_strassen(i0, j0, k0):
            nonlocal col
            # M1
            U[i0*m+j0, col] += 1; U[(i0+1)*m+(j0+1), col] += 1
            V[j0*p+k0, col] += 1; V[(j0+1)*p+(k0+1), col] += 1
            W[k0*n+i0, col] += 1; W[(k0+1)*n+(i0+1), col] += 1
            col += 1
            # M2
            U[(i0+1)*m+j0, col] += 1; U[(i0+1)*m+(j0+1), col] += 1
            V[j0*p+k0, col] += 1
            W[k0*n+(i0+1), col] += 1; W[(k0+1)*n+(i0+1), col] -= 1
            col += 1
            # M3
            U[i0*m+j0, col] += 1
            V[j0*p+(k0+1), col] += 1; V[(j0+1)*p+(k0+1), col] -= 1
            W[(k0+1)*n+i0, col] += 1; W[(k0+1)*n+(i0+1), col] += 1
            col += 1
            # M4
            U[(i0+1)*m+(j0+1), col] += 1
            V[(j0+1)*p+k0, col] += 1; V[j0*p+k0, col] -= 1
            W[k0*n+i0, col] += 1; W[k0*n+(i0+1), col] += 1
            col += 1
            # M5
            U[i0*m+j0, col] += 1; U[i0*m+(j0+1), col] += 1
            V[(j0+1)*p+(k0+1), col] += 1
            W[k0*n+i0, col] -= 1; W[(k0+1)*n+i0, col] += 1
            col += 1
            # M6
            U[(i0+1)*m+j0, col] += 1; U[i0*m+j0, col] -= 1
            V[j0*p+k0, col] += 1; V[j0*p+(k0+1), col] += 1
            W[(k0+1)*n+(i0+1), col] += 1
            col += 1
            # M7
            U[i0*m+(j0+1), col] += 1; U[(i0+1)*m+(j0+1), col] -= 1
            V[(j0+1)*p+k0, col] += 1; V[(j0+1)*p+(k0+1), col] += 1
            W[k0*n+i0, col] += 1
            col += 1
        
        add_strassen(0, 0, 0)
        add_strassen(0, 0, 2)
        add_strassen(0, 2, 0)
        add_strassen(0, 2, 2)
        add_strassen(0, 0, 4)
        add_strassen(0, 2, 4)
        
        for i in range(2):
            for j in range(4):
                U[i*m+j, col] = 1
                V[j*p+6, col] = 1
                W[6*n+i, col] = 1
                col += 1
        
        return U, V, W, rank
    
    U50, V50, W50, rank50 = build_rank50()
    
    if not verify_decomposition(U50, V50, W50, n, m, p):
        return {"U": U50.tolist(), "V": V50.tolist(), "W": W50.tolist()}
    
    best_U, best_V, best_W = U50.copy(), V50.copy(), W50.copy()
    best_rank = rank50
    
    # Structural approach: exploit n=2 structure
    # For n=2, W has shape (14, rank). Key insight: W[k*2+i, r] for i=0,1
    # The constraint is: for each (i,k), sum_j sum_r U[i,j,r]*V[j,k,r]*W[k*2+i,r] = sum_j A[i,j]*B[j,k]
    # This must hold for all A, B, so we need:
    # For each (i,k): sum_r W[k*2+i,r] * (sum_j U[i,j,r]*V[j,k,r]) = 1 if i=k else 0... no
    # Actually: sum_j U[i,j,r]*V[j,k,r] * W[k*2+i,r] summed over r must equal delta-like structure
    
    # Better structural approach: try to find rank-r solutions by fixing W structure
    # and solving for U, V. With n=2, we can use the fact that the tensor is bilinear in A and B.
    
    # Try various ranks from low to high, using randomized initialization
    def try_rank(r, max_attempts=50):
        """Try to find a rank-r decomposition using randomized algebraic approach."""
        if r < 1:
            return None
        
        # The tensor T[i,j,k] = 1 for all i,j,k (it's the identity multiplication tensor)
        # We need: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Note: W is indexed as W[k*n+i, r] = W[k*2+i, r]
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(attempt * 137 + 42)
            
            # Initialize with sparse random integer values
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            
            # Randomly assign non-zero entries
            for r_idx in range(r):
                # U: each row (i,j) gets a random value from {-1, 0, 1}
                for idx in range(n*m):
                    if np.random.random() < 0.4:
                        U[idx, r_idx] = np.random.choice([-1, 1])
                
                # V: each row (j,k) gets a random value
                for idx in range(m*p):
                    if np.random.random() < 0.4:
                        V[idx, r_idx] = np.random.choice([-1, 1])
                
                # W: each row (k,i) gets a random value
                for idx in range(p*n):
                    if np.random.random() < 0.5:
                        W[idx, r_idx] = np.random.choice([-1, 1])
            
            # Verify
            if verify_decomposition(U, V, W, n, m, p, num_tests=5):
                return U, V, W
        
        return None
    
    # Try ranks from 28 up to 49
    for r in range(28, min(50, rank50)):
        if time.time() > deadline - 30:
            break
        
        result = try_rank(r, max_attempts=30)
        if result is not None:
            U_r, V_r, W_r = result
            best_U, best_V, best_W = U_r, V_r, W_r
            best_rank = r
            break  # Found the lowest rank, stop
    
    # If no lower rank found, try local search on the rank-50 solution
    # to see if we can eliminate some terms
    if best_rank == rank50 and time.time() < deadline - 20:
        # Try to zero out columns that are linearly dependent in a tensor sense
        # For each subset of columns to remove, check if the remaining still works
        # This is expensive, so only try small removals
        
        # Try removing 1 column at a time
        for col_to_remove in range(rank50):
            if time.time() > deadline - 15:
                break
            
            U_test = np.delete(U50, col_to_remove, axis=1)
            V_test = np.delete(V50, col_to_remove, axis=1)
            W_test = np.delete(W50, col_to_remove, axis=1)
            
            if verify_decomposition(U_test, V_test, W_test, n, m, p, num_tests=5):
                best_U, best_V, best_W = U_test, V_test, W_test
                best_rank = rank50 - 1
                break
    
    # Final verification
    if not verify_decomposition(best_U, best_V, best_W, n, m, p):
        best_U, best_V, best_W = U50, V50, W50
        best_rank = rank50
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
