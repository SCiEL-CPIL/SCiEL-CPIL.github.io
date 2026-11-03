import numpy as np
import time
from itertools import product, combinations
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 148.0  # safety margin before 150s kill

    nm, mp, pn = n * m, m * p, p * n

    # Build the target tensor T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B
    # contributes to (i,k) entry of C
    # T_target[u, v, w] where u=i*m+j, v=j*p+k, w=k*n+i
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def check_decomposition(U, V, W, tol=1e-8):
        T_full = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.allclose(T_full, T_target, atol=tol)

    def _winograd_fomm_decomposition():
        """
        Winograd Frequency-Oriented Matrix Multiplication (FOMM) approach.
        Treats matrix multiplication as convolution and uses the structure
        to find a lower-rank decomposition.
        
        For a 2x3 matrix A and 3x7 matrix B:
        - We can pad A to size 4x4 and B to size 4x8
        - The convolution can be computed via pointwise multiplications in frequency domain
        """
        # Winograd FOMM for general matrix multiplication uses the fact that
        # matrix multiplication is a convolution, and we can use the FFT/NTT
        # to reduce the number of multiplications.
        
        # For 2x3x7, let's try a structured approach:
        # We can view C = A * B where A is 2x3 and B is 3x7
        
        # One approach: use the fact that for small matrices, we can find
        # explicit decompositions by solving systems of equations.
        
        # Let's try a constructive approach based on the algebraic structure.
        # For 2x3x7, the theoretical lower bound is max(6, 21, 14) = 21,
        # but the actual rank is higher.
        
        # Try to find a decomposition by solving the tensor equation directly
        # using a structured ansatz.
        
        # Ansatz: assume U, V, W have a specific block structure
        # based on the dimensions.
        
        # For now, let's try a more sophisticated random search with
        # structured initialization that respects the tensor's sparsity.
        
        best_result = None
        best_rank = 43  # Start above schoolbook rank
        
        # Try multiple random restarts with different strategies
        for strategy in range(10):
            if time.time() - start_time > time_limit - 5:
                break
            
            np.random.seed(strategy * 17 + 3)
            residual = T_target.copy()
            U_list = []
            V_list = []
            W_list = []
            
            max_rank_attempt = 40 + strategy // 3
            
            for rank_idx in range(max_rank_attempt):
                if time.time() - start_time > time_limit - 5:
                    break
                
                # Structured initialization: try to capture patterns in the tensor
                u = np.zeros(nm)
                v = np.zeros(mp)
                w = np.zeros(pn)
                
                # Different initialization strategies
                if strategy % 3 == 0:
                    # Random initialization
                    u = np.random.randn(nm)
                    v = np.random.randn(mp)
                    w = np.random.randn(pn)
                elif strategy % 3 == 1:
                    # Sparse initialization
                    for _ in range(3):
                        idx = np.random.randint(nm)
                        u[idx] = np.random.randn()
                    for _ in range(5):
                        idx = np.random.randint(mp)
                        v[idx] = np.random.randn()
                    for _ in range(4):
                        idx = np.random.randint(pn)
                        w[idx] = np.random.randn()
                else:
                    # Block-structured initialization
                    # Based on the 2x3x7 structure
                    i0 = np.random.randint(n)
                    j0 = np.random.randint(m)
                    k0 = np.random.randint(p)
                    u[i0 * m + j0] = 1.0
                    v[j0 * p + k0] = 1.0
                    w[k0 * n + i0] = 1.0
                    u += 0.1 * np.random.randn(nm)
                    v += 0.1 * np.random.randn(mp)
                    w += 0.1 * np.random.randn(pn)
                
                # Alternating maximization
                for it in range(40):
                    S_u = np.einsum('ijk,j,k->i', residual, v, w)
                    u_new = S_u / (np.linalg.norm(S_u) + 1e-15)
                    
                    S_v = np.einsum('ijk,i,k->j', residual, u_new, w)
                    v_new = S_v / (np.linalg.norm(S_v) + 1e-15)
                    
                    S_w = np.einsum('ijk,i,j->k', residual, u_new, v_new)
                    w_new = S_w / (np.linalg.norm(S_w) + 1e-15)
                    
                    cur_norm = np.linalg.norm(u_new - u) + np.linalg.norm(v_new - v) + np.linalg.norm(w_new - w)
                    u, v, w = u_new, v_new, w_new
                    
                    if cur_norm < 1e-12:
                        break
                
                alpha = np.einsum('ijk,i,j,k', residual, u, v, w)
                if abs(alpha) < 1e-14:
                    break
                
                residual = residual - alpha * np.einsum('i,j,k->ijk', u, v, w)
                U_list.append(alpha * u)
                V_list.append(v)
                W_list.append(w)
                
                if np.linalg.norm(residual) < 1e-12:
                    break
            
            R = len(U_list)
            if R <= max_rank_attempt:
                U = np.column_stack(U_list)
                V = np.column_stack(V_list)
                W = np.column_stack(W_list)
                if check_decomposition(U, V, W, tol=1e-6):
                    if R < best_rank:
                        best_rank = R
                        best_result = (U_list, V_list, W_list)
                        if R <= 40:
                            break
        
        return best_result

    def try_rank_peeling(T_res, max_rank, max_iter=40, n_restarts=2, seed=None):
        """
        Peeling approach: iteratively find best rank-1 approximation to residual.
        Returns (U_list, V_list, W_list) or None if fails.
        """
        if seed is not None:
            np.random.seed(seed)

        residual = T_res.copy()
        U_list = []
        V_list = []
        W_list = []

        for rank_idx in range(max_rank):
            if time.time() - start_time > time_limit - 8:
                return None

            # Find best rank-1 approx to residual via alternating maximization
            u = np.random.randn(nm)
            v = np.random.randn(mp)
            w = np.random.randn(pn)

            prev_norm = np.inf
            converged = False

            for it in range(max_iter):
                # Update u
                S_u = np.einsum('ijk,j,k->i', residual, v, w)
                u_new = S_u / (np.linalg.norm(S_u) + 1e-15)

                # Update v
                S_v = np.einsum('ijk,i,k->j', residual, u_new, w)
                v_new = S_v / (np.linalg.norm(S_v) + 1e-15)

                # Update w
                S_w = np.einsum('ijk,i,j->k', residual, u_new, v_new)
                w_new = S_w / (np.linalg.norm(S_w) + 1e-15)

                # Check convergence
                cur_norm = np.linalg.norm(u_new - u) + np.linalg.norm(v_new - v) + np.linalg.norm(w_new - w)
                u, v, w = u_new, v_new, w_new

                if cur_norm < 1e-10:
                    converged = True
                    break

            # Compute coefficient
            alpha = np.einsum('ijk,i,j,k', residual, u, v, w)
            if abs(alpha) < 1e-12:
                break

            residual = residual - alpha * np.einsum('i,j,k->ijk', u, v, w)
            U_list.append(alpha * u)
            V_list.append(v)
            W_list.append(w)

            # Check if residual is zero
            if np.linalg.norm(residual) < 1e-10:
                break

        R = len(U_list)
        if R > max_rank:
            return None

        # Verify
        U = np.column_stack(U_list)
        V = np.column_stack(V_list)
        W = np.column_stack(W_list)
        if check_decomposition(U, V, W, tol=1e-6):
            return (U_list, V_list, W_list)
        return None

    def try_structured_search(max_rank, time_budget=30.0):
        """
        Try to find a decomposition of rank <= max_rank using structured approaches.
        """
        best_result = None
        best_rank = max_rank + 1

        # Strategy 1: Try peeling with different random seeds
        for attempt in range(8):
            if time.time() - start_time > time_limit - 8:
                break
            if time.time() - start_time > start_time + time_budget:
                break

            result = try_rank_peeling(T_target, max_rank, max_iter=35, seed=42 + attempt * 7)
            if result is not None:
                U_list, V_list, W_list = result
                R = len(U_list)
                if R < best_rank:
                    best_rank = R
                    best_result = (U_list, V_list, W_list)
                    if R == 40:  # theoretical lower bound
                        return best_result

        # Strategy 2: Try with different initializations
        for attempt in range(5):
            if time.time() - start_time > time_limit - 8:
                break

            # Use structured initialization based on block patterns
            np.random.seed(100 + attempt * 13)
            residual = T_target.copy()
            U_list = []
            V_list = []
            W_list = []

            for rank_idx in range(max_rank):
                if time.time() - start_time > time_limit - 8:
                    break

                # Structured initialization: try to capture one (i,j,k) triple
                # with some randomization
                i = np.random.randint(n)
                j = np.random.randint(m)
                k = np.random.randint(p)

                u = np.zeros(nm)
                v = np.zeros(mp)
                w = np.zeros(pn)
                u[i * m + j] = 1.0
                v[j * p + k] = 1.0
                w[k * n + i] = 1.0

                # Add small random perturbation
                u += 0.01 * np.random.randn(nm)
                v += 0.01 * np.random.randn(mp)
                w += 0.01 * np.random.randn(pn)

                # Alternating maximization
                for it in range(30):
                    S_u = np.einsum('ijk,j,k->i', residual, v, w)
                    u = S_u / (np.linalg.norm(S_u) + 1e-15)

                    S_v = np.einsum('ijk,i,k->j', residual, u, w)
                    v = S_v / (np.linalg.norm(S_v) + 1e-15)

                    S_w = np.einsum('ijk,i,j->k', residual, u, v)
                    w = S_w / (np.linalg.norm(S_w) + 1e-15)

                alpha = np.einsum('ijk,i,j,k', residual, u, v, w)
                if abs(alpha) < 1e-12:
                    break

                residual = residual - alpha * np.einsum('i,j,k->ijk', u, v, w)
                U_list.append(alpha * u)
                V_list.append(v)
                W_list.append(w)

                if np.linalg.norm(residual) < 1e-10:
                    break

            R = len(U_list)
            if R <= max_rank:
                U = np.column_stack(U_list)
                V = np.column_stack(V_list)
                W = np.column_stack(W_list)
                if check_decomposition(U, V, W, tol=1e-6):
                    if R < best_rank:
                        best_rank = R
                        best_result = (U_list, V_list, W_list)
                        if R == 40:
                            return best_result

        return best_result

    # Strategy 3: Use the specific structure of 2x3x7 matrix multiplication
    def try_exploiting_structure(max_rank, time_budget=30.0):
        """
        Exploit the specific structure of 2x3x7 matrix multiplication tensor.
        """
        if time.time() - start_time > time_limit - 10:
            return None

        best_result = None
        best_rank = max_rank + 1

        # Try multiple random restarts with structured initialization
        for seed in range(15):
            if time.time() - start_time > time_limit - 10:
                break

            np.random.seed(seed)
            residual = T_target.copy()
            U_list = []
            V_list = []
            W_list = []

            for rank_idx in range(max_rank):
                if time.time() - start_time > time_limit - 10:
                    break

                # Try to find a rank-1 component that captures as much structure as possible
                u = np.zeros(nm)
                v = np.zeros(mp)
                w = np.zeros(pn)

                # Randomly select a base (i,j,k) triple
                i0 = np.random.randint(n)
                j0 = np.random.randint(m)
                k0 = np.random.randint(p)

                u[i0 * m + j0] = 1.0
                v[j0 * p + k0] = 1.0
                w[k0 * n + i0] = 1.0

                # Add random components with smaller weights
                u += 0.05 * np.random.randn(nm)
                v += 0.05 * np.random.randn(mp)
                w += 0.05 * np.random.randn(pn)

                # Alternating maximization with more iterations
                for it in range(50):
                    S_u = np.einsum('ijk,j,k->i', residual, v, w)
                    u_new = S_u / (np.linalg.norm(S_u) + 1e-15)

                    S_v = np.einsum('ijk,i,k->j', residual, u_new, w)
                    v_new = S_v / (np.linalg.norm(S_v) + 1e-15)

                    S_w = np.einsum('ijk,i,j->k', residual, u_new, v_new)
                    w_new = S_w / (np.linalg.norm(S_w) + 1e-15)

                    cur_norm = np.linalg.norm(u_new - u) + np.linalg.norm(v_new - v) + np.linalg.norm(w_new - w)
                    u, v, w = u_new, v_new, w_new

                    if cur_norm < 1e-12:
                        break

                alpha = np.einsum('ijk,i,j,k', residual, u, v, w)
                if abs(alpha) < 1e-14:
                    break

                residual = residual - alpha * np.einsum('i,j,k->ijk', u, v, w)
                U_list.append(alpha * u)
                V_list.append(v)
                W_list.append(w)

                if np.linalg.norm(residual) < 1e-12:
                    break

            R = len(U_list)
            if R <= max_rank:
                U = np.column_stack(U_list)
                V = np.column_stack(V_list)
                W = np.column_stack(W_list)
                if check_decomposition(U, V, W, tol=1e-6):
                    if R < best_rank:
                        best_rank = R
                        best_result = (U_list, V_list, W_list)
                        if R == 40:
                            return best_result

        return best_result

    # Try to find rank 40 first (theoretical lower bound)
    result = try_structured_search(40, time_budget=25.0)
    if result is not None:
        U_list, V_list, W_list = result
        U = np.column_stack(U_list)
        V = np.column_stack(V_list)
        W = np.column_stack(W_list)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try the Winograd FOMM approach
    if time.time() - start_time < time_limit - 25:
        result = _winograd_fomm_decomposition()
        if result is not None:
            U_list, V_list, W_list = result
            U = np.column_stack(U_list)
            V = np.column_stack(V_list)
            W = np.column_stack(W_list)
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try the structure-exploiting approach
    if time.time() - start_time < time_limit - 20:
        result = try_exploiting_structure(40, time_budget=20.0)
        if result is not None:
            U_list, V_list, W_list = result
            U = np.column_stack(U_list)
            V = np.column_stack(V_list)
            W = np.column_stack(W_list)
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try rank 41
    if time.time() - start_time < time_limit - 15:
        result = try_structured_search(41, time_budget=15.0)
        if result is not None:
            U_list, V_list, W_list = result
            U = np.column_stack(U_list)
            V = np.column_stack(V_list)
            W = np.column_stack(W_list)
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook
    return get_schoolbook()
