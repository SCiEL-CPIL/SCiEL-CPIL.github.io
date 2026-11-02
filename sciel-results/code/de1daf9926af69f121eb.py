import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 4, 5]

def construct():
    """
    Find a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 2x4 matrix by a 4x5 matrix.
    
    Strategy:
    1. Construct a rank-36 solution using Strassen for 2x2x2 blocks + schoolbook for 2x2x1.
    2. Attempt to reduce the rank to 35, 34, etc. using least-squares optimization.
    3. Verify exactness with integer/half-integer snapping.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    nm, mp, pn = n * m, m * p, p * n
    
    start_time = time.time()
    time_limit = 115  # Leave margin for 120s limit
    
    # --- Helper to verify a solution ---
    def verify_solution(U, V, W):
        # Check if U, V, W satisfy the tensor equations exactly (with tolerance)
        R = U.shape[1]
        for i in range(n):
            for k in range(p):
                for a in range(n):
                    for b in range(m):
                        for c in range(m):
                            for d in range(p):
                                val = 0.0
                                for mu in range(R):
                                    val += U[a*m+b, mu] * V[c*p+d, mu] * W[k*n+i, mu]
                                if b == c and a == i and d == k:
                                    if abs(val - 1.0) > 1e-9:
                                        return False
                                else:
                                    if abs(val - 0.0) > 1e-9:
                                        return False
        return True

    # --- Helper to check if entries are close to half-integers ---
    def snap_to_half_integers(U, V, W):
        U_snapped = np.round(U * 2) / 2.0
        V_snapped = np.round(V * 2) / 2.0
        W_snapped = np.round(W * 2) / 2.0
        return U_snapped, V_snapped, W_snapped

    # --- Precompute constraint structure ---
    # indices: (i, k, a, b, c, d, target)
    cons_list = []
    for i in range(n):
        for k in range(p):
            for a in range(n):
                for b in range(m):
                    for c in range(m):
                        for d in range(p):
                            target = 1.0 if (b == c and a == i and d == k) else 0.0
                            cons_list.append((i, k, a, b, c, d, target))
    
    # --- Fast residual calculation (vectorized) ---
    # Precompute index arrays for vectorized access
    cons_idx = np.array([(i, k, a, b, c, d, target) for (i, k, a, b, c, d, target) in cons_list])
    cons_i = cons_idx[:, 0].astype(int)
    cons_k = cons_idx[:, 1].astype(int)
    cons_a = cons_idx[:, 2].astype(int)
    cons_b = cons_idx[:, 3].astype(int)
    cons_c = cons_idx[:, 4].astype(int)
    cons_d = cons_idx[:, 5].astype(int)
    cons_target = cons_idx[:, 6]
    
    def residual_vector_fast(x, R):
        size_U = nm * R
        size_V = mp * R
        size_W = pn * R
        
        U_flat = x[0:size_U]
        V_flat = x[size_U:size_U+size_V]
        W_flat = x[size_U+size_V:size_U+size_V+size_W]
        
        U = U_flat.reshape(nm, R)
        V = V_flat.reshape(mp, R)
        W = W_flat.reshape(pn, R)
        
        # Vectorized: compute U[a,b,:] * V[c,d,:] * W[k,i,:] for each constraint
        U_sel = U[cons_a * m + cons_b, :]  # (N, R)
        V_sel = V[cons_c * p + cons_d, :]  # (N, R)
        W_sel = W[cons_k * n + cons_i, :]  # (N, R)
        
        # Element-wise product then sum over R
        res = np.einsum('ir,ir,ir->i', U_sel, V_sel, W_sel) - cons_target
        return res

    # --- Build rank-36 solution ---
    def get_initial_solution_rank36():
        rank = 36
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        
        # Strassen 2x2x2 block
        def add_strassen_222(i0, j0, k0):
            nonlocal col
            # M1: (A00+A11)(B00+B11)
            U[i0*m + j0][col] += 1; U[(i0+1)*m + (j0+1)][col] += 1
            V[j0*p + k0][col] += 1; V[(j0+1)*p + (k0+1)][col] += 1
            W[k0*n + i0][col] += 1; W[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            
            # M2: (A10+A11)B00
            U[(i0+1)*m + j0][col] += 1; U[(i0+1)*m + (j0+1)][col] += 1
            V[j0*p + k0][col] += 1
            W[k0*n + (i0+1)][col] += 1; W[(k0+1)*n + (i0+1)][col] -= 1
            col += 1
            
            # M3: A00(B10-B11)
            U[i0*m + j0][col] += 1
            V[j0*p + (k0+1)][col] += 1; V[(j0+1)*p + (k0+1)][col] -= 1
            W[(k0+1)*n + i0][col] += 1; W[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            
            # M4: A11(B01-B00)
            U[(i0+1)*m + (j0+1)][col] += 1
            V[(j0+1)*p + k0][col] += 1; V[j0*p + k0][col] -= 1
            W[k0*n + i0][col] += 1; W[k0*n + (i0+1)][col] += 1
            col += 1
            
            # M5: (A00+A01)B11
            U[i0*m + j0][col] += 1; U[i0*m + (j0+1)][col] += 1
            V[(j0+1)*p + (k0+1)][col] += 1
            W[k0*n + i0][col] -= 1; W[(k0+1)*n + i0][col] += 1
            col += 1
            
            # M6: (A10-A00)(B00+B01)
            U[(i0+1)*m + j0][col] += 1; U[i0*m + j0][col] -= 1
            V[j0*p + k0][col] += 1; V[j0*p + (k0+1)][col] += 1
            W[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            
            # M7: (A01-A11)(B10+B11)
            U[i0*m + (j0+1)][col] += 1; U[(i0+1)*m + (j0+1)][col] -= 1
            V[(j0+1)*p + k0][col] += 1; V[(j0+1)*p + (k0+1)][col] += 1
            W[k0*n + i0][col] += 1
            col += 1
            
            return col
        
        # Schoolbook 2x2x1 block
        def add_schoolbook_2x2x1(i0, j0, k0):
            nonlocal col
            for di in range(2):
                for dj in range(2):
                    U[(i0 + di)*m + (j0 + dj)][col] = 1
                    V[(j0 + dj)*p + k0][col] = 1
                    W[k0*n + (i0 + di)][col] = 1
                    col += 1
            return col
        
        # 4 Strassen blocks for the 4x4 part + 2 schoolbook blocks for the 4x1 part
        # Rows 0-1 of A, Cols 0-1 of B, Cols 0-1 of C
        add_strassen_222(0, 0, 0)
        # Rows 0-1 of A, Cols 2-3 of B, Cols 0-1 of C
        add_strassen_222(0, 2, 0)
        # Rows 0-1 of A, Cols 0-1 of B, Cols 2-3 of C
        add_strassen_222(0, 0, 2)
        # Rows 0-1 of A, Cols 2-3 of B, Cols 2-3 of C
        add_strassen_222(0, 2, 2)
        # Rows 0-1 of A, Cols 0-1 of B, Col 4 of C
        add_schoolbook_2x2x1(0, 0, 4)
        # Rows 0-1 of A, Cols 2-3 of B, Col 4 of C
        add_schoolbook_2x2x1(0, 2, 4)
        
        return U, V, W

    U36, V36, W36 = get_initial_solution_rank36()
    
    # Verify the rank-36 solution
    if verify_solution(U36, V36, W36):
        U_snap, V_snap, W_snap = snap_to_half_integers(U36, V36, W36)
        if verify_solution(U_snap, V_snap, W_snap):
            # Try to reduce rank
            best_rank = 36
            best_solution = (U_snap, V_snap, W_snap)
            
            # Attempt to find lower ranks
            for R in range(35, 20, -1):
                if time.time() - start_time > time_limit:
                    break
                
                # Try more random initializations with the faster residual
                for attempt in range(5):
                    if time.time() - start_time > time_limit:
                        break
                    
                    if attempt == 0:
                        # Truncate rank 36 solution
                        U_init = U36[:, :R]
                        V_init = V36[:, :R]
                        W_init = W36[:, :R]
                    else:
                        # Random
                        U_init = np.random.normal(0, 0.5, (nm, R))
                        V_init = np.random.normal(0, 0.5, (mp, R))
                        W_init = np.random.normal(0, 0.5, (pn, R))
                    
                    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                    
                    try:
                        result = opt.least_squares(residual_vector_fast, x0, args=(R,), method='lm', max_nfev=5000)
                        if result.success:
                            x_opt = result.x
                            size_U = nm * R
                            size_V = mp * R
                            size_W = pn * R
                            U_opt = x_opt[0:size_U].reshape(nm, R)
                            V_opt = x_opt[size_U:size_U+size_V].reshape(mp, R)
                            W_opt = x_opt[size_U+size_V:size_U+size_V+size_W].reshape(pn, R)
                            
                            # Check residual
                            res = residual_vector_fast(x_opt, R)
                            if np.max(np.abs(res)) < 1e-4:
                                # High-precision polish: run a tighter local optimization
                                if time.time() - start_time < time_limit - 5:
                                    try:
                                        result2 = opt.least_squares(residual_vector_fast, x_opt, args=(R,), method='lm', max_nfev=2000, xtol=1e-12, ftol=1e-12)
                                        if result2.success:
                                            x_opt = result2.x
                                            U_opt = x_opt[0:size_U].reshape(nm, R)
                                            V_opt = x_opt[size_U:size_U+size_V].reshape(mp, R)
                                            W_opt = x_opt[size_U+size_V:size_U+size_V+size_W].reshape(pn, R)
                                            res = residual_vector_fast(x_opt, R)
                                    except Exception:
                                        pass
                            
                            if np.max(np.abs(res)) < 1e-6:
                                # Snap to half-integers
                                U_snap, V_snap, W_snap = snap_to_half_integers(U_opt, V_opt, W_opt)
                                if verify_solution(U_snap, V_snap, W_snap):
                                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                    except Exception as e:
                        pass
            
            # Return the best solution found (rank 36)
            return {"U": best_solution[0].tolist(), "V": best_solution[1].tolist(), "W": best_solution[2].tolist()}
        else:
            # If snapping failed, return original
            return {"U": U36.tolist(), "V": V36.tolist(), "W": W36.tolist()}
    else:
        # Fallback: schoolbook
        rank = nm * mp
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
