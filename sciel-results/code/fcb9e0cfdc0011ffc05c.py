import numpy as np
import time
from fractions import Fraction
import sys

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline (10s safety margin)
    
    def elapsed():
        return time.time() - start_time
    
    def time_left():
        return deadline - time.time()

    # Verification function
    def verify(U, V, W, rank):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def check_half_integer(U, V, W):
        """Check if all entries are half-integers (multiples of 0.5)"""
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        return True

    # Build rank-28 decomposition
    def build_rank_28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        def add_strassen(i_off, j_off, k_off):
            nonlocal col
            # M1
            U[(0 + i_off) * m + (0 + j_off)][col] = 1
            U[(1 + i_off) * m + (1 + j_off)][col] = 1
            V[(0 + j_off) * p + (0 + k_off)][col] = 1
            V[(1 + j_off) * p + (1 + k_off)][col] = 1
            W[(0 + k_off) * n + (0 + i_off)][col] = 1
            W[(1 + k_off) * n + (1 + i_off)][col] = 1
            col += 1
            # M2
            U[(1 + i_off) * m + (0 + j_off)][col] = 1
            U[(1 + i_off) * m + (1 + j_off)][col] = 1
            V[(0 + j_off) * p + (0 + k_off)][col] = 1
            W[(0 + k_off) * n + (1 + i_off)][col] = 1
            W[(1 + k_off) * n + (1 + i_off)][col] = -1
            col += 1
            # M3
            U[(0 + i_off) * m + (0 + j_off)][col] = 1
            V[(0 + j_off) * p + (1 + k_off)][col] = 1
            V[(1 + j_off) * p + (1 + k_off)][col] = -1
            W[(1 + k_off) * n + (0 + i_off)][col] = 1
            W[(1 + k_off) * n + (1 + i_off)][col] = 1
            col += 1
            # M4
            U[(1 + i_off) * m + (1 + j_off)][col] = 1
            V[(1 + j_off) * p + (0 + k_off)][col] = 1
            V[(0 + j_off) * p + (0 + k_off)][col] = -1
            W[(0 + k_off) * n + (0 + i_off)][col] = 1
            W[(0 + k_off) * n + (1 + i_off)][col] = 1
            col += 1
            # M5
            U[(0 + i_off) * m + (0 + j_off)][col] = 1
            U[(0 + i_off) * m + (1 + j_off)][col] = 1
            V[(1 + j_off) * p + (1 + k_off)][col] = 1
            W[(0 + k_off) * n + (0 + i_off)][col] = -1
            W[(1 + k_off) * n + (0 + i_off)][col] = 1
            col += 1
            # M6
            U[(1 + i_off) * m + (0 + j_off)][col] = 1
            U[(0 + i_off) * m + (0 + j_off)][col] = -1
            V[(0 + j_off) * p + (0 + k_off)][col] = 1
            V[(0 + j_off) * p + (1 + k_off)][col] = 1
            W[(1 + k_off) * n + (1 + i_off)][col] = 1
            col += 1
            # M7
            U[(0 + i_off) * m + (1 + j_off)][col] = 1
            U[(1 + i_off) * m + (1 + j_off)][col] = -1
            V[(1 + j_off) * p + (0 + k_off)][col] = 1
            V[(1 + j_off) * p + (1 + k_off)][col] = 1
            W[(0 + k_off) * n + (0 + i_off)][col] = 1
            col += 1

        # Block 1: j=0 (schoolbook: 2*5 = 10 terms)
        for i in range(n):
            for k in range(p):
                U[i * m + 0][col] = 1
                V[0 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
        # Block 2: j=1,2, k=0,1 (Strassen: 7 terms)
        add_strassen(0, 1, 0)
        
        # Block 3: j=1,2, k=2 (schoolbook: 2*2 = 4 terms)
        for i in range(n):
            for j in [1, 2]:
                U[i * m + j][col] = 1
                V[j * p + 2][col] = 1
                W[2 * n + i][col] = 1
                col += 1
                
        # Block 4: j=1,2, k=3,4 (Strassen: 7 terms)
        add_strassen(0, 1, 3)
        
        assert col == rank, f"Column count mismatch: {col} != {rank}"
        return {"U": U, "V": V, "W": W}

    # Try to find rank-27 using rational linear algebra (slice method)
    def try_find_rank_27():
        """
        Use the slice method: fix one factor (say W) to random rational values,
        then the problem becomes finding a rank-27 decomposition of a matrix
        (the "slice"), which can be attacked with linear algebra over Q.
        
        More specifically: if we fix W's columns to be e_r (standard basis),
        then U and V must satisfy: sum_r U[:,r] V[:,r]^T (with W[:,r] = e_r)
        gives the tensor. But this is over-constrained.
        
        Alternative: Use the fact that for n=2, we can view the tensor as
        a linear map from (3x5 matrices) to (2x5 matrices) and use
        border rank / flattening arguments.
        
        Practical approach: Try to find a rank-27 solution by starting from
        rank-28 and attempting to merge two terms, or by direct search with
        rational constraints.
        """
        rank = 27
        
        # Strategy: Start from rank-28 and try to eliminate one term
        # by expressing it as a linear combination of the others.
        # This is equivalent to finding a rank-27 decomposition.
        
        # Use iterative refinement with rational arithmetic
        # Initialize with random rational values (small integers / 2)
        
        best_err = float('inf')
        best_solution = None
        
        np.random.seed(12345)
        
        # Try multiple random initializations
        for attempt in range(3):
            if time_left() < 5:
                break
            
            # Initialize with random small rational values
            U = np.random.randint(-2, 3, size=(n * m, rank)).astype(float) * 0.5
            V = np.random.randint(-2, 3, size=(m * p, rank)).astype(float) * 0.5
            W = np.random.randint(-2, 3, size=(p * n, rank)).astype(float) * 0.5
            
            # ALS with rational snapping
            max_iters = 30
            for it in range(max_iters):
                if time_left() < 5:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        M_ij = np.zeros((p, rank))
                        for k in range(p):
                            M_ij[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        try:
                            U[i * m + j, :] = np.linalg.lstsq(M_ij, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        M_jk = np.zeros((n, rank))
                        for i in range(n):
                            M_jk[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        try:
                            V[j * p + k, :] = np.linalg.lstsq(M_jk, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        M_ki = np.zeros((m, rank))
                        for j in range(m):
                            M_ki[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            W[k * n + i, :] = np.linalg.lstsq(M_ki, np.ones(m), rcond=None)[0]
                        except:
                            pass
                
                # Compute error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                            err += (val - 1.0) ** 2
                
                if err < best_err:
                    best_err = err
                    best_solution = (U.copy(), V.copy(), W.copy())
                
                if err < 1e-10:
                    break
            
            if best_err < 1e-10:
                break
        
        if best_solution is not None and best_err < 1e-8:
            U_sol, V_sol, W_sol = best_solution
            
            # Try to snap to half-integers
            U_snap = np.round(U_sol * 2) / 2
            V_snap = np.round(V_sol * 2) / 2
            W_snap = np.round(W_sol * 2) / 2
            
            # Verify snapped solution
            err_snap = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_snap[i * m + j, :], V_snap[j * p + k, :] * W_snap[k * n + i, :])
                        err_snap += (val - 1.0) ** 2
            
            if err_snap < 1e-6:
                return {
                    "U": U_snap.tolist(),
                    "V": V_snap.tolist(),
                    "W": W_snap.tolist()
                }
        
        return None

    # Try to find rank-27 using a different approach: 
    # Exploit the structure by fixing some entries and solving linear systems
    def try_find_rank_27_structured():
        """
        Structured approach: Use the known rank-28 decomposition and try to
        replace two terms with one by finding a rank-1 update.
        
        If we have a rank-28 decomposition, we can try to find vectors
        u', v', w' such that the sum of two existing terms equals
        u' ⊗ v' ⊗ w' (in the tensor sense).
        """
        rank = 27
        U28, V28, W28 = build_rank_28()["U"], build_rank_28()["V"], build_rank_28()["W"]
        
        # Try all pairs of terms to see if their sum can be expressed as rank 1
        # This is a very expensive search, so we'll use a heuristic
        
        # Alternative: Use the fact that the multiplication tensor for 2x3x5
        # has a known structure. Try to find a rank-27 decomposition by
        # solving a system of polynomial equations.
        
        # For now, use a grid search over small rational values for a subset
        # of free parameters
        
        # This is a placeholder for a more sophisticated approach
        # that would use Groebner bases or similar algebraic methods.
        
        # Given time constraints, we'll rely on the ALS approach above.
        return None

    # First, try the quick rank-27 search
    if time_left() > 30:
        res_27 = try_find_rank_27()
        if res_27:
            # Verify
            U = np.array(res_27["U"], dtype=float)
            V = np.array(res_27["V"], dtype=float)
            W = np.array(res_27["W"], dtype=float)
            if verify(res_27["U"], res_27["V"], res_27["W"], 27):
                return res_27
    
    # Try the structured approach if time permits
    if time_left() > 20:
        res_27s = try_find_rank_27_structured()
        if res_27s:
            U = np.array(res_27s["U"], dtype=float)
            V = np.array(res_27s["V"], dtype=float)
            W = np.array(res_27s["W"], dtype=float)
            if verify(res_27s["U"], res_27s["V"], res_27s["W"], 27):
                return res_27s
    
    # Fallback to rank 28
    res_28 = build_rank_28()
    
    # Verify rank-28 solution
    U = np.array(res_28["U"], dtype=float)
    V = np.array(res_28["V"], dtype=float)
    W = np.array(res_28["W"], dtype=float)
    if verify(res_28["U"], res_28["V"], res_28["W"], 28):
        return res_28
    
    # Ultimate fallback: schoolbook rank 30
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
