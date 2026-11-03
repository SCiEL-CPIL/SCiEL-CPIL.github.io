import time
import numpy as np
from fractions import Fraction

def _build_44_rank(n, m, p):
    """Build the known 44-rank decomposition."""
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    rank = 44
    
    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]
    
    col = 0
    
    # Part 1: 2x1x8 part (j=0)
    for k in range(p):
        U[0 * m + 0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        U[1 * m + 0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1
        
    # Part 2: 2x2x8 part (j=1,2)
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # P1: (A[0,1] + A[1,2]) * (B[1,k0] + B[2,k1])
        U[0 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # P2: (A[1,1] + A[1,2]) * B[1,k0]
        U[1 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1
        
        # P3: A[0,1] * (B[1,k1] - B[2,k1])
        U[0 * m + 1][col] = 1
        V[1 * p + k1][col] = 1
        V[2 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # P4: A[1,2] * (B[2,k0] - B[1,k0])
        U[1 * m + 2][col] = 1
        V[2 * p + k0][col] = 1
        V[1 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # P5: (A[0,1] + A[0,2]) * B[2,k1]
        U[0 * m + 1][col] = 1
        U[0 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # P6: (A[1,1] - A[0,1]) * (B[1,k0] + B[1,k1])
        U[1 * m + 1][col] = 1
        U[0 * m + 1][col] = -1
        V[1 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # P7: (A[0,2] - A[1,2]) * (B[2,k0] + B[2,k1])
        U[0 * m + 2][col] = 1
        U[1 * m + 2][col] = -1
        V[2 * p + k0][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}

def _try_rank(R, n, m, p, deadline):
    """
    Try to find a rank-R decomposition by fixing U and V patterns
    and solving for W via linear algebra.
    
    Returns (U, V, W) if found, None otherwise.
    """
    if time.time() > deadline:
        return None
    
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # The tensor T[i,j,k] = 1 if (i,j,k) corresponds to a valid multiplication
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    
    # For our problem, T[i,j,k] = 1 for all valid (i,j,k) in the multiplication tensor
    # Actually, the multiplication tensor has T[i,j,k] = 1 when the output C[i,k] 
    # gets a contribution from A[i,j]*B[j,k]
    
    # Let's use a different approach: try specific U,V patterns and solve for W
    
    # Pattern: use Strassen-like blocks for the 2x2x8 part
    # First 16 terms: j=0 part (mandatory)
    # Remaining R-16 terms: for j=1,2 part
    
    R_rem = R - 16
    if R_rem < 0:
        return None
    
    # Build U and V for the j=0 part
    U = np.zeros((dim_U, R), dtype=object)
    V = np.zeros((dim_V, R), dtype=object)
    
    col = 0
    for k in range(p):
        U[0 * m + 0, col] = 1
        V[0 * p + k, col] = 1
        col += 1
        U[1 * m + 0, col] = 1
        V[0 * p + k, col] = 1
        col += 1
    
    # For the remaining R_rem terms, use a systematic pattern
    # Try: group columns into pairs, use 7 terms per pair (Strassen)
    # If R_rem < 28, we need a more efficient scheme
    
    # Let's try: use R_rem terms for the 2x2x8 part
    # We'll set up U and V with a pattern, then solve for W
    
    # Strategy: for the 2x2x8 part, use a "Winograd" style
    # Process all 8 columns with shared A-side combinations
    
    # Try different patterns
    patterns = []
    
    # Pattern 1: 7 terms per pair of columns (standard Strassen)
    if R_rem == 28:
        for b in range(4):
            k0 = 2 * b
            k1 = 2 * b + 1
            base = 16 + 7 * b
            
            U[0*m+1, base+0] = 1
            U[1*m+2, base+0] = 1
            V[1*p+k0, base+0] = 1
            V[2*p+k1, base+0] = 1
            
            U[1*m+1, base+1] = 1
            U[1*m+2, base+1] = 1
            V[1*p+k0, base+1] = 1
            
            U[0*m+1, base+2] = 1
            V[1*p+k1, base+2] = 1
            V[2*p+k1, base+2] = -1
            
            U[1*m+2, base+3] = 1
            V[2*p+k0, base+3] = 1
            V[1*p+k0, base+3] = -1
            
            U[0*m+1, base+4] = 1
            U[0*m+2, base+4] = 1
            V[2*p+k1, base+4] = 1
            
            U[1*m+1, base+5] = 1
            U[0*m+1, base+5] = -1
            V[1*p+k0, base+5] = 1
            V[1*p+k1, base+5] = 1
            
            U[0*m+2, base+6] = 1
            U[1*m+2, base+6] = -1
            V[2*p+k0, base+6] = 1
            V[2*p+k1, base+6] = 1
    
    # Pattern 2: Try 6 terms per pair for some pairs
    # This is complex; let's just try to solve for W given the U,V above
    
    # Set up linear system for W
    # For each (i,j,k), we need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # This is linear in W entries.
    
    # Collect all equations
    A_mat = []
    b_vec = []
    w_indices = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq = []
                for r in range(R):
                    u_val = U[i*m+j, r]
                    v_val = V[j*p+k, r]
                    if u_val != 0 and v_val != 0:
                        eq.append((r, u_val * v_val))
                # The equation is: sum over (r, coeff) of coeff * W[k*n+i, r] = 1
                # But W[k*n+i, r] is the unknown
                row = np.zeros(R, dtype=object)
                for r, coeff in eq:
                    row[r] = coeff
                A_mat.append(row)
                b_vec.append(1)
                w_indices.append((k*n+i, list(range(R))))
    
    # Actually, let me set this up more carefully
    # Unknowns: W[idx, r] for idx in range(dim_W), r in range(R)
    # Total unknowns: dim_W * R
    
    # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    num_eqs = n * m * p
    num_unk = dim_W * R
    
    A_sys = np.zeros((num_eqs, num_unk), dtype=object)
    b_sys = np.zeros(num_eqs, dtype=object)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq_idx = i * m * p + j * p + k
                b_sys[eq_idx] = 1
                for r in range(R):
                    u_val = U[i*m+j, r]
                    v_val = V[j*p+k, r]
                    if u_val != 0 and v_val != 0:
                        unk_idx = (k*n+i) * R + r
                        A_sys[eq_idx, unk_idx] = u_val * v_val
    
    # Solve the linear system
    # Convert to float for solving, then check if solution is exact
    A_float = np.zeros((num_eqs, num_unk), dtype=float)
    b_float = np.zeros(num_eqs, dtype=float)
    
    for i in range(num_eqs):
        b_float[i] = float(b_sys[i])
        for j in range(num_unk):
            if A_sys[i, j] != 0:
                A_float[i, j] = float(A_sys[i, j])
    
    try:
        # Use least squares since the system may be underdetermined
        sol, residuals, rank, sv = np.linalg.lstsq(A_float, b_float, rcond=None)
    except:
        return None
    
    # Check if solution is close to integer/half-integer
    W_float = sol.reshape(dim_W, R)
    
    # Check if the solution is approximately integer
    W_check = np.round(W_float * 2) / 2  # Round to nearest half-integer
    W_check = np.where(np.abs(W_float - W_check) < 1e-6, W_check, W_float)
    
    # Verify the solution
    W_list = [[0] * R for _ in range(dim_W)]
    for idx in range(dim_W):
        for r in range(R):
            val = W_check[idx, r]
            # Check if it's close to integer or half-integer
            val_rounded = round(val * 2) / 2
            if abs(val - val_rounded) < 1e-6:
                W_list[idx][r] = val_rounded
            else:
                # Not a valid half-integer solution
                return None
    
    # Final verification
    U_list = [[int(U[i, r]) if U[i, r] == int(U[i, r]) else float(U[i, r]) for r in range(R)] for i in range(dim_U)]
    V_list = [[int(V[i, r]) if V[i, r] == int(V[i, r]) else float(V[i, r]) for r in range(R)] for i in range(dim_V)]
    
    if _verify_decomposition(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _build_42_rank(n, m, p):
    """Try a 42-rank decomposition using linear algebra search."""
    deadline = time.time() + 30
    return _try_rank(42, n, m, p, deadline)

def _build_43_rank(n, m, p):
    """Try a 43-rank decomposition using linear algebra search."""
    deadline = time.time() + 30
    return _try_rank(43, n, m, p, deadline)

def _verify_decomposition(U, V, W, n, m, p):
    """Check that the decomposition is exact."""
    R = len(U[0]) if U else 0
    if R == 0:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each (i, j, k), check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(R):
                    val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(val - 1.0) > 1e-6:
                    return False
    
    # Also check that the decomposition doesn't produce spurious outputs
    # For the multiplication tensor, T[i,j,k] should be 1 only when
    # the output C[i,k] receives contribution from A[i,j]*B[j,k]
    # Actually, for matrix multiplication, every (i,j,k) triple contributes
    # to C[i,k], so we need T[i,j,k] = 1 for all valid triples.
    # But we also need to make sure there are no spurious contributions.
    # The check above verifies that each (i,j,k) gives exactly 1.
    # We should also verify that the bilinear form is correct.
    
    # Let's do a full check: for random A and B, verify C = A*B
    # Use a few test cases
    import random
    random.seed(42)
    for trial in range(3):
        A_test = np.array([[random.randint(-5, 5) for _ in range(m)] for _ in range(n)])
        B_test = np.array([[random.randint(-5, 5) for _ in range(p)] for _ in range(m)])
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for r in range(R):
            A_proj = U_arr[:, r] @ A_test.reshape(-1)  # This isn't right
            # Actually: U is (n*m, R), so U[:, r] is a vector of length n*m
            # A is (n, m), so A.reshape(-1) is length n*m
            # The product U[:, r] @ A.reshape(-1) gives the r-th "factor" of A
            pass
        
        # Let me redo this properly
        # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
        # Wait, that's not right either. The decomposition is:
        # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j' V[j'*p+k, r] * B[j',k])
        # Hmm, actually the standard tensor decomposition for matrix mult is:
        # C[i,k] = sum_r U[i, r] * V[r, k] where U and V are the factor matrices
        # But here we have a 3-tensor decomposition.
        
        # The correct interpretation:
        # For each r, compute:
        #   a_r = sum_{i,j} U[i*m+j, r] * A[i,j]  (but U only has entries for specific i,j)
        #   b_r = sum_{j,k} V[j*p+k, r] * B[j,k]  (but V only has entries for specific j,k)
        #   c_r = a_r * b_r
        # Then C[i,k] = sum_r W[k*n+i, r] * c_r
    
        C_computed = np.zeros((n, p))
        for r in range(R):
            a_r = 0.0
            for i in range(n):
                for j in range(m):
                    a_r += U_arr[i*m+j, r] * A_test[i, j]
            b_r = 0.0
            for j in range(m):
                for k in range(p):
                    b_r += V_arr[j*p+k, r] * B_test[j, k]
            c_r = a_r * b_r
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += W_arr[k*n+i, r] * c_r
        
        if not np.allclose(C_computed, C_expected, atol=1e-6):
            return False
    
    return True

def construct():
    n, m, p = 2, 3, 8
    start_time = time.time()
    deadline = start_time + 100  # Safety margin (evaluator kills at 150s)
    
    # Start with the known 44-rank solution
    best_result = _build_44_rank(n, m, p)
    
    # Try to find lower rank solutions by searching R from 43 down to 30
    for R in range(43, 29, -1):
        if time.time() > deadline - 20:
            break
        result = _try_rank(R, n, m, p, deadline - 5)
        if result is not None:
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                best_result = result
                break  # Found a valid lower-rank solution, stop
    
    return best_result

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
    if _verify_decomposition(res["U"], res["V"], res["W"], 2, 3, 8):
        print("Verification PASSED")
    else:
        print("Verification FAILED")
