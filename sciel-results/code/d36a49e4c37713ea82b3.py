"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    def check_exact(U, V, W):
        """Verify if U, V, W exactly decompose the tensor using Fraction arithmetic."""
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(R):
                        s += Fraction(U[i*m+j, r]).limit_denominator(1000000) * \
                             Fraction(V[j*p+k, r]).limit_denominator(1000000) * \
                             Fraction(W[k*n+i, r]).limit_denominator(1000000)
                    if s != 1:
                        return False
        return True

    def check_exact_float(U, V, W, tol=1e-10):
        """Fast float-based check."""
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def build_A_matrix(U, V, R):
        """Build the linear system A * W_flat = b for W given U and V."""
        A = np.zeros((n*m*p, p*n*R))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row_idx = i*m*p + j*p + k
                    for r in range(R):
                        col_idx = (k*n + i)*R + r
                        A[row_idx, col_idx] = U[i*m+j, r] * V[j*p+k, r]
        return A

    def solve_for_W(U, V, R):
        """Solve for W given U and V. Returns W matrix or None."""
        A = build_A_matrix(U, V, R)
        b = np.ones(n*m*p)
        
        try:
            W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=1e-12)
        except np.linalg.LinAlgError:
            return None
        
        # Check residual
        if len(residuals) > 0:
            if residuals[0] > 1e-8:
                return None
        
        W = W_flat.reshape(p*n, R)
        return W

    def try_snap_and_verify(U, V, W, R):
        """Try to snap W to half-integers and verify."""
        # Try snapping to nearest half-integer
        W_snapped = np.round(W * 2) / 2.0
        
        if check_exact_float(U, V, W_snapped, tol=1e-9):
            # Verify with exact arithmetic
            if check_exact(U, V, W_snapped):
                return W_snapped
        
        # Try snapping to nearest integer
        W_int = np.round(W)
        if check_exact_float(U, V, W_int, tol=1e-9):
            if check_exact(U, V, W_int):
                return W_int
        
        return None

    def generate_U_candidates(R, num_candidates):
        """Generate diverse U matrices."""
        candidates = []
        nm = n * m
        
        # Candidate 1: All ones
        candidates.append(np.ones((nm, R)))
        
        # Candidate 2: Identity-like pattern
        U_id = np.zeros((nm, R))
        for idx in range(min(nm, R)):
            U_id[idx, idx] = 1
        candidates.append(U_id)
        
        # Candidate 3: Random with more structure
        for _ in range(num_candidates):
            U = np.random.randint(-2, 3, size=(nm, R)).astype(float)
            # Ensure each row has at least one non-zero
            for row in range(nm):
                if np.all(U[row] == 0):
                    U[row, np.random.randint(R)] = np.random.choice([-1, 1])
            candidates.append(U)
        
        return candidates

    def generate_V_candidates(R, num_candidates):
        """Generate diverse V matrices."""
        candidates = []
        mp = m * p
        
        # Candidate 1: All ones
        candidates.append(np.ones((mp, R)))
        
        # Candidate 2: Identity-like pattern
        V_id = np.zeros((mp, R))
        for idx in range(min(mp, R)):
            V_id[idx, idx] = 1
        candidates.append(V_id)
        
        # Candidate 3: Random with more structure
        for _ in range(num_candidates):
            V = np.random.randint(-2, 3, size=(mp, R)).astype(float)
            for row in range(mp):
                if np.all(V[row] == 0):
                    V[row, np.random.randint(R)] = np.random.choice([-1, 1])
            candidates.append(V)
        
        return candidates

    # Search for minimal rank
    # Lower bound: max(n, m, p) = 8, but also nm=6, mp=24, pn=16
    # The rank must be at least max(nm, mp, pn) / something... 
    # Actually, the rank of the matmul tensor for n x m x p is at least max(n*m, m*p, p*n) is NOT true.
    # The correct lower bound is more subtle. Let's just try from 8 upward.
    
    for R in range(8, 35):
        if time.time() > deadline - 15:
            break
        
        U_cands = generate_U_candidates(R, 20)
        V_cands = generate_V_candidates(R, 20)
        
        found = False
        for U in U_cands:
            if time.time() > deadline - 10:
                break
            for V in V_cands:
                if time.time() > deadline - 10:
                    break
                
                W = solve_for_W(U, V, R)
                if W is None:
                    continue
                
                W_snapped = try_snap_and_verify(U, V, W, R)
                if W_snapped is not None:
                    best_rank = R
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W_snapped.tolist()
                    }
                    found = True
                    break
            if found:
                break
        
        if found:
            # Found a solution at rank R; since we search upward, this is minimal
            return best_result

    # If no better rank found via the above, try a more targeted approach
    # Use the structure: for each output (i,k), we need sum_j A[i,j]B[j,k] = C[i,k]
    # This means the tensor is "all ones" on the valid index set.
    
    # Alternative: Try to find a decomposition where W has a specific structure
    # e.g., W[k*n+i, r] is non-zero only for specific (k,i) pairs
    
    for R in range(8, 30):
        if time.time() > deadline - 5:
            break
        
        # Try sparse W patterns
        for attempt in range(30):
            if time.time() > deadline - 5:
                break
            
            # Random sparse U, V
            U = np.random.choice([0, 0, 0, 1, -1], size=(n*m, R)).astype(float)
            V = np.random.choice([0, 0, 0, 1, -1], size=(m*p, R)).astype(float)
            
            W = solve_for_W(U, V, R)
            if W is None:
                continue
            
            W_snapped = try_snap_and_verify(U, V, W, R)
            if W_snapped is not None:
                best_rank = R
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W_snapped.tolist()
                }
                return best_result

    return best_result

# EVOLVE-BLOCK-END
