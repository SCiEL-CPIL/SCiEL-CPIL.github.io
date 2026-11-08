import numpy as np
import time
from itertools import product
from fractions import Fraction

TARGET_DEADLINE = time.time() + 110  # Safety margin

def _verify(U, V, W, n, m, p):
    R = U.shape[1]
    # Check non-zero entries
    for i in range(n):
        for k in range(p):
            for j in range(m):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                val = np.dot(W[c_idx, :], np.multiply(U[a_idx, :], V[b_idx, :]))
                if abs(val - 1.0) > 1e-9:
                    return False
    # Check zero entries (sample)
    for i in range(n):
        for k in range(p):
            c_idx = k * n + i
            for j in range(m):
                a_idx = i * m + j
                for j_test in range(m):
                    if j_test == j:
                        continue
                    b_idx = j_test * p + k
                    val = np.dot(W[c_idx, :], np.multiply(U[a_idx, :], V[b_idx, :]))
                    if abs(val) > 1e-9:
                        return False
    return True

def _try_rank_structured(R, n, m, p, deadline):
    """Try to find a rank-R decomposition with structured U, V."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    # For each rank r, the term contributes to C[k*n+i] only if j matches
    # U has shape (n*m, R), V has shape (m*p, R), W has shape (p*n, R)
    
    # Strategy: assign each of the R terms to a specific j value
    # For j=0: U rows are (0*3+0, 1*3+0) = (0, 3), V rows are (0*10+k for k in 0..9) = (0..9)
    # For j=1: U rows are (1, 4), V rows are (10..19)
    # For j=2: U rows are (2, 5), V rows are (20..29)
    
    # Let's try: assign r_j terms to each j value
    # The constraint is that for output C[k*n+i], we need sum over j of (terms with that j)
    # to equal 1.
    
    # Simple structured approach: use the same pattern for each j
    # For each j, we have n*p = 2*10 = 20 output entries that depend on j
    # But actually, each output C[k*n+i] depends on all j values
    
    # Let's try a simpler approach: random search for U, V, then solve for W
    np.random.seed(42)
    
    for attempt in range(100):
        if time.time() > deadline:
            return None
        
        # Generate random sparse U, V
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        
        for r in range(R):
            # Choose which j this term is associated with
            j = r % m
            # For U: only rows (i*m + j) for i in range(n) can be non-zero
            for i in range(n):
                if np.random.random() < 0.5:
                    U[i * m + j, r] = np.random.choice([-1, 1])
            # For V: only rows (j*p + k) for k in range(p) can be non-zero
            for k in range(p):
                if np.random.random() < 0.5:
                    V[j * p + k, r] = np.random.choice([-1, 1])
        
        # Build the linear system for W
        # For each (i, k), we need: sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1 for the correct j
        # But actually, the constraint is: for each (i, k), sum over j of (sum over r where term r is for j) W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1
        
        # This is more complex. Let's use a different approach.
        # For each output entry c_idx = k*n+i, the equation is:
        # sum_{r=0}^{R-1} W[c_idx, r] * (sum_{j=0}^{m-1} U[i*m+j, r] * V[j*p+k, r]) = 1
        
        # Wait, that's not right either. The tensor decomposition is:
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For matrix multiplication, T[i,j,k] = 1 if the indices are consistent, 0 otherwise
        # Actually, T[i,j,k] = 1 for all (i,j,k) in the valid range, and the multiplication is:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor is T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor in this context)
        
        # Wait, I need to re-read the problem. The tensor for matrix multiplication has:
        # T[a_idx, b_idx, c_idx] where a_idx = i*m+j, b_idx = j'*p+k, c_idx = k'*n+i'
        # T is 1 if j=j', k=k', i=i', else 0.
        
        # So the constraint is:
        # For all (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # For all (i, j, k, j') with j != j': sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = 0
        
        # This is a linear system in W for fixed U, V.
        
        # Let's build the system properly
        # Unknowns: W[c_idx, r] for c_idx in range(p*n), r in range(R)
        # Total unknowns: p*n*R = 20*R
        
        # Equations:
        # 1. For each (i, j, k) with i in range(n), j in range(m), k in range(p):
        #    sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1
        #    That's n*m*p = 60 equations
        
        # 2. For each (i, k) and each pair (j1, j2) with j1 != j2:
        #    sum_r W[k*n+i, r] * U[i*m+j1, r] * V[j2*p+k, r] = 0
        #    That's n * p * m * (m-1) = 2 * 10 * 3 * 2 = 120 equations
        
        # Total: 180 equations, 20*R unknowns
        
        # Build the system
        num_eq = 0
        eq_list = []
        
        # Equation 1: diagonal terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c_idx = k * n + i
                    a_idx = i * m + j
                    b_idx = j * p + k
                    # Coefficients for W[c_idx, r]
                    coeffs = np.array([U[a_idx, r] * V[b_idx, r] for r in range(R)])
                    # This equation only involves W[c_idx, :]
                    eq_list.append((c_idx, coeffs, 1.0))
                    num_eq += 1
        
        # Equation 2: off-diagonal terms (should be zero)
        for i in range(n):
            for k in range(p):
                c_idx = k * n + i
                for j1 in range(m):
                    a_idx = i * m + j1
                    for j2 in range(m):
                        if j1 == j2:
                            continue
                        b_idx = j2 * p + k
                        coeffs = np.array([U[a_idx, r] * V[b_idx, r] for r in range(R)])
                        eq_list.append((c_idx, coeffs, 0.0))
                        num_eq += 1
        
        # Now solve the linear system
        # The system is block diagonal by c_idx!
        # For each c_idx, we have a system in W[c_idx, :]
        
        W = np.zeros((pn, R))
        valid = True
        
        for c_idx in range(pn):
            # Get all equations for this c_idx
            A_block = []
            b_block = []
            for (eq_c_idx, coeffs, rhs) in eq_list:
                if eq_c_idx == c_idx:
                    A_block.append(coeffs)
                    b_block.append(rhs)
            
            if not A_block:
                continue
            
            A_block = np.array(A_block)
            b_block = np.array(b_block)
            
            # Solve A_block @ W[c_idx, :] = b_block
            try:
                sol, residuals, rank, sv = np.linalg.lstsq(A_block, b_block, rcond=None)
                W[c_idx, :] = sol
                # Check if it's an exact solution
                if np.max(np.abs(A_block @ sol - b_block)) > 1e-9:
                    valid = False
                    break
            except:
                valid = False
                break
        
        if not valid:
            continue
        
        # Verify
        if _verify(U, V, W, n, m, p):
            return U, V, W
    
    return None

def construct():
    n, m, p = 2, 3, 10
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Trivial decomposition
    R_trivial = n * m * p
    U_trivial = np.zeros((nm, R_trivial))
    V_trivial = np.zeros((mp, R_trivial))
    W_trivial = np.zeros((pn, R_trivial))
    
    for i in range(n):
        for k in range(p):
            for j in range(m):
                r = (i * p + k) * m + j
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U_trivial[a_idx, r] = 1.0
                V_trivial[b_idx, r] = 1.0
                W_trivial[c_idx, r] = 1.0
                
    best_U, best_V, best_W = U_trivial, V_trivial, W_trivial
    best_rank = R_trivial
    
    # Try to find lower rank solutions
    for R in range(30, R_trivial):
        if time.time() > TARGET_DEADLINE - 10:
            break
        
        result = _try_rank_structured(R, n, m, p, TARGET_DEADLINE - 10)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = R
                break
    
    return {
        'U': best_U.tolist(),
        'V': best_V.tolist(),
        'W': best_W.tolist()
    }
