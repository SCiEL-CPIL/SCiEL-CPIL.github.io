import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [4, 5, 6]

# Fano plane (finite projective plane of order 2): 7 points, 7 lines
# Points: 0..6, Lines are triples
FANO_LINES = [
    (0, 1, 2),
    (0, 3, 4),
    (0, 5, 6),
    (1, 3, 5),
    (1, 4, 6),
    (2, 3, 6),
    (2, 4, 5),
]
# Incidence: point p is on line l if p in FANO_LINES[l]
FANO_INC = np.zeros((7, 7), dtype=int)
for l, (a, b, c) in enumerate(FANO_LINES):
    FANO_INC[a, l] = 1
    FANO_INC[b, l] = 1
    FANO_INC[c, l] = 1

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin before 150s kill

    # Build the target tensor structure
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # U shape: (n*m, r), V shape: (m*p, r), W shape: (p*n, r)
    # Constraint: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]

    nonzero_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero_entries.append((a, b, c))

    nz_set = set(nonzero_entries)
    nm = n * m  # 20
    mp = m * p  # 30
    pn = p * n  # 24

    # Schoolbook decomposition (rank 120)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    fallback = schoolbook()

    # Verify a decomposition numerically
    def verify(U, V, W, tol=1e-6):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)

        max_err = 0.0
        # Check nonzero entries
        for (a, b, c) in nonzero_entries:
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            max_err = max(max_err, abs(val - 1.0))

        # Check zero entries (sample for speed, but check all if small)
        # Full check: 20*30*24 = 14400 entries, minus 120 nonzero = 14280
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if (a, b, c) not in nz_set:
                        val = np.dot(U[a, :], V[b, :] * W[c, :])
                        max_err = max(max_err, abs(val))

        return max_err <= tol, max_err

    # Strategy: Fix a sparse structure for U and V, then solve for W
    # Key insight: If we fix U and V with a specific sparse pattern,
    # the constraint on W becomes a system of linear equations.
    #
    # For each nonzero (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] = 1
    # For each zero (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] = 0
    #
    # This is linear in W[c,:] for each c.
    # For fixed c, the equations involving W[c,:] are:
    #   For each (a,b) such that (a,b,c) is in the tensor:
    #     sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    #
    # So for each c, we have a linear system: M_c * W[c,:] = t_c
    # where M_c[r, (a,b)] = U[a,r]*V[b,r] and t_c[(a,b)] = T[a,b,c]
    #
    # But the system must be consistent across all c, and W[c,:] are coupled
    # through the shared U and V. So we need to find U, V such that for each c,
    # the system is consistent.
    #
    # Alternative approach: Use the fact that the tensor has a specific structure.
    # Let's try a structured approach where U and V have a block structure.
    
    # Approach: Try to find a decomposition where U and V have a specific
    # low-rank structure, then verify if W can be solved.
    
    # Let's try a different approach: use the known structure of matrix multiplication.
    # For M_{n,m,p}, a natural decomposition uses the "outer product" structure.
    
    # Try rank r with a structured initialization and Newton's method
    
    def try_structured(r, max_iters=100):
        """Try to find a rank-r decomposition using structured initialization."""
        if time.time() > deadline:
            return None
        
        # Initialize U, V, W with a structured pattern
        # Use the schoolbook as a starting point but with rank r
        # We'll use random initialization with some structure
        
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for attempt in range(3):
            if time.time() > deadline:
                return None
            
            # Random initialization
            np.random.seed(42 + attempt)
            U = np.random.randn(nm, r) * 0.3
            V = np.random.randn(mp, r) * 0.3
            W = np.random.randn(pn, r) * 0.3
            
            # ALS iterations
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Update U: for each a, solve linear system
                for a in range(nm):
                    i = a // m
                    j = a % m
                    # Equations: for each k, U[a,:] dot (V[j*p+k,:] * W[k*n+i,:]) = 1
                    A = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                    b_vec = np.ones(p)
                    U[a, :] = np.linalg.lstsq(A, b_vec, rcond=None)[0]
                
                # Update V
                for b in range(mp):
                    j = b // p
                    k = b % p
                    A = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                    b_vec = np.ones(n)
                    V[b, :] = np.linalg.lstsq(A, b_vec, rcond=None)[0]
                
                # Update W
                for c in range(pn):
                    k = c // n
                    i = c % n
                    A = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                    b_vec = np.ones(m)
                    W[c, :] = np.linalg.lstsq(A, b_vec, rcond=None)[0]
            
            # Verify
            valid, err = verify(U, V, W, tol=1e-5)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if valid:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None
    
    # Try a more direct approach: exploit the rectangular structure
    # Fix U and V to have a specific sparse pattern, then solve for W
    
    def try_sparse_fix(r, max_attempts=5):
        """
        Fix a sparse pattern for U and V, then solve for W as a linear system.
        """
        if time.time() > deadline:
            return None
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Create a sparse U and V with rank r
            # Each row of U has a few nonzeros, similarly for V
            U = np.zeros((nm, r))
            V = np.zeros((mp, r))
            
            # Assign nonzeros randomly
            np.random.seed(100 + attempt)
            for a in range(nm):
                # Each row of U gets 2-3 nonzeros
                num_nonzero = min(r, np.random.randint(2, 4))
                cols = np.random.choice(r, size=num_nonzero, replace=False)
                for c in cols:
                    U[a, c] = np.random.choice([-1, 1])
            
            for b in range(mp):
                num_nonzero = min(r, np.random.randint(2, 4))
                cols = np.random.choice(r, size=num_nonzero, replace=False)
                for c in cols:
                    V[b, c] = np.random.choice([-1, 1])
            
            # Now solve for W
            # For each c, we have a linear system:
            # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all (a,b,c)
            
            # Group equations by c
            # For fixed c, the equations are over (a,b)
            # Let's build the system for all c simultaneously
            
            # Actually, let's solve for W[c,:] one at a time
            # For fixed c, the equations are:
            # For each (a,b) with (a,b,c) in tensor: U[a,:] dot V[b,:] * W[c,:] = 1
            # For each (a,b) with (a,b,c) not in tensor: U[a,:] dot V[b,:] * W[c,:] = 0
            
            # This is: M_c * W[c,:] = t_c
            # where M_c is a matrix with rows indexed by (a,b) and columns by r
            # M_c[(a,b), r] = U[a,r] * V[b,r]
            # t_c[(a,b)] = 1 if (a,b,c) in nz_set else 0
            
            W = np.zeros((pn, r))
            consistent = True
            
            for c in range(pn):
                # Build M_c and t_c
                # Number of equations: nm * mp = 600
                # Number of unknowns: r
                M_c = np.zeros((nm * mp, r))
                t_c = np.zeros(nm * mp)
                
                idx = 0
                for a in range(nm):
                    for b in range(mp):
                        M_c[idx, :] = U[a, :] * V[b, :]
                        if (a, b, c) in nz_set:
                            t_c[idx] = 1.0
                        else:
                            t_c[idx] = 0.0
                        idx += 1
                
                # Solve M_c * W[c,:] = t_c
                # This is overdetermined (600 equations, r unknowns)
                # Use least squares
                W[c, :], residuals, rank, sv = np.linalg.lstsq(M_c, t_c, rcond=None)
                
                # Check if the solution is consistent
                if len(residuals) > 0:
                    if residuals[0] > 1e-6:
                        consistent = False
                        break
            
            if not consistent:
                continue
            
            # Verify the full decomposition
            valid, err = verify(U, V, W, tol=1e-5)
            if valid:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None
    
    # Try ranks from a lower bound upward
    # Lower bound: rank >= max(n*m, m*p, p*n) is not correct
    # But rank >= n + m + p - 2 is a known lower bound for some cases
    # For 4x5x6, let's try ranks from 20 upward
    
    best_result = None
    best_rank = n * m * p  # 120
    
    # Strategy: Use Fano plane combinatorial design to define sparse support
    # The Fano plane has 7 points and 7 lines with the property that any two
    # points determine exactly one line, and any two lines meet in exactly one point.
    # This structure allows sharing of intermediate products across non-adjacent
    # tensor entries, potentially reducing rank below schoolbook.
    
    def try_fano_design(r, max_attempts=10):
        """
        Use Fano plane incidence to define sparse patterns for U and V,
        then solve for W via linear systems.
        """
        if time.time() > deadline:
            return None
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            np.random.seed(200 + attempt)
            
            U = np.zeros((nm, r))
            V = np.zeros((mp, r))
            
            # Map rows of U to Fano points via modular arithmetic
            # nm = 20 rows, 7 Fano points
            # Use a structured mapping: row a maps to Fano point (a % 7)
            # Each row gets nonzeros on Fano lines incident to its point
            # Plus some random columns for flexibility
            
            for a in range(nm):
                fano_pt = a % 7
                # Lines incident to fano_pt
                incident_lines = [l for l in range(7) if FANO_INC[fano_pt, l] == 1]
                
                # Assign nonzeros: use Fano structure + random
                num_fano = min(len(incident_lines), r // 3)
                num_random = min(r - num_fano, max(0, r // 4))
                
                cols = set()
                for l in incident_lines[:num_fano]:
                    col = (l * 3 + attempt) % r
                    cols.add(col)
                
                for _ in range(num_random):
                    col = np.random.randint(0, r)
                    cols.add(col)
                
                for c in cols:
                    U[a, c] = np.random.choice([-1, 1])
            
            for b in range(mp):
                fano_pt = b % 7
                incident_lines = [l for l in range(7) if FANO_INC[fano_pt, l] == 1]
                
                num_fano = min(len(incident_lines), r // 3)
                num_random = min(r - num_fano, max(0, r // 4))
                
                cols = set()
                for l in incident_lines[:num_fano]:
                    col = (l * 3 + 1 + attempt) % r
                    cols.add(col)
                
                for _ in range(num_random):
                    col = np.random.randint(0, r)
                    cols.add(col)
                
                for c in cols:
                    V[b, c] = np.random.choice([-1, 1])
            
            # Solve for W
            W = np.zeros((pn, r))
            consistent = True
            
            for c in range(pn):
                M_c = np.zeros((nm * mp, r))
                t_c = np.zeros(nm * mp)
                
                idx = 0
                for a in range(nm):
                    for b in range(mp):
                        M_c[idx, :] = U[a, :] * V[b, :]
                        if (a, b, c) in nz_set:
                            t_c[idx] = 1.0
                        idx += 1
                
                W[c, :], residuals, rank_c, sv = np.linalg.lstsq(M_c, t_c, rcond=None)
                
                if len(residuals) > 0:
                    if residuals[0] > 1e-6:
                        consistent = False
                        break
            
            if not consistent:
                continue
            
            valid, err = verify(U, V, W, tol=1e-5)
            if valid:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None
    
    # Try Fano design approach first for various ranks
    for r in range(20, 51):
        if time.time() > deadline - 20:
            break
        
        result = try_fano_design(r, max_attempts=5)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    # Then try the sparse fix approach for various ranks
    if best_result is None:
        for r in range(20, 61):
            if time.time() > deadline - 15:
                break
            
            result = try_sparse_fix(r, max_attempts=3)
            if result is not None:
                best_result = result
                best_rank = r
                break
    
    # If not found, try ALS for various ranks
    if best_result is None:
        for r in range(30, 81):
            if time.time() > deadline - 10:
                break
            
            result = try_structured(r, max_iters=50)
            if result is not None:
                best_result = result
                best_rank = r
                break
    
    if best_result is None:
        best_result = fallback
    
    return best_result
