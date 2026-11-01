import numpy as np
import time
import random
from fractions import Fraction
import itertools

DIMENSIONS = [2, 3, 6]

def verify_decomposition(U, V, W, n, m, p):
    """
    Verifies if U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    r = U.shape[1]
    # Convert to numpy for speed, but we assume U,V,W are lists of lists of Fractions or floats
    # For exactness in final check, we use Fraction if possible, else float
    U_arr = np.array(U, dtype=object)
    V_arr = np.array(V, dtype=object)
    W_arr = np.array(W, dtype=object)
    
    # Check dimensions
    if U_arr.shape != (n*m, r) or V_arr.shape != (m*p, r) or W_arr.shape != (p*n, r):
        return False

    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for i in range(n):
                        for k in range(p):
                            target = 1 if (a == i and c == k and b == d) else 0
                            # s = sum_{t=0}^{r-1} U[a*m+b, t] * V[c*p+d, t] * W[k*n+i, t]
                            s = Fraction(0, 1)
                            for t in range(r):
                                s += U_arr[a * m + b, t] * V_arr[c * p + d, t] * W_arr[k * n + i, t]
                            if s != target:
                                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    
    # The target tensor T has indices (a, b, c, d, i, k) -> 1 if a=i, c=k, b=d else 0.
    # We seek rank r decomposition.
    # U[a,b,t], V[c,d,t], W[k,i,t]
    # Constraint: Sum_t U[a,b,t] V[c,d,t] W[k,i,t] = delta_{a,i} delta_{c,k} delta_{b,d}
    
    # Strategy:
    # 1. Try to find a rank 11 decomposition for 2x3x3 (M_233).
    #    If found, we can potentially combine two copies for 2x3x6 (rank 22).
    #    However, the current best is 34 (2x17). 22 is a huge improvement.
    #    Rank 12 for 2x3x3 -> Rank 24 for 2x3x6.
    #    Rank 13 for 2x3x3 -> Rank 26 for 2x3x6.
    #    
    #    Actually, let's look at the structure of 2x3x6 directly.
    #    It is equivalent to multiplying two 2x3 matrices by a 3x6 matrix? No.
    #    M_{2,3,6} = M_{2,3,3} \otimes M_{1,1,2}? No.
    #    M_{n,m,p} rank is at most rank(M_{n,m,p1}) + rank(M_{n,m,p2}) if we split p.
    #    So rank(M_{2,3,6}) <= 2 * rank(M_{2,3,3}).
    #    Current seed uses rank 17 for M_{2,3,3} (which is schoolbook 18 minus 1? No, 17 is not standard schoolbook).
    #    Standard schoolbook for 2x3x3 is 18.
    #    The seed program found a rank 17 decomposition for 2x3x3? 
    #    Wait, the seed program's fallback is rank 34 for 2x3x6, which implies it used rank 17 for 2x3x3.
    #    Is rank 17 possible for 2x3x3?
    #    Lower bound for M_{2,3,3}: max(2*3, 3*3, 3*2) = 9.
    #    Strassen for 2x2x2 is 7.
    #    For 2x3x3, we can embed 2x2x2.
    #    Known results: rank(M_{2,3,3}) = 17? Or 18?
    #    If we can find rank 16 for 2x3x3, we get 32 for 2x3x6.
    #    If we can find rank 15 for 2x3x3, we get 30 for 2x3x6.
    
    # Let's try to solve for M_{2,3,3} with rank r using exact rational algebra.
    # Variables: U (6 x r), V (9 x r), W (6 x r).
    # Equations: For each (a,b,c,d,i,k), Sum_t U[a,b,t]V[c,d,t]W[k,i,t] = delta.
    # This is a system of bilinear equations.
    
    # We will use a "sparse ansatz" approach.
    # Assume U, V, W are sparse.
    # We will try to construct a solution for M_{2,3,3} with rank 16, 15, 14, 13, 12, 11.
    
    def solve_m233(rank_r):
        """
        Attempts to find a rank r decomposition for M_{2,3,3} using exact rational arithmetic.
        Returns (U, V, W) as lists of lists of Fractions, or None.
        """
        if time.time() > deadline:
            return None
            
        n2, m2, p2 = 2, 3, 3
        U_dim = n2 * m2  # 6
        V_dim = m2 * p2  # 9
        W_dim = p2 * n2  # 6
        
        # We fix a gauge to reduce symmetry.
        # We can assume W is in row echelon form or has a specific structure.
        # A common technique: Fix W to be the identity matrix in the first r rows if possible?
        # No, W is 6x r. If r > 6, we can't make it identity.
        
        # Let's try a structured search:
        # We assume the decomposition is built from "terms".
        # Each term t has a support in U, V, W.
        
        # Instead of general search, let's try to exploit the known structure.
        # M_{2,3,3} can be seen as M_{2,3,2} + M_{2,3,1}.
        # Rank(M_{2,3,2})?
        # M_{2,3,2} is 2x3 * 3x2.
        # Rank(M_{2,3,2}) <= Rank(M_{2,2,2}) + Rank(M_{1,1,2})? No.
        # Let's just try to solve the linear system for fixed U, V.
        
        # Strategy: 
        # 1. Randomly initialize U and V with sparse rational entries (mostly 0, some 1, -1, 1/2).
        # 2. The equations are linear in W given U and V.
        #    Sum_t W[k,i,t] * (U[a,b,t] * V[c,d,t]) = delta_{a,i} delta_{c,k} delta_{b,d}
        #    This is a linear system for W.
        #    Number of unknowns in W: 6 * r.
        #    Number of equations: 2*3*3*3*2*3 = 324? No.
        #    The tensor has size 2*3*3*3*2*3 = 324 entries.
        #    But many are 0.
        #    Non-zero targets: a=i, c=k, b=d.
        #    a in 0..1, i in 0..1 -> 2 pairs.
        #    c in 0..2, k in 0..2 -> 3 pairs.
        #    b in 0..2, d in 0..2 -> 3 pairs.
        #    Total non-zero targets: 2 * 3 * 3 = 18.
        #    So we have 18 equations.
        #    Unknowns: 6*r.
        #    If r=16, 96 unknowns. Underdetermined.
        #    We can solve for W using least squares (exact rational).
        #    Then we check if the solution W satisfies all 324 equations (the zeros must be 0).
        
        # This is a good strategy!
        # 1. Pick random sparse U, V.
        # 2. Solve linear system for W.
        # 3. Check if W satisfies the full tensor equation.
        
        # To make it work, we need the "zero" equations to also be satisfied.
        # This imposes constraints on U and V.
        
        # Let's define the linear system for W.
        # For each (a, c, b), we have target delta_{a,i} delta_{c,k} delta_{b,d} for all i, k, d.
        # Actually, for a fixed (a, c, b), the target is non-zero only if i=a, k=c, d=b.
        # So for each (a, c, b), we have 18 equations (for all i, k, d), but only 1 is non-zero.
        # Wait, the equation is for ALL (a,b,c,d,i,k).
        # Let's group by (a, c, b).
        # For a fixed (a, c, b), the LHS is:
        # Sum_t U[a,b,t] V[c,d,t] W[k,i,t]
        # This must equal 1 if (i=a, k=c, d=b) and 0 otherwise.
        
        # So for each (a, c, b), we have a system of 18 equations for the vector W_col = (W[0,0,t]...W[5,5,t])?
        # No, W is shared across all (a,b,c,d).
        # The linearity is in W.
        # Let's form the matrix A of size (18 * 18) x (6 * r)?
        # No, the equations are coupled.
        
        # Let's vectorize W into a vector w of length 6*r.
        # Each equation is linear in w.
        # There are 324 equations.
        # We can solve the linear system A w = b.
        # If a solution exists, we check if it's valid (it will be, by construction of A and b).
        # BUT, the system is underdetermined if 6*r > 324.
        # For r=16, 96 < 324. It's overdetermined!
        # So for a random U, V, the system A w = b will likely have NO solution.
        # We need to find U, V such that the system is consistent.
        
        # This is hard.
        
        # Alternative: Fix W to be a basis?
        # If we fix W to have rank 6 (identity in first 6 cols), then r >= 6.
        # If we fix W to be sparse, we reduce degrees of freedom.
        
        # Let's try a different approach:
        # We know M_{2,3,3} has rank 18 (schoolbook).
        # The seed found 17.
        # Can we find 16?
        
        # Let's try to construct M_{2,3,3} from M_{2,2,2} (Strassen, rank 7) and extra terms.
        # M_{2,3,3} = M_{2,2,2} + M_{1,1,1} + ...
        
        # Let's just use the seed's rank 17 solution for M_{2,3,3} and see if we can improve it to 16.
        # Or better, let's try to find a rank 16 solution for M_{2,3,3} using a specific ansatz.
        
        # Ansatz:
        # U, V, W are 0-1 matrices with some -1s.
        
        # Given the time constraint, let's just return the seed's solution (rank 34) if we can't find better quickly.
        # But we can try to verify if rank 33 is possible using the "Strassen 2x2x2" insight from the prompt.
        # The prompt says: "Using block decomposition with Strassen's algorithm for 2x2x2 sub-problems reduces the rank from 36 to 33".
        # This implies a rank 33 solution exists!
        # The current seed is rank 34.
        # If we can construct the rank 33 solution, we improve the score.
        
        # How to construct rank 33?
        # M_{2,3,6} can be split into M_{2,3,3} + M_{2,3,3}.
        # If we use rank 16 for each, we get 32.
        # If we use rank 17 for one and 16 for the other, we get 33.
        # Or maybe there's a direct rank 33 construction.
        
        # Let's try to find a rank 16 decomposition for M_{2,3,3}.
        # If we find it, we can use it twice to get rank 32.
        
        # Let's implement a quick search for rank 16 M_{2,3,3} using the linear system consistency check.
        
        def check_m233_solution(U, V, W):
            """U, V, W are numpy arrays of Fractions or floats. Returns True if valid."""
            n2, m2, p2 = 2, 3, 3
            r = U.shape[1]
            for a in range(n2):
                for b in range(m2):
                    for c in range(m2):
                        for d in range(p2):
                            for i in range(n2):
                                for k in range(p2):
                                    target = 1 if (a == i and c == k and b == d) else 0
                                    s = 0
                                    for t in range(r):
                                        s += U[a * m2 + b, t] * V[c * p2 + d, t] * W[k * n2 + i, t]
                                    if s != target:
                                        return False
            return True

        # We will try to find a rank 16 solution for M_{2,3,3}.
        # We will use a heuristic search with exact rational arithmetic.
        
        # We fix the structure of U and V to be sparse.
        # We will try to solve for W.
        
        # To speed up, we will use floating point for the search, and exact rational for verification.
        
        found = None
        for r in [16, 15, 14]:
            if time.time() > deadline - 10:
                break
            for restart in range(100):
                if time.time() > deadline - 5:
                    break
                
                # Random sparse init
                U = np.zeros((6, r))
                V = np.zeros((9, r))
                W = np.zeros((6, r))
                
                # Fill with random -1, 0, 1
                for t in range(r):
                    for i in range(6):
                        U[i, t] = random.choice([-1, 0, 0, 1])
                    for i in range(9):
                        V[i, t] = random.choice([-1, 0, 0, 1])
                    for i in range(6):
                        W[i, t] = random.choice([-1, 0, 0, 1])
                
                # Check if this random sparse solution works? Unlikely.
                # Instead, let's try to refine it.
                
                # We will use the fact that the equations are linear in W.
                # For a fixed U, V, we can solve for W.
                # Let's form the linear system.
                
                # Variables: W[0..5, 0..r-1]
                # Equations: for each (a,b,c,d,i,k), sum_t U[a,b,t]V[c,d,t]W[k,i,t] = target
                # This is a system of 324 equations in 6*r unknowns.
                # We can use numpy linalg.lstsq (float) to find a candidate W.
                # Then check if it's exact.
                
                # Build A matrix (324 x 6r)
                # Index for W: k*2 + i + 6*t? No, W is (6, r).
                # Let's flatten W to vector w of length 6*r.
                # w[idx] = W[i, t] where idx = t*6 + i.
                
                A = np.zeros((324, 6*r))
                b = np.zeros(324)
                
                eq_idx = 0
                for a in range(2):
                    for b_ in range(3):
                        for c in range(3):
                            for d in range(3):
                                for i in range(2):
                                    for k in range(3):
                                        target = 1 if (a == i and c == k and b_ == d) else 0
                                        b[eq_idx] = target
                                        for t in range(r):
                                            coeff = U[a*3+b_, t] * V[c*3+d, t]
                                            if coeff != 0:
                                                # W[k,i,t] is at index t*6 + (k*2+i)
                                                A[eq_idx, t*6 + k*2 + i] = coeff
                                        eq_idx += 1
                                        
                # Solve A w = b
                # Use least squares
                w, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                
                # Check residual
                if np.linalg.norm(A @ w - b) < 1e-9:
                    # Reshape w to W
                    W_sol = w.reshape((r, 6)).T # (6, r)
                    # Check if W_sol is close to integers or simple rationals
                    # For now, just check if it's a valid solution
                    if check_m233_solution(U, V, W_sol):
                        found = (U, V, W_sol, r)
                        break
                            
                # If not found, try next restart
            if found:
                break
                
        if found:
            U233, V233, W233, r233 = found
            # Construct M_{2,3,6} with rank 2*r233
            r_total = 2 * r233
            U_comb = np.zeros((6, r_total))
            V_comb = np.zeros((18, r_total))
            W_comb = np.zeros((12, r_total))
            
            U_comb[:, :r233] = U233
            U_comb[:, r233:] = U233
            
            for j in range(3):
                for k in range(3):
                    V_comb[j * 6 + k, :r233] = V233[j * 3 + k]
                    V_comb[j * 6 + 3 + k, r233:] = V233[j * 3 + k]
                    
            for k in range(3):
                for i in range(2):
                    W_comb[k * 2 + i, :r233] = W233[k * 2 + i]
                    W_comb[(k + 3) * 2 + i, r233:] = W233[k * 2 + i]
                    
            # Verify
            if verify_decomposition(U_comb.tolist(), V_comb.tolist(), W_comb.tolist(), n, m, p):
                return {
                    "U": U_comb.tolist(),
                    "V": V_comb.tolist(),
                    "W": W_comb.tolist()
                }
                
        # Fallback to seed's rank 34 solution
        U233_seed = np.zeros((6, 17))
        V233_seed = np.zeros((9, 17))
        W233_seed = np.zeros((6, 17))
        
        U233_seed[0, 0] = 1; U233_seed[4, 0] = 1
        V233_seed[0, 0] = 1; V233_seed[4, 0] = 1
        W233_seed[0, 0] = 1
        W233_seed[3, 0] = 1
        
        U233_seed[3, 1] = 1; U233_seed[4, 1] = 1
        V233_seed[0, 1] = 1
        W233_seed[1, 1] = 1
        W233_seed[3, 1] = -1
        
        U233_seed[0, 2] = 1
        V233_seed[1, 2] = 1; V233_seed[4, 2] = -1
        W233_seed[2, 2] = 1
        W233_seed[3, 2] = 1
        
        U233_seed[4, 3] = 1
        V233_seed[3, 3] = 1; V233_seed[0, 3] = -1
        W233_seed[0, 3] = 1
        W233_seed[1, 3] = 1
        
        U233_seed[0, 4] = 1; U233_seed[1, 4] = 1
        V233_seed[4, 4] = 1
        W233_seed[0, 4] = -1
        W233_seed[2, 4] = 1
        
        U233_seed[3, 5] = 1; U233_seed[0, 5] = -1
        V233_seed[0, 5] = 1; V233_seed[1, 5] = 1
        W233_seed[3, 5] = 1
        
        U233_seed[1, 6] = 1; U233_seed[4, 6] = -1
        V233_seed[3, 6] = 1; V233_seed[4, 6] = 1
        W233_seed[0, 6] = 1
        
        U233_seed[0, 7] = 1
        V233_seed[2, 7] = 1
        W233_seed[4, 7] = 1
        
        U233_seed[1, 8] = 1
        V233_seed[5, 8] = 1
        W233_seed[4, 8] = 1
        
        U233_seed[3, 9] = 1
        V233_seed[2, 9] = 1
        W233_seed[5, 9] = 1
        
        U233_seed[4, 10] = 1
        V233_seed[5, 10] = 1
        W233_seed[5, 10] = 1
        
        U233_seed[2, 11] = 1
        V233_seed[6, 11] = 1
        W233_seed[0, 11] = 1
        
        U233_seed[2, 12] = 1
        V233_seed[7, 12] = 1
        W233_seed[2, 12] = 1
        
        U233_seed[2, 13] = 1
        V233_seed[8, 13] = 1
        W233_seed[4, 13] = 1
        
        U233_seed[5, 14] = 1
        V233_seed[6, 14] = 1
        W233_seed[1, 14] = 1
        
        U233_seed[5, 15] = 1
        V233_seed[7, 15] = 1
        W233_seed[3, 15] = 1
        
        U233_seed[5, 16] = 1
        V233_seed[8, 16] = 1
        W233_seed[5, 16] = 1
        
        U_comb = np.zeros((6, 34))
        V_comb = np.zeros((18, 34))
        W_comb = np.zeros((12, 34))
        
        U_comb[:, :17] = U233_seed
        U_comb[:, 17:] = U233_seed
        
        for j in range(3):
            for k in range(3):
                V_comb[j * 6 + k, :17] = V233_seed[j * 3 + k]
                V_comb[j * 6 + 3 + k, 17:] = V233_seed[j * 3 + k]
                
        for k in range(3):
            for i in range(2):
                W_comb[k * 2 + i, :17] = W233_seed[k * 2 + i]
                W_comb[(k + 3) * 2 + i, 17:] = W233_seed[k * 2 + i]
                
        return {
            "U": U_comb.tolist(),
            "V": V_comb.tolist(),
            "W": W_comb.tolist()
        }

if __name__ == "__main__":
    construct()
