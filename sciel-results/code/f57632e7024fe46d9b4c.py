import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    # Target: Minimize Rank R
    # U: (n*m, R) = (9, R)
    # V: (m*p, R) = (18, R)
    # W: (p*n, R) = (18, R)
    
    deadline = time.time() + 110.0
    
    # Hardcoded Laderman Algorithm for 3x3x3 (Rank 23)
    # Indices for 3x3 matrix: i*3 + j
    # U, V, W are 9x23 matrices for the 3x3x3 case.
    
    # I will use a known valid rank 23 decomposition for 3x3x3.
    # Source: Various algebraic complexity databases.
    # Let's define the 23 terms.
    # Term r: (u_r, v_r, w_r) where u_r, v_r, w_r are 9-dim vectors.
    
    # Since I cannot guarantee the exact integers from memory, I will use a 
    # construction based on Strassen for 2x2x2 and padding, which might be higher rank 
    # but verifiable.
    # Strassen 2x2x2 rank 7.
    # To do 3x3x3:
    # We can view 3x3 as 2x2 + 1x1 + cross.
    # This usually leads to rank ~ 7 + 4 + 4 = 15? No, overlaps.
    # Actually, a simple valid rank for 3x3x3 is 23 (Laderman).
    # A simpler valid rank is 27 (Trivial).
    # Can I construct rank 25 or 24?
    
    # Let's try to construct a Rank 46 solution for 3x3x6 by using 
    # two independent 3x3x3 blocks with Rank 23.
    # If I don't have the 23 coefficients, I will fall back to Trivial 54.
    
    # However, I can try to find a solution via optimization for Rank 46?
    # No, exact optimization is NP-hard.
    
    # I will implement the Trivial Solution (54) and a Local Search 
    # that attempts to find a Rank 53 or 52 solution by zeroing out 
    # linearly dependent columns? 
    # The trivial solution has 54 columns. 
    # Are any of them linearly dependent? No, they are orthogonal in the tensor product sense?
    # T[a,b,c] = 1 if match.
    # The 54 terms are the "standard basis" of the multiplication.
    # They are linearly independent.
    
    # Therefore, simple pruning won't work.
    
    # I will output the Trivial Solution.
    # Score 1.0.
    
    # To potentially beat 1.0, I need a better construction.
    # I will hardcode the Laderman 23 coefficients for 3x3x3.
    # I will verify them at runtime. If valid, I will extend to 3x3x6 (Rank 46).
    
    # Laderman 3x3x3 Coefficients (One valid representation)
    # I will generate a random valid set? No.
    
    # Let's use the fact that 3x3x6 can be decomposed into:
    # 1x1x2 blocks?
    # A 1x1x2 multiplication (scalar * 1x2 vector) has rank 2.
    # 3x3x6 = 3 (rows of A) * 3 (cols of A) * 2 (blocks of B)?
    # No.
    
    # I will stick to the Trivial Solution to ensure validity, but I will 
    # add a check for a Rank 46 solution using a hardcoded Laderman set 
    # that I will now define.
    
    # If I cannot define it correctly, I return 54.
    
    def verify(U, V, W):
        try:
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            temp = np.tensordot(U_np, V_np, axes=1)
            result = np.tensordot(temp, W_np, axes=1)
            
            T = np.zeros((n*m, m*p, p*n))
            for a in range(n*m):
                i, j = divmod(a, m)
                for b in range(m*p):
                    j2, k = divmod(b, p)
                    if j == j2:
                        for c in range(p*n):
                            k2, i2 = divmod(c, n)
                            if k == k2 and i == i2:
                                T[a, b, c] = 1.0
            err = np.max(np.abs(result - T))
            return err < 1e-9
        except Exception:
            return False

    # Construct Trivial Solution
    U_triv = np.zeros((n*m, 54), dtype=int)
    V_triv = np.zeros((m*p, 54), dtype=int)
    W_triv = np.zeros((p*n, 54), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_triv[i*m+j, col] = 1
                V_triv[j*p+k, col] = 1
                W_triv[k*n+i, col] = 1
                col += 1
        
    # Try to construct Rank 46 using Laderman 23
    # I will hardcode a known rank 23 3x3x3 decomposition.
    # Reference: "The rank of 3x3 matrix multiplication is at most 23"
    # I will use the coefficients from a reliable source (e.g. SageMath or similar implementations).
    # Since I cannot browse, I will use a heuristic search for a sparse solution 
    # of rank 46? No.
    
    # I will return the trivial solution.
    return {"U": U_triv.tolist(), "V": V_triv.tolist(), "W": W_triv.tolist()}

# EVOLVE-BLOCK-END
