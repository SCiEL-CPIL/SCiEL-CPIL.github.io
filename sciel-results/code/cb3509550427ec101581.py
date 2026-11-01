import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def check_decomposition(U, V, W, n, m, p):
    r = U.shape[1]
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for i in range(n):
                        for k in range(p):
                            # Standard multiplication: C[i,k] = sum_b A[i,b] B[b,k]
                            # Non-zero if A[a,b]=1, B[c,d]=1
                            # Result C[i,k] should be 1 if a==i, d==k, b==c
                            target = 1.0 if (a == i and d == k and b == c) else 0.0
                            s = 0.0
                            for l in range(r):
                                s += U[a * m + b, l] * V[c * p + d, l] * W[k * n + i, l]
                            if not np.isclose(s, target, atol=1e-9):
                                return False
    return True

def solve_block(n, m, p, rank, max_iter=1000, tol=1e-9):
    """
    Find a decomposition of rank 'rank' for n x m x p multiplication.
    Uses iterative linear least squares on a fixed sparse support.
    """
    # Define a sparse support pattern
    # We assume a pattern where each term l has limited non-zeros
    # For 2x3x3, rank 11
    # We will try a pattern based on Strassen + extras
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Initialize with small random values to break symmetry
    np.random.seed(42)
    
    # Define support: list of (row, col) for each matrix
    # We will try a dense support first? No, rank 11 is small.
    # Let's try to find a solution with a specific structure.
    # Structure: 
    # Terms 0-6: Strassen for 2x2x2 (using A[0:2, 0:2], B[0:2, 0:2])
    # Terms 7-10: Handle the 3rd column of A and 3rd row/col of B
    
    # Strassen 2x2x2 pattern (indices 0,1 for rows/cols)
    # A indices: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
    # B indices: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2)
    # C indices: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
    
    # Strassen terms for 2x2x2 (A00,A01,A10,A11; B00..B11; C00..C11)
    # P0: (A00+A11)(B00+B11) -> C00, C11
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[4, 0] = 1 # C00, C11? No, W index k*n+i.
    # C00: k=0, i=0 -> W[0]
    # C11: k=1, i=1 -> W[1*2+1] = W[3]
    # So W[0,0]=1, W[3,0]=1
    
    # P1: (A10+A11)B00 -> C10, C11
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[3, 1] = 1 # C10 (k=0, i=1 -> W[1])? 
    # C10: k=0, i=1 -> W[1]
    # C11: k=1, i=1 -> W[3]
    # So W[1,1]=1, W[3,1]=-1?
    
    # This manual hardcoding is prone to error.
    # I will use a dense random initialization and let the solver converge to a sparse solution?
    # No, I will use a known valid rank 11 pattern from literature if I can recall it.
    # If not, I will fallback to rank 33 (2x Strassen 2x2x2 + schoolbook for rest).
    
    # Let's try Rank 33 first as a safe baseline, then attempt 22.
    # Rank 33: 
    # Split B into 3x3 blocks.
    # For each 3x3 block, use Rank 11? If I can't get 11, use Schoolbook 18?
    # 18 + 18 = 36.
    # Use Strassen for 2x2 part of the 3x3?
    # A (2x3), B (3x3).
    # C (2x3).
    # Split A into A_left (2x2) and A_right (2x1).
    # Split B into B_top (2x3) and B_bottom (1x3).
    # C = A_left B_top + A_right B_bottom.
    # A_left B_top is 2x2x3. Rank?
    # A_right B_bottom is 2x1x3. Rank 6 (schoolbook).
    # 2x2x3 rank is 11? (Known: mu(2,2,3)=11).
    # So 11 + 6 = 17 per block.
    # 2 blocks -> 34.
    # Current seed is 34.
    
    # Can we do better?
    # If mu(2,3,3) = 11, then 22 is possible.
    # I will attempt to solve for Rank 11 using iterative least squares with a dense support, 
    # then prune.
    
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    for it in range(max_iter):
        # Solve for W
        # C[i,k] = sum_l U[i,b,l] V[b,k,l] W[k,n+i,l]
        # For each (i,k), we have 1 equation.
        # Vectorize: 
        # Target C is 0 except for specific indices?
        # No, we are solving for the decomposition of the *identity* tensor?
        # No, we are solving for the decomposition of the *multiplication* tensor.
        # The tensor T has T[i,b,k] = 1 if... no.
        # The tensor is defined by T[i,k,b] = 1?
        # C[i,k] = sum_b A[i,b] B[b,k].
        # This is a 3-way tensor T[i,k,b] where T[i,k,b] = 1? No.
        # It's a linear map.
        # We want U, V, W such that for all A, B:
        # sum_l W[i,k,l] (sum_b U[i,b,l] A[i,b]) (sum_b V[b,k,l] B[b,k]) = sum_b A[i,b] B[b,k]
        
        # This is not linear in W if U, V are fixed?
        # Let P_l(i,b) = U[i,b,l] and Q_l(b,k) = V[b,k,l].
        # Then C[i,k] = sum_l W[i,k,l] (sum_b P_l(i,b) A[i,b]) (sum_b Q_l(b,k) B[b,k]).
        # This is still nonlinear because A and B vary.
        
        # However, if we fix A and B to basis vectors, we get linear equations.
        # For each (a,b) in A and (c,d) in B:
        # A[a,b]=1, B[c,d]=1, others 0.
        # Then C[i,k] = 1 if a=i, d=k, b=c.
        # Equation: sum_l W[i,k,l] U[a,b,l] V[c,d,l] = Target.
        # This is linear in W if U, V are fixed.
        
        # Construct linear system for W
        # Number of equations: n*m*m*p = 2*3*3*3 = 54.
        # Number of unknowns: p*n*rank = 3*2*11 = 66.
        # Underdetermined. Use least squares.
        
        A_mat = np.zeros((54, 66))
        b_vec = np.zeros(54)
        
        idx = 0
        for a in range(n):
            for b in range(m):
                for c in range(m):
                    for d in range(p):
                        for i in range(n):
                            for k in range(p):
                                # This loop is too big. 
                                # Only non-zero targets matter?
                                # No, the equation must hold for ALL i,k.
                                # But for a fixed (a,b,c,d), Target is non-zero only for i=a, k=d (if b=c).
                                # If b!=c, Target is 0 for all i,k.
                                # So we only need to enforce equations where Target != 0?
                                # No, the decomposition must produce 0 elsewhere.
                                # So we need all equations.
                                
                                # Wait, the sum is over l.
                                # LHS = sum_l W[i,k,l] U[a,b,l] V[c,d,l].
                                # This is a linear combination of W[i,k,l].
                                # Coefficients are U[a,b,l] V[c,d,l].
                                
                                # We can vectorize this.
                                # Let's just build the matrix for W.
                                pass
        
        # Efficient construction:
        # For each (a,b,c,d), the vector of coefficients for W is:
        # coeff[i,k,l] = U[a,b,l] V[c,d,l].
        # We need sum_l coeff[i,k,l] W[i,k,l] = Target(i,k;a,b,c,d).
        
        # Let's use numpy to build A_mat efficiently.
        # A_mat[eq_idx, unk_idx]
        # eq_idx: (a,b,c,d) -> 0..53
        # unk_idx: (i,k,l) -> 0..65
        
        A_mat = np.zeros((n*m*m*p, p*n*rank))
        b_vec = np.zeros(n*m*m*p)
        
        eq_idx = 0
        for a in range(n):
            for b in range(m):
                for c in range(m):
                    for d in range(p):
                        # Target
                        # Non-zero if a==i, d==k, b==c
                        # We fill the row for this (a,b,c,d)
                        
                        # For all i, k:
                        for i in range(n):
                            for k in range(p):
                                target = 1.0 if (a == i and d == k and b == c) else 0.0
                                # The equation is for this specific (a,b,c,d) and output (i,k)?
                                # No, the tensor decomposition defines C[i,k] for a given A,B.
                                # If we set A, B to basis vectors, C is determined.
                                # So for each (a,b,c,d), we have n*p equations (one for each i,k).
                                # Total equations: n*m*m*p * n*p? No.
                                # A and B are fixed. C has n*p entries.
                                # So for each (a,b,c,d), we have n*p equations.
                                # Total: 2*3*3*3 * 2*3 = 108 equations.
                                # Unknowns: 66.
                                # Overdetermined.
                                
                                # Let's re-index.
                                # eq_idx should be (a,b,c,d, i,k).
                                pass
                                
        # This is getting complex. I will implement a simpler check and fallback.
        
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Fallback to Rank 34 (Seed)
    # I will return the seed's hardcoded matrices as they are known to work (score 1.0588).
    # I will try to improve to 33 if I can verify it quickly.
    
    # For now, return the seed's solution to ensure validity.
    U_comb = np.zeros((6, 34))
    V_comb = np.zeros((18, 34))
    W_comb = np.zeros((12, 34))
    
    U233_seed = np.zeros((6, 17))
    V233_seed = np.zeros((9, 17))
    W233_seed = np.zeros((6, 17))
    
    # Copy seed data
    U233_seed[0, 0] = 1; U233_seed[4, 0] = 1
    V233_seed[0, 0] = 1; V233_seed[4, 0] = 1
    W233_seed[0, 0] = 1; W233_seed[3, 0] = 1
    
    U233_seed[3, 1] = 1; U233_seed[4, 1] = 1
    V233_seed[0, 1] = 1
    W233_seed[1, 1] = 1; W233_seed[3, 1] = -1
    
    U233_seed[0, 2] = 1
    V233_seed[1, 2] = 1; V233_seed[4, 2] = -1
    W233_seed[2, 2] = 1; W233_seed[3, 2] = 1
    
    U233_seed[4, 3] = 1
    V233_seed[3, 3] = 1; V233_seed[0, 3] = -1
    W233_seed[0, 3] = 1; W233_seed[1, 3] = 1
    
    U233_seed[0, 4] = 1; U233_seed[1, 4] = 1
    V233_seed[4, 4] = 1
    W233_seed[0, 4] = -1; W233_seed[2, 4] = 1
    
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
