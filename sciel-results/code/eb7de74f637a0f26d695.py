import numpy as np
import time

n, m, p = 2, 5, 6
nm, mp, pn = n * m, m * p, p * n


def verify_decomposition(U_mat, V_mat, W_mat, tol=1e-10):
    """Verify that the decomposition correctly computes matrix multiplication.
    
    The matrix multiplication tensor M_{i,j,k} = 1 for all (i,j,k) in range.
    A rank-r decomposition satisfies:
        sum_u U[i*m+j, u] * V[j*p+k, u] * W[k*n+i, u] = 1 for all (i,j,k)
    """
    recon = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                recon[i, j, k] = np.sum(
                    U_mat[i * m + j, :] * V_mat[j * p + k, :] * W_mat[k * n + i, :]
                )
    original = np.ones((n, m, p), dtype=np.float64)
    return np.allclose(recon, original, atol=tol)


def strassen_2x2_decomposition():
    """Return the Strassen decomposition for 2x2 matrix multiplication.
    
    Strassen's algorithm multiplies two 2x2 matrices using 7 scalar multiplications
    instead of 8. This gives a rank-7 decomposition of the 2x2x2 matrix multiplication tensor.
    
    For A = [[a,b],[c,d]] and B = [[e,f],[g,h]], C = A*B:
    C[0,0] = a*e + b*g
    C[0,1] = a*f + b*h
    C[1,0] = c*e + d*g
    C[1,1] = c*f + d*h
    
    Strassen's 7 products:
    P1 = (a+d)(e+h)
    P2 = (c+d)e
    P3 = a(f-h)
    P4 = d(g-e)
    P5 = (a+b)h
    P6 = (c-a)(e+f)
    P7 = (b-d)(g+h)
    
    Then:
    C[0,0] = P1 - P4 + P5 - P7
    C[0,1] = P3 + P4
    C[1,0] = P2 + P5
    C[1,1] = P1 - P2 + P3 + P6
    """
    # Each term is (U_idx, V_idx, W_idx, coeff)
    # U indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # V indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # W indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    terms = []
    # P1 = (a+d)(e+h): U has 1 at (0,0) and (1,1), V has 1 at (0,0) and (0,1)
    # This contributes to all 4 output positions with different signs
    # We need to decompose this into rank-1 terms over the full tensor space
    
    # Actually, for a general n x m x p multiplication, we use Strassen on 2x2 blocks
    # For 2x5x6, we can partition:
    # A is 2x5: split into [A1 | A2] where A1 is 2x2, A2 is 2x3
    # B is 5x6: split into [B1; B2] where B1 is 2x6, B2 is 3x6
    # C = A1*B1 + A2*B2
    
    # This doesn't directly give us a Strassen decomposition for the whole thing.
    # Instead, let's use a block approach.
    
    # For now, return None to indicate we need a different approach
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    # Schoolbook decomposition is the baseline
    rank = n * m * p
    U = np.zeros((nm, rank), dtype=np.float64)
    V = np.zeros((mp, rank), dtype=np.float64)
    W = np.zeros((pn, rank), dtype=np.float64)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

    best_U = U
    best_V = V
    best_W = W
    best_rank = rank

    # Try to find a better decomposition using structured methods
    
    if time.time() < deadline - 5.0:
        # Strategy: Use the fact that for small matrices, we can sometimes
        # find decompositions with rank less than n*m*p by exploiting
        # algebraic structure.
        
        # For 2x5x6, one approach is to use a block decomposition.
        # Split A (2x5) into [A_left (2x2) | A_right (2x3)]
        # Split B (5x6) into [B_top (2x6); B_bottom (3x6)]
        # C = A_left @ B_top + A_right @ B_bottom
        
        # The schoolbook for this block structure still gives rank 60.
        # But we can try to use Strassen on the 2x2 block if we can
        # embed it properly.
        
        # Alternative: try to find a rank-54 decomposition by combining
        # the 2x2 Strassen (rank 7) with schoolbook for the remaining parts.
        
        # Let's try a direct approach: search for a low-rank decomposition
        # using an iterative method.
        
        # Method: Start with the schoolbook and try to reduce rank by
        # finding linear dependencies.
        
        # First, verify the schoolbook works
        if not verify_decomposition(U, V, W):
            # This shouldn't happen, but just in case
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Try to reduce rank by looking for structure
        # For 2x5x6, the minimum possible rank is at least max(2*5, 5*6, 6*2) = 30
        # and at most 60 (schoolbook).
        
        # Let's try a greedy approach: try to merge pairs of schoolbook terms
        # by finding if their sum can be represented as a single rank-1 term.
        
        # This is complex, so instead let's try a randomized search for
        # a low-rank decomposition.
        
        # Use a different strategy: try to find a decomposition with rank
        # between 30 and 60 using an iterative optimization approach.
        
        # For simplicity, let's try to find if rank 54 is achievable
        # by using a block Strassen approach.
        
        # Block approach for 2x5x6:
        # We can write the multiplication as:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # 
        # Split j into two groups: j=0,1 (left) and j=2,3,4 (right)
        # C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{4} A[i,j]*B[j,k]
        #
        # The first part is a 2x2x6 multiplication (rank 12 schoolbook)
        # The second part is a 2x3x6 multiplication (rank 18 schoolbook)
        # Total: 30
        #
        # But this is just the schoolbook again. We need to find a better
        # decomposition.
        
        # Let's try a different approach: use the known result that
        # for 2xn matrix multiplication, the rank is at most 3n-2 for n>=2.
        # For 2x5, this would give rank 13, but that's for square matrices.
        
        # Actually, for multiplying a 2xm matrix by an mxp matrix,
        # the rank is at most 2mp - something.
        
        # Let's just try to find a decomposition with rank < 60 using
        # a search approach.
        
        # Try rank 59, 58, ... down to some minimum
        for target_rank in range(59, max(30, 59 - 30), -1):
            if time.time() > deadline - 5.0:
                break
            
            # Try a few random attempts at this rank
            for attempt in range(10):
                if time.time() > deadline - 5.0:
                    break
                
                # Create random factor matrices with some structure
                # Start from the schoolbook and perturb
                U_rand = U[:, :target_rank].copy()
                V_rand = V[:, :target_rank].copy()
                W_rand = W[:, :target_rank].copy()
                
                # Add random perturbations
                U_rand += np.random.randn(nm, target_rank) * 0.1
                V_rand += np.random.randn(mp, target_rank) * 0.1
                W_rand += np.random.randn(pn, target_rank) * 0.1
                
                # Verify if this decomposition works
                if verify_decomposition(U_rand, V_rand, W_rand):
                    best_U = U_rand
                    best_V = V_rand
                    best_W = W_rand
                    best_rank = target_rank
                    print(f"Found rank {target_rank} decomposition!")
                    break
            
            if best_rank < rank:
                break

    # Return the best result found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
