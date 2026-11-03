import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n, m, p = 3, 4, 8
    
    # Target tensor T[i, j, k] = 1 for all i, j, k
    # We want T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
    # Note: The problem statement says W[k*n+i, r].
    # So W is indexed by (k, i).
    
    def get_schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n, m, rank))
        V = np.zeros((m, p, rank))
        W = np.zeros((p, n, rank))
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i, j, idx] = 1
                    V[j, k, idx] = 1
                    W[k, i, idx] = 1
                    idx += 1
        return U, V, W

    def verify_decomp(U, V, W, n, m, p, tol=1e-9):
        # U: (n, m, r)
        # V: (m, p, r)
        # W: (p, n, r)
        
        # Check exactness using random matrices
        rng = np.random.default_rng(42)
        for _ in range(3):
            A = rng.standard_normal((n, m))
            B = rng.standard_normal((m, p))
            C = A @ B
            
            # C[i, k] = sum_j A[i, j] B[j, k]
            # Decomp: C[i, k] = sum_r W[k, i, r] * ( sum_j U[i, j, r] A[i, j] ) * ( sum_j V[j, k, r] B[j, k] )
            # Wait, the decomposition is:
            # C[i, k] = sum_r U[i, j, r] A[i, j] V[j, k, r] W[k, i, r] ?? No.
            
            # The standard tensor decomposition for matrix multiplication is:
            # C[i, k] = sum_r M_r
            # M_r = (sum_j U[i, j, r] A[i, j]) * (sum_j V[j, k, r] B[j, k])
            # Then C[i, k] = sum_r W[k, i, r] M_r ?
            
            # Let's check the definition in the prompt:
            # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
            # "W[k*n+i,r]"
            
            # The tensor T for matmul is T[i, j, k] = 1.
            # We want to write T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
            # If we plug this into C = T(A, B):
            # C[i, k] = sum_{i, j, k} T[i, j, k] A[i, j] B[j, k]
            # Wait, T contracts with A and B.
            # C[i, k] = sum_j A[i, j] B[j, k]
            
            # If T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
            # Then C[i, k] = sum_{i', j', k'} (sum_r U[i', j', r] V[j', k', r] W[k', i', r]) A[i', j'] B[j', k']
            # This doesn't look right. The indices must match.
            
            # Let's look at the schoolbook case:
            # U[i, j, r] is non-zero only if r corresponds to (i, j, k).
            # Actually, the standard decomposition is:
            # C[i, k] = sum_r (sum_j U[i, j, r] A[i, j]) (sum_j V[j, k, r] B[j, k]) W[k, i, r] ?
            # No, usually it is:
            # C = sum_r W_r * (U_r^T A) * (V_r B) ? No, dimensions don't match.
            
            # Let's re-read the prompt carefully.
            # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
            # "W[k*n+i,r]"
            
            # The tensor T is defined such that C = T(A, B).
            # T is a 3-tensor of size n x m x p? Or n x p x m?
            # C[i, k] = sum_j T[i, j, k] A[i, j] B[j, k]
            # So T[i, j, k] = 1.
            
            # The decomposition is T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
            # Note the indices:
            # U is indexed by i, j.
            # V is indexed by j, k.
            # W is indexed by k, i.
            
            # Let's verify with schoolbook:
            # r corresponds to (i0, j0, k0).
            # U[i, j, r] = 1 if i==i0 and j==j0.
            # V[j, k, r] = 1 if j==j0 and k==k0.
            # W[k, i, r] = 1 if k==k0 and i==i0.
            
            # Sum_r U[i, j, r] V[j, k, r] W[k, i, r]
            # Non-zero term only when r=(i, j, k).
            # Then U[i, j, (i, j, k)] = 1, V[j, k, (i, j, k)] = 1, W[k, i, (i, j, k)] = 1.
            # Product is 1.
            # So T[i, j, k] = 1. Correct.
            
            # Now, how do we compute C from U, V, W?
            # C[i, k] = sum_{i', j', k'} T[i', j', k'] A[i', j'] B[j', k'] ? No.
            # C[i, k] = sum_j A[i, j] B[j, k].
            # This is a specific contraction.
            
            # Let's check the decomposition of the contraction:
            # C[i, k] = sum_j A[i, j] B[j, k]
            # Substitute T:
            # We want to express the operation "multiply A and B" as a sum of rank-1 operations.
            # A rank-1 operation in this context is:
            # C[i, k] += W[k, i, r] * (sum_j U[i, j, r] A[i, j]) * (sum_j V[j, k, r] B[j, k])
            
            # Let's verify this formula with schoolbook:
            # r = (i0, j0, k0).
            # Term = W[k0, i0, r] * (sum_j U[i, j, r] A[i, j]) * (sum_j V[j, k, r] B[j, k])
            # U[i, j, r] is 1 only if i=i0, j=j0. So sum is A[i0, j0].
            # V[j, k, r] is 1 only if j=j0, k=k0. So sum is B[j0, k0].
            # W[k0, i0, r] is 1.
            # Term = A[i0, j0] B[j0, k0].
            
            # Sum over all r=(i0, j0, k0):
            # Sum_{i0, j0, k0} A[i0, j0] B[j0, k0] * (1 if i=i0, k=k0)
            # = A[i, j0] B[j0, k] summed over j0.
            # = sum_{j0} A[i, j0] B[j0, k] = C[i, k].
            
            # Yes! The formula is:
            # C[i, k] = sum_r W[k, i, r] * (U[:, :, r] @ A[:, :]) ... wait.
            # U is (n, m). A is (n, m).
            # sum_j U[i, j, r] A[i, j] is an inner product of row i of U_r and row i of A.
            # Let's call this u_r(i).
            # sum_j V[j, k, r] B[j, k] is an inner product of col k of V_r and col k of B.
            # Let's call this v_r(k).
            
            # So C[i, k] = sum_r W[k, i, r] u_r(i) v_r(k).
            
            # This allows us to check the decomposition efficiently.
            
            # Compute u_r(i) for all i, r
            # U is (n, m, r). A is (n, m).
            # u = np.einsum('ijr,ij->ir', U, A)
            
            # Compute v_r(k) for all k, r
            # V is (m, p, r). B is (m, p).
            # v = np.einsum('jkr,jk->kr', V, B)
            
            # Compute C_dec
            # C_dec[i, k] = sum_r W[k, i, r] u_r(i) v_r(k)
            # W is (p, n, r).
            # C_dec = np.einsum('kir,ir,kr->ik', W, u, v)
            
            u = np.einsum('ijr,ij->ir', U, A)
            v = np.einsum('jkr,jk->kr', V, B)
            C_dec = np.einsum('kir,ir,kr->ik', W, u, v)
            
            if np.max(np.abs(C - C_dec)) > tol:
                return False
        return True

    def try_rank(r, max_iters=500):
        # Randomized search for rank r decomposition
        # We fix U and V to be random (or structured), and solve for W.
        # Or fix U, solve for V, solve for W?
        
        # A better approach:
        # We want T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
        # This is a multilinear equation.
        
        # Let's use the "rank revealing" idea.
        # Fix U and V to random matrices.
        # Then for each (i, j, k), we have:
        # 1 = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
        
        # This is linear in W if U and V are fixed.
        # Let M_r(i, j, k) = U[i, j, r] V[j, k, r].
        # We want sum_r M_r(i, j, k) W[k, i, r] = 1.
        
        # This is a linear system for W.
        # Variables: W[k, i, r] for all k, i, r.
        # Number of variables: p * n * r.
        # Number of equations: n * m * p (one for each i, j, k).
        
        # If p*n*r >= n*m*p, we might have a solution.
        # r >= m.
        
        # Let's set up the linear system.
        # A_flat * W_flat = b_flat
        # b_flat is all 1s.
        
        # Construct the matrix A_flat.
        # Row index corresponds to (i, j, k).
        # Col index corresponds to (k, i, r).
        
        # This matrix is sparse.
        # Entry A[(i, j, k), (k', i', r')] is non-zero only if k==k' and i==i'.
        # Then A = U[i, j, r'] V[j, k, r'].
        
        # So for a fixed (i, j, k), the non-zero entries are at columns (k, i, r) for all r.
        # The value is U[i, j, r] V[j, k, r].
        
        # We can solve this system.
        # However, the system might be underdetermined or ill-conditioned.
        # We can use least squares.
        
        # To improve chances, we can fix some structure in U and V.
        # For example, make U and V have sparse entries or be orthogonal.
        
        # Let's try random U and V first.
        
        rng = np.random.default_rng(123)
        
        # Initialize U and V
        # We can scale them to avoid numerical issues.
        U = rng.standard_normal((n, m, r))
        V = rng.standard_normal((m, p, r))
        
        # Construct the linear system
        # Num equations: n * m * p
        # Num unknowns: p * n * r
        
        num_eqs = n * m * p
        num_unk = p * n * r
        
        A_mat = np.zeros((num_eqs, num_unk))
        b_vec = np.ones(num_eqs)
        
        # Fill A_mat
        # Index mapping:
        # eq_idx = (i * m + j) * p + k
        # unk_idx = (k * n + i) * r + r_idx
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = (i * m + j) * p + k
                    # Non-zero columns for this eq are (k, i, r_idx)
                    # unk_idx = (k * n + i) * r + r_idx
                    base_unk = (k * n + i) * r
                    for r_idx in range(r):
                        val = U[i, j, r_idx] * V[j, k, r_idx]
                        A_mat[eq_idx, base_unk + r_idx] = val
        
        # Solve least squares
        try:
            W_flat, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        except:
            return None
            
        # Reshape W_flat to W (p, n, r)
        W = W_flat.reshape(p, n, r)
        
        # Check if the solution is exact (within tolerance)
        # And if it can be cleaned to integers/half-integers
        
        # Verify the decomposition
        if verify_decomp(U, V, W, n, m, p, tol=1e-8):
            # Try to clean
            # The entries of U, V, W might be close to integers or half-integers
            # due to the algebraic structure.
            
            # Let's try rounding to nearest integer
            U_clean = np.round(U)
            V_clean = np.round(V)
            W_clean = np.round(W)
            
            if verify_decomp(U_clean, V_clean, W_clean, n, m, p, tol=1e-9):
                return U_clean, V_clean, W_clean
                
            # Try rounding to nearest half-integer
            U_half = np.round(U * 2) / 2
            V_half = np.round(V * 2) / 2
            W_half = np.round(W * 2) / 2
            
            if verify_decomp(U_half, V_half, W_half, n, m, p, tol=1e-9):
                return U_half, V_half, W_half
                
            # If not clean, we might still accept it if it's "close enough" for the evaluator?
            # The prompt says "integer/half-integer entries".
            # If we can't clean, this rank r might not work with random U, V.
            
        return None

    # Strategy:
    # 1. Try ranks from a lower bound up to 96.
    # 2. Lower bound for 3x4x8?
    #    Schoolbook is 96.
    #    Strassen 2x2x2 is 7.
    #    3x4x8 can be split into 1x4x8 (rank 32) and 2x4x8.
    #    2x4x8 can be split into 2x2x8 (rank 28) and 2x2x8 (rank 28) -> 56.
    #    Total 88.
    #    So 88 is a known valid rank.
    #    Can we do better?
    #    Maybe 80? 72?
    
    # Let's try ranks from 80 to 96.
    # We have 145 seconds.
    # Each rank attempt involves solving a linear system of size (96, 3*r).
    # For r=80, num_unk = 240. 96x240 system. Fast.
    # For r=96, num_unk = 288. Fast.
    
    # We can afford many iterations per rank.
    
    best_result = None
    best_rank = 96
    
    # Start with a known good rank to ensure we have a fallback
    U_sb, V_sb, W_sb = get_schoolbook(n, m, p)
    best_result = (U_sb, V_sb, W_sb)
    
    # Try lower ranks
    for r in range(80, 96):
        if time.time() > deadline - 10:
            break
            
        found = try_rank(r, max_iters=10)
        if found is not None:
            U, V, W = found
            best_result = (U, V, W)
            best_rank = r
            # If we find a lower rank, we can break or continue to find even lower?
            # The task asks for "rank-minimal". So we should try to go lower.
            # But 80 might be the minimum. Let's see if we can find 79, 78...
            # We can continue the loop downwards if we found one.
            # But the loop is upwards. Let's change the loop to downwards from 96?
            # No, we want the lowest rank. So we should try 80, 81...
            # If we find 80, we should try 79?
            # Let's restructure: try 80, if found, try 79, etc.
            # But we don't know the lower bound.
            
            # Let's just try a few random restarts for each rank.
            # If we find a valid one, we update best.
            # Then we try lower ranks.
            
    # Actually, let's try ranks in increasing order of "difficulty" or just increasing rank.
    # Since we want the MINIMAL rank, we should start from a low rank and go up?
    # No, if we start from 64 and fail, we try 65...
    # But if 88 is the minimum, we waste time on 64-87.
    # However, 64-87 are likely impossible.
    # Let's start from 80 and go up to 96.
    # If we find 80, we can stop? No, maybe 79 exists.
    # But 79 is very unlikely.
    # Let's just find the first valid rank in the range [80, 96] and return it.
    # Or find the lowest valid rank in a small window.
    
    # Let's refine:
    # Try r = 80. If found, return it.
    # Try r = 81. If found, return it.
    # ...
    # This finds the first valid rank. Since we go 80, 81..., it finds the lowest in this range.
    
    # What if 79 is valid? We miss it.
    # But 79 is extremely unlikely for 3x4x8.
    # The known lower bounds are usually multiples of something or related to Strassen.
    # 3x4x8 -> 3*4*8 = 96.
    # 2x4x8 -> 2*4*8 = 64. Strassen 2x2x2 helps.
    # 2x2x8 -> 2*2*8 = 32. Strassen 2x2x2 (7) x 4 blocks?
    # 2x2x8 can be 4 blocks of 2x2x2. Rank 4*7 = 28.
    # 2x4x8 can be 2 blocks of 2x2x8. Rank 2*28 = 56.
    # 3x4x8 can be 1x4x8 (32) + 2x4x8 (56) = 88.
    
    # So 88 is a strong candidate for the minimum or near-minimum.
    # Let's try 80, 81, ..., 88.
    
    for r in range(80, 89):
        if time.time() > deadline - 10:
            break
            
        # Try multiple random initializations for this rank
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            
            # Modify try_rank to accept a seed or rng
            found = try_rank(r, max_iters=1) # try_rank currently uses fixed seed 123.
            # We need to vary the seed.
            
            # Let's create a helper that takes a seed.
            
            # I'll just re-implement the core logic here with a variable seed.
            
            rng = np.random.default_rng(123 + attempt)
            U = rng.standard_normal((n, m, r))
            V = rng.standard_normal((m, p, r))
            
            num_eqs = n * m * p
            num_unk = p * n * r
            A_mat = np.zeros((num_eqs, num_unk))
            b_vec = np.ones(num_eqs)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = (i * m + j) * p + k
                        base_unk = (k * n + i) * r
                        for r_idx in range(r):
                            val = U[i, j, r_idx] * V[j, k, r_idx]
                            A_mat[eq_idx, base_unk + r_idx] = val
            
            try:
                W_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W = W_flat.reshape(p, n, r)
                
                if verify_decomp(U, V, W, n, m, p, tol=1e-8):
                    U_clean = np.round(U)
                    V_clean = np.round(V)
                    W_clean = np.round(W)
                    if verify_decomp(U_clean, V_clean, W_clean, n, m, p, tol=1e-9):
                        return {"U": U_clean.reshape(n*m, r).tolist(), 
                                "V": V_clean.reshape(m*p, r).tolist(), 
                                "W": W_clean.reshape(p*n, r).tolist()}
                    
                    U_half = np.round(U * 2) / 2
                    V_half = np.round(V * 2) / 2
                    W_half = np.round(W * 2) / 2
                    if verify_decomp(U_half, V_half, W_half, n, m, p, tol=1e-9):
                        return {"U": U_half.reshape(n*m, r).tolist(), 
                                "V": V_half.reshape(m*p, r).tolist(), 
                                "W": W_half.reshape(p*n, r).tolist()}
            except:
                pass
                
    # Fallback to schoolbook
    U_sb, V_sb, W_sb = get_schoolbook(n, m, p)
    return {"U": U_sb.reshape(n*m, n*m*p).tolist(), 
            "V": V_sb.reshape(m*p, n*m*p).tolist(), 
            "W": W_sb.reshape(p*n, n*m*p).tolist()}

# Run construct to ensure it works
if __name__ == "__main__":
    res = construct()
    rank = len(res['U'][0])
    print(f"Found rank: {rank}")
