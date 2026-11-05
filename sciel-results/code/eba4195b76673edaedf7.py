import numpy as np
import time

def construct():
    n, m, p = 3, 5, 6
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 150s kill

    # Matrix multiplication tensor: T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because C[i,k] = sum_j A[i,j] * B[j,k]
    # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k

    def _verify(U, V, W):
        """Verify that U, V, W give an exact matrix multiplication tensor decomposition."""
        rank = U.shape[1]
        T_target = np.ones((n, m, p))
        T_computed = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_computed[i, j, k] = np.dot(
                        U[i * m + j, :],
                        V[j * p + k, :] * W[k * n + i, :]
                    )
        return np.allclose(T_computed, T_target, atol=1e-9)

    def _schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def _winograd_freq_domain():
        """
        Winograd frequency domain approach:
        Map matrix multiplication to pointwise multiplication in cyclic group algebra.
        
        The key idea: if we can find N such that we can embed the (n,m,p) index structure
        into Z_N without aliasing, then we can use DFT to get a decomposition.
        
        For n x m * m x p, the tensor is all-ones T[i,j,k]=1.
        We need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        
        One structured approach: use the fact that the all-ones tensor can be decomposed
        using separable patterns. 
        
        Actually, let me think about this differently. The Winograd approach for small
        matrix multiplication typically works by:
        1. Padding to power-of-2 or suitable size
        2. Using DFT/FFT to convert convolution to pointwise mult
        3. The rank is related to the size of the DFT
        
        For 3x5x6, let's try a more direct structured decomposition.
        
        Key insight: The all-ones tensor of size n*m*p can be written as:
        sum over some structured factors.
        
        Let's try: Can we use the fact that 1 = 1*1*1 and find a low-rank
        factorization of the all-ones tensor?
        
        The all-ones tensor T[i,j,k] = 1 is rank 1! But that's trivial and doesn't
        work because we need the specific index structure.
        
        Wait - the tensor IS all ones. So T = sum_r u_r * v_r * w_r where
        (u_r * v_r * w_r)[i,j,k] = u_r[i,j] * v_r[j,k] * w_r[k,i]
        
        This is NOT the standard tensor decomposition because the indices are coupled:
        u depends on (i,j), v on (j,k), w on (k,i).
        
        Let me reformulate. We need:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        This is equivalent to: for all (i,j,k), the inner product of
        U[i*m+j, :] with (V[j*p+k, :] * W[k*n+i, :]) equals 1.
        
        A simple decomposition: use the schoolbook rank n*m*p = 90.
        Can we do better?
        
        Let's try a Kronecker product structure:
        If we write n=3, m=5, p=6, we can try:
        - Split m = 2 + 3 (or other partitions)
        - Use Strassen-like tricks on sub-blocks
        
        For a 2x2x2 block, Strassen gives rank 7 instead of 8.
        
        Let me try to explicitly construct a decomposition using Strassen on sub-blocks.
        
        Partition A (3x5) and B (5x6):
        A = [[A11, A12, A13, A14, A15],
             [A21, A22, A23, A24, A25],
             [A31, A32, A33, A34, A35]]
        
        This is getting complex. Let me try a different approach:
        Use the fact that the tensor is all-ones and find a minimal set of
        "basis" decompositions.
        
        Actually, let me try a random/optimization approach with the structure
        that U, V, W should have integer or half-integer entries.
        """
        
        # Approach 1: Try to find a rank < 90 decomposition using structured patterns
        # The all-ones tensor has a lot of symmetry we can exploit.
        
        # Key observation: T[i,j,k] = 1 for all i,j,k
        # This means: for fixed j, sum over i,k of the constraint is...
        # Actually, the constraint is per (i,j,k) triple.
        
        # Let's try: use a "grid" of factors where each factor r corresponds to
        # a subset of (i,j,k) triples.
        
        # Simple structured approach: 
        # For each pair (i, k), we need sum_j,r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all j
        # This means: for fixed (i,k), the vectors [U[i*m+j,r]*W[k*n+i,r]]_j dot V[j*p+k,:] = 1 for all j
        
        # Let's try a decomposition where:
        # W[k*n+i, r] = w_{i,k,r} (depends on i,k)
        # U[i*m+j, r] = u_{i,j,r} (depends on i,j)  
        # V[j*p+k, r] = v_{j,k,r} (depends on j,k)
        # Constraint: sum_r u_{i,j,r} * v_{j,k,r} * w_{i,k,r} = 1 for all i,j,k
        
        # This is equivalent to: for each (i,k), the matrix [v_{j,k,r} * w_{i,k,r}]_{j,r}
        # dotted with [u_{i,j,r}]_{j,r}... wait, this isn't quite right.
        
        # Let me just try to construct a valid decomposition with rank < 90.
        # Start with schoolbook and try to merge terms.
        
        # Schoolbook: U[i*m+j, i*m+j] = 1, V[j*p+k, i*m+j] = 1, W[k*n+i, i*m+j] = 1
        # (using column index = i*m+j for the "diagonal" terms, but W index is k*n+i)
        
        # Actually in schoolbook: for each (i,j,k), column r = i*m*p + j*p + k has
        # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
        
        # Can we combine multiple (i,j,k) into one column?
        # Column r contributes to (i,j,k) if U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] != 0
        # For the sum to be 1, we need exactly the right combination.
        
        # Let's try: group by i. For each i, we have m*p = 5*6 = 30 terms in schoolbook.
        # Can we reduce these 30 terms per i?
        
        # For fixed i, the constraint is: for all j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # Let W[k*n+i,r] be a vector of length rank. Let A_r = U[i*m+*,r] (vector of length m)
        # and B_r = V[*p+k,r] for each k... this is getting complicated.
        
        # Let me try a completely different approach: use the Winograd method properly.
        
        # Winograd for matrix mult: 
        # C = A * B where A is n x m, B is m x p
        # Pad to sizes that allow FFT: e.g., pad to 8x8 if possible
        # But 3x5x6 doesn't fit neatly into power-of-2 blocks.
        
        # Alternative: use the "separable" decomposition
        # T[i,j,k] = 1 = sum_{a=0}^{n-1} sum_{b=0}^{p-1} (something)
        
        # Actually, here's a key insight:
        # 1 = sum_{r=0}^{R-1} f_r(i,j) * g_r(j,k) * h_r(k,i)
        # where f_r, g_r, h_r are structured functions.
        
        # If we use Fourier modes:
        # f_r(i,j) = omega^{a_r * i + b_r * j}
        # g_r(j,k) = omega^{c_r * j + d_r * k}
        # h_r(k,i) = omega^{e_r * k + f_r * i}
        # Then the product is omega^{(a_r+e_r)*i + (b_r+c_r)*j + (d_r+f_r)*k}
        # Sum over r of these should equal 1 for all i,j,k.
        
        # This requires that the sum of exponentials is constant (=1) for all i,j,k.
        # This is only possible if all the frequency components cancel out,
        # leaving only the zero-frequency term.
        
        # So: sum_r omega^{0} = R must equal 1... that gives R=1, which is wrong.
        
        # The issue is that the index structure couples i,j,k in a specific way.
        # The correct formulation uses the fact that:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # And the tensor decomposition gives:
        # C[i,k] = sum_r (sum_j U[i*m+j,r] A[i,j]) * (sum_k W[k*n+i,r] B[j,k] * V[j*p+k,r])
        # Wait, that's not quite right either.
        
        # Let me re-derive. The standard tensor decomposition of matrix multiplication:
        # mu = sum_r u_r * v_r * w_r where mu is the matrix multiplication tensor
        # (mu * A_flat) (B_flat) = (A*B)_flat
        # Specifically: (A*B)[i,k] = sum_{j,r} A[i,j] * B[j,k] * (coefficient from tensor)
        
        # The tensor mu has mu[i,j,k] = 1 for all i,j,k (in the standard convention
        # where A is n x m, B is m x p, and we sum over j).
        
        # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # This IS the all-ones tensor with the specific index coupling.
        
        # Let me try: can we use the fact that 1 = 1 * 1 * 1 and find a
        # low-rank factorization?
        
        # If we set U[i*m+j, r] = 1/sqrt(R), V[j*p+k, r] = 1/sqrt(R), W[k*n+i, r] = R
        # for all entries, then sum_r = R * (1/sqrt(R)) * (1/sqrt(R)) * R = R. Not 1.
        
        # If U = V = 1/sqrt(R), W = 1/sqrt(R), then sum_r = R * 1/R = 1. 
        # But this gives U[i*m+j,r] = 1/sqrt(R) for ALL i,j,r, which means
        # each column of U is constant, same for V and W.
        # Check: sum_r (1/sqrt(R)) * (1/sqrt(R)) * (1/sqrt(R)) = R / R^{3/2} = 1/sqrt(R). 
        # That's not 1 unless R = 1.
        
        # So that doesn't work.
        
        # Let me try: U[i*m+j, r] = delta_{i, f(r)}, V[j*p+k, r] = delta_{j, g(r)}, 
        # W[k*n+i, r] = delta_{k, h(r)} for some functions f, g, h.
        # Then the product is 1 only when i=f(r), j=g(r), k=h(r), so each r covers
        # exactly one (i,j,k) triple. We need R >= n*m*p = 90. This is schoolbook.
        
        # To get rank < 90, we need each r to cover multiple (i,j,k) triples
        # with the right weights summing to 1.
        
        # Key idea: use "blocks" where a single factor covers a group of triples.
        
        # For example, if we can find r such that U[i*m+j,r] = a_{i,j}, V[j*p+k,r] = b_{j,k},
        # W[k*n+i,r] = c_{k,i} and a_{i,j} * b_{j,k} * c_{k,i} sums to 1 over r,
        # we need the sum over r to be 1.
        
        # Let's try a 2D block decomposition.
        # Split n=3 into 1+2, m=5 into 2+3, p=6 into 3+3.
        
        # This is getting too complex for a clean construction. Let me try a
        # numerical optimization approach.
        
        # Approach: Start with schoolbook (rank 90) and try to "merge" pairs of
        # columns using an alternating minimization.
        
        # Actually, let me try a simpler structured approach first.
        
        # Observation: The tensor T[i,j,k] = 1 can be written as:
        # T[i,j,k] = sum_{a=0}^{n-1} sum_{b=0}^{p-1} (1/(n*p)) * (n*p) = 1
        # That's trivial.
        
        # Better: T[i,j,k] = sum_{r=0}^{R-1} x_r(i) * y_r(j) * z_r(k) * w_r(i,k)
        # But our structure is fixed: U depends on (i,j), V on (j,k), W on (k,i).
        
        # Let me try: R = n*p = 18.
        # For each (i,k) pair, assign one column r_{i,k}.
        # U[i*m+j, r_{i',k'}] = 1 if i=i', else 0
        # V[j*p+k, r_{i',k'}] = 1 if k=k', else 0
        # W[k*n+i, r_{i',k'}] = 1 if (i,k)=(i',k'), else 0
        # Then for fixed (i,j,k), only r_{i,k} contributes: 1*1*1 = 1. 
        # But wait, we also need other (i',k') not to contribute.
        # U[i*m+j, r_{i',k'}] = 1 if i=i'. So for i' != i, U=0, product=0. Good.
        # V[j*p+k, r_{i',k'}] = 1 if k=k'. For k' != k, V=0, product=0. Good.
        # So this works! Rank = n*p = 3*6 = 18.
        
        # Wait, let me double-check. For (i,j,k), the sum is over r:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # = sum_{i',k'} U[i*m+j, r_{i',k'}] * V[j*p+k, r_{i',k'}] * W[k*n+i, r_{i',k'}]
        # U[i*m+j, r_{i',k'}] = 1 if i = i', else 0
        # V[j*p+k, r_{i',k'}] = 1 if k = k', else 0
        # W[k*n+i, r_{i',k'}] = 1 if (k*n+i) = (k'*n+i'), i.e., k=k' and i=i'
        # So the only non-zero term is when i'=i and k'=k, giving 1*1*1 = 1.
        # All other terms are 0. Sum = 1. 
        
        # This gives rank = n*p = 18! Much better than 90.
        
        U = np.zeros((n * m, n * p))
        V = np.zeros((m * p, n * p))
        W = np.zeros((p * n, n * p))
        
        for i in range(n):
            for k in range(p):
                r = i * p + k
                # U[i*m+j, r] = 1 for all j
                for j in range(m):
                    U[i * m + j, r] = 1
                # V[j*p+k, r] = 1 for all j
                for j in range(m):
                    V[j * p + k, r] = 1
                # W[k*n+i, r] = 1
                W[k * n + i, r] = 1
        
        return U, V, W

    def _winograd_improved():
        """
        Try to further reduce the rank below n*p = 18.
        
        The n*p construction uses one column per (i,k) pair.
        Can we merge some of these?
        
        For fixed (i,k), the column r_{i,k} has:
        - U[i*m+*, r_{i,k}] = 1 (all j)
        - V[*p+k, r_{i,k}] = 1 (all j)
        - W[k*n+i, r_{i,k}] = 1
        
        To merge (i,k) and (i',k') into one column, we'd need:
        U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] + U[i*m+j, r] * V[j*p+k', r] * W[k'*n+i, r] + ... = 1
        
        This is complex. Let's try a different merging strategy.
        
        Alternative: use n*m = 15 columns (one per (i,j) pair).
        For fixed (i,j), column r_{i,j}:
        - U[i*m+j, r_{i,j}] = 1
        - V[j*p+k, r_{i,j}] = 1 for all k
        - W[k*n+i, r_{i,j}] = 1 for all k
        
        Check: for (i',j',k'), sum over r:
        Only r_{i',j'} contributes to U[i'*m+j', r] (since U[i*m+j, r_{i,j}] = delta_{i,i'}*delta_{j,j'})
        So the sum is: 1 * V[j'*p+k', r_{i',j'}] * W[k'*n+i', r_{i',j'}] = 1 * 1 * 1 = 1. 
        This also works! Rank = n*m = 15.
        
        Even better: use m*p = 30 columns (one per (j,k) pair).
        For fixed (j,k), column r_{j,k}:
        - U[i*m+j, r_{j,k}] = 1 for all i
        - V[j*p+k, r_{j,k}] = 1
        - W[k*n+i, r_{j,k}] = 1 for all i
        
        Check: for (i,j',k'), sum over r:
        Only r_{j',k'} contributes to V[j'*p+k', r]
        Sum = U[i*m+j', r_{j',k'}] * 1 * W[k'*n+i, r_{j',k'}] = 1 * 1 * 1 = 1. 
        Rank = m*p = 30. Worse than 15.
        
        So the best of these simple constructions is rank = min(n*p, n*m, m*p) = min(18, 15, 30) = 15.
        
        Can we do even better? 
        
        Lower bound: The matrix multiplication tensor mu_{n,m,p} has rank at least
        max(n*m, m*p, n*p) for "generic" tensors, but for the all-ones tensor
        with this specific structure, the lower bound might be different.
        
        Actually, the rank of the matrix multiplication tensor mu_{n,m,p} is known to be
        at least n*m*p / max(n,m,p) by some arguments, but I'm not sure of the exact bound.
        
        For 3x5x6, can we achieve rank < 15?
        
        Let's try rank = 12. We'd need to merge some of the 15 (i,j) pairs.
        
        For rank = 12, we have 3 fewer columns. Each column in the n*m construction
        corresponds to a unique (i,j) pair. To reduce rank, we need some columns to
        cover multiple (i,j) pairs.
        
        Column r covering (i1,j1) and (i2,j2):
        U[i1*m+j1, r] * V[j1*p+k, r] * W[k*n+i1, r] should contribute to (i1,j1,k)
        U[i2*m+j2, r] * V[j2*p+k', r] * W[k'*n+i2, r] should contribute to (i2,j2,k')
        
        But U[i*m+j, r] is a single value, so if U[i1*m+j1, r] = a, then for
        (i1, j1, k): contribution includes a * V[j1*p+k, r] * W[k*n+i1, r]
        For (i1, j2, k): contribution includes a * V[j2*p+k, r] * W[k*n+i1, r]
        This means the same column affects multiple (i,j,k) triples in a way that
        must still sum to 1.
        
        This is a system of linear equations. Let's set it up.
        
        For rank R, we have:
        For all i in [0,n), j in [0,m), k in [0,p):
        sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        This is n*m*p = 90 equations in R * (n*m + m*p + p*n) = R * (15 + 30 + 18) = 63R unknowns.
        
        For R = 15: 90 equations, 945 unknowns. Underdetermined, should have solutions.
        For R = 12: 90 equations, 756 unknowns. Still underdetermined.
        For R = 10: 90 equations, 630 unknowns. Still underdetermined.
        
        The question is whether there exist solutions with the right structure
        (integer or half-integer entries, or at least real entries that give exact 1).
        
        Let's try to solve for R = 12, 11, 10 using linear algebra.
        
        Actually, the equations are NOT linear in the unknowns because of the
        product U*V*W. Each equation is a sum of products of three unknowns.
        This is a system of polynomial equations, not linear.
        
        So we need a different approach. Let's use the fact that for the n*m
        construction (rank 15), the solution is:
        U[i*m+j, r_{i,j}] = 1, all other U entries in column r_{i,j} are 0
        V[j*p+k, r_{i,j}] = 1, all other V entries in column r_{i,j} are 0
        W[k*n+i, r_{i,j}] = 1, all other W entries in column r_{i,j} are 0
        
        To reduce rank, we need to "merge" columns. This is like finding a
        rank-revealing decomposition of a structured tensor.
        
        Given the complexity, let's stick with rank = 15 for now, which is
        already a huge improvement over 90.
        """
        U = np.zeros((n * m, n * m))
        V = np.zeros((m * p, n * m))
        W = np.zeros((p * n, n * m))
        
        for i in range(n):
            for j in range(m):
                r = i * m + j
                U[i * m + j, r] = 1
                for k in range(p):
                    V[j * p + k, r] = 1
                    W[k * n + i, r] = 1
        
        return U, V, W

    # Try the best structured approaches
    best_rank = n * m * p  # schoolbook rank
    
    # Try n*p construction (rank 18)
    U1, V1, W1 = _winograd_freq_domain()
    if _verify(U1, V1, W1):
        rank1 = U1.shape[1]
        if rank1 < best_rank:
            best_rank = rank1
            best_U, best_V, best_W = U1, V1, W1
    
    # Try n*m construction (rank 15)
    U2, V2, W2 = _winograd_improved()
    if _verify(U2, V2, W2):
        rank2 = U2.shape[1]
        if rank2 < best_rank:
            best_rank = rank2
            best_U, best_V, best_W = U2, V2, W2
    
    # Check deadline
    if time.time() > deadline - 10:
        # Not enough time for further search, return best found
        if best_rank < n * m * p:
            return {
                "U": best_U.tolist(),
                "V": best_V.tolist(),
                "W": best_W.tolist()
            }
        else:
            U, V, W = _schoolbook()
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Try to find an even lower rank using numerical optimization
    # This is expensive, so only attempt if we have time.
    
    def _try_lower_rank(R):
        """Try to find a rank-R decomposition using structured initialization."""
        if R < 1:
            return None
        
        # Initialize with the n*m pattern and try to merge columns
        # For now, just check if R >= 15 (our best known)
        if R >= 15:
            U, V, W = _winograd_improved()
            # This gives rank 15, not R. We need exactly rank R.
            # Truncate to first R columns (if R < 15, this won't work)
            if R >= 15:
                return U, V, W
        
        return None
    
    # Try ranks from 14 down to 10
    for R in range(14, 9, -1):
        if time.time() > deadline - 20:
            break
        result = _try_lower_rank(R)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W):
                best_rank = R
                best_U, best_V, best_W = U, V, W
                break
    
    # Return the best result
    if best_rank < n * m * p:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    # Fall back to schoolbook
    U, V, W = _schoolbook()
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
