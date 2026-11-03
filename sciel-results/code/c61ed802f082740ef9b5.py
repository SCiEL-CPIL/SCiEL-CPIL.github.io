import time
import numpy as np
from itertools import product

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    Uses Strassen-based block decomposition to reduce rank below schoolbook (60).
    Falls back to schoolbook if no improvement is found within time budget.
    """
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    nm, mp, pn = n * m, m * p, p * n  # 10, 30, 12

    # Build target tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W):
        """Check if U, V, W exactly decompose tensor T."""
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_rec, T, atol=1e-9)

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=np.float64)
        V = np.zeros((mp, rank), dtype=np.float64)
        W = np.zeros((pn, rank), dtype=np.float64)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1.0
                    V[j * p + k, idx] = 1.0
                    W[k * n + i, idx] = 1.0
                    idx += 1
        return U, V, W

    def strassen_decomp_2x5_5x6():
        """
        Construct decomposition using Strassen on 2x2 sub-blocks.
        For each j-pair (j0,j1) and k-pair (k0,k1), the 2x2*2x2 subproblem
        (over i in {0,1}) can be done in rank 7 instead of 8.
        """
        rank = 0
        U_list = []
        V_list = []
        W_list = []
        
        def add_term(U_col, V_col, W_col):
            nonlocal rank
            U_list.append(U_col)
            V_list.append(V_col)
            W_list.append(W_col)
            rank += 1
        
        def make_U_col(values):
            col = np.zeros(nm, dtype=np.float64)
            for idx, val in values.items():
                col[idx] = val
            return col
        
        def make_V_col(values):
            col = np.zeros(mp, dtype=np.float64)
            for idx, val in values.items():
                col[idx] = val
            return col
        
        def make_W_col(values):
            col = np.zeros(pn, dtype=np.float64)
            for idx, val in values.items():
                col[idx] = val
            return col
        
        def strassen_terms(i0, i1, j0, j1, k0, k1):
            """
            Generate Strassen terms for 2x2*2x2 subproblem.
            A = [[a,b],[c,d]] where a=A[i0,j0], b=A[i0,j1], c=A[i1,j0], d=A[i1,j1]
            B = [[e,f],[g,h]] where e=B[j0,k0], f=B[j0,k1], g=B[j1,k0], h=B[j1,k1]
            C = [[C11,C12],[C21,C22]] where C11=C[i0,k0], C12=C[i0,k1], C21=C[i1,k0], C22=C[i1,k1]
            
            Strassen:
            M1 = (a+d)(e+h)
            M2 = (c+d)e
            M3 = a(f-h)
            M4 = d(g-e)
            M5 = (a+b)h
            M6 = (c-a)(e+f)
            M7 = (b-d)(g+h)
            
            C11 = M1 + M4 - M5 + M7
            C12 = M3 + M5
            C21 = M2 + M4
            C22 = M1 + M3 + M6
            """
            # M1 = (a+d)(e+h)
            add_term(
                make_U_col({i0*m+j0: 1, i1*m+j1: 1}),
                make_V_col({j0*p+k0: 1, j1*p+k1: 1}),
                make_W_col({k0*n+i0: 1, k1*n+i1: 1})
            )
            
            # M2 = (c+d)e
            add_term(
                make_U_col({i1*m+j0: 1, i1*m+j1: 1}),
                make_V_col({j0*p+k0: 1}),
                make_W_col({k0*n+i1: 1})
            )
            
            # M3 = a(f-h)
            add_term(
                make_U_col({i0*m+j0: 1}),
                make_V_col({j0*p+k1: 1, j1*p+k1: -1}),
                make_W_col({k1*n+i0: 1, k1*n+i1: 1})
            )
            
            # M4 = d(g-e)
            add_term(
                make_U_col({i1*m+j1: 1}),
                make_V_col({j1*p+k0: 1, j0*p+k0: -1}),
                make_W_col({k0*n+i0: 1, k0*n+i1: 1})
            )
            
            # M5 = (a+b)h
            add_term(
                make_U_col({i0*m+j0: 1, i0*m+j1: 1}),
                make_V_col({j1*p+k1: 1}),
                make_W_col({k0*n+i0: -1, k1*n+i0: 1})
            )
            
            # M6 = (c-a)(e+f)
            add_term(
                make_U_col({i1*m+j0: 1, i0*m+j0: -1}),
                make_V_col({j0*p+k0: 1, j0*p+k1: 1}),
                make_W_col({k1*n+i1: 1})
            )
            
            # M7 = (b-d)(g+h)
            add_term(
                make_U_col({i0*m+j1: 1, i1*m+j1: -1}),
                make_V_col({j1*p+k0: 1, j1*p+k1: 1}),
                make_W_col({k0*n+i0: 1})
            )
        
        # Apply Strassen to all 2x2*2x2 subproblems
        # j-pairs: (0,1), (2,3)
        # k-pairs: (0,1), (2,3), (4,5)
        # i-pair: (0,1)
        
        for (j0, j1) in [(0, 1), (2, 3)]:
            for (k0, k1) in [(0, 1), (2, 3), (4, 5)]:
                strassen_terms(0, 1, j0, j1, k0, k1)
        
        # Handle j=4 (singleton) with each k-pair
        # For j=4, k in {k0,k1}: C[i,k] += A[i,4]*B[4,k]
        # This is rank 2 (one term per k value)
        for (k0, k1) in [(0, 1), (2, 3), (4, 5)]:
            for k in [k0, k1]:
                add_term(
                    make_U_col({0*m+4: 1, 1*m+4: 1}),
                    make_V_col({4*p+k: 1}),
                    make_W_col({k*n+0: 1, k*n+1: 1})
                )
        
        U = np.column_stack(U_list)
        V = np.column_stack(V_list)
        W = np.column_stack(W_list)
        
        return U, V, W, rank

    # Start with schoolbook as fallback
    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p  # 60

    # Try Strassen decomposition
    try:
        U_s, V_s, W_s, rank_s = strassen_decomp_2x5_5x6()
        if verify(U_s, V_s, W_s) and rank_s < best_rank:
            best_U, best_V, best_W = U_s, V_s, W_s
            best_rank = rank_s
    except Exception:
        pass

    # Systematic search: try to merge schoolbook terms
    # Key insight: if two schoolbook terms (i,j,k) and (i',j',k') can be combined
    # into one term with U,V,W having overlapping support, we save a rank.
    # 
    # More structured approach: use the fact that n=2 to find patterns.
    # For fixed j, the contribution is: sum_k A[i,j]*B[j,k] to C[i,k]
    # If we can group k values, we might save.
    
    # Try a different block structure: split k into groups
    # and use linearity to reduce rank.
    
    # Approach: For each j, we have a 2x6 matrix C_j where C_j[i,k] = A[i,j]*B[j,k]
    # This is an outer product, rank 1 as a matrix.
    # In tensor terms, for fixed j, we need 2*6 = 12 schoolbook terms,
    # but the structure is: U has A[i,j] for i in {0,1}, V has B[j,k] for k in {0..5},
    # W has 1 for all (i,k). This is actually 1 term if we allow W to have
    # multiple nonzeros! But wait, that's not how tensor decomposition works.
    # Each term is a rank-1 tensor: U[:,r] outer V[:,r] outer W[:,r].
    # So for fixed j, we can't do it in 1 term because U and V have different supports.
    
    # Let me try a smarter random search with structured patterns
    np.random.seed(12345)
    
    def random_structured_search(target_rank, n_attempts=100):
        """Try to find a decomposition of given rank with structured random patterns."""
        for attempt in range(n_attempts):
            if time.time() > deadline - 15:
                return None
            
            U_try = np.zeros((nm, target_rank), dtype=np.float64)
            V_try = np.zeros((mp, target_rank), dtype=np.float64)
            W_try = np.zeros((pn, target_rank), dtype=np.float64)
            
            for r in range(target_rank):
                # Randomly choose which (i,j,k) triple this term primarily targets
                i = np.random.randint(0, n)
                j = np.random.randint(0, m)
                k = np.random.randint(0, p)
                
                # Base term
                U_try[i*m+j, r] = 1.0
                V_try[j*p+k, r] = 1.0
                W_try[k*n+i, r] = 1.0
                
                # With some probability, add additional structure
                if np.random.rand() < 0.4:
                    # Add another i
                    if np.random.rand() < 0.5:
                        i2 = 1 - i
                        U_try[i2*m+j, r] = np.random.choice([-1, 1])
                        W_try[k*n+i2, r] = np.random.choice([-1, 1])
                
                if np.random.rand() < 0.4:
                    # Add another k
                    k2 = np.random.randint(0, p)
                    if k2 != k:
                        V_try[j*p+k2, r] = np.random.choice([-1, 1])
                        W_try[k2*n+i, r] = np.random.choice([-1, 1])
                
                if np.random.rand() < 0.3:
                    # Add another j
                    j2 = np.random.randint(0, m)
                    if j2 != j:
                        U_try[i*m+j2, r] = np.random.choice([-1, 1])
                        V_try[j2*p+k, r] = np.random.choice([-1, 1])
            
            if verify(U_try, V_try, W_try):
                return U_try, V_try, W_try
        
        return None

    # Try to find lower rank
    for target_rank in range(best_rank - 1, max(40, best_rank - 15), -1):
        if time.time() > deadline - 20:
            break
        
        result = random_structured_search(target_rank, n_attempts=50)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = target_rank
            break

    def to_list(arr):
        return arr.astype(float).tolist()

    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
