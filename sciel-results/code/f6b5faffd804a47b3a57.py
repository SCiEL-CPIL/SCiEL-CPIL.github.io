"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product as iproduct

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s limit

    n, m, p = DIMENSIONS

    def time_left():
        return deadline - time.time()

    # Build target tensor T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W):
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T_rec - T)
        return err < 1e-9

    def make_strassen_42():
        """Create the standard Strassen-42 decomposition."""
        rank = 42
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        # Strassen 2x2x2 factors for local indices
        prods = [
            # M1: (a00+a11)(b00+b11) -> c00+=1, c11+=1
            ([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)]),
            # M2: (a10+a11)b00 -> c10+=1
            ([(2,1), (3,1)], [(0,1)], [(2,1)]),
            # M3: a00(b01-b11) -> c01+=1
            ([(0,1)], [(1,1), (3,-1)], [(1,1)]),
            # M4: a11(b10-b00) -> c00+=1, c10+=1
            ([(3,1)], [(2,1), (0,-1)], [(0,1), (2,1)]),
            # M5: (a00+a01)b11 -> c00-=1, c01+=1
            ([(0,1), (1,1)], [(3,1)], [(0,-1), (1,1)]),
            # M6: (a10-a00)(b00+b01) -> c11+=1
            ([(2,1), (0,-1)], [(0,1), (1,1)], [(3,1)]),
            # M7: (a01-a11)(b10+b11) -> c00-=1, c11+=1
            ([(1,1), (3,-1)], [(2,1), (3,1)], [(0,-1), (3,1)]),
        ]

        for t in range(6):
            for s in range(7):
                col = 7 * t + s
                u_entries, v_entries, w_entries = prods[s]

                for idx, coeff in u_entries:
                    U[idx, col] = coeff

                for idx, coeff in v_entries:
                    j = idx // 2
                    k_loc = idx % 2
                    k = 2 * t + k_loc
                    V[j * p + k, col] = coeff

                for idx, coeff in w_entries:
                    k_loc = idx // 2
                    i = idx % 2
                    k = 2 * t + k_loc
                    W[k * n + i, col] = coeff

        return U, V, W, rank

    U42, V42, W42, rank42 = make_strassen_42()
    
    # Verify baseline
    if not verify(U42, V42, W42):
        # Fallback to schoolbook
        rank_fb = 48
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U_fb[a, col] = 1
                    V_fb[b, col] = 1
                    W_fb[c, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    best_U, best_V, best_W = U42, V42, W42
    best_rank = 42

    # ------------------------------------------------------------------
    # NEW STRATEGY: Find a lower rank decomposition for the 2x2x3 sub-tensor.
    # If we can decompose 2x2x3 with rank R_sub, then 2x2x12 can be done with 4*R_sub.
    # We suspect R_sub might be 9 or 10.
    #
    # Let's define the 2x2x3 tensor T_sub.
    # Indices: a in {0,1,2,3} (i*m+j), b in {0..5} (j*3+k), c in {0..5} (k*2+i)
    # Wait, for 2x2x3: n=2, m=2, p=3.
    # a = i*2 + j
    # b = j*3 + k
    # c = k*2 + i
    # ------------------------------------------------------------------

    def find_sub_decomp(rank_sub):
        """
        Try to find a rank-rank_sub decomposition for the 2x2x3 tensor.
        We use a structured approach: assume the decomposition is a modification
        of the Strassen 7 terms plus some extra terms, or a fully generic search
        with rational constraints.
        
        For rank 9, we have 9 terms.
        U: 4x9, V: 6x9, W: 6x9.
        This is a system of 48 equations (non-zero entries in T_sub) with 9*(4+6+6) = 126 variables.
        It's underdetermined. We can fix some variables to integers to simplify.
        
        Let's try a specific ansatz:
        The first 7 terms are the Strassen terms for the first 2 columns (k=0,1).
        The last 2 terms handle the 3rd column (k=2) and potentially adjust the first 7.
        
        Actually, let's just use a generic linear solver approach on the flattened tensor.
        We want sum_r U[:,r] V[:,r] W[:,r]^T = T_sub.
        
        We will try to find integer solutions.
        """
        if time_left() < 5:
            return None

        # T_sub dimensions: 4 x 6 x 6
        T_sub = np.zeros((4, 6, 6))
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    a = i * 2 + j
                    b = j * 3 + k
                    c = k * 2 + i
                    T_sub[a, b, c] = 1.0

        # We will try to construct the decomposition block by block.
        # Block 1: columns 0,1 of B (k=0,1). This is 2x2x2. Rank 7.
        # Block 2: column 2 of B (k=2). This is 2x2x1. Rank 4.
        # Total 11. Can we merge?
        
        # Let's try a rank 9 ansatz.
        # We define 9 rank-1 tensors.
        # We will use a randomized search with local optimization, but constrained to integers.
        
        # Initialize with Strassen 7 + 2 random terms
        U = np.zeros((4, rank_sub))
        V = np.zeros((6, rank_sub))
        W = np.zeros((6, rank_sub))
        
        # Fill first 7 with Strassen
        prods = [
            ([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)]),
            ([(2,1), (3,1)], [(0,1)], [(2,1)]),
            ([(0,1)], [(1,1), (3,-1)], [(1,1)]),
            ([(3,1)], [(2,1), (0,-1)], [(0,1), (2,1)]),
            ([(0,1), (1,1)], [(3,1)], [(0,-1), (1,1)]),
            ([(2,1), (0,-1)], [(0,1), (1,1)], [(3,1)]),
            ([(1,1), (3,-1)], [(2,1), (3,1)], [(0,-1), (3,1)]),
        ]
        
        for s in range(7):
            u_entries, v_entries, w_entries = prods[s]
            for idx, coeff in u_entries:
                U[idx, s] = coeff
            for idx, coeff in v_entries:
                # b index for 2x2x3: j*3+k. Here k is 0 or 1.
                # In prods, idx is local (j, k_loc) where k_loc is 0 or 1.
                j = idx // 2
                k_loc = idx % 2
                b = j * 3 + k_loc
                V[b, s] = coeff
            for idx, coeff in w_entries:
                # c index for 2x2x3: k*2+i.
                k_loc = idx // 2
                i = idx % 2
                c = k_loc * 2 + i
                W[c, s] = coeff

        # The last rank_sub - 7 terms are free.
        # We want to adjust U, V, W to minimize error.
        # Let's use a simple gradient step in float space, then round.
        
        np.random.seed(42)
        # Initialize extra terms randomly
        for r in range(7, rank_sub):
            U[:, r] = np.random.randint(-2, 3, size=4)
            V[:, r] = np.random.randint(-2, 3, size=6)
            W[:, r] = np.random.randint(-2, 3, size=6)
            
        # Optimize
        best_err = np.linalg.norm(np.einsum('ar,br,cr->abc', U, V, W) - T_sub)
        
        for _ in range(1000):
            if time_left() < 2:
                break
            # Pick a random term to modify
            r = np.random.randint(0, rank_sub)
            # Try small perturbations
            for delta in [-1, 1]:
                # Perturb U
                idx = np.random.randint(0, 4)
                orig = U[idx, r]
                U[idx, r] += delta
                err = np.linalg.norm(np.einsum('ar,br,cr->abc', U, V, W) - T_sub)
                if err < best_err:
                    best_err = err
                else:
                    U[idx, r] = orig
                # Perturb V
                idx = np.random.randint(0, 6)
                orig = V[idx, r]
                V[idx, r] += delta
                err = np.linalg.norm(np.einsum('ar,br,cr->abc', U, V, W) - T_sub)
                if err < best_err:
                    best_err = err
                else:
                    V[idx, r] = orig
                # Perturb W
                idx = np.random.randint(0, 6)
                orig = W[idx, r]
                W[idx, r] += delta
                err = np.linalg.norm(np.einsum('ar,br,cr->abc', U, V, W) - T_sub)
                if err < best_err:
                    best_err = err
                else:
                    W[idx, r] = orig
                    
            if best_err < 1e-9:
                break

        if best_err < 1e-9:
            return U, V, W

        return None

    # Try to find a rank 9 or 10 decomposition for the 2x2x3 sub-tensor
    for r_sub in [9, 10, 11]:
        if time_left() < 10:
            break
        U_sub, V_sub, W_sub = find_sub_decomp(r_sub)
        if U_sub is not None:
            # Construct full 2x2x12 decomposition
            rank_total = 4 * r_sub
            U_full = np.zeros((n * m, rank_total))
            V_full = np.zeros((m * p, rank_total))
            W_full = np.zeros((p * n, rank_total))
            
            for t in range(4):
                # Map sub-block indices to full indices
                # Sub-block t corresponds to columns 3*t to 3*t+2 of B
                # And rows 3*t to 3*t+2 of C (in terms of k)
                
                for r in range(r_sub):
                    col = t * r_sub + r
                    
                    # U is the same for all blocks (A is 2x2)
                    U_full[:, col] = U_sub[:, r]
                    
                    # V: b = j*p + k. In sub, b_sub = j*3 + k_sub.
                    # k = 3*t + k_sub
                    for j in range(2):
                        for k_sub in range(3):
                            b_sub = j * 3 + k_sub
                            b_full = j * p + (3 * t + k_sub)
                            V_full[b_full, col] = V_sub[b_sub, r]
                            
                    # W: c = k*n + i. In sub, c_sub = k_sub*2 + i.
                    # k = 3*t + k_sub
                    for k_sub in range(3):
                        for i in range(2):
                            c_sub = k_sub * 2 + i
                            c_full = (3 * t + k_sub) * n + i
                            W_full[c_full, col] = W_sub[c_sub, r]
                            
            if verify(U_full, V_full, W_full):
                best_U, best_V, best_W = U_full, V_full, W_full
                best_rank = rank_total
                break

    # If we found a better rank, return it
    if best_rank < 42:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }

    # Otherwise, return the Strassen-42 solution
    return {
        "U": U42.tolist(),
        "V": V42.tolist(),
        "W": W42.tolist()
    }
# EVOLVE-BLOCK-END
