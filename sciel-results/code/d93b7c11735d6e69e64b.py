import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _bkr_decompose(n, m, p):
    """
    Perform Bilinear Complexity Reduction (BKR) to find a low-rank decomposition.
    Uses exact arithmetic (fractions) to ensure correctness.
    """
    # Initialize the tensor T with the matrix multiplication structure.
    # T[i, j, k] = 1 if (i, j, k) corresponds to (a, b, c) in C_ab = sum A_ak B_kb.
    # Indices:
    # i in [0, n*m)  -> a = i // m, b = i % m
    # j in [0, m*p)  -> b = j // p, c = j % p
    # k in [0, p*n)  -> c = k // n, a = k % n
    
    T = {}
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T[(i, j, k)] = Fraction(1)
    
    # Lists to store the decomposition vectors
    U_vecs = [] # List of lists, each of length n*m
    V_vecs = [] # List of lists, each of length m*p
    W_vecs = [] # List of lists, each of length p*n
    
    # BKR Algorithm
    # We iterate, finding a non-zero entry in T, and eliminating it.
    # To improve rank, we can try to find entries that allow for more efficient elimination,
    # but the standard greedy BKR picks the first non-zero or a random one.
    # We will use a deterministic order for reproducibility, but could randomize for search.
    
    # To make it a "search", we can randomize the order of picking pivot elements.
    # However, BKR is deterministic given the pivot choice.
    # A simple strategy: iterate through all non-zero entries, pick one, eliminate.
    
    # To potentially find better ranks, we might need to try different pivot strategies.
    # For now, we implement the standard greedy BKR.
    
    # We need to track the current state of T.
    # T is a dict of (i, j, k) -> value.
    
    # Optimization: We can represent T as a 3D array of Fractions for faster access,
    # but given small dimensions (125 entries), dict is fine.
    
    while T:
        # Pick a pivot (i0, j0, k0) with non-zero value.
        # Strategy: Pick the first one to be deterministic, or random.
        # Let's pick the first one for now.
        pivot_key = next(iter(T.keys()))
        i0, j0, k0 = pivot_key
        val = T[pivot_key]
        
        # Create new vectors u, v, w
        u = [Fraction(0)] * (n * m)
        v = [Fraction(0)] * (m * p)
        w = [Fraction(0)] * (p * n)
        
        u[i0] = Fraction(1)
        v[j0] = Fraction(1)
        w[k0] = Fraction(1)
        
        # Eliminate the pivot
        del T[pivot_key]
        
        # For all other non-zero entries (i, j, k) in T:
        # If i == i0 or j == j0 or k == k0, we can modify T to eliminate the contribution of the new term.
        # The term is u_i * v_j * w_k.
        # We want to subtract val * u_i * v_j * w_k from T? No.
        # The decomposition is T = sum u_r v_r w_r.
        # We have chosen u_r, v_r, w_r such that u_r[i0]=1, v_r[j0]=1, w_r[k0]=1.
        # This term contributes val to T[i0, j0, k0] if we scale it by val.
        # So let's scale the vectors by val.
        u = [x * val for x in u]
        v = [x * val for x in v]
        w = [x * val for x in w]
        
        # Now, for any other (i, j, k) in T:
        # If i == i0, the term u[i0]*v[j]*w[k] = val * v[j] * w[k] is added to T[i0, j, k].
        # But we want to keep T[i0, j, k] unchanged in the *remaining* tensor?
        # Actually, the standard BKR update is:
        # T_new[i, j, k] = T[i, j, k] - (T[i0, j, k] * T[i, j0, k] * T[i, j, k0]) / T[i0, j0, k0] ? No, that's for rank 1 updates in a different context.
        
        # Correct BKR Update:
        # We are constructing a decomposition.
        # We pick a pivot (i0, j0, k0) with value v.
        # We define u, v, w such that u[i0]=1, v[j0]=1, w[k0]=1.
        # We add this term to the decomposition, scaled by v.
        # So the term is v * e_i0 * e_j0 * e_k0.
        # This term has non-zero components at:
        # (i0, j, k) with value v * v_vec[j] * w_vec[k]
        # (i, j0, k) with value v * u_vec[i] * w_vec[k]
        # (i, j, k0) with value v * u_vec[i] * v_vec[j]
        # (i0, j0, k) with value v * v_vec[k]  (wait, u[i0]=1, v[j0]=1, so term is v * 1 * v_vec[j] * 1 * w_vec[k]?)
        
        # Let's restart the logic clearly.
        # Target: T = sum_{r} u_r \otimes v_r \otimes w_r.
        # Current residual: T_res.
        # Pick (i0, j0, k0) from T_res. Let val = T_res[i0, j0, k0].
        # Set u_r[i0] = 1, v_r[j0] = 1, w_r[k0] = 1. Others 0 initially.
        # This term contributes:
        # T_contrib[i, j, k] = val * (1 if i==i0 else 0) * (1 if j==j0 else 0) * (1 if k==k0 else 0) ?
        # No, we can set other components of u_r, v_r, w_r to non-zero values to cancel other parts of T_res.
        
        # Standard BKR step:
        # 1. Pick pivot (i0, j0, k0).
        # 2. Define u, v, w with u[i0]=1, v[j0]=1, w[k0]=1.
        # 3. For all (i, j, k) in T_res such that i==i0 and j!=j0 and k!=k0:
        #    We can set v[j] and w[k] to cancel T_res[i0, j, k]?
        #    Actually, the term is u[i] v[j] w[k].
        #    If i=i0, term is 1 * v[j] * w[k].
        #    We want to match T_res[i0, j, k].
        #    But we only have one degree of freedom per (j, k) pair? No, v[j] and w[k] are shared.
        
        # The algorithm is:
        # 1. Pick (i0, j0, k0).
        # 2. Set u[i0]=1, v[j0]=1, w[k0]=1.
        # 3. For all j != j0, k != k0:
        #    If T_res[i0, j, k] != 0, we can't necessarily cancel it with just v[j] and w[k] unless we have more freedom.
        #    Actually, the standard BKR eliminates the "slice" through the pivot.
        
        # Let's use the explicit formula for BKR update:
        # T_new[i, j, k] = T[i, j, k] - (T[i0, j, k] * T[i, j0, k] * T[i, j, k0]) / T[i0, j0, k0] ?
        # This formula is for the rank-1 update in the context of reducing the tensor to a "core" tensor.
        
        # Let's implement the standard loop:
        # We maintain T as a dict.
        # We pick (i0, j0, k0).
        # We create u, v, w vectors.
        # u[i0] = 1, v[j0] = 1, w[k0] = 1.
        # For all (i, j, k) in T:
        #   If i == i0 and j != j0 and k != k0:
        #      We want to cancel T[i0, j, k].
        #      The term is u[i0] * v[j] * w[k] = 1 * v[j] * w[k].
        #      We can choose v[j] and w[k] arbitrarily?
        #      No, v[j] is shared across all k, and w[k] across all i.
        #      
        #      Actually, the standard BKR sets:
        #      v[j] = T[i0, j, k0] / T[i0, j0, k0]  (if T[i0, j0, k0] != 0)
        #      w[k] = T[i0, j0, k] / T[i0, j0, k0]
        #      This cancels the terms T[i0, j, k0] and T[i0, j0, k].
        #      Then it updates T[i0, j, k] for j!=j0, k!=k0:
        #      T_new[i0, j, k] = T[i0, j, k] - v[j] * w[k]
        #      
        #      Similarly for other slices?
        #      The full BKR step:
        #      1. Pick (i0, j0, k0).
        #      2. Set u[i0]=1, v[j0]=1, w[k0]=1.
        #      3. For all j: v[j] = T[i0, j, k0] / T[i0, j0, k0]
        #      4. For all k: w[k] = T[i0, j0, k] / T[i0, j0, k0]
        #      5. For all i: u[i] = T[i, j0, k0] / T[i0, j0, k0]
        #      6. Update T for all (i, j, k):
        #         T_new[i, j, k] = T[i, j, k] - (u[i] * v[j] * w[k] * T[i0, j0, k0])
        #         Wait, the term added is u[i] v[j] w[k].
        #         We want T_new = T - u v w.
        #         So T_new[i, j, k] = T[i, j, k] - u[i] v[j] w[k].
        
        # Let's verify this logic.
        # If we set u, v, w as above, does T_new[i0, j0, k0] become 0?
        # T_new[i0, j0, k0] = T[i0, j0, k0] - u[i0] v[j0] w[k0]
        # = T[i0, j0, k0] - 1 * 1 * 1 = T[i0, j0, k0] - 1.
        # This is not 0 unless T[i0, j0, k0] = 1.
        
        # Correction:
        # We should scale the vectors by the pivot value.
        # Let val = T[i0, j0, k0].
        # Set u[i0] = 1, v[j0] = 1, w[k0] = 1.
        # The term is val * u v w.
        # So we subtract val * u[i] v[j] w[k] from T.
        
        # Revised BKR Step:
        # 1. Pick (i0, j0, k0). val = T[i0, j0, k0].
        # 2. Initialize u, v, w to 0.
        # 3. Set u[i0] = 1, v[j0] = 1, w[k0] = 1.
        # 4. For all j != j0:
        #    If T[i0, j, k0] != 0:
        #       v[j] = T[i0, j, k0] / val
        # 5. For all k != k0:
        #    If T[i0, j0, k] != 0:
        #       w[k] = T[i0, j0, k] / val
        # 6. For all i != i0:
        #    If T[i, j0, k0] != 0:
        #       u[i] = T[i, j0, k0] / val
        # 7. For all (i, j, k) in T:
        #    T_new[i, j, k] = T[i, j, k] - val * u[i] * v[j] * w[k]
        #    If T_new[i, j, k] == 0, remove from dict.
        # 8. Add u, v, w (scaled by val? No, the term is val * u v w. So we store u, v, w and remember the scalar val? 
        #    Or just store val*u, val*v, val*w. Let's store the vectors such that their product gives the term.
        #    Term = (val*u) \otimes v \otimes w.
        #    So we can store U_r = val*u, V_r = v, W_r = w.
        
        # Let's implement this.
        
        u = [Fraction(0)] * (n * m)
        v = [Fraction(0)] * (m * p)
        w = [Fraction(0)] * (p * n)
        
        u[i0] = Fraction(1)
        v[j0] = Fraction(1)
        w[k0] = Fraction(1)
        
        # Fill v
        for j in range(m * p):
            if j != j0:
                if (i0, j, k0) in T:
                    v[j] = T[(i0, j, k0)] / val
                # else 0
        
        # Fill w
        for k in range(p * n):
            if k != k0:
                if (i0, j0, k) in T:
                    w[k] = T[(i0, j0, k)] / val
        
        # Fill u
        for i in range(n * m):
            if i != i0:
                if (i, j0, k0) in T:
                    u[i] = T[(i, j0, k0)] / val
        
        # Scale u by val to form the actual vector for decomposition
        # The term is val * u[i] * v[j] * w[k]
        # So we can define U_r[i] = val * u[i], V_r[j] = v[j], W_r[k] = w[k]
        U_r = [x * val for x in u]
        V_r = v
        W_r = w
        
        # Update T
        new_T = {}
        for key, t_val in T.items():
            i, j, k = key
            subtract = U_r[i] * V_r[j] * W_r[k]
            new_val = t_val - subtract
            if new_val != 0:
                new_T[key] = new_val
        
        T = new_T
        
        # Add to decomposition
        U_vecs.append(U_r)
        V_vecs.append(V_r)
        W_vecs.append(W_r)
        
        # Check time
        if time.time() > deadline:
            break
            
    # Convert to lists of floats/ints for output
    # The problem requires integer/half-integer entries.
    # Fractions are exact. We can convert them to floats if they are integers or half-integers.
    # If not, the decomposition is valid over Q, but the interface might require specific types?
    # "integer/half-integer entries" suggests we should output numbers that are k/2.
    # BKR over Q produces rationals. If the rank is minimal, they might be integers.
    # If not, we might need to scale? No, the tensor is fixed.
    # We will output the fractions as floats. The verifier checks if 2*x is integer.
    
    U_out = []
    for vec in U_vecs:
        U_out.append([float(x) for x in vec])
    V_out = []
    for vec in V_vecs:
        V_out.append([float(x) for x in vec])
    W_out = []
    for vec in W_vecs:
        W_out.append([float(x) for x in vec])
        
    return U_out, V_out, W_out

def _verify(U, V, W, n, m, p, rank):
    """Verify tensor decomposition exactly (for integer/half-integer entries)."""
    def scale2(mat):
        out = np.array(mat, dtype=np.float64) * 2
        if np.max(np.abs(out - np.round(out))) > 1e-9:
            return None
        return np.round(out).astype(np.int64)
    U2 = scale2(U)
    V2 = scale2(V)
    W2 = scale2(W)
    if U2 is None or V2 is None or W2 is None:
        return False
    # Build target tensor
    target = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                target[a*m + b, b*p + c, c*n + a] = 1
    T_computed = np.einsum('ic,jc,kc->ijk', U2, V2, W2, optimize=True)
    T_target_8 = target * 8
    if np.max(np.abs(T_computed - T_target_8)) > 1e-6:
        return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for a in range(n):
        for b in range(m):
            for c in range(p):
                U[a*m + b][col] = 1
                V[b*p + c][col] = 1
                W[c*n + a][col] = 1
                col += 1
    return U, V, W

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS
    
    # Try BKR
    try:
        U, V, W = _bkr_decompose(n, m, p)
        rank = len(U[0]) if U else 0
        if rank < 125: # Better than schoolbook
             if _verify(U, V, W, n, m, p, rank):
                 return {"U": U, "V": V, "W": W}
    except Exception as e:
        pass

    # Fallback to schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END
