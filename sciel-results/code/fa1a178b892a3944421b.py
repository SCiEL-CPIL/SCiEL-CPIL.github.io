"""
Rank-minimal tensor decomposition for 2x4 * 4x8 matrix multiplication.

Strategy:
  1. Start with the known rank-56 Strassen-based decomposition (8 independent
     2x2x2 sub-tensors, each handled with rank 7).
  2. Use NumPy to form the full multiplication tensor T (shape 2x4x8, all ones)
     and the factor matrices U, V, W as arrays.
  3. Attempt to reduce rank by identifying and removing linearly dependent
     rank-1 terms (columns of the Khatri-Rao product U ⊙ V ⊙ W).
  4. Use a greedy / randomised search over subsets of terms to find a
     lower-rank decomposition that still reproduces T exactly.
  5. If no improvement is found within the time budget, return the best
     verified decomposition (at worst rank 56).
"""
import time
import numpy as np
from fractions import Fraction
from itertools import combinations
from collections import defaultdict

_DIMENSIONS = (2, 4, 8)
n, m, p = _DIMENSIONS
RANK_SCHOOLBOOK = n * m * p  # 64

_DEADLINE = 135.0
_T0 = time.monotonic()


def _elapsed():
    return time.monotonic() - _T0


def _schoolbook():
    """Rank-64 schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _strassen_56():
    """
    Rank-56 decomposition: 8 independent 2x2x2 Strassen blocks.
    U: (8, 56), V: (32, 56), W: (16, 56)
    """
    F = Fraction
    rank = 56
    U = [[F(0)] * rank for _ in range(n * m)]
    V = [[F(0)] * rank for _ in range(m * p)]
    W = [[F(0)] * rank for _ in range(p * n)]

    col = 0
    for a in range(2):      # j-block index: j = 2*a + b
        for c in range(4):  # k-block index: k = 2*c + d

            # M1: (a00+a11)(b00+b11)
            U[0 * 4 + 2 * a + 0][col] += F(1)
            U[1 * 4 + 2 * a + 1][col] += F(1)
            V[(2 * a + 0) * 8 + 2 * c + 0][col] += F(1)
            V[(2 * a + 1) * 8 + 2 * c + 1][col] += F(1)
            W[4 * c + 0][col] += F(1)
            W[4 * c + 1][col] += F(-1)
            W[4 * c + 2][col] += F(1)
            W[4 * c + 3][col] += F(1)
            col += 1

            # M2: (a10+a11)b00
            U[1 * 4 + 2 * a + 0][col] += F(1)
            U[1 * 4 + 2 * a + 1][col] += F(1)
            V[(2 * a + 0) * 8 + 2 * c + 0][col] += F(1)
            W[4 * c + 1][col] += F(1)
            W[4 * c + 2][col] += F(-1)
            W[4 * c + 3][col] += F(1)
            col += 1

            # M3: a00(b10-b11)
            U[0 * 4 + 2 * a + 0][col] += F(1)
            V[(2 * a + 1) * 8 + 2 * c + 0][col] += F(1)
            V[(2 * a + 1) * 8 + 2 * c + 1][col] += F(-1)
            W[4 * c + 0][col] += F(1)
            col += 1

            # M4: a11(b01-b00)
            U[1 * 4 + 2 * a + 1][col] += F(1)
            V[(2 * a + 0) * 8 + 2 * c + 1][col] += F(1)
            V[(2 * a + 0) * 8 + 2 * c + 0][col] += F(-1)
            W[4 * c + 0][col] += F(-1)
            W[4 * c + 1][col] += F(1)
            col += 1

            # M5: (a00+a01)b11
            U[0 * 4 + 2 * a + 0][col] += F(1)
            U[0 * 4 + 2 * a + 1][col] += F(1)
            V[(2 * a + 1) * 8 + 2 * c + 1][col] += F(1)
            W[4 * c + 2][col] += F(1)
            W[4 * c + 3][col] += F(-1)
            col += 1

            # M6: (a10-a00)(b00+b01)
            U[1 * 4 + 2 * a + 0][col] += F(1)
            U[0 * 4 + 2 * a + 0][col] += F(-1)
            V[(2 * a + 0) * 8 + 2 * c + 0][col] += F(1)
            V[(2 * a + 0) * 8 + 2 * c + 1][col] += F(1)
            W[4 * c + 0][col] += F(-1)
            W[4 * c + 1][col] += F(1)
            col += 1

            # M7: (a11-a01)(b01+b11)
            U[1 * 4 + 2 * a + 1][col] += F(1)
            U[0 * 4 + 2 * a + 1][col] += F(-1)
            V[(2 * a + 0) * 8 + 2 * c + 1][col] += F(1)
            V[(2 * a + 1) * 8 + 2 * c + 1][col] += F(1)
            W[4 * c + 2][col] += F(-1)
            W[4 * c + 3][col] += F(1)
            col += 1

    assert col == 56, f"Expected 56 terms, got {col}"
    return {"U": U, "V": V, "W": W}


def _to_float_matrices(decomp):
    """Convert a decomposition dict of Fraction lists to numpy float arrays."""
    U = np.array([[float(x) for x in row] for row in decomp["U"]])
    V = np.array([[float(x) for x in row] for row in decomp["V"]])
    W = np.array([[float(x) for x in row] for row in decomp["W"]])
    return U, V, W


def _verify_numpy(U, V, W, tol=1e-9):
    """
    Verify tensor decomposition: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _verify_fraction(U, V, W):
    """Exact verification using Fractions."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True


def _try_reduce_rank(decomp, max_time=120.0):
    """
    Attempt to reduce the rank of a decomposition by removing linearly
    dependent columns from the Khatri-Rao product.
    
    Returns a potentially lower-rank decomposition, or None if no improvement.
    """
    U_f, V_f, W_f = _to_float_matrices(decomp)
    rank = U_f.shape[1]
    
    if rank <= 7:
        return None
    
    # Build the Khatri-Rao product: for each column c, the vector
    # U[:,c] ⊗ V[:,c] ⊗ W[:,c] should equal the tensor T flattened.
    # T is all-ones of shape (n, m, p).
    
    # Instead of forming the full tensor, we work with the factor matrices.
    # A column c is "redundant" if it can be expressed as a linear combination
    # of other columns in a way that preserves the tensor.
    
    # Simpler approach: try removing one column at a time and check if the
    # remaining columns still span the tensor.
    
    # For each column c, check if removing it still allows exact reconstruction.
    # We do this by checking if the tensor equation still holds.
    
    best_rank = rank
    best_indices = list(range(rank))
    
    # Greedy: try removing columns one by one
    current_indices = list(range(rank))
    
    # Sort columns by "importance" (norm of the Khatri-Rao vector)
    norms = []
    for c in range(rank):
        # Approximate norm of the rank-1 term
        u_norm = np.linalg.norm(U_f[:, c])
        v_norm = np.linalg.norm(V_f[:, c])
        w_norm = np.linalg.norm(W_f[:, c])
        norms.append(u_norm * v_norm * w_norm)
    
    # Try removing columns from least to most important
    order = sorted(range(rank), key=lambda c: norms[c])
    
    for c_remove in order:
        if _elapsed() > max_time:
            break
        
        # Skip if this column is already removed
        if c_remove not in current_indices:
            continue
        
        # Try removing this column
        new_indices = [c for c in current_indices if c != c_remove]
        
        # Check if the remaining columns still decompose the tensor
        U_sub = U_f[:, new_indices]
        V_sub = V_f[:, new_indices]
        W_sub = W_f[:, new_indices]
        
        if _verify_numpy(U_sub, V_sub, W_sub):
            current_indices = new_indices
            best_rank = len(current_indices)
            best_indices = list(current_indices)
    
    if best_rank < rank:
        # Extract the reduced decomposition
        U_red = [[U_f[i, c] for c in best_indices] for i in range(n * m)]
        V_red = [[V_f[i, c] for c in best_indices] for i in range(m * p)]
        W_red = [[W_f[i, c] for c in best_indices] for i in range(p * n)]
        
        # Convert back to Fractions if possible
        def to_frac(x):
            if abs(x - round(x)) < 1e-9:
                return Fraction(int(round(x)))
            if abs(2 * x - round(2 * x)) < 1e-9:
                return Fraction(int(round(2 * x)), 2)
            return Fraction(x)
        
        U_frac = [[to_frac(x) for x in row] for row in U_red]
        V_frac = [[to_frac(x) for x in row] for row in V_red]
        W_frac = [[to_frac(x) for x in row] for row in W_red]
        
        return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    return None


def _try_kronecker_structure(decomp):
    """
    Try to exploit Kronecker product structure to find a lower-rank decomposition.
    
    The 2x4x8 tensor can be viewed as a Kronecker product structure:
    - The 2 dimension (i) is independent
    - The 4 dimension (j) can be split into two 2-dimensions
    - The 8 dimension (k) can be split into four 2-dimensions
    
    This gives a structure where we might be able to share Strassen factors
    across the j and k dimensions.
    """
    # For now, this is a placeholder for more sophisticated structural search.
    # The main structural insight is that the 8 independent 2x2x2 blocks
    # might have shared structure if we allow cross-block terms.
    
    # Try: can we merge two Strassen blocks that share the same i-index?
    # Block (a=0, c=0) and block (a=0, c=1) both use i=0 and i=1.
    # If we allow cross-terms between c=0 and c=1, we might reduce rank.
    
    # This is complex; for now, return None and rely on the greedy reduction.
    return None


def construct():
    best = _schoolbook()
    best_rank = RANK_SCHOOLBOOK

    # Try the rank-56 Strassen decomposition
    if _elapsed() < _DEADLINE - 10:
        s2 = _strassen_56()
        if s2 is not None:
            if _verify_fraction(s2["U"], s2["V"], s2["W"]):
                r = len(s2["U"][0])
                if r < best_rank:
                    best = s2
                    best_rank = r

    # Try to reduce rank further via linear dependence elimination
    if _elapsed() < _DEADLINE - 5:
        reduced = _try_reduce_rank(best, max_time=_DEADLINE - _elapsed() - 5)
        if reduced is not None:
            if _verify_fraction(reduced["U"], reduced["V"], reduced["W"]):
                r = len(reduced["U"][0])
                if r < best_rank:
                    best = reduced
                    best_rank = r

    # Try Kronecker structure exploitation
    if _elapsed() < _DEADLINE - 3:
        kron = _try_kronecker_structure(best)
        if kron is not None:
            if _verify_fraction(kron["U"], kron["V"], kron["W"]):
                r = len(kron["U"][0])
                if r < best_rank:
                    best = kron
                    best_rank = r

    return best
