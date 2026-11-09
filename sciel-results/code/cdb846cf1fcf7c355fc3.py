"""Tensor decomposition for 3x5x5 matrix multiplication.

We search for a rank-minimal exact decomposition of the matmul tensor
T_{i,j,k} = delta_{j,k}  with dimensions n=3, m=5, p=5.

Strategy:
  1. Start from the schoolbook decomposition (rank = 75).
  2. Iteratively remove columns (terms) that are redundant or can be
     absorbed, while preserving exactness via a linear-algebra test.
  3. Use randomised rank-reduction attempts: propose removing a random
     subset of columns, then check whether the remaining columns can be
     re-scaled (or a small correction added) to still represent the tensor.
  4. If time remains, try local optimisation / Gaussian-elimination based
     sparsification to find a lower-rank representation.

The output must be a dict {'U', 'V', 'W'} with shapes (n*m, rank),
(m*p, rank), (p*n, rank).  Entries may be integer or half-integer.
"""

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import sys
import random
import copy
from fractions import Fraction

def _verify(U, V, W, n, m, p):
    """Verify that sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] == delta_{j,k}."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for t in range(rank):
                    s += Fraction(U[i*m+j][t]) * Fraction(V[j*p+k][t]) * Fraction(W[k*n+i][t])
                expected = 1 if j == k else 0
                if s != expected:
                    return False
    return True

def _rank_check(U, V, W, n, m, p):
    """Check exactness using Fraction arithmetic. Returns True if valid."""
    return _verify(U, V, W, n, m, p)

def _build_schoolbook(n, m, p):
    """Build the schoolbook (rank = n*m*p) decomposition."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _try_remove_column(U, V, W, col_idx, n, m, p):
    """
    Try to remove column col_idx by checking if it can be expressed as a
    linear combination of the other columns in a way that preserves the
    tensor. This is a necessary condition; we do a quick check by seeing
    if setting that column to zero and then checking if the tensor is still
    representable. A simpler approach: just remove it and verify.
    """
    rank = len(U[0])
    if rank <= 1:
        return None
    new_rank = rank - 1
    U2 = []
    V2 = []
    W2 = []
    for i in range(len(U)):
        U2.append(U[i][:col_idx] + U[i][col_idx+1:])
    for i in range(len(V)):
        V2.append(V[i][:col_idx] + V[i][col_idx+1:])
    for i in range(len(W)):
        W2.append(W[i][:col_idx] + W[i][col_idx+1:])
    return U2, V2, W2

def _randomize_columns(U, V, W, n, m, p, rng):
    """
    Randomly perturb the decomposition: for each column, multiply U-col, V-col,
    W-col by (a, 1/a, 1) or similar transformations that preserve the product.
    Specifically, if we scale U_col by a, V_col by b, W_col by c with a*b*c=1,
    the contribution is unchanged.
    """
    rank = len(U[0])
    U2 = copy.deepcopy(U)
    V2 = copy.deepcopy(V)
    W2 = copy.deepcopy(W)
    for t in range(rank):
        # Choose a random scaling factor from {1, -1, 2, 1/2, 3, 1/3, -2, -1/2}
        a = rng.choice([1, -1, 2, 3, 4])
        b = rng.choice([1, -1])
        # c = 1/(a*b)
        # For simplicity use a, b, c such that a*b*c = 1
        # Let's just scale U by a, V by b, W by 1/(a*b)
        # To keep integers, choose a,b such that a*b divides nicely
        pass
    # Simpler: just shuffle columns
    order = list(range(rank))
    rng.shuffle(order)
    U2 = [[U[i][order[t]] for t in range(rank)] for i in range(len(U))]
    V2 = [[V[i][order[t]] for t in range(rank)] for i in range(len(V))]
    W2 = [[W[i][order[t]] for t in range(rank)] for i in range(len(W))]
    return U2, V2, W2

def _greedy_reduce(U, V, W, n, m, p, rng, deadline):
    """
    Greedy column removal: repeatedly try to remove a random column and
    check if the remaining columns still span the tensor. We do this by
    checking if the removed column's contribution is zero (trivial) or
    by attempting to solve for the removed column as a combination of others.
    
    A more practical approach: try removing each column and verify.
    Keep the best (fewest columns) valid decomposition found.
    """
    best_U, best_V, best_W = copy.deepcopy(U), copy.deepcopy(V), copy.deepcopy(W)
    best_rank = len(best_U[0])
    
    current_U, current_V, current_W = copy.deepcopy(U), copy.deepcopy(V), copy.deepcopy(W)
    current_rank = len(current_U[0])
    
    while current_rank > 1 and time.time() < deadline:
        removed = False
        # Try removing random columns
        cols_to_try = list(range(current_rank))
        rng.shuffle(cols_to_try)
        for col_idx in cols_to_try:
            if time.time() > deadline:
                break
            new_U, new_V, new_W = _try_remove_column(current_U, current_V, current_W, col_idx, n, m, p)
            if new_U is None:
                continue
            # Verify the new decomposition
            if _rank_check(new_U, new_V, new_W, n, m, p):
                current_U, current_V, current_W = new_U, new_V, new_W
                current_rank -= 1
                removed = True
                # Update best
                best_U, best_V, best_W = copy.deepcopy(current_U), copy.deepcopy(current_V), copy.deepcopy(current_W)
                best_rank = current_rank
                break
        if not removed:
            # Try random reordering and retry
            current_U, current_V, current_W = _randomize_columns(current_U, current_V, current_W, n, m, p, rng)
            if time.time() > deadline - 2:
                break
    
    return best_U, best_V, best_W, best_rank

def construct():
    """Return a rank-minimal exact tensor decomposition for 3x5x5 matmul."""
    start_time = time.time()
    deadline = start_time + 115  # 5 second safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rng = random.Random(42)
    
    # Start with schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    print(f"Schoolbook rank: {len(U[0])}", file=sys.stderr)
    
    # Verify schoolbook is correct
    assert _rank_check(U, V, W, n, m, p), "Schoolbook decomposition failed verification"
    
    best_rank = len(U[0])
    best_U, best_V, best_W = copy.deepcopy(U), copy.deepcopy(V), copy.deepcopy(W)
    
    # Try greedy reduction
    gU, gV, gW, g_rank = _greedy_reduce(U, V, W, n, m, p, rng, deadline)
    if g_rank < best_rank:
        best_rank = g_rank
        best_U, best_V, best_W = gU, gV, gW
        print(f"Greedy reduction achieved rank: {best_rank}", file=sys.stderr)
    
    # Try multiple random restarts
    restarts = 0
    while time.time() < deadline - 5:
        restarts += 1
        # Create a randomised variant
        rU, rV, rW = _randomize_columns(best_U, best_V, best_W, n, m, p, rng)
        if not _rank_check(rU, rV, rW, n, m, p):
            continue
        # Try to reduce
        gU, gV, gW, g_rank = _greedy_reduce(rU, rV, rW, n, m, p, rng, deadline - 5)
        if g_rank < best_rank:
            best_rank = g_rank
            best_U, best_V, best_W = gU, gV, gW
            print(f"Restart {restarts} achieved rank: {best_rank}", file=sys.stderr)
    
    print(f"Final best rank: {best_rank} (restarted {restarts} times)", file=sys.stderr)
    
    # Final verification
    assert _rank_check(best_U, best_V, best_W, n, m, p), "Final decomposition failed verification"
    
    result = {"U": best_U, "V": best_V, "W": best_W}
    return result
# EVOLVE-BLOCK-END
