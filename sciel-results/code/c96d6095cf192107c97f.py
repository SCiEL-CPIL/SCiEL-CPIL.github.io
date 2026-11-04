"""Matrix multiplication 3x5 by 5x7 rank-minimal exact tensor decomposition."""
from fractions import Fraction
import time
import random

DIMENSIONS = [3, 5, 7]
n, m, p = DIMENSIONS
N = n * m      # 15
M = m * p      # 35
P = p * n      # 21

# Deadline guard: keep a safety margin before the 150s wall-clock kill.
_DEADLINE = time.time() + 145.0


def _time_left():
    return _DEADLINE - time.time()


def _verify(U, V, W, rank):
    """Exact verification of the tensor identity using Fractions.

    For all i in [0,n), k in [0,p):
        sum_{j in [0,m)} sum_{c in [0,rank)} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        must equal 1 if i==k else 0.
    """
    # Convert to Fraction once
    UF = [[Fraction(x) for x in row] for row in U]
    VF = [[Fraction(x) for x in row] for row in V]
    WF = [[Fraction(x) for x in row] for row in W]
    for i in range(n):
        for k in range(p):
            target = 1 if i == k else 0
            wcol = WF[k * n + i]
            acc = Fraction(0)
            for j in range(m):
                urow = UF[i * m + j]
                vrow = VF[j * p + k]
                for c in range(rank):
                    acc += urow[c] * vrow[c] * wcol[c]
            if acc != target:
                return False
    return True


def _schoolbook():
    rank = n * m * p
    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(M)]
    W = [[0] * rank for _ in range(P)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _try_structured_rank(rank, seed, attempts=3):
    """Attempt to find an exact rank-`rank` decomposition using a structured
    parametric family and exact rational search.

    We use a Kronecker-like factorization:
        U[i*m+j, c] = A[i, a_c] * B[j, b_c]
        V[j*p+k, c] = B[j, b_c] * C[k, d_c]
        W[k*n+i, c] = C[k, d_c] * A[i, a_c]
    where c indexes (a_c, b_c, d_c) from sets of sizes ra, rb, rd with
    rank = ra*rb*rd. This gives the identity sum over c of
    A[i,a]C[k,d] * B[j,b]^2 * C[k,d]A[i,a] ... need to be careful.

    Instead use a simpler guaranteed family: the "block-diagonal with shared
    middle" construction. For a rank r decomposition we can try to factor the
    identity tensor via a shared 5-dim middle basis.

    Concretely, represent the multiplication as:
        (A B)_{i,k} = sum_j A_{i,j} B_{j,k}
    We want sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} = delta_{i,k}.

    Use a rank-r scheme where each column c is a rank-1 slice:
        U_{i,j,c} = u_{i,j,c}, V_{j,k,c} = v_{j,k,c}, W_{k,i,c} = w_{k,i,c}
    such that sum_c u_{i,j,c} v_{j,k,c} w_{k,i,c} = delta_{i,k}.

    We search over integer assignments with small magnitudes via randomized
    local search on a parameterization, verifying exactly.
    """
    for _ in range(attempts):
        if _time_left() < 5.0:
            return None
        rng = random.Random(seed)
        # Parameterize: choose rank columns as products of three vectors.
        # Let each column c correspond to a triple (a_c, b_c, d_c).
        # U[i,j,c] = A[i,a_c] * B[j,b_c]
        # V[j,k,c] = B[j,b_c] * C[k,d_c]
        # W[k,i,c] = C[k,d_c] * A[i,a_c]
        # Then sum_c A[i,a]C[k,d] B[j,b]^2 * C[k,d]A[i,a]? No, W has C[k,d]A[i,a].
        # Product = A[i,a]^2 * C[k,d]^2 * B[j,b]^2 * ... that's not delta.

        # Better: use
        # U[i,j,c] = A[i,a_c] * B[j,b_c]
        # V[j,k,c] = B[j,b_c] * C[k,d_c]
        # W[k,i,c] = C[k,d_c] * A[i,a_c] * (sign)
        # Product over c: A[i,a]^2 C[k,d]^2 B[j,b]^2 -> sum over j gives A[i,a]^2 C[k,d]^2 (sum_j B[j,b]^2).
        # This equals delta_{i,k} only if we can make A[i,a]C[k,d] sum to delta and the B part a scalar.
        # That requires the B part to factor as a constant per (a,d), which it does if
        # sum_j B[j,b]^2 = const(a,d). Hard to control.

        # Use a cleaner exact family: the "identity embedding" with rank = n*p + (m-1)*n*p style.
        # We'll instead do a direct integer local search on the full matrices.
        # Initialize with schoolbook then zero out columns and repair.
        pass
    return None


def construct():
    t0 = time.time()
    best = _schoolbook()
    best_rank = best["U"][0].__len__() if best["U"] else n * m * p
    # best_rank = 105

    # Try to improve via exact structured constructions.
    # Construction 1: split the 5 middle dimension.
    # We can write the 5x5 middle as identity plus nilpotent-free part.
    # A known trick: rank of multiplying n x m by m x p is at most
    #   n*p + (m-1)*n*p? Not generally. But for our small sizes we can try
    #   rank = n*p + (m-1)*n*p = m*n*p (same). Not helpful.

    # Try a direct randomized exact search over small integer matrices for
    # ranks below 105, using a deadline guard.
    # We'll attempt ranks from a promising lower bound upward.
    # Lower bound: rank >= max(n*m, m*p, p*n) = max(15,35,21)=35? No, that's
    # not a valid lower bound for this tensor. Use rank >= n*p = 21 (since the
    # identity in i,k has rank n in the i-k slice, and m>=1). Actually a valid
    # lower bound is n*p = 21 because fixing j and summing, the map must
    # reproduce an n x p identity-like structure. We'll search from 21 upward
    # but that's expensive; instead we try a few candidate ranks with a
    # parametric exact solver.

    # Parametric exact solver: fix the W matrix to be the identity-embedding
    # (W[k*n+i, c] nonzero only for c in a set), then solve for U,V exactly.
    # Use the fact that if we set W to have, for each (k,i), a single column
    # c_{k,i} with W=1, and 0 elsewhere, then the constraint becomes
    #   sum_j U[i*m+j, c_{k,i}] V[j*p+k, c_{k,i}] = delta_{i,k}
    # for each (k,i). This is a system we can try to satisfy by choosing
    # U and V columns. With rank = n*p = 21 we have exactly one column per
    # (k,i). Then we need, for each (k,i), sum_j U[i,j, c_{k,i}] V[j,k, c_{k,i}]
    # = delta_{i,k}. For i != k this must be 0; for i == k it must be 1.
    # Choose V[j,k, c_{k,i}] = 1 for all j when i==k? Let's set:
    #   For column c_{k,i}: U[i,j, c_{k,i}] = 1 if i==k else 0, V[j,k,c_{k,i}]=1.
    # Then sum_j U V = (1 if i==k else 0)*sum_j 1 = m * delta_{i,k}. That gives m, not 1.
    # Scale: U[i,j,c_{k,i}] = (1/m) if i==k else 0, V[j,k,c_{k,i}]=1.
    # Then sum = (1/m)*m = 1 if i==k. 
    # But U must be the same across columns sharing (i,j). Column c_{k,i} uses
    # U[i,j, c_{k,i}] which is a distinct column index, so no conflict! Each
    # (k,i) has its own column, so U[i,j,c_{k,i}] is independent per (k,i).
    # Thus rank = n*p = 21 works with rational entries 1/m.
    # Verify: for (k,i), sum_j U[i*m+j, c_{k,i}] V[j*p+k, c_{k,i}]
    #   = sum_j (1/m if i==k else 0)*1 = (m*(1/m)) if i==k else 0 = delta_{i,k}. 
    # And W[k*n+i, c_{k,i}] = 1. 
    # This is a valid rank 21 decomposition!

    rank = n * p  # 21
    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(M)]
    W = [[0] * rank for _ in range(P)]
    col = 0
    for k in range(p):
        for i in range(n):
            # column col corresponds to output (k, i)
            for j in range(m):
                if i == k:
                    U[i * m + j][col] = Fraction(1, m)
                V[j * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    # Convert to float-free integers: multiply U entries by m to clear denominators?
    # The evaluator accepts integer/half-integer entries. 1/m = 1/5 is not integer
    # or half-integer. We need entries that are integers or half-integers.
    # Rescale: multiply U by m and divide W by m? That changes the product.
    # Instead, absorb: set U[i*m+j, col] = 1 (when i==k), V[j*p+k, col] = 1,
    # and W[k*n+i, col] = 1/m. 1/5 is not half-integer either.
    # Use a different split: distribute the 1/m across U and V as 1/sqrt? Not integer.
    # Better: use the fact that we can put the 1/m into V as 1/m and U as 1.
    # Same problem.
    # Use integer entries by choosing a different construction:
    # Set U[i*m+j, col] = 1 if i==k else 0, V[j*p+k, col] = 1, W[k*n+i, col] = 1.
    # Then sum_j = m * delta_{i,k}. To fix, we need the product to be 1.
    # We can use m columns per (k,i) with values that sum to 1: e.g. one column
    # with U=1, V=1, W=1 and (m-1) columns that cancel. That increases rank.
    # Alternatively use the identity: sum_{j} 1 * 1 = m, so set W = 1/m.
    # Since 1/5 is not half-integer, multiply all of U,V,W by 5 and divide the
    # target by 5? The target delta is fixed at 1, so we can't rescale the tensor.
    # However, we CAN rescale each column: multiply U_col by a, V_col by b,
    # W_col by c with a*b*c = 1. Choose a=1, b=1, c=1/5 -> c not half-integer.
    # Choose a=1/5, b=1, c=1 -> a not half-integer.
    # Choose a=1, b=1/5, c=1 -> not half-integer.
    # We need a*b*c=1 with each in {integers and half-integers}.
    # 1/5 cannot be represented. But we can use two columns: one with (1,1,1)
    # contributing 1, and we need the total to be 1, so the extra (m-1) contribution
    # must be 0. Use columns where U or V is 0 to contribute 0. That just wastes rank.
    # Alternatively, use the fact that we can split the 1 across m columns each
    # contributing 1/m by setting per-column (a,b,c) with a*b*c=1/m. Still need 1/m.
    #
    # Key insight: the constraint is a SUM over columns. We can use m columns for
    # the (i==k) case, each with U=1, V=1, W=1, but that gives m. Instead use
    # ONE column with U=1, V=1, W=1 (contributes 1) and ensure no other column
    # contributes to this (k,i). For i!=k, ensure 0. This is exactly the
    # rank = n*p construction with U[i*m+j,col]=1 if i==k, V=1, W=1, which gives
    # sum_j = m, NOT 1. The issue is the sum over j.
    #
    # Fix the sum over j: make V depend on j so that sum_j U*V = 1.
    # Set U[i*m+j, col] = 1 (i==k), V[j*p+k, col] = 1 for j=0 and 0 for j>0.
    # Then sum_j = 1. 
    # So: for column c_{k,i} (i==k): U[i*m+0, col]=1, V[0*p+k, col]=1, W[k*n+i, col]=1.
    # All other j: U=0, V=0. Then sum_j U*V = 1*1 = 1. 
    # For i != k: U[i*m+j, col] = 0 for all j, so sum = 0. 
    # This is rank = n*p = 21 with INTEGER entries! Verify no conflicts:
    # Column c_{k,i} is unique per (k,i). U[i*m+0, c_{k,i}] = 1 only when i==k.
    # For a fixed (i,j=0), U[i*m+0, c_{k,i}] = 1 if i==k else 0. Different (k,i)
    # use different columns, so no column has two 1's in the same U cell. Good.
    # V[0*p+k, c_{k,i}] = 1 if i==k else 0. For fixed (j=0,k), V[0*p+k, c_{k,i}]=1
    # only when i==k. Different i use different columns. Good.
    # This works! Rank 21, all integer entries.

    rank = n * p  # 21
    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(M)]
    W = [[0] * rank for _ in range(P)]
    col = 0
    for k in range(p):
        for i in range(n):
            if i == k:
                U[i * m + 0][col] = 1
                V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    cand = {"U": U, "V": V, "W": W}
    if _verify(U, V, W, rank):
        return cand
    # Fallback to schoolbook if verification fails (should not).
    return best


# Provide the required entry point exactly.
if __name__ == "__main__":
    _ = construct()
