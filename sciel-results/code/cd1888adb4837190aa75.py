"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
import time
import math

def _build_strassen():
    """Return the verified rank-7 Strassen decomposition."""
    n, m, p = DIMENSIONS
    r = 7
    U = [[0.0] * r for _ in range(4)]
    V = [[0.0] * r for _ in range(4)]
    W = [[0.0] * r for _ in range(4)]

    # M1 = (A00+A11)*(B00+B11)
    U[0][0] = 1.0; U[3][0] = 1.0
    V[0][0] = 1.0; V[3][0] = 1.0
    W[0][0] = 1.0; W[3][0] = 1.0

    # M2 = (A10+A11)*B00
    U[2][1] = 1.0; U[3][1] = 1.0
    V[0][1] = 1.0
    W[1][1] = 1.0; W[3][1] = -1.0

    # M3 = A00*(B01-B11)
    U[0][2] = 1.0
    V[1][2] = 1.0; V[3][2] = -1.0
    W[2][2] = 1.0; W[3][2] = 1.0

    # M4 = A11*(B10-B00)
    U[3][3] = 1.0
    V[2][3] = 1.0; V[0][3] = -1.0
    W[0][3] = 1.0; W[1][3] = 1.0

    # M5 = (A00+A01)*B11
    U[0][4] = 1.0; U[1][4] = 1.0
    V[3][4] = 1.0
    W[0][4] = -1.0; W[2][4] = 1.0

    # M6 = (A10-A00)*(B00+B01)
    U[2][5] = 1.0; U[0][5] = -1.0
    V[0][5] = 1.0; V[1][5] = 1.0
    W[3][5] = 1.0

    # M7 = (A01-A11)*(B10+B11)
    U[1][6] = 1.0; U[3][6] = -1.0
    V[2][6] = 1.0; V[3][6] = 1.0
    W[0][6] = 1.0

    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition exactly reproduces matrix multiplication.
    Returns True if all entries match, False otherwise.
    """
    # The tensor for matrix multiplication has indices (i, k, j) -> C[i,j] += A[i,k]*B[k,j]
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition says: sum_k U[i*m+k, s] * V[k*p+j, s] * W[j*n+i, s] = delta[i,k,j]
    # Wait, the convention is W[k*n+i, r] per the task description?
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # Let's re-read carefully: "The output factor follows the official trace convention W[k*n+i,r]."
    # This implies the index of W is (k*n + i).
    # In the seed code:
    # U rows: 0=A00, 1=A01, 2=A10, 3=A11  -> row = i*m + k
    # V rows: 0=B00, 1=B01, 2=B10, 3=B11  -> row = k*p + j
    # W rows: Mapped to output C elements.
    #   W[0] -> C00, W[1] -> C10, W[2] -> C01, W[3] -> C11
    # If W index is k*n + i:
    # C00 -> i=0, j=0. Sum over k. Term is U[0*m+k, s] V[k*p+0, s] W[0*n+0, s].
    # C10 -> i=1, j=0. Sum over k. Term is U[1*m+k, s] V[k*p+0, s] W[0*n+1, s].
    # C01 -> i=0, j=1. Sum over k. Term is U[0*m+k, s] V[k*p+1, s] W[1*n+0, s].
    # C11 -> i=1, j=1. Sum over k. Term is U[1*m+k, s] V[k*p+1, s] W[1*n+1, s].
    
    # So W index w = j*n + i? Or k*n + i?
    # The task says W[k*n+i]. But k is not in the output C. The output is C[i,j].
    # Usually, the third factor in matrix multiplication tensor decomposition is indexed by the output index.
    # If the tensor is T_{ijk}, and we decompose T_{ijk} = sum_s U_{is} V_{js} W_{ks}, then the order is different.
    # Standard form: C_{ij} = sum_k A_{ik} B_{kj}.
    # Tensor T_{i,k,j} = delta... no, T is the multiplication tensor.
    # T_{i,k,j} contributes to C_{i,j}.
    # Decomposition: T_{i,k,j} = sum_s U_{i,k,s} V_{k,j,s} W_{j,i,s}?
    # Let's check the seed code's W mapping again.
    # Seed: W[0] -> C00, W[1] -> C10, W[2] -> C01, W[3] -> C11.
    # If W index is j*n + i:
    # C00 (i=0,j=0) -> idx 0.
    # C10 (i=1,j=0) -> idx 0*2 + 1 = 1.
    # C01 (i=0,j=1) -> idx 1*2 + 0 = 2.
    # C11 (i=1,j=1) -> idx 1*2 + 1 = 3.
    # This matches the seed's comment: W[0]->C00, W[1]->C10, W[2]->C01, W[3]->C11.
    # So W index is j*n + i.
    # The task text says "W[k*n+i]". This might be a typo in the task description or "k" is used as the first index of the output pair in some convention. Given the seed code works (score 1.14), I will trust the seed code's mapping: W index = j*n + i.

    rank = len(U[0])
    # Check all 8 tensor entries (i, k, j) for n=m=p=2.
    # The tensor T_{i,k,j} should equal 1 if it's a valid "multiplication" tensor?
    # Actually, the tensor is defined such that C_{ij} = sum_k A_{ik} B_{kj}.
    # The decomposition is: sum_s U[i,k,s] V[k,j,s] W[j,i,s] = T[i,k,j].
    # For matrix multiplication, T[i,k,j] is 1 if the indices correspond to a valid term A[i,k]*B[k,j] contributing to C[i,j].
    # Wait, the tensor is fixed. It's the structure of matrix multiplication.
    # T[i,k,j] = 1 for all i,k,j? No.
    # The tensor T has dimensions (n*m) x (m*p) x (p*n).
    # T[i*m+k, k*p+j, j*n+i] = 1.
    # All other entries are 0.
    # So we check: for all i, k, j:
    #   sum_s U[i*m+k, s] * V[k*p+j, s] * W[j*n+i, s] == 1
    # And for all other (row_U, row_V, row_W) not corresponding to (i*m+k, k*p+j, j*n+i):
    #   sum_s ... == 0
    
    for i in range(n):
        for k in range(m):
            for j in range(p):
                # Valid entry
                idx_u = i * m + k
                idx_v = k * p + j
                idx_w = j * n + i
                val = 0.0
                for s in range(rank):
                    val += U[idx_u][s] * V[idx_v][s] * W[idx_w][s]
                if abs(val - 1.0) > 1e-9:
                    return False
                
                # Invalid entries (just a few checks to save time, or all?)
                # For small dims, checking all is cheap.
                # Total entries: (n*m)*(m*p)*(p*n) = 4*4*4 = 64.
                # Valid: 8. Invalid: 56.
    # Check invalid entries
    valid_set = set()
    for i in range(n):
        for k in range(m):
            for j in range(p):
                valid_set.add((i*m+k, k*p+j, j*n+i))
    
    for ru in range(n*m):
        for rv in range(m*p):
            for rw in range(p*n):
                if (ru, rv, rw) not in valid_set:
                    val = 0.0
                    for s in range(rank):
                        val += U[ru][s] * V[rv][s] * W[rw][s]
                    if abs(val) > 1e-9:
                        return False
    return True

# Rank 6 is theoretically impossible for 2x2x2 matrix multiplication.
# No search is performed.

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s safety margin
    
    # Rank 6 is impossible. We proceed directly to the optimal Rank 7 Strassen decomposition.
    res7 = _build_strassen()
    
    # Verify Strassen (should always pass)
    if _verify_decomposition(res7['U'], res7['V'], res7['W'], 2, 2, 2):
        return res7
    else:
        # This should never happen
        raise RuntimeError("Strassen decomposition verification failed")

# EVOLVE-BLOCK-END
