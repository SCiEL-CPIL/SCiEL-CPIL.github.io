"""Rewritten matrix multiplication 2x3 x 3x10 tensor decomposition program."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 10
    total = 55

    # Pre-allocate dense factor matrices
    U = [[0] * total for _ in range(n * m)]
    V = [[0] * total for _ in range(m * p)]
    W = [[0] * total for _ in range(p * n)]

    # Build the rank-55 decomposition in a single pass over 10 output columns.
    # For each output column j = 0..9 we emit five tensor slices.
    # The first three slices are shared across all columns (rank 3 * 5 = 15).
    # The last two slices are column-specific (rank 2 * 10 = 20).
    # Total = 35, but we need 55. Let me recount.

    # Actually, let me use the known rank-55 decomposition pattern.
    # I'll construct it directly from the bilinear form structure.

    # For a 2x3 * 3x10 multiplication, C[i,j] = sum_k A[i,k]*B[k,j]
    # We need to decompose the multiplication tensor of rank 55.

    # Using a structured approach:
    # Group the 10 columns of B (and corresponding C columns) into 5 pairs.
    # For each pair (2j, 2j+1), we use 11 rank-1 tensors.

    idx = 0
    for j in range(5):
        k0 = 2 * j
        k1 = 2 * j + 1

        # Slice 1: (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1]) -> C[0,k0]+C[1,k1]
        U[0][idx] += 1  # A[0,0]
        U[4][idx] += 1  # A[1,1]
        V[k0][idx] += 1    # B[0,k0]
        V[10+k1][idx] += 1 # B[1,k1]
        W[k0][idx] += 1    # C[0,k0]
        W[2+k1][idx] += 1  # C[1,k1]
        idx += 1

        # Slice 2: (A[1,0]+A[1,1]) * B[0,k0] -> C[1,k0] - C[1,k1]
        U[3][idx] += 1  # A[1,0]
        U[4][idx] += 1  # A[1,1]
        V[k0][idx] += 1    # B[0,k0]
        W[2+k0][idx] += 1  # C[1,k0]
        W[2+k1][idx] -= 1  # C[1,k1]
        idx += 1

        # Slice 3: A[0,0] * (B[0,k1]-B[1,k1]) -> C[0,k1]+C[1,k1]
        U[0][idx] += 1  # A[0,0]
        V[10+k1][idx] += 1  # B[1,k1]
        V[k1][idx] -= 1     # B[0,k1]
        W[k1][idx] += 1     # C[0,k1]
        W[2+k1][idx] += 1   # C[1,k1]
        idx += 1

        # Slice 4: A[1,1] * (B[1,k0]-B[0,k0]) -> C[0,k0]+C[1,k0]
        U[4][idx] += 1  # A[1,1]
        V[10+k0][idx] += 1  # B[1,k0]
        V[k0][idx] -= 1     # B[0,k0]
        W[k0][idx] += 1     # C[0,k0]
        W[2+k0][idx] += 1   # C[1,k0]
        idx += 1

        # Slice 5: (A[0,0]+A[0,1]) * B[1,k1] -> -C[0,k0]+C[0,k1]
        U[0][idx] += 1  # A[0,0]
        U[1][idx] += 1  # A[0,1]
        V[10+k1][idx] += 1  # B[1,k1]
        W[k0][idx] -= 1     # C[0,k0]
        W[k1][idx] += 1     # C[0,k1]
        idx += 1

        # Slice 6: (A[1,0]-A[0,0]) * (B[0,k0]+B[0,k1]) -> C[1,k1]
        U[3][idx] += 1   # A[1,0]
        U[0][idx] -= 1   # A[0,0]
        V[k0][idx] += 1  # B[0,k0]
        V[k1][idx] += 1  # B[0,k1]
        W[2+k1][idx] += 1  # C[1,k1]
        idx += 1

        # Slice 7: (A[0,1]-A[1,1]) * (B[1,k0]+B[1,k1]) -> C[0,k0]
        U[1][idx] += 1   # A[0,1]
        U[4][idx] -= 1   # A[1,1]
        V[10+k0][idx] += 1  # B[1,k0]
        V[10+k1][idx] += 1  # B[1,k1]
        W[k0][idx] += 1  # C[0,k0]
        idx += 1

        # Slice 8: A[0,2] * B[2,k0] -> C[0,k0]
        U[2][idx] += 1   # A[0,2]
        V[20+k0][idx] += 1  # B[2,k0]
        W[k0][idx] += 1  # C[0,k0]
        idx += 1

        # Slice 9: A[0,2] * B[2,k1] -> C[0,k1]
        U[2][idx] += 1   # A[0,2]
        V[20+k1][idx] += 1  # B[2,k1]
        W[k1][idx] += 1  # C[0,k1]
        idx += 1

        # Slice 10: A[1,2] * B[2,k0] -> C[1,k0]
        U[5][idx] += 1   # A[1,2]
        V[20+k0][idx] += 1  # B[2,k0]
        W[2+k0][idx] += 1  # C[1,k0]
        idx += 1

        # Slice 11: A[1,2] * B[2,k1] -> C[1,k1]
        U[5][idx] += 1   # A[1,2]
        V[20+k1][idx] += 1  # B[2,k1]
        W[2+k1][idx] += 1  # C[1,k1]
        idx += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
