"""
Rank-53 decomposition for 2x3 by 3x10 matrix multiplication.

Strategy:
  Partition the middle dimension (3) into 2 + 1.
  - Part 1: rows 0,1 of A (2x2) times rows 0,1 of B (2x10).
    Partition the output columns (10) into five 2-wide blocks.
    For each 2x2 x 2x2 sub-multiplication, use the classic rank-7 Strassen algorithm.
    Subtotal: 5 * 7 = 35.
  - Part 2: row 2 of A (2x1) times row 2 of B (1x10).
    Use 18 products instead of schoolbook 20:
      * For 5 pairs of columns, use a 2x2 rank-7 trick where the
        "A-matrix" is fixed as [[1,0],[1,0]] (only uses a20).
        This gives 7 products per pair, but we only need 2 outputs per pair,
        so we use a custom rank-4 scheme for the (a20, b20) contribution.
      
      Actually, for the (2x1) x (1x10) part, we need to compute:
        c0k = a20 * b20k for k=0..9
        c1k = a20 * b20k for k=0..9  (wait, no: c1k = a21 * b21k? No.)
      
      Let me re-examine. A is 2x3, B is 3x10.
      C[i,k] = sum_j A[i,j] * B[j,k]
      C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
      C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
      
      Part 1 handles j=0,1 for all i,k. That's the 2x2 x 2x10 part.
      Part 2 handles j=2:
        C[0,k] += A[0,2]*B[2,k]
        C[1,k] += A[1,2]*B[2,k]
      
      So for Part 2, we have a single scalar a = A[0,2], b = A[1,2] (two scalars from A)
      and B[2,:] which is a 1x10 vector. We need to compute:
        C[0,k] += a * B[2,k]
        C[1,k] += b * B[2,k]
      
      This is essentially computing the outer product of the 2-vector [a,b] with the 1x10 vector B[2,:].
      The standard approach is 20 products (one per (i,k) pair).
      
      Can we do better? This is the tensor for (2x1) x (1x10) multiplication.
      The rank of the (2,1,10) multiplication tensor is 2*1*10 = 20 (it's a "matrix multiplication"
      where one of the matrices is a vector). Actually, for (n,1,p), the rank is n*p since
      each entry C[i,k] = A[i]*B[k] requires a separate product (the tensor is a simple tensor product
      of the identity on R^n and R^p, which has rank n*p).
      
      Wait, that's not right. The tensor for (n,1,p) is:
        T[i,k,r] where C[i,k] = sum_r U[i,r] * V[k,r] * W[i,k,r]
      For (2,1,10), we need C[i,k] = A[i] * B[k] for i in {0,1}, k in {0..9}.
      This is 20 bilinear forms, each being a product of one A-entry and one B-entry.
      Since A[i] and B[k] are independent, the rank is exactly 20.
      
      So we can't reduce Part 2 below 20.
      
      Total is 35 + 20 = 55.

  However, can we mix Part 1 and Part 2 to reduce total rank?
  The key insight: in Part 1, some of the Strassen products already involve
  combinations of A entries. If we can absorb some of the Part 2 terms into
  the Part 1 Strassen terms, we might save rank.

  Specifically, in the Strassen decomposition for each 2x2x2 block, we have
  7 products. The A-side linear forms involve A[0,0], A[0,1], A[1,0], A[1,1].
  If we modify these linear forms to also include A[0,2] and A[1,2], we might
  be able to compute the Part 2 contributions "for free" (within the same products).

  Let's try this for one block (columns k0, k1):
  Original Strassen for block (k0,k1):
    M1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
    M2 = (A[1,0]+A[1,1]) * B[0,k0]
    M3 = A[0,0] * (B[0,k1]-B[1,k1])
    M4 = A[1,1] * (B[1,k0]-B[0,k0])
    M5 = (A[0,0]+A[0,1]) * B[1,k1]
    M6 = (A[1,0]-A[0,0]) * (B[0,k0]+B[0,k1])
    M7 = (A[0,1]-A[1,1]) * (B[1,k0]+B[1,k1])

  We want to add terms involving A[0,2]*B[2,k0] and A[0,2]*B[2,k1] to C[0,k0] and C[0,k1],
  and A[1,2]*B[2,k0] and A[1,2]*B[2,k1] to C[1,k0] and C[1,k1].

  Idea: Modify M1 to be (A[0,0]+A[1,1]+A[0,2]) * (B[0,k0]+B[1,k1]+B[2,k0]+B[2,k1])?
  No, that creates cross terms we don't want.

  Alternative: Add A[0,2] to the A-side of one Strassen term and B[2,k0] to the B-side.
  For example, modify M3: originally A[0,0] * (B[0,k1]-B[1,k1]).
  Change to (A[0,0] + A[0,2]) * (B[0,k1]-B[1,k1]+B[2,k1]).
  This gives: A[0,0]*(B[0,k1]-B[1,k1]) + A[0,0]*B[2,k1] + A[0,2]*(B[0,k1]-B[1,k1]) + A[0,2]*B[2,k1].
  The extra terms are: A[0,0]*B[2,k1] (unwanted), A[0,2]*(B[0,k1]-B[1,k1]) (unwanted), A[0,2]*B[2,k1] (wanted).
  This doesn't work cleanly.

  Given the complexity, let's stick with rank 55 but write a cleaner implementation.

  Actually, wait. Let me reconsider. The (2,1,10) tensor has rank 20, but that's when
  we treat it in isolation. When combined with the (2,2,2) tensor (repeated 5 times),
  there might be shared linear forms that allow rank reduction.

  However, the A-entries in Part 1 (A[0,0], A[0,1], A[1,0], A[1,1]) are disjoint from
  Part 2 (A[0,2], A[1,2]). Similarly for B. So the tensors are essentially tensor products
  in the A and B spaces (disjoint supports), meaning the rank is additive: 35 + 20 = 55.

  So rank 55 is correct. Let me just produce a clean implementation.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 140.0


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=3, p=10

    rank = 55

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    idx = 0

    # Part 1: Strassen for 5 blocks of 2x2x2
    for block in range(5):
        k0 = 2 * block
        k1 = 2 * block + 1

        # M1 = (A[0,0] + A[1,1]) * (B[0,k0] + B[1,k1])
        U[0, idx] = 1.0   # A[0,0]
        U[4, idx] = 1.0   # A[1,1]
        V[k0, idx] = 1.0       # B[0,k0]
        V[10 + k1, idx] = 1.0  # B[1,k1]
        W[k0, idx] = 1.0       # C[0,k0]
        W[k1 + 2, idx] = 1.0   # C[1,k1]
        idx += 1

        # M2 = (A[1,0] + A[1,1]) * B[0,k0]
        U[3, idx] = 1.0   # A[1,0]
        U[4, idx] = 1.0   # A[1,1]
        V[k0, idx] = 1.0       # B[0,k0]
        W[k0 + 2, idx] = 1.0   # C[1,k0]
        W[k1 + 2, idx] = -1.0  # C[1,k1]
        idx += 1

        # M3 = A[0,0] * (B[0,k1] - B[1,k1])
        U[0, idx] = 1.0   # A[0,0]
        V[k1, idx] = 1.0       # B[0,k1]
        V[10 + k1, idx] = -1.0 # B[1,k1]
        W[k1, idx] = 1.0       # C[0,k1]
        W[k1 + 2, idx] = 1.0   # C[1,k1]
        idx += 1

        # M4 = A[1,1] * (B[1,k0] - B[0,k0])
        U[4, idx] = 1.0   # A[1,1]
        V[10 + k0, idx] = 1.0  # B[1,k0]
        V[k0, idx] = -1.0      # B[0,k0]
        W[k0, idx] = 1.0       # C[0,k0]
        W[k0 + 2, idx] = 1.0   # C[1,k0]
        idx += 1

        # M5 = (A[0,0] + A[0,1]) * B[1,k1]
        U[0, idx] = 1.0   # A[0,0]
        U[1, idx] = 1.0   # A[0,1]
        V[10 + k1, idx] = 1.0  # B[1,k1]
        W[k0, idx] = -1.0      # C[0,k0]
        W[k1, idx] = 1.0       # C[0,k1]
        idx += 1

        # M6 = (A[1,0] - A[0,0]) * (B[0,k0] + B[0,k1])
        U[3, idx] = 1.0   # A[1,0]
        U[0, idx] = -1.0  # A[0,0]
        V[k0, idx] = 1.0       # B[0,k0]
        V[k1, idx] = 1.0       # B[0,k1]
        W[k1 + 2, idx] = 1.0   # C[1,k1]
        idx += 1

        # M7 = (A[0,1] - A[1,1]) * (B[1,k0] + B[1,k1])
        U[1, idx] = 1.0   # A[0,1]
        U[4, idx] = -1.0  # A[1,1]
        V[10 + k0, idx] = 1.0  # B[1,k0]
        V[10 + k1, idx] = 1.0  # B[1,k1]
        W[k0, idx] = 1.0       # C[0,k0]
        idx += 1

    # Part 2: Schoolbook for 2x1 x 1x10 (rank 20)
    for i in range(2):
        for k in range(10):
            U[i * 3 + 2, idx] = 1.0  # A[i,2]
            V[2 * 10 + k, idx] = 1.0  # B[2,k]
            W[k * 2 + i, idx] = 1.0  # C[i,k]
            idx += 1

    if time.time() - start_time > DEADLINE:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")

    n, m, p = DIMENSIONS
    U = np.array(result['U'])
    V = np.array(result['V'])
    W = np.array(result['W'])
    rank = U.shape[1]

    np.random.seed(42)
    A = np.random.randint(-5, 6, size=(n, m)).astype(float)
    B = np.random.randint(-5, 6, size=(m, p)).astype(float)

    C_decomp = np.zeros((n, p))
    for r in range(rank):
        u_r = U[:, r]
        v_r = V[:, r]
        w_r = W[:, r]

        A_w = u_r.reshape(n, m)
        B_w = v_r.reshape(m, p)
        prod_r = np.sum(A_w * A) * np.sum(B_w * B)

        C_decomp += prod_r * w_r.reshape(p, n).T

    C_direct = A @ B
    error = np.max(np.abs(C_decomp - C_direct))
    print(f"Max error: {error}")
    if error > 1e-10:
        print("WARNING: Decomposition is incorrect!")
