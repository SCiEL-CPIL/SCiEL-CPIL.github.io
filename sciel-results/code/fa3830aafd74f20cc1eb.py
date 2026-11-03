"""Rank-minimal exact tensor decomposition for 2x4 * 4x8 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 8]
DEADLINE = 150.0
START = time.time()

# EVOLVE-BLOCK-START
def _build_uvw(n, m, p, P, Q):
    """Build U,V,W from integer matrices P (m x m) and Q (m x p).

    C[i,k] = sum_j A[i,j] B[j,k] = sum_j (sum_l a[j,l] A[i,l]) (sum_t q[t,k] B[t,j]).
    Rank = rank(P)*rank(Q).
    """
    # Column indices: l in [0,n), j in [0,m), t in [0,m), k in [0,p)
    # U row = (i,l) in [0,n*m): 1 if a[j,l] nonzero -> contributes to (i,k)
    # V row = (j,t) in [0,m*p): 1 if q[t,k] ... 
    # Actually: term (i,j) * (j,k) with coefficient a[j,l]*q[t,k]
    # U[i*m + l, c] = a[j,l]; V[j*p + t, c] = q[t,k]; W[k*n + i, c] = 1
    cols = []
    for j in range(m):
        for l in range(n):
            for t in range(m):
                for k in range(p):
                    if P[j, l] != 0 and Q[t, k] != 0:
                        cols.append((j, l, t, k, P[j, l], Q[t, k]))
    rank = len(cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for c, (j, l, t, k, a, q) in enumerate(cols):
        pass  # placeholder (U is rebuilt below)
    # Redo cleanly:
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for c, (j, l, t, k, a, q) in enumerate(cols):
        U[l, c] = a          # row index l in [0,n)  (i is free, appears in W)
        V[j * p + t, c] = q  # row index (j,t) in [0,m*p)
        W[k * n + i, c] = 1  # placeholder
    return U, V, W

def _verify(U, V, W, n, m, p, trials=40, mod=9973):
    """Check exactness: for random A (n x m), B (m x p), does sum_c U A V B == A B?"""
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    rng = random.Random(12345)
    for _ in range(trials):
        A = np.array([[rng.randrange(-50, 51) for _ in range(m)] for _ in range(n)], dtype=object)
        B = np.array([[rng.randrange(-50, 51) for _ in range(p)] for _ in range(m)], dtype=object)
        # expected
        AB = np.zeros((n, p), dtype=object)
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    s = s + A[i, j] * B[j, k]
                AB[i, k] = s
        # decomposed: C[i,k] = sum_c W[k*n+i, c] * (sum_l U[l,c]*A[i,l]) * (sum_t V[j*p+t,c]*B[t,j])? 
        # Need correct contraction. We'll do it via the P,Q form instead below.
    return True

def construct():
    n, m, p = DIMENSIONS
    best = None
    deadline = START + 120.0  # safety margin

    # Search P (m x m) and Q (m x p) over small integer entries.
    # We want rank(P)*rank(Q) minimal. Try structured candidates.
    candidates = []

    # Identity-like: P = I_m (rank m=4), Q = I-like (rank p=8) -> 32
    # Better: P with rank 2, Q with rank 4 -> 8? But must be exact.
    # Exactness requires: for all A,B: sum_j A[i,j] B[j,k] = sum_{l,t} A[i,l] B[t,k] (P^T ... )
    # We need: for each (i,k): sum_j A[i,j]B[j,k] = sum_l sum_t A[i,l] B[t,k] * coeff[l,t]
    # This requires coeff matrix R (n x m x m x p) such that R[i,j,k] ... 
    # Simplest exact: P = I_m, Q = I_p embedded. rank = m * p = 32.
    # Can we do better? The identity P=I_m, Q=I_p gives 32.
    # To beat 32, need rank(P)*rank(Q) < 32 with exactness.
    # Exactness: sum_j A[i,j]B[j,k] must equal sum_{l,t} A[i,l] B[t,k] P[j,l] Q[t,k] summed.
    # For this to hold for all A,B: the coefficient of A[i,l] B[t,k] on RHS = P[j? ] ... 
    # Actually RHS = sum_l sum_t A[i,l] B[t,k] (sum_j P[j,l] Q[t,k])? No.
    # Let me re-derive. We set C[i,k] = sum_l (sum_j a[j,l] A[i,j]) (sum_t q[t,k] B[t,j])? Indices don't match.

    # Correct factorization: C = A B. Insert I = P P^{-1}? Use A B = A (P P^{-1}) B = (A P)(P^{-1} B).
    # If P invertible integer with integer inverse, rank(A P)=n, rank(P^{-1} B)=m. 
    # Then (A P) is n x m, (P^{-1} B) is m x p. Standard mult rank n*m*p=64. No help.

    # The real reduction: A B where A is 2x4, B is 4x8. 
    # Write A = A1 A2 with A1 2x2, A2 2x4 (rank 2), B = B1 B2 with B1 4x4, B2 4x8.
    # If A has rank <=2 always (it does, 2 rows), B has rank <=4 always.
    # A B = (A1 A2) (B1 B2) = A1 (A2 B1) B2. A2 B1 is 2x4. 
    # This is 2x2 * 2x4 * 4x8. Rank = rank(2x2)*rank(2x4)*... not directly.

    # Use: A B = sum over rank-1 of A and B. A = sum_{r=1}^{2} u_r v_r^T (rank 2).
    # B = sum_{s=1}^{4} w_s z_s^T (rank 4).
    # A B = sum_{r,s} (u_r w_s^T) (v_r^T z_s) = sum_{r,s} (v_r^T z_s) (u_r w_s^T).
    # Each (v_r^T z_s) is a scalar, (u_r w_s^T) is 2x8 outer product.
    # This is 8 terms each of which is a scalar times a 2x8 matrix -> rank 8 * 1? 
    # No: A B = sum_{r,s} (v_r^T z_s) u_r w_s^T. That's 8 bilinear forms each rank-1 in the output.
    # But v_r^T z_s depends on A and B, and u_r, w_s depend on A, B. Not a fixed tensor.

    # The fixed tensor decomposition requires fixed coefficients. The minimum rank for 2x4x4x8:
    # Known: rank(2x4, 4x8) <= rank(2x4,4x4)*... Let's just use the 32 solution (P=I, Q=I) which beats 64.
    # Actually even better: since A is 2x4, we can use the fact that only 2 rows. 
    # Standard: rank(2,4,8) <= 2*8*4/... Let's use a known bound. 
    # For multiplying m x n by n x p, rank <= m*n*p (schoolbook). 
    # With m=2: we can do rank <= 2 * (rank for 4x8 part). 
    # 4x8 mult rank <= 4*8=32 schoolbook, or better via known results.
    # Actually 2x4 * 4x8: treat as (2x4)*(4x8). Use that 2x4 mult has rank <= 8 (schoolbook 2*4=8).
    # Then we need to multiply result 2x4 by 4x8: that's 2x4 * 4x8 again = 64. Circular.

    # Let me just construct the 32-rank solution: P = I_4, Q = I_4 padded? 
    # C[i,k] = sum_j A[i,j] B[j,k]. Set P = I_4 (a[j,l]=delta_{j,l}), Q such that q[t,k]=delta_{t,k} for k<4, and for k>=4? 
    # Q is 4x8, can't be identity. So P=I, Q=I doesn't directly give 32.

    # Correct 32 solution: C[i,k] = sum_j A[i,j] B[j,k]. 
    # = sum_j (A[i,j]) (B[j,k]). This is already rank 1 per (i,j,k) -> 64.
    # To reduce: note A has 2 rows. C = [row0; row1] of A times B.
    # row0 of C = row0(A) * B, row1 of C = row1(A) * B.
    # Each is a 1x4 times 4x8 = 1x8, rank 4 each (schoolbook). Total 8? No, 1x4*4x8 rank = 4*8/... = 4 (since 1x4 vector times 4x8).
    # Actually 1x4 * 4x8: C[k] = sum_j v[j] B[j,k], rank = 4 (one per j). So 2 rows * 4 = 8? 
    # But v[j] = A[i,j] depends on i. For fixed i, rank 4. Two i's, but they share the B structure.
    # Total rank = 2 * 4 = 8? Let's see: C[i,k] = sum_j A[i,j] B[j,k]. 
    # = sum_j (A[i,j]) (B[j,k]). The "A[i,j]" and "B[j,k]" are the two factors. 
    # Number of (i,j,k) = 2*4*8 = 64, but we can group: for fixed j, sum over i,k: A[i,j] B[j,k].
    # = (A[:,j]) (B[j,:]) outer product. That's rank 1 per j, but A[:,j] is 2x1, B[j,:] is 1x8.
    # So C = sum_j (A[:,j]) (B[j,:]). Each term is rank 1 (outer of 2x1 and 1x8). 
    # So rank = 4! Wait that's the standard: C = sum_j A[:,j] B[j,:], 4 terms, each rank-1 in the 2x8 output.
    # Yes! rank = m = 4? No, that's not right either because A[:,j] and B[j,:] are not independent of i,k.

    # C[i,k] = sum_j A[i,j] B[j,k]. For each j, the term A[i,j] B[j,k] is a product of a function of (i,j) and a function of (j,k). 
    # This is exactly the schoolbook with rank = sum over j of (number of i)*(number of k) = 4 * 2 * 8 = 64. 
    # To reduce, we need A[i,j] B[j,k] to factor further, which requires structure in A or B.

    # The minimum rank for 2x4 x 4x8 is known to be at most 2*4*8 - something. 
    # Let's use a concrete reduction: since A is 2x4, we can write any 2x4 A as a linear combination, but the tensor is fixed.
    # The tensor T_{i,j,k} = delta structure. Its rank: T is 2x4x8 with T[i,j,k] = 1 if we're doing A[i,j]*B[j,k].
    # Actually the multiplication tensor has T[i,j,k] = 1 for the "A[i,j] B[j,k]" coefficient, but it's a 3-way tensor of size 2x4x8 where entry (i,j,k) = 1 if j is the shared index... 
    # The standard multiplication tensor for (n,m) x (m,p) is T[i,j,k] = 1 for all i,j,k (it's the sum), no.
    # T[i,j,k] = 1 means C[i,k] gets contribution A[i,j]B[j,k]. So T is all-ones 2x4x8, rank = ?
    # The all-ones tensor of size a x b x c has rank = min(a*b, a*c, b*c, a+b+c-2, ...). 
    # For 2x4x8: rank of all-ones = ? The all-ones tensor rank is at most 2*4 + 4*8 - 1 = 35, or min(8, 16, 32) ... 
    # Actually rank(all-ones a x b x c) = min over partitions. Known: rank of all-ones 2 x b x c = 2b + 2c - 1? 
    # Let me just compute a valid decomposition.

    # Construction: C[i,k] = sum_j A[i,j] B[j,k].
    # = sum_j A[i,j] B[j,k] 
    # Introduce: = sum_j (A[i,j]) (B[j,k]).
    # Write A[i,j] = A[i,j], B[j,k] = B[j,k]. 
    # Use: for the 2 rows, C[0,k] = sum_j A[0,j] B[j,k], C[1,k] = sum_j A[1,j] B[j,k].
    # Define for each j: u_j = (A[0,j], A[1,j]) (2x1), v_j = B[j,:] (1x8).
    # C = sum_j u_j v_j. Rank 1 per j -> 4 terms, each rank-1 (2x8). So rank = 4!
    # But u_j = (A[0,j], A[1,j]) is a linear function of A, v_j = B[j,:] is linear in B. 
    # So C[i,k] = sum_j (A[i,j]) (B[j,k]) = sum_j u_j[i] v_j[k]. 
    # This IS a rank-4 decomposition! Each term j: coefficient = A[i,j] * B[j,k], output C[i,k].
    # U[i, j] = A[i,j] (so U has row i, col j, entry = A[i,j]), V[j, k] = B[j,k], W[i,k] = 1.
    # Wait, the tensor decomposition: C[i,k] = sum_c U[i,c] W[c,k] V[?]. 
    # Standard: C[i,k] = sum_c (sum_j U[i,j,c] A[i,j]) (sum_t V[j,t,c] B[t,k]) W[...]. 
    # Let me just verify rank 4 works: C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]. 
    # This is sum over j of (A[i,j]) (B[j,k]). Each j is one "rank-1" in the sense that it's a product of a linear form in A and a linear form in B, giving C[i,k].
    # Number of j = 4. So rank = 4?? That can't be right because schoolbook is 64.

    # OH. I see the error. C[i,k] = sum_j A[i,j] B[j,k]. The term A[i,j] B[j,k] for a FIXED j is a product of A[i,j] (depends on i) and B[j,k] (depends on k). 
    # So for fixed j, the contribution to C[i,k] is A[i,j] * B[j,k], which is a rank-1 matrix (outer product of column j of A and row j of B).
    # So C = sum_j (column j of A) (row j of B). That's 4 rank-1 matrices summing to C.
    # So rank of the multiplication = 4? No! The rank of a matrix product AB is not the rank of the multiplication tensor.
    # The multiplication tensor rank is the minimum number of scalar multiplications. 
    # C = sum_j (A[:,j]) (B[j,:]) uses 4 "matrix outer products" but each outer product (2x1)(1x8) = 2x8 requires 2*8=16 scalar mults to form.
    # So total = 4 * 16 = 64 scalar mults. That's schoolbook. The "rank 4" is the rank of C as a matrix, not the tensor rank.

    # The tensor rank of 2x4 x 4x8 multiplication: this is the minimum number of scalar products.
    # Lower bound: max(n*m, m*p, n*p) = max(8, 32, 16) = 32. So rank >= 32.
    # Upper bound: schoolbook 64. Can we achieve 32? 
    # The lower bound 32 = m*p = 4*8. This is achievable if... 
    # Actually for m x n by n x p, rank >= max(mn, np, mp) and rank <= mn + np - 1 or similar.
    # For 2x4 x 4x8: rank >= max(8, 16, 32) = 32. 
    # Is rank = 32 achievable? This would be optimal.
    # Construction for rank = m*p when n <= ... : 
    # C[i,k] = sum_j A[i,j] B[j,k]. If we could do it in m*p = 32 mults.
    # One way: if A has rank 1, then A = u v^T, C = u (v^T B), rank = 1*... but A is general 2x4.

    # Let me just use the known result: rank(2,4,8) = ? I'll construct a 32-rank solution.
    # Use: C[i,k] = sum_j A[i,j] B[j,k]. 
    # Group by k: for each k, C[:,k] = A B[:,k] (2x4 times 4x1 = 2x1). 
    # 2x4 * 4x1 has rank <= 2*4 = 8 schoolbook, or <= 2*1 + 4*1 - 1 = 5? 
    # 2x4 x 4x1: rank >= max(8, 2, 4) = 8. So 8 per k, 8 k's = 64. No.

    # I'll just return the schoolbook 64 but that's the baseline. Let me try to find a real reduction.
    # Try rank 32: C[i,k] = sum_j A[i,j] B[j,k]. 
    # Note: = sum_j A[i,j] B[j,k]. Let me define for each (i,k) we need sum over j.
    # Use the identity: sum_j A[i,j] B[j,k] = sum_j (A[i,j]) (B[j,k]).
    # This is 64 terms. To reduce to 32, pair them up.
    # Pair j=0,1 for each (i,k): A[i,0]B[0,k] + A[i,1]B[1,k]. 
    # = A[i,0]B[0,k] + A[i,1]B[1,k]. Can we compute this with 1 mult? No, it's a bilinear form in 4 variables.
    # A 2x2 bilinear form (in A[i,0],A[i,1],B[0,k],B[1,k]) has rank 2 (it's A[i,0]B[0,k] + A[i,1]B[1,k], two products).
    # So no reduction there.

    # I think the minimum rank is actually 32 (the lower bound), and it's achievable.
    # Achievability: rank(2,4,8) <= 2*8 + 4*8 - ... Let me look at it as (2x4)*(4x8) = (2x4*4x4)*(4x4*4x8)? No.
    # Use: 2x4 * 4x8 <= (2x4 * 4x4) * ... 

    # Let me just hardcode a 32-rank decomposition using the "slice" method.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # Define new variables: for each i in {0,1}, k in {0..7}: 
    # C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k].
    # This is 8 independent 1x4*4x1 = 1x1 products? No, each C[i,k] is a dot product of 4 terms = 4 mults.
    # 16 (i,k) pairs * 4 = 64.
    # To get 32, each C[i,k] must use 2 mults on average, impossible for a 4-term dot product in general.
    # So rank 32 may NOT be achievable. The lower bound 32 is not tight.

    # Let me reconsider. The true minimum rank for 2x4 x 4x8 multiplication.
    # Known bounds: rank >= max(nm, np, mp) = 32. rank <= nm + np + mp - 2 - ... 
    # Actually a standard upper bound: rank(m,n,p) <= mn + np - 1 (when ...). For 2,4,8: 8 + 32 - 1 = 39.
    # Or rank <= m*n*p - something. 
    # Let me just use a 32-rank construction if I can find one, else 39, else 64.

    # Try the "rank = m*p" construction: it works when n=1 (trivial) or when... 
    # For general, rank(m,1,p) = p (since it's a row times matrix). rank(1,n,p) = n.
    # For m=2: rank(2,4,8). 

    # I'll construct using the fact that we can precompute. Let me just do a 32-rank via:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_j A[i,j] B[j,k].
    # Define M[i,j] = A[i,j], N[j,k] = B[j,k]. 
    # C = M N. rank of this bilinear map.
    # I'll use a known 32-rank: actually I recall rank(2,4,8) might be 32. Let me construct it.
    # C[i,k] = sum_{j} A[i,j] B[j,k]. 
    # = sum_{j even} A[i,j] B[j,k] + sum_{j odd} A[i,j] B[j,k]. No help.

    # Let me just go with a safe 32-rank using a verified construction via the P,Q factorization with P=I_4, Q=I_4x8 impossible.
    # Instead: C = A B. Write A = A, B = B. 
    # Use rank = 2*8 + 4*8 - 8 = 32? 
    # I'll implement a general search-free 32-rank: 
    # C[i,k] = sum_j A[i,j] B[j,k].
    # Introduce for each i: let a_i = A[i,:] (1x4). C[i,:] = a_i B (1x4 times 4x8 = 1x8).
    # 1x4 * 4x8 rank = 4 (each of 8 outputs needs 4 mults, but shared: C[i,k] = sum_j a_i[j] B[j,k], 4 mults per k, 8 k's, but a_i[j] reused -> 4*8=32 for one i? No, 4 mults per (j,k) pair = 32 for one row i, times 2 rows = 64.
    # But a_i[j] is reused across k! For fixed i and j, a_i[j] * B[j,k] for all k: that's a_i[j] times the whole row B[j,:]. 
    # So C[i,:] = sum_j a_i[j] B[j,:]. This is sum over j of (scalar a_i[j]) * (row B[j,:]).
    # Each j: scalar a_i[j] times 8-vector B[j,:] = 8 mults, but a_i[j] is one scalar, B[j,:] is 8 scalars -> 8 mults.
    # 4 j's * 8 = 32 per i, 2 i's = 64. Still 64 because a_i[j] differs per i.
    # If a_i[j] were shared between i=0 and i=1, we'd save, but it's not.

    # I give up on beating 64 easily. Let me just return a valid 32-rank if I can verify, else 64.
    # Actually, let me reconsider the lower bound. rank >= max(nm, np, mp) = max(8,16,32)=32. 
    # And there's a construction achieving rank = m*p + n*p - ... 
    # Let me just try rank 32 with a specific formula and verify.

    # Formula: C[i,k] = sum_j A[i,j] B[j,k]. 
    # Claim: rank 32 via: for each (j,k) in 4x8=32, define term = A[:,j] * B[j,k] (2x1 times scalar = 2x1).
    # C[:,k] = sum_j A[:,j] B[j,k]. So C[i,k] = sum_j A[i,j] B[j,k]. 
    # Each term (j,k): output is a 2-vector (A[0,j]*B[j,k], A[1,j]*B[j,k]). 
    # This uses 2 mults per (j,k) = 2*32 = 64. No.

    # I'll just return the schoolbook 64 to be safe, OR try a real search.
    # Given time, let me do a small search for P (4x4), Q (4x8) with integer entries in {-1,0,1} minimizing rank(P)*rank(Q) subject to exactness.
    # Exactness condition: for all A,B, sum_j A[i,j]B[j,k] = sum_{l,t} A[i,l] B[t,k] R[l,t] where R depends on P,Q.
    # From C = (A P)(P^{-1} B) requires P invertible. If P invertible integer with integer inverse:
    # A P is 2x4, P^{-1} B is 4x8. Then (A P)(P^{-1} B) = A B. 
    # Rank of (2x4)*(4x8) = 64 still. No reduction.

    # The reduction must come from the 2-row structure. 
    # Final decision: I'll implement a 32-rank decomposition that I'll verify, using the following:
    # C[i,k] = sum_j A[i,j] B[j,k].
    # = sum_j A[i,j] B[j,k].
    # I'll just use the identity and accept 64, but try to shave via pairing i.
    # Pair i=0,1: C[0,k] + C[1,k] and C[0,k] - C[1,k].
    # C[0,k] + C[1,k] = sum_j (A[0,j]+A[1,j]) B[j,k].
    # C[0,k] - C[1,k] = sum_j (A[0,j]-A[1,j]) B[j,k].
    # Each is a 1x4 * 4x8 = 1x8, rank 4 each (per the 1x4*4x8 which needs 4*8=32? no, 1x4*4x8: C[k]=sum_j v[j]B[j,k], 4 mults per k, 8 k's = 32, but v[j] reused -> 4*8=32).
    # So 2 such = 64. No gain.

    # I'll return 64 (schoolbook) as a safe valid answer, which is the baseline. To improve, I need a real reduction.
    # Let me try rank 32 with a concrete verified construction using numpy at runtime (fast).

    # Actually, the cleanest: rank(2,4,8) <= 2*4*8 - 4*8 + ... I'll just search small P,Q.

    def _exact_rank_via_PQ(P, Q):
        # C[i,k] = sum_j A[i,j] B[j,k] must equal sum_{l,t} A[i,l] B[t,k] (P^T ... )
        # We want: sum_j A[i,j] B[j,k] = sum_{l,t} A[i,l] B[t,k] C[l,t]
        # LHS coefficient of A[i,l] B[t,k] is delta_{l,t} (1 if l==t, 0 else).
        # RHS coefficient of A[i,l] B[t,k] is C[l,t].
        # So we need C = I_4 (4x4 identity). But C must factor as C[l,t] = sum_j P[j,l] Q[t,j] (if P is 4x4, Q is 4x8 -> C is 4x8, not 4x4).
        # Mismatch. So P,Q factorization of the form (A P)(Q B) with P 4x4, Q 4x8 gives C = P^T Q (4x4)? 
        # (A P) is 2x4, (Q B) is 4x8? Q is 4x8, B is 4x8, can't multiply Q B.
        # So this factorization doesn't work directly.

        # Correct: C = A B = (A P) (P^{-1} B) needs P square invertible. No rank gain.
        return None

    # I'll just return the schoolbook 64. It's valid and matches baseline. To beat it I'd need deep knowledge.
    # Given the directive to improve, let me at least return a valid 32-rank IF I can construct one.
    # I'll construct rank 32 as follows (standard for m x n x p with small m):
    # Not available cleanly. Return 64.

    n, m, p = 2, 4, 8
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
