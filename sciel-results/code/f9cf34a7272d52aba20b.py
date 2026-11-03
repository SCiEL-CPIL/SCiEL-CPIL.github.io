"""Tensor decomposition for 3x4 * 4x7 matrix multiplication.

Uses the known rank-10 decomposition for (2,2,4) and combines via
block-splitting (additive rank construction) to reach rank 51.
"""
import time

DIMENSIONS = [3, 4, 7]


def construct():
    deadline = time.time() + 110  # keep well under the 120 s / 150 s limits

    n, m, p = DIMENSIONS

    # ---------- rank-10 decomposition for (2,2,4) ----------
    # Indexing: A is 2x2 (i=0,1; j=0,1), B is 2x4 (j=0,1; k=0..3),
    #           C is 2x4 (i=0,1; k=0..3)
    # Strassen for the two 2x2 blocks in the last two columns of B,
    # plus 4 scalar products for the first two columns.
    _u_224 = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    _v_224 = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 1, 1, 1, 1, 1],
    ]
    _w_224 = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
    ]

    # ---------- build (2,4,7) with rank 51 ----------
    # Split p=7 into p1=4 and p2=3.
    # (2,4,4) rank 20: split m=4 into 2+2, two copies of (2,2,4).
    # (2,4,3) rank 31: split p=3 into 2+1.
    #   (2,4,2) rank 14: split m=4 into 2+2, two copies of (2,2,2) (Strassen rank 7).
    #   (2,4,1) rank 17: hmm, (2,4,1) rank should be 8. Let me recompute.

    # Actually, let me use the straightforward split more carefully.
    # (2,4,7):
    #   Split p: p=4 + p=3
    #   (2,4,4): split m: m=2 + m=2 -> two (2,2,4) each rank 10 -> total 20
    #   (2,4,3): split p: p=2 + p=1
    #     (2,4,2): split m: m=2 + m=2 -> two (2,2,2) each rank 7 -> total 14
    #     (2,4,1): split m: m=2 + m=2 -> two (2,2,1) each rank 4 -> total 8
    #   (2,4,3) rank = 14 + 8 = 22
    #   (2,4,7) rank = 20 + 22 = 42

    # (3,4,7): split n: n=2 + n=1
    #   (2,4,7) rank 42
    #   (1,4,7) rank 7 (scalar: A is 1x4, B is 4x7, C is 1x7, rank = 7)
    #   Total rank = 42 + 7 = 49

    # Let's build it all via helper functions.

    def strassen_222():
        """Rank-7 decomposition for (2,2,2)."""
        u = [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 1],
        ]
        v = [
            [1, 0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1, 0],
        ]
        w = [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 1, 1],
        ]
        return u, v, w

    def scalar_221():
        """Rank-4 decomposition for (2,2,1)."""
        u = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ]
        v = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ]
        w = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ]
        return u, v, w

    def scalar_1mp(m, p):
        """Rank-p decomposition for (1,m,p). A is 1xm, B is mxp, C is 1xp."""
        u = [[1] * p]  # 1 x p  (wait, U shape is n*m x rank = 1*m x p)
        # U shape: (n*m, rank) = (m, p)
        # For (1,m,p): U is (m, p), V is (m*p, p), W is (p*1, p) = (p, p)
        # U[i, k] = delta(i, k) for i in 0..m-1, k in 0..p-1
        # Actually: C[k] = sum_j A[j] * B[j, k]
        # Term c: A[j] * B[j, c] for j in 0..m-1, c in 0..p-1
        # U[j, c] = delta(j, c) -- but j ranges 0..m-1 and c ranges 0..p-1
        # This only works if m == p. For general m != p we need rank = m*p.
        # Wait, (1,m,p): C is 1xp. C[k] = sum_{j=0}^{m-1} A[j] B[j,k].
        # Each term is A[j] * B[j,k]. There are m*p such terms, so rank <= m*p.
        # But we can do better: for each k, C[k] = dot(A, B[:,k]).
        # This is a single inner product, which has rank m (or p, whichever is smaller? No.)
        # Actually the tensor is sum_j sum_k A_j B_{j,k} C_k.
        # This is the "bilinear" form. The rank is min(m*p, ...) hmm.
        # Let me think: the matrix multiplication tensor T_{i,j,k} = delta(i,0) * 1 for A, B, C.
        # For (1,m,p): T has entries T[0, j, k] = 1 for all j,k. That's not right.
        # T[i, j, k] corresponds to A[i,j] * B[j,k] * C[i,k].
        # For i=0: T[0, j, k] = 1 for all j in 0..m-1, k in 0..p-1.
        # So the tensor is all-ones of shape (1, m, p).
        # Rank of all-ones tensor (1,m,p) is min(m, p) if we can factor, but actually
        # it's the rank of the matrix of ones (m x p) which is 1 if m,p >= 1? No.
        # The tensor T_{j,k} = 1 for all j,k. As a matrix (m x p) it's all ones, rank 1.
        # But as a 3-tensor with the first dimension being 1, the "rank" (tensor rank) is
        # the minimum r such that T = sum_{c=1}^r u_c v_c w_c where u_c is (1,), v_c is (m,), w_c is (p,).
        # Since u_c is a scalar, T_{j,k} = sum_c u_c v_{c,j} w_{c,k}.
        # This is sum_c (u_c v_{c,j}) w_{c,k} = (V W^T)_{j,k} where V is m x r, W is p x r.
        # So we need an m x p matrix of all ones as a sum of r rank-1 matrices.
        # The matrix rank of all-ones (m x p) is 1, so r = 1 suffices!
        # u_0 = 1, v_0 = [1,1,...,1] (m entries), w_0 = [1,1,...,1] (p entries).
        # T_{j,k} = 1 * 1 * 1 = 1. Yes!
        # So rank of (1,m,p) is 1? That seems too good. Let me verify.
        # A is 1xm, B is mxp, C is 1xp.
        # C[0, k] = sum_j A[0, j] B[j, k].
        # With the decomposition: C[0,k] = sum_c U[0,c] * V[:,c]^T * B[:,k]... no.
        # The formula is C = sum_c U_c * V_c^T ... no.
        # Standard: C_{i,k} = sum_c U_{i,c} (sum_j ... ) hmm.
        # Actually the tensor decomposition: T_{i,j,k} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}
        # Wait no, the convention is:
        # C_{i,k} = sum_c (sum_j U_{i*m+j, c} * V_{j*p+k, c}) * W_{k*n+i, c}
        # For (1,m,p): i=0 only.
        # C_{0,k} = sum_c (sum_j U_{j, c} * V_{j*p+k, c}) * W_{k, c}
        # We need this to equal sum_j A_{0,j} B_{j,k} for all A, B.
        # So for each c, the contribution is W_{k,c} * (sum_j U_{j,c} V_{j*p+k,c}).
        # The total is sum_c W_{k,c} * (sum_j U_{j,c} V_{j*p+k,c}).
        # We need: sum_c W_{k,c} * (sum_j U_{j,c} V_{j*p+k,c}) = sum_j A_j B_{j,k}
        # for all A, B. This means the bilinear form sum_c W_{k,c} sum_j U_{j,c} V_{j*p+k,c}
        # must equal sum_j A_j B_{j,k}.
        # So the matrix M_{j,k} = sum_c U_{j,c} V_{j*p+k,c} W_{k,c} must equal the identity in some sense?
        # No, it must be such that sum_j M_{j,k} A_j B_{j,k}... that's not how it works.
        # Let me re-derive. The matrix multiplication is C = A B where A is 1xm, B is mxp.
        # C_k = sum_j A_j B_{j,k}.
        # In the tensor decomposition framework:
        # C_{i,k} = sum_c [ (U^T_c * A) * (V_c * B) ] * W_c  -- this isn't standard.
        # Let me use the standard formulation:
        # C_{i,k} = sum_c (sum_j U_{i,j,c}) (sum_k' V_{j,k',c}) ... no.
        # The correct formulation: C = sum_c (U_c A) (V_c B) W_c  -- not quite.
        # Standard: for matrix multiplication A (n x m) times B (m x p) to get C (n x p):
        # C_{i,k} = sum_{j} A_{i,j} B_{j,k}
        # Tensor decomposition: C_{i,k} = sum_c sum_j U_{i,j,c} B_{j,k} * (something)
        # Actually: C_{i,k} = sum_c (sum_j U_{i,j,c} A_{i,j}) * (sum_k V_{j,k,c} B_{j,k}) * W_{i,k,c}
        # No, that's not right either. Let me use the standard 3-tensor formulation.
        # The matrix multiplication tensor T has T_{i,j,k} = 1 if the indices form a valid
        # multiplication path, 0 otherwise. Specifically for multiplying A (n x m) by B (m x p):
        # T_{i,j,k} = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
        # Wait, that's the all-ones tensor, which has rank min(n*p, m) or something.
        # No! The tensor is: T[i, j, k] = delta(i, i') delta(j, j') delta(k, k') ... 
        # I'm confusing myself. Let me go back to basics.
        # The matrix product C = AB has C_{ik} = sum_j A_{ij} B_{jk}.
        # We want to write C_{ik} = sum_{c=1}^{r} l_c(A) * m_c(B) * w_{ik,c}
        # where l_c is a linear functional on A, m_c is a linear functional on B, w_{ik,c} is a scalar.
        # l_c(A) = sum_{i,j} U_{ij,c} A_{ij}
        # m_c(B) = sum_{j,k} V_{jk,c} B_{jk}
        # w_{ik,c} = W_{ik,c}
        # We need: sum_j A_{ij} B_{jk} = sum_c (sum_{i',j'} U_{i'j',c} A_{i'j'}) (sum_{j'',k''} V_{j''k'',c} B_{j''k''}) W_{ik,c}
        # For this to hold for all A, B, we need:
        # delta_{ij} delta_{jk} (as a function of i,j,k) = sum_c U_{ij,c} V_{jk,c} W_{ik,c}
        # So the tensor T_{ijk} = delta_{ij} delta_{jk} (which is 1 when i,j,k form a valid chain, but since
        # j is shared, T_{ijk} = 1 for all valid i,j,k) -- wait, T_{ijk} should be such that
        # sum_{i,j,k} T_{ijk} A_{ij} B_{jk} = sum_{i,k} C_{ik} A_{i,k}... 
        # I think the right way: the "matrix multiplication tensor" is defined by
        # <T, A \otimes B \otimes C> = sum_{i,j,k} A_{ij} B_{jk} C_{ki} (or similar).
        # For our purposes, the factorization condition is:
        # For all A, B: AB = sum_c (U_c \bullet A) (V_c \bullet B) W_c
        # where (U_c \bullet A) is a scalar (sum of U_c[i,j]*A[i,j]),
        # (V_c \bullet B) is a scalar (sum of V_c[j,k]*B[j,k]),
        # and W_c is an n x p matrix.
        # Wait, that gives C = sum_c scalar_c * W_c, where scalar_c depends on both A and B.
        # That means C_{ik} = sum_c (sum_{i',j'} U_{i'j',c} A_{i'j'}) (sum_{j'',k''} V_{j''k'',c} B_{j''k''}) W_{ik,c}.
        # We need this to equal sum_j A_{ij} B_{jk} for all i, k.
        # So: sum_c U_{ij,c} V_{jk,c} W_{ik,c} = delta (as appropriate).
        # More precisely, the coefficient of A_{ij} B_{jk} on the LHS is sum_c U_{ij,c} V_{jk,c} W_{ik,c}.
        # But the LHS also has terms where i' != i or k'' != k, etc. Since A and B are arbitrary,
        # we need the LHS to be a bilinear form in A and B that equals sum_j A_{ij} B_{jk} for each (i,k).
        # This means: for each pair (i, k), the bilinear form sum_{i',j',j'',k''} U_{i'j',c} V_{j''k'',c} W_{ik,c} A_{i'j'} B_{j''k''}
        # must equal sum_j A_{ij} B_{jk}.
        # Since this must hold for all A, B, the coefficient of A_{i'j'} B_{j''k''} must be delta_{i'i} delta_{j'j} delta_{j''j} delta_{k''k} summed over j...
        # This is getting complicated. Let me just use the standard formulation from the seed code.
        # In the seed code: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
        # The condition is: for all A (n x m), B (m x p):
        # C[i, k] = sum_c (sum_j U[i*m+j, c] * A[i, j]) ... no.
        # Looking at the seed: U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for the term (i,j,k).
        # The product for term c is U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
        # The formula must be: C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
        # No, that would give C[i,k] = sum_c W[k*n+i,c] * l_c(A) * m_c(B) where l_c and m_c are scalars.
        # Let me check with the schoolbook: for term c corresponding to (i0, j0, k0):
        # U[i*m+j, c] = 1 iff i=i0 and j=j0.
        # V[j*p+k, c] = 1 iff j=j0 and k=k0.
        # W[k*n+i, c] = 1 iff k=k0 and i=i0.
        # So l_c(A) = A[i0, j0], m_c(B) = B[j0, k0], W[k*n+i0, c] = 1 iff k=k0.
        # C[i, k] = sum_c W[k*n+i, c] * A[i0,c, j0,c] * B[j0,c, k0,c]
        # For the term (i0, j0, k0): W[k*n+i, c] = 1 iff k=k0 and i=i0.
        # So C[i, k] gets contribution A[i0,j0]*B[j0,k0] only when i=i0 and k=k0.
        # Thus C[i,k] = sum_{j} A[i,j] B[j,k]. Correct!
        # So the formula is:
        # C[i,k] = sum_c W[k*n+i, c] * (sum_{j} U[i*m+j, c] * A[i, j]) ... 
        # Wait, U[i*m+j, c] is indexed by (i, j) and c. For a fixed c, U[:, c] is a vector of length n*m.
        # l_c(A) = sum_{i,j} U[i*m+j, c] * A[i, j] -- this is a scalar.
        # But in the schoolbook, U[i*m+j, c] is 1 only for one (i,j) pair. So l_c(A) = A[i0, j0].
        # Similarly m_c(B) = sum_{j,k} V[j*p+k, c] * B[j, k] = B[j0, k0].
        # And C[i,k] = sum_c W[k*n+i, c] * l_c(A) * m_c(B).
        # Yes! That's the formula. So:
        # C[i,k] = sum_c W[k*n+i, c] * l_c(A) * m_c(B)
        # where l_c(A) = sum_{i',j'} U[i'*m+j', c] A[i',j'] and m_c(B) = sum_{j',k'} V[j'*p+k', c] B[j',k'].

    # OK so with this understanding, let me build the decompositions.

    def make_222():
        """Rank-7 Strassen for (2,2,2)."""
        u = [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 1],
        ]
        v = [
            [1, 0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1, 0],
        ]
        w = [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 1, 1],
        ]
        return u, v, w

    def make_224():
        """Rank-10 for (2,2,4): two Strassen blocks for cols 2,3 and 4 scalar products."""
        u222, v222, w222 = make_222()
        rank = 10
        u = [[0]*rank for _ in range(4)]  # n*m = 2*2 = 4
        v = [[0]*rank for _ in range(8)]  # m*p = 2*4 = 8
        w = [[0]*rank for _ in range(8)]  # p*n = 4*2 = 8

        # Strassen block 1: columns k=2,3 (indices 2,3 in B)
        # A is 2x2, B_block is 2x2 (rows j=0,1; cols k=2,3)
        # C_block is 2x2 (rows i=0,1; cols k=2,3)
        for c in range(7):
            for idx in range(4):  # U has 4 entries: (i,j) with i=0,1 j=0,1
                u[idx][c] = u222[idx][c]
            for j in range(2):
                for kk in range(2):
                    v[j*4 + (2+kk)][c] = v222[j*2 + kk][c]
            for kk in range(2):
                for i in range(2):
                    w[(2+kk)*2 + i][c] = w222[kk*2 + i][c]

        # 4 scalar products for k=0,1
        for s in range(4):
            c = 7 + s
            i, j = s // 2, s % 2
            u[i*2 + j][c] = 1
            v[j*4 + s][c] = 1  # j*p + k where k = s (0 or 1)
            w[s*2 + i][c] = 1
        return u, v, w

    def make_1mp(m, p):
        """Rank-1 for (1,m,p)."""
        # C[0,k] = sum_j A[0,j] B[j,k]
        # l_c(A) = sum_j A[0,j] = sum of all A entries (scalar)
        # m_c(B) = sum_{j,k} B[j,k] ... no, that doesn't work.
        # We need: C[0,k] = W[0,k] * l(A) * m(B) where W[0,k] = 1.
        # l(A) * m(B) must equal sum_j A[j] B[j,k] for each k.
        # This is not possible with rank 1 unless m=p=1.
        # So (1,m,p) has rank m*p? No...
        # Let me think again. C[0,k] = sum_j A[j] B[j,k].
        # This is an inner product between A (length m) and the k-th column of B.
        # As a bilinear form, it's the standard dot product. The tensor is:
        # T[j, k] = 1 for all j, k (since C[0,k] = sum_j A[j] B[j,k]).
        # Wait no: the tensor T_{ijk} where i=0, j=0..m-1, k=0..p-1.
        # T_{0,j,k} = 1 for all j, k.
        # Factorization: T_{0,j,k} = sum_c U[j,c] V[j*p+k, c] W[k, c]
        # (since n=1, i=0, U index is i*m+j = j, V index is j*p+k, W index is k*n+i = k)
        # So: 1 = sum_c U[j,c] V[j*p+k, c] W[k,c] for all j, k.
        # This means the m x p matrix of all ones = (U V^T W^T) in some sense.
        # More precisely: for each (j,k), sum_c U[j,c] V[j*p+k,c] W[k,c] = 1.
        # Let's try rank 1: U[j,0] = a_j, V[j*p+k,0] = b_{j,k}, W[k,0] = c_k.
        # sum_j,k: a_j b_{j,k} c_k = 1 for all j,k.
        # If we set a_j = 1, c_k = 1, then b_{j,k} = 1 for all j,k.
        # Check: a_j * b_{j,k} * c_k = 1*1*1 = 1. Yes!
        # So rank 1 works!
        u = [[1]*1 for _ in range(m)]       # m x 1
        v = [[1]*1 for _ in range(m*p)]     # m*p x 1
        w = [[1]*1 for _ in range(p)]       # p x 1
        return u, v, w

    def make_221():
        """Rank-4 for (2,2,1)."""
        # C[i,0] = sum_j A[i,j] B[j,0]
        # Rank 4: one term per (i,j) pair.
        u = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]  # 4x4 (n*m=4)
        v = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]  # 4x4 (m*p=4)
        w = [[1,0,0,0],[0,0,0,0],     # k=0, i=0
             [0,1,0,0],[0,0,1,0]]     # k=0, i=1 -- wait, w index is k*n+i = 0*2+i = i
        # w has p*n = 1*2 = 2 rows.
        w = [[1,0,0,0],   # k=0, i=0
             [0,1,0,0]]   # k=0, i=1
        # Term c=0: U[0,0]=1 (i=0,j=0), V[0,0]=1 (j=0,k=0), W[0,0]=1 (k=0,i=0)
        #   Contribution to C[0,0]: A[0,0]*B[0,0]. Good.
        # Term c=1: U[1,1]=1 (i=0,j=1), V[2,1]=1 (j=1,k=0), W[0,1]=1 (k=0,i=0)
        #   Contribution to C[0,0]: A[0,1]*B[1,0]. Good.
        # Term c=2: U[2,2]=1 (i=1,j=0), V[0,2]=1 (j=0,k=0), W[1,2]=1 (k=0,i=1)
        #   Contribution to C[1,0]: A[1,0]*B[0,0]. Good.
        # Term c=3: U[3,3]=1 (i=1,j=1), V[2,3]=1 (j=1,k=0), W[1,3]=1 (k=0,i=1)
        #   Contribution to C[1,0]: A[1,1]*B[1,0]. Good.
        return u, v, w

    def concatenate_decomps(decomps, n, m, p):
        """Concatenate multiple decompositions of (n,m,p) by stacking in the rank direction."""
        total_rank = sum(d[0].__len__() for d in decomps) if decomps else 0
        # Actually, each decomp is (U, V, W) where U is (n*m) x rank_c.
        total_rank = 0
        for U, V, W in decomps:
            total_rank += len(U[0]) if U else 0
        U_full = [[0]*total_rank for _ in range(n*m)]
        V_full = [[0]*total_rank for _ in range(m*p)]
        W_full = [[0]*total_rank for _ in range(p*n)]
        col = 0
        for U, V, W in decomps:
            r = len(U[0]) if U else 0
            for i in range(n*m):
                for c in range(r):
                    U_full[i][col+c] = U[i][c]
            for i in range(m*p):
                for c in range(r):
                    V_full[i][col+c] = V[i][c]
            for i in range(p*n):
                for c in range(r):
                    W_full[i][col+c] = W[i][c]
            col += r
        return U_full, V_full, W_full

    def embed_split_m(decomp_list, n, m, p):
        """
        Split m into m1+m2. decomp_list = [(U1,V1,W1), (U2,V2,W2)] for (n,m1,p) and (n,m2,p).
        Return decomposition of (n,m,p).
        """
        (U1, V1, W1), (U2, V2, W2) = decomp_list
        m1 = len(U1) // n
        m2 = len(U2) // n
        r1 = len(U1[0])
        r2 = len(U2[0])
        total_rank = r1 + r2
        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*p)]
        W = [[0]*total_rank for _ in range(p*n)]
        # U1: rows (i, j) with j in 0..m1-1 -> place in U at (i, j)
        for i in range(n):
            for j in range(m1):
                for c in range(r1):
                    U[i*m + j][c] = U1[i*m1 + j][c]
        # U2: rows (i, j) with j in 0..m2-1 -> place in U at (i, m1+j)
        for i in range(n):
            for j in range(m2):
                for c in range(r2):
                    U[i*m + m1 + j][r1 + c] = U2[i*m2 + j][c]
        # V1: rows (j, k) with j in 0..m1-1 -> place in V at (j, k)
        for j in range(m1):
            for k in range(p):
                for c in range(r1):
                    V[j*p + k][c] = V1[j*m1 + k][c]  # V1 is (m1*p) x r1, index j*m1+k
        # Wait, V1 shape is (m1 * p) x r1. Index is j*m1 + k? No, it's j*p + k for the sub-matrix.
        # V1 is for (n, m1, p), so V1 has m1*p rows, indexed by j*p + k for j in 0..m1-1, k in 0..p-1.
        for j in range(m1):
            for k in range(p):
                for c in range(r1):
                    V[j*p + k][c] = V1[j*p + k][c]
        # V2: rows (j, k) with j in 0..m2-1 -> place in V at (m1+j, k)
        for j in range(m2):
            for k in range(p):
                for c in range(r2):
                    V[(m1+j)*p + k][r1 + c] = V2[j*p + k][c]
        # W: same as W1 and W2
        for i in range(p*n):
            for c in range(r1):
                W[i][c] = W1[i][c]
            for c in range(r2):
                W[i][r1 + c] = W2[i][c]
        return U, V, W

    def embed_split_n(decomp_list, n, m, p):
        """
        Split n into n1+n2. decomp_list = [(U1,V1,W1), (U2,V2,W2)] for (n1,m,p) and (n2,m,p).
        Return decomposition of (n,m,p).
        """
        (U1, V1, W1), (U2, V2, W2) = decomp_list
        n1 = len(U1) // m
        n2 = len(U2) // m
        r1 = len(U1[0])
        r2 = len(U2[0])
        total_rank = r1 + r2
        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*p)]
        W = [[0]*total_rank for _ in range(p*n)]
        for i in range(n1):
            for j in range(m):
                for c in range(r1):
                    U[i*m + j][c] = U1[i*m + j][c]
        for i in range(n2):
            for j in range(m):
                for c in range(r2):
                    U[(n1+i)*m + j][r1 + c] = U2[i*m + j][c]
        for j in range(m):
            for k in range(p):
                for c in range(r1):
                    V[j*p + k][c] = V1[j*p + k][c]
                for c in range(r2):
                    V[j*p + k][r1 + c] = V2[j*p + k][c]
        for i in range(p):
            for ii in range(n1):
                for c in range(r1):
                    W[i*n + ii][c] = W1[i*n1 + ii][c]
            for ii in range(n2):
                for c in range(r2):
                    W[i*n + n1 + ii][r1 + c] = W2[i*n2 + ii][c]
        return U, V, W

    def embed_split_p(decomp_list, n, m, p):
        """
        Split p into p1+p2. decomp_list = [(U1,V1,W1), (U2,V2,W2)] for (n,m,p1) and (n,m,p2).
        Return decomposition of (n,m,p).
        """
        (U1, V1, W1), (U2, V2, W2) = decomp_list
        p1 = len(W1) // n
        p2 = len(W2) // n
        r1 = len(U1[0])
        r2 = len(U2[0])
        total_rank = r1 + r2
        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*p)]
        W = [[0]*total_rank for _ in range(p*n)]
        for i in range(n*m):
            for c in range(r1):
                U[i][c] = U1[i][c]
            for c in range(r2):
                U[i][r1 + c] = U2[i][c]
        for j in range(m):
            for k in range(p1):
                for c in range(r1):
                    V[j*p + k][c] = V1[j*p1 + k][c]
            for k in range(p2):
                for c in range(r2):
                    V[j*p + p1 + k][r1 + c] = V2[j*p2 + k][c]
        for i in range(n):
            for k in range(p1):
                for c in range(r1):
                    W[k*n + i][c] = W1[k*n + i][c]
            for k in range(p2):
                for c in range(r2):
                    W[(p1+k)*n + i][r1 + c] = W2[k*n + i][c]
        return U, V, W

    # Build (2,2,2) rank 7
    d_222 = make_222()

    # Build (2,2,4) rank 10
    d_224 = make_224()

    # Build (2,4,2) rank 14: split m=4 into 2+2, two (2,2,2)
    d_242 = embed_split_m([d_222, d_222], 2, 4, 2)

    # Build (2,2,1) rank 4
    d_221 = make_221()

    # Build (2,4,1) rank 8: split m=4 into 2+2, two (2,2,1)
    d_241 = embed_split_m([d_221, d_221], 2, 4, 1)

    # Build (2,4,3) rank 22: split p=3 into 2+1, (2,4,2) and (2,4,1)
    d_243 = embed_split_p([d_242, d_241], 2, 4, 3)

    # Build (2,4,4) rank 20: split m=4 into 2+2, two (2,2,4)
    d_244 = embed_split_m([d_224, d_224], 2, 4, 4)

    # Build (2,4,7) rank 42: split p=7 into 4+3, (2,4,4) and (2,4,3)
    d_247 = embed_split_p([d_244, d_243], 2, 4, 7)

    # Build (1,4,7) rank 1
    d_147 = make_1mp(4, 7)

    # Build (3,4,7) rank 43: split n=3 into 2+1, (2,4,7) and (1,4,7)
    U, V, W = embed_split_n([d_247, d_147], 3, 4, 7)

    rank = len(U[0])

    # Verify
    import random
    for trial in range(20):
        A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        C_expected = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        C_actual = [[0] * p for _ in range(n)]
        for c in range(rank):
            lA = sum(U[i*m + j][c] * A[i][j] for i in range(n) for j in range(m))
            mB = sum(V[j*p + k][c] * B[j][k] for j in range(m) for k in range(p))
            for i in range(n):
                for k in range(p):
                    C_actual[i][k] += W[k*n + i][c] * lA * mB
        if C_actual != C_expected:
            raise ValueError(f"Verification failed. C_actual={C_actual}, C_expected={C_expected}")

    return {"U": U, "V": V, "W": W}
