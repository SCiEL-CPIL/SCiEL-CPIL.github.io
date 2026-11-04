import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 5]

def _verify(n, m, p, U, V, W):
    """Check exact decomposition: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] == 1 for all i,j,k"""
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    S = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return bool(np.allclose(S, 1.0, atol=1e-9))

def _rank_of(U, V, W):
    return U.shape[1]

def _make_schoolbook(n, m, p):
    r = n * m * p
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1
                V[j*p+k, c] = 1
                W[k*n+i, c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start = time.time()
    deadline = start + 130  # safety margin under 150s

    best = _make_schoolbook(n, m, p)
    best_rank = n * m * p

    def try_candidate(U, V, W):
        nonlocal best, best_rank
        r = U.shape[1]
        if r >= best_rank:
            return False
        if _verify(n, m, p, U, V, W):
            best = {"U": U, "V": V, "W": W}
            best_rank = r
            return True
        return False

    # ------------------------------------------------------------------
    # 1) Kronecker construction: (4x4 x 4x5) = (2x2 x 2x2)  ⊗  (2x2 x 2x1)
    #    rank = rA * rB
    # ------------------------------------------------------------------
    # Strassen rank-7 for 2x2 x 2x2.
    # Convention: U indexed (i,j), V (j,k), W (k,i) for i,j,k in {0,1}
    # S = U[i,j,c]*V[j,k,c]*W[k,i,c] must equal 1 iff (i,j,k) is the
    # product position (i,k) of A[i,j]*B[j,k]... but our tensor is the
    # multiplication tensor: T[i,j,k] = 1 for the "A[i,j] * B[j,k] -> C[i,k]"
    # So the target is: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for ALL (i,j,k)
    # where the triple (i,j,k) is a valid path. Actually the matmul tensor
    # M[i,j,k] = 1 for every (i,j,k) with i in [n), j in [m), k in [p)
    # because C[i,k] = sum_j A[i,j] B[j,k]; the tensor has a 1 at each
    # (i,j,k) representing the term A[i,j]*B[j,k].
    #
    # For 2x2 x 2x2 (n=2,m=2,p=2): target T[i,j,k]=1 for all 8 triples.
    # Strassen gives rank 7. Let's build it via the standard 7-term formula.

    def strassen_2x2x2x2():
        # Returns U (2x2 x 7), V (2x2 x 7), W (2x2 x 7)
        # Using the standard Strassen 7 products for C = A B (2x2):
        # A = [[a00,a01],[a10,a11]], B=[[b00,b01],[b10,b11]]
        # m1 = (a00+a11)(b00+b11)
        # m2 = (a10+a11) b00
        # m3 = a00 (b01-b11)
        # m4 = a11 (b10-b00)
        # m5 = (a00+a01) b11
        # m6 = (a10-a00)(b00+b10)
        # m7 = (a01-a11)(b01+b11)
        # C00 = m1+m4-m5+m7
        # C01 = m3+m5
        # C10 = m2+m4
        # C11 = m1-m2+m3+m6
        #
        # In tensor form: U[i,j,c] is the coefficient of A[i,j] in m_c
        # V[j,k,c] is coefficient of B[j,k] in m_c
        # W[k,i,c] is coefficient of m_c in C[i,k]
        U = np.zeros((2,2,7))
        V = np.zeros((2,2,7))
        W = np.zeros((2,2,7))
        # c=0: m1 = (a00+a11)(b00+b11)
        U[0,0,0]=1; U[1,1,0]=1
        V[0,0,0]=1; V[1,1,0]=1
        W[0,0,0]=1; W[1,1,0]=1
        # c=1: m2 = (a10+a11) b00
        U[1,0,1]=1; U[1,1,1]=1
        V[0,0,1]=1
        W[1,0,1]=1; W[1,1,1]=1
        # c=2: m3 = a00 (b01-b11)
        U[0,0,2]=1
        V[0,1,2]=1; V[1,1,2]=-1
        W[0,1,2]=1; W[1,1,2]=1
        # c=3: m4 = a11 (b10-b00)
        U[1,1,3]=1
        V[1,0,3]=1; V[0,0,3]=-1
        W[0,0,3]=1; W[1,0,3]=1
        # c=4: m5 = (a00+a01) b11
        U[0,0,4]=1; U[0,1,4]=1
        V[1,1,4]=1
        W[0,0,4]=-1; W[0,1,4]=1
        # c=5: m6 = (a10-a00)(b00+b10)
        U[1,0,5]=1; U[0,0,5]=-1
        V[0,0,5]=1; V[1,0,5]=1
        W[1,1,5]=1
        # c=6: m7 = (a01-a11)(b01+b11)
        U[0,1,6]=1; U[1,1,6]=-1
        V[0,1,6]=1; V[1,1,6]=1
        W[0,0,6]=1
        return U, V, W

    # 2x2 x 2x1 (n=2, m=2, p=1): C[i,0] = sum_j A[i,j] B[j,0]
    # This is just a 2x2 times 2x1. Rank? It's a 2x2x1 tensor.
    # T[i,j,0]=1 for i,j in {0,1}. This is the outer product of (i) and (j),
    # i.e., T = e_i e_j^T summed... Actually T[i,j,0]=1 is the all-ones 2x2
    # matrix at k=0. That's rank 1! Because T = u v^T where u=1_2 (over i),
    # v=1_2 (over j), and the k-dim is a single 1.
    # Wait: T[i,j,k] with k only 0. T[i,j,0]=1 = (1)*(1)*(1). So U[i,j,0]=1,
    # V[j,0,0]=1, W[0,i,0]=1. Rank 1.
    def lin_2x2x2x1():
        U = np.ones((2,2,1))
        V = np.ones((2,1,1))
        W = np.ones((1,2,1))
        return U, V, W

    # Kronecker product of two decompositions.
    # If T1 (n1,m1,p1) has rank r1 with (U1,V1,W1) and T2 (n2,m2,p2) has
    # rank r2 with (U2,V2,W2), then T1⊗T2 (n1*n2, m1*m2, p1*p2) has rank
    # r1*r2. The new indices:
    #   i = i1*n2 + i2, j = j1*m2 + j2, k = k1*p2 + k2
    #   U_new[i,j,c] = U1[i1,j1,c1]*U2[i2,j2,c2]  where c = c1*r2 + c2
    #   V_new[j,k,c] = V1[j1,k1,c1]*V2[j2,k2,c2]
    #   W_new[k,i,c] = W1[k1,i1,c1]*W2[k2,i2,c2]
    # The target T_new[i,j,k] = T1[i1,j1,k1]*T2[i2,j2,k2].
    #
    # For our case: 4x4x5 = (2x2x2) ⊗ (2x2x1)?? That gives n=4,m=4,p=2. No.
    # We need p=5. 5 is prime, can't factor as product of small ints nicely.
    # 5 = 5. So (2x2x2)⊗(2x1x5)? n=4,m=2,p=10. No.
    # We need n=4,m=4,p=5. Factor n=4=2*2, m=4=2*2, p=5. 5 is prime.
    # So pure Kronecker of two integer-dim tensors can't give p=5 unless one
    # factor has p=5. So we need a base tensor with p=5.
    #
    # Alternative: 4x4x5 = (2x2x5) ⊗ (2x2x1). n=4, m=4, p=5. 
    # rank = rank(2x2x5) * rank(2x2x1=1) = rank(2x2x5).
    # So we just need a low-rank 2x2x5. 2x2x5: C[i,k]=sum_j A[i,j]B[j,k],
    # i in {0,1}, j in {0,1}, k in {0,1,2,3,4}. This is 2x2x5 tensor, all
    # ones at valid (i,j,k). That's just two independent 2x1x5? No.
    # T[i,j,k]=1 for all i,j,k. This is the all-ones tensor of shape 2x2x5,
    # which has rank 1 (it's u v w^T with u=1_2, v=1_2, w=1_5). Rank 1!
    # So 2x2x5 has rank 1?? That can't be right for matmul.
    #
    # WAIT. I've been confusing myself. The matrix multiplication tensor
    # M_{n,m,p} has M[i,j,k]=1 for ALL i in [n), j in [m), k in [p). It is
    # NOT the all-ones tensor in the sense of rank 1. Let me reconsider.
    #
    # Actually M[i,j,k]=1 for every (i,j,k). As a 3-tensor, is it rank 1?
    # A rank-1 tensor is T[i,j,k]=a_i b_j c_k. For M, T[i,j,k]=1 = 1*1*1.
    # So yes, M is the all-ones tensor and has rank 1!
    #
    # But that contradicts Strassen. The resolution: the matrix multiplication
    # tensor is NOT the all-ones tensor. The correct matmul tensor is:
    #   M[i,j,k] = 1 if the triple (i,j,k) corresponds to a term in the product,
    # but the tensor that encodes "C[i,k] = sum_j A[i,j] B[j,k]" is the
    # tensor with M[i,j,k] = 1 for all (i,j,k) — and this IS rank 1 as a
    # 3-tensor?! No.
    #
    # Let me re-derive. The multiplication map is a bilinear map
    #   (A, B) -> C,  C[i,k] = sum_j A[i,j] B[j,k].
    # As a tensor in V_A^* ⊗ V_B^* ⊗ V_C, it is:
    #   M = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k  (where e_i is the functional
    #     picking A[i,j]... no.
    #
    # Standard: M = sum_{i,j,k}  A^i_j ⊗ B^j_k ⊗ C_{i,k}  where A^i_j is
    # the functional extracting A[i,j], B^j_k extracts B[j,k], C_{i,k} is
    # the vector that adds to C[i,k]. So M = sum_{i,j,k} (A^i_j ⊗ B^j_k ⊗ C_{i,k}).
    # This is a sum of n*m*p rank-1 terms. Its rank (tensor rank) is what
    # Strassen bounds. The tensor is M[i,j,k] = 1 for all i,j,k in the
    # sense that each "slot" (i,j,k) has coefficient 1. But the three index
    # sets are DIFFERENT spaces (A-indices, B-indices, C-indices). So M is
    # the all-ones tensor over the three index spaces, and its rank is the
    # matmul rank. It is NOT rank 1 because the three index spaces are
    # distinct; "rank 1" would mean M[i,j,k] = a_i b_j c_k for vectors a,
    # b, c — and indeed 1 = 1*1*1 satisfies that! 
    #
    # The catch: the three index spaces have DIFFERENT dimensions in general,
    # but here for 2x2x2 they're all 2. M[i,j,k]=1 = (1,1,1)_i (1,1,1)_j
    # (1,1,1)_k. So as a 3-tensor over the three spaces, it's rank 1?!
    #
    # No! I think I'm making an error. Let me reconsider what the tensor
    # actually is. The issue: the SAME index j appears in both A and B. In
    # the tensor M, the index j in "A[i,j]" and the index j in "B[j,k]" are
    # the SAME j (they're contracted in the product). So the tensor is:
    #   M = sum_j (sum_i A^i_j) ⊗ ... no.
    #
    # Correct formulation: M ∈ V_{ij}^* ⊗ V_{jk}^* ⊗ V_{ik}, where V_{ij}
    # is the space of A-matrices (dim n*m), V_{jk} is space of B-matrices
    # (dim m*p), V_{ik} is space of C-matrices (dim n*p). The tensor is:
    #   M = sum_{i,j,k} A^*_{ij} ⊗ B^*_{jk} ⊗ C_{ik}
    # where A^*_{ij} extracts the (i,j) entry of A, etc.
    # As a 3-tensor with indices (i,j), (j,k), (k,i) [or (i,j,k) grouping],
    # M[(i,j),(j,k),(k,i)] = 1 for all i,j,k. 
    #
    # Now, is this rank 1? A rank-1 element of V1^*⊗V2^*⊗V3 is f⊗g⊗h
    # where f∈V1^*, g∈V2^*, h∈V3. Then M[(i,j),(j,k),(k,i)] = f(i,j) g(j,k)
    # h(k,i). For this to equal 1 for ALL i,j,k, we need f(i,j)=1, g(j,k)=1,
    # h(k,i)=1 for all, which is possible (f=1, g=1, h=1). So M IS rank 1?!
    #
    # This is clearly wrong because matmul rank is known to be > 1. The error:
    # the three spaces V_{ij}, V_{jk}, V_{ik} — the index j in V_{ij} and the
    # index j in V_{jk} are the SAME underlying index, but in the tensor product
    # V_{ij}^* ⊗ V_{jk}^* ⊗ V_{ik}, they are treated as independent. The tensor
    # M has M[(i,j),(j,k),(k,i)] = 1. For rank 1, M[(i,j),(j,k),(k,i)] =
    # f(i,j) g(j,k) h(k,i). Setting f(i,j)=1 for all, g(j,k)=1 for all,
    # h(k,i)=1 for all gives M=1 everywhere. So M = 1⊗1⊗1 is rank 1.
    #
    # But that means the bilinear map (A,B)->C has a rank-1 tensor, which
    # would mean C[i,k] = (sum_{i,j} f(i,j) A[i,j]) (sum_{j,k} g(j,k) B[j,k])
    # h(k,i)?? That's not how matmul works. The resolution is that the tensor
    # M is NOT 1⊗1⊗1. Let me recompute M[(i,j),(j,k),(k,i)].
    #
    # The bilinear map is B(A,B) = C where C[i,k] = sum_j A[i,j] B[j,k].
    # The associated tensor M satisfies: for all A,B, C = M(A,B) means
    #   C[i,k] = sum_{(i,j),(j,k)} M[(i,j),(j,k),(i,k)] A[i,j] B[j,k].
    # So M[(i,j),(j,k),(i,k)] = 1 if the indices are consistent, i.e., the
    # (i,k) output slot matches. But the output index is (i,k), and the input
    # slots are (i,j) and (j,k). So M[(i,j),(j,k),(i,k)] = 1 for all i,j,k.
    # Note the output index is (i,k), NOT (k,i) or (i,j,k) independently.
    #
    # So the tensor has THREE index pairs: (i,j), (j,k), (i,k), and
    # M[(i,j),(j,k),(i,k)] = 1. Now for rank 1: M[(i,j),(j,k),(i,k)] =
    # f(i,j) g(j,k) h(i,k). For this to be 1 for all i,j,k: f=1, g=1, h=1.
    # So M = 1⊗1⊗1, rank 1. STILL rank 1?!
    #
    # I must be fundamentally misunderstanding the tensor. Let me look at it
    # differently. The point is: the SAME physical index j is shared between
    # the first and second slots. In the tensor product V_{ij}^* ⊗ V_{jk}^*,
    # the two copies of the j-index are DIFFERENT (one is in the ij-space,
    # one in the jk-space). The tensor M[(i,j),(j,k),(i,k)] = 1 means: for
    # every combination of (i,j), (j,k), (i,k) — but these are NOT independent!
    # Given (i,j) and (j,k), the (i,k) is determined (it's (i,k)). So the
    # tensor is really a function on triples where the third is determined by
    # the first two. So M is not a tensor on the full product space; it's a
    # tensor where the indices are linked.
    #
    # The correct tensor: M ∈ (V_A^*) ⊗ (V_B^*) ⊗ V_C where V_A has basis
    # {E_{ij}}, V_B has basis {F_{jk}}, V_C has basis {G_{ik}}. Then
    # M = sum_{i,j,k} E^*_{ij} ⊗ F^*_{jk} ⊗ G_{ik}. This is a sum of n*m*p
    # rank-1 terms. The tensor rank is the minimum number of rank-1 terms to
    # express M. And M ≠ 1⊗1⊗1 because 1⊗1⊗1 = (sum_i,j E^*_{ij}) ⊗
    # (sum_j,k F^*_{jk}) ⊗ (sum_i,k G_{ik}) = sum_{i,j,j',k,k'} E^*_{ij}⊗
    # F^*_{j'k} ⊗ G_{ik'}, which has terms where j≠j' and i,k don't match.
    # So 1⊗1⊗1 ≠ M. Good, so M is NOT rank 1. My earlier "M=1" was wrong
    # because I was conflating the linked indices.
    #
    # OK so back to construction. M_{4,4,5} = sum_{i,j,k} E^*_{ij} ⊗ F^*_{jk}
    # ⊗ G_{ik} with i∈[4], j∈[4], k∈[5]. Rank = ?
    #
    # Kronecker: M_{a,b,c} ⊗ M_{d,e,f} = M_{ad, be, cf}. Proof:
    # M_{ad,be,cf} = sum_{i1,i2,j1,j2,k1,k2} E^*_{i1d+i2, j1e+j2} ⊗
    # F^*_{j1e+j2, k1f+k2} ⊗ G_{i1d+i2, k1f+k2}
    # = (sum_{i1,j1,k1} E^*_{i1,j1}⊗F^*_{j1,k1}⊗G_{i1,k1}) ⊗
    #   (sum_{i2,j2,k2} ...)  [using Kronecker product of tensors]
    # = M_{a,b,c} ⊗ M_{d,e,f}.
    # And rank(M⊗N) ≤ rank(M)*rank(N).
    #
    # So M_{4,4,5} = M_{2,2,5} ⊗ M_{2,2,1}. rank ≤ rank(2,2,5)*rank(2,2,1).
    # rank(2,2,1): M_{2,2,1} = sum_{i,j,k=0} E^*_{ij}⊗F^*_{j0}⊗G_{i0}
    # = sum_{i,j} E^*_{ij}⊗F^*_{j0}⊗G_{i0}. This is a 2x2x1 tensor.
    # Its rank: it's the "2x2 matmul with 1 column". This is just the
    # bilinear map A(2x2), B(2x1) -> C(2x1). C[i,0] = sum_j A[i,j] B[j,0].
    # This tensor M_{2,2,1}[i,j,0] = 1 for all i,j. As a 3-tensor with
    # indices (i,j), (j,0), (i,0): M[(i,j),(j,0),(i,0)] = 1. 
    # Is this rank 1? M = sum_{i,j} E^*_{ij} ⊗ F^*_{j0} ⊗ G_{i0}.
    # Can we write it as f⊗g⊗h? f(i,j)g(j,0)h(i,0)=1 for all i,j.
    # f(i,j)=1, g(j,0)=1, h(i,0)=1 works. So M_{2,2,1} = 1⊗1⊗1? 
    # But 1⊗1⊗1 = (sum_{i,j}E^*_{ij})(sum_j F^*_{j0})(sum_i G_{i0})
    # = sum_{i,j,j',i'} E^*_{ij}⊗F^*_{j'0}⊗G_{i'0}. This has terms with
    # j≠j' and i≠i'. So 1⊗1⊗1 ≠ M_{2,2,1}. 
    # The linked indices again. M_{2,2,1} has the j-index linked between
    # slot1 and slot2, and i-index linked between slot1 and slot3.
    # So M_{2,2,1} is the "2x2 times 2x1" matmul tensor. Its rank is known:
    # for 2x2 x 2x1, the rank is 2 (it's just two independent dot products:
    # C[0,0]=A[0,0]B[0,0]+A[0,1]B[1,0], C[1,0]=A[1,0]B[0,0]+A[1,1]B[1,0]).
    # Actually this is the tensor sum_{i,j} E^*_{ij}⊗F^*_{j0}⊗G_{i0}.
    # We can write it as: (E^*_{00}⊗F^*_{00}⊗G_{00} + E^*_{01}⊗F^*_{10}⊗G_{00})
    # + (E^*_{10}⊗F^*_{00}⊗G_{10} + E^*_{11}⊗F^*_{10}⊗G_{10}).
    # That's 4 terms. Can we do 2? 
    # M = sum_j (sum_i E^*_{ij}⊗G_{i0}) ⊗ F^*_{j0}. For j=0: (E^*_{00}⊗G_{00}
    # + E^*_{10}⊗G_{10})⊗F^*_{00}. For j=1: (E^*_{01}⊗G_{00}+E^*_{11}⊗G_{10})
    # ⊗F^*_{10}. So M = (E^*_{00}⊗G_{00}+E^*_{10}⊗G_{10})⊗F^*_{00} +
    # (E^*_{01}⊗G_{00}+E^*_{11}⊗G_{10})⊗F^*_{10}. That's 2 terms, each of
    # which is a rank-1 tensor in (V_A⊗V_C)⊗V_B. But as a 3-tensor, each
    # term (E^*_{00}⊗G_{00}+E^*_{10}⊗G_{10})⊗F^*_{00} is a sum of two
    # rank-1 3-tensors. So total 4 rank-1 terms. Hmm.
    #
    # Actually the rank of M_{2,2,1}: this is the tensor for multiplying a
    # 2x2 by a 2x1. This is equivalent to two independent 2x1 dot products
    # sharing the B-matrix. The tensor is M[i,j,k=0]=1. The rank of the
    # 2x2x1 all-ones... I recall that the rank of the 2x2x2 matmul tensor is
    # 7 (Strassen). For 2x2x1, I believe the rank is 2. Let me just compute
    # it: M = sum_{i,j} E^*_{ij} ⊗ F^*_{j0} ⊗ G_{i0}.
    # Claim: rank = 2. 
    # M = (E^*_{00}⊗F^*_{00}⊗G_{00}) + (E^*_{10}⊗F^*_{00}⊗G_{10}) +
    #     (E^*_{01}⊗F^*_{10}⊗G_{00}) + (E^*_{11}⊗F^*_{10}⊗G_{10}).
    # Can we combine? Group by F: 
    # F^*_{00} ⊗ (E^*_{00}⊗G_{00} + E^*_{10}⊗G_{10}) + F^*_{10}⊗(E^*_{01}⊗G_{00}
    # + E^*_{11}⊗G_{10}). Each parenthesis is a rank-2 tensor in V_A⊗V_C.
    # So M is a sum of 2 terms, each of the form f_j ⊗ (rank-2 in A,C).
    # As a 3-tensor, this is 2 * 2 = 4 rank-1 terms. So rank ≤ 4.
    # Can we do better? The 2x2x1 matmul tensor... I'll just use rank 4 to be
    # safe, or compute it. Actually, let me just use the schoolbook for the
    # small factors and rely on Kronecker for the structure, then verify.
    #
    # Let me just directly build M_{4,4,5} via Kronecker of M_{2,2,5} and
    # M_{2,2,1}, using schoolbook for both, then the rank is 20*2=40.
    # Schoolbook M_{2,2,5} has rank 2*2*5=20. Schoolbook M_{2,2,1} has rank 2.
    # So Kronecker gives rank 40. That's better than 80 (schoolbook 4*4*5=80).
    #
    # Even better: use Strassen (rank 7) for M_{2,2,2} and build M_{2,2,5}
    # from M_{2,2,2} and M_{2,2,3}? 5=2+3, not a product. 
    # M_{2,2,5} can't be a Kronecker of smaller integer-dim tensors unless
    # 5 factors. So M_{2,2,5} must be built directly or as a sum.
    #
    # Let me just do: M_{4,4,5} = M_{2,2,5} ⊗ M_{2,2,1}.
    # Build M_{2,2,5} with schoolbook (rank 20), M_{2,2,1} with schoolbook
    # (rank 2). Kronecker rank = 40. Verify.
    #
    # Actually, I can do M_{2,2,1} with rank 2 if I find it. Let me just
    # use schoolbook for now and get rank 40, which is a solid improvement
    # over 80.

    def kron_decomp(U1, V1, W1, U2, V2, W2):
        """Kronecker product of two decompositions."""
        n1, m1, r1 = U1.shape[0], U1.shape[1] // U1.shape[0], U1.shape[2]
        # U1 is (n1, m1, r1), V1 is (m1, p1, r1), W1 is (p1, n1, r1)
        p1 = V1.shape[1]
        n2, m2, r2 = U2.shape[0], U2.shape[1] // U2.shape[0], U2.shape[2]
        p2 = V2.shape[1]
        n, m, p = n1*n2, m1*m2, p1*p2
        r = r1 * r2
        U = np.zeros((n, m, r))
        V = np.zeros((m, p, r))
        W = np.zeros((p, n, r))
        for i1 in range(n1):
            for j1 in range(m1):
                for k1 in range(p1):
                    for i2 in range(n2):
                        for j2 in range(m2):
                            for k2 in range(p2):
                                i = i1*n2 + i2
                                j = j1*m2 + j2
                                k = k1*p2 + k2
                                for c1 in range(r1):
                                    for c2 in range(r2):
                                        c = c1*r2 + c2
                                        U[i, j, c] += U1[i1, j1, c1] * U2[i2, j2, c2]
                                        V[j, k, c] += V1[j1, k1, c1] * V2[j2, k2, c2]
                                        W[k, i, c] += W1[k1, i1, c1] * W2[k2, i2, c2]
        return U, V, W

    def schoolbook_3d(n, m, p):
        r = n*m*p
        U = np.zeros((n, m, r))
        V = np.zeros((m, p, r))
        W = np.zeros((p, n, r))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i, j, c] = 1
                    V[j, k, c] = 1
                    W[k, i, c] = 1
                    c += 1
        return U, V, W

    # Build M_{2,2,5} schoolbook (rank 20) and M_{2,2,1} schoolbook (rank 2)
    if time.time() < deadline:
        U1, V1, W1 = schoolbook_3d(2, 2, 5)  # rank 20
        U2, V2, W2 = schoolbook_3d(2, 2, 1)  # rank 2
        U, V, W = kron_decomp(U1, V1, W1, U2, V2, W2)
        # Flatten to (n*m, r), (m*p, r), (p*n, r)
        Uf = U.reshape(4*4, 40)
        Vf = V.reshape(4*5, 40)
        Wf = W.reshape(5*4, 40)
        try_candidate(Uf, Vf, Wf)

    # Try to improve M_{2,2,1} to rank 2 (if possible)
    # M_{2,2,1}: C[i,0] = A[i,0]B[0,0] + A[i,1]B[1,0]
    # = (A[i,0]B[0,0] + A[i,1]B[1,0])
    # This is rank 2: 
    # Term 1: i=0 part: A[0,0]B[0,0] + A[0,1]B[1,0] -> U[0,0,0]=1,U[0,1,0]=1; V[0,0,0]=1,V[1,0,0]=1; W[0,0,0]=1
    # Term 2: i=1 part: A[1,0]B[0,0] + A[1,1]B[1,0] -> U[1,0,1]=1,U[1,1,1]=1; V[0,0,1]=1,V[1,0,1]=1; W[0,1,1]=1
    # So rank 2!
    def m_221_rank2():
        U = np.zeros((2,2,2))
        V = np.zeros((2,1,2))
        W = np.zeros((1,2,2))
        U[0,0,0]=1; U[0,1,0]=1; V[0,0,0]=1; V[1,0,0]=1; W[0,0,0]=1
        U[1,0,1]=1; U[1,1,1]=1; V[0,0,1]=1; V[1,0,1]=1; W[0,1,1]=1
        return U, V, W

    if time.time() < deadline:
        U1, V1, W1 = schoolbook_3d(2, 2, 5)  # rank 20
        U2, V2, W2 = m_221_rank2()  # rank 2
        U, V, W = kron_decomp(U1, V1, W1, U2, V2, W2)
        Uf = U.reshape(16, 40)
        Vf = V.reshape(20, 40)
        Wf = W.reshape(20, 40)
        try_candidate(Uf, Vf, Wf)

    # Now try to improve M_{2,2,5}. 
    # M_{2,2,5} = M_{2,2,2} ⊗ M_{2,2,?} ... 5 is prime.
    # M_{2,2,5} = M_{2,2,2} + M_{2,2,3}? No, Kronecker is product not sum.
    # Direct: M_{2,2,5} schoolbook rank 20. Can we do better?
    # M_{2,2,5}: C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k], i∈{0,1}, k∈{0..4}.
    # This is 2 rows of 2x5 matmul. Each row is a 2x5 matmul (rank 2*5=10
    # schoolbook, or can it be less?). 2x5 matmul: C[i,k]=sum_j A[i,j]B[j,k],
    # j∈{0,1}. This is just 2 dot products of length 2 for each k. The tensor
    # M_{2,2,5}[i,j,k]=1. Rank? It's the "2x2 x 2x5" tensor. 
    # I think the rank of 2x2 x 2x5 is 2*5=10 (schoolbook) or possibly less.
    # Actually 2x2 x 2xp: C[i,k]=A[i,0]B[0,k]+A[i,1]B[1,k]. This is a sum
    # over j of A[i,j]B[j,k]. The tensor is sum_{i,j,k} E^*_{ij}⊗F^*_{jk}⊗G_{ik}.
    # For fixed i, it's a 2xp matmul. The whole thing: I'll just use schoolbook
    # rank 10 for 2x2x5... wait, 2*2*5=20. 
    #
    # Let me just try rank-40 via Kronecker (done above) and also try a direct
    # structured approach for the full 4x4x5.
    #
    # Alternative: M_{4,4,5} = M_{4,4,2} ⊕ M_{4,4,3} (direct sum, rank adds).
    # But we want Kronecker (rank multiplies).
    #
    # Let me also try: build M_{4,4,5} directly with a structured ansatz and
    # verify. Use the fact that 4=2*2 and 5 is prime.
    #
    # For now, rank 40 is a good result. Let me also attempt to find a lower
    # rank via a more clever construction.
    #
    # Try: M_{4,4,5} via M_{2,2,5} (rank 20) ⊗ M_{2,2,1} (rank 2) = rank 40.
    # Can M_{2,2,5} be done in rank < 20? 
    # M_{2,2,5}: I'll try rank 10 (half of schoolbook) via a structured guess.
    # Actually, let me just try to verify rank 40 and move on. If time permits,
    # try lower.

    # Try a direct rank-40 verification with the Kronecker result (already done).
    # Now try to beat 40.
    #
    # Idea: M_{4,4,5} = M_{2,4,5} ⊗ M_{2,1,1}? n=4, m=4, p=5. 
    # M_{2,4,5} has n=2,m=4,p=5, rank schoolbook 40. ⊗ M_{2,1,1} (rank 2)
    # = rank 80. Worse.
    #
    # M_{4,4,5} = M_{4,2,5} ⊗ M_{1,2,1}? n=4,m=4,p=5. M_{4,2,5} n=4,m=2,p=5
    # rank 40. ⊗ M_{1,2,1} rank 2 = 80. Worse.
    #
    # The best Kronecker factorization for 4,4,5 with 5 prime is 2*2*5:
    # M_{2,2,5} ⊗ M_{2,2,1}. That's what we have.
    #
    # To beat rank 40, we need rank(M_{2,2,5}) * rank(M_{2,2,1}) < 40.
    # rank(M_{2,2,1}) = 2 (we found). So we need rank(M_{2,2,5}) < 20.
    # Can M_{2,2,5} be done in rank < 20? 
    # M_{2,2,5}: 2x2 times 2x5. Let me try to find a low-rank decomposition.
    # C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k], i=0,1; k=0..4.
    # This is 10 independent outputs, each a dot product of length 2.
    # The tensor M[i,j,k]=1 for i∈{0,1}, j∈{0,1}, k∈{0..4}.
    # Rank of this tensor: it's the all-ones 2x2x5 tensor with linked indices.
    # I believe the rank is 2*5 = 10 (each (i,k) output needs its own term,
    # but j is shared). Actually, let me think: 
    # M = sum_{i,k} (sum_j E^*_{ij}⊗F^*_{jk}) ⊗ G_{ik}.
    # For each (i,k), sum_j E^*_{ij}⊗F^*_{jk} is a rank-2 tensor in V_A⊗V_B.
    # So M is a sum of 10 terms, each rank-2 in (A,B) times rank-1 in C.
    # As a 3-tensor, that's 10*2 = 20 rank-1 terms. So rank ≤ 20.
    # Can we do better? The 2x2 x 2x5 tensor... I think rank 10 might be
    # achievable if we can share. But I'm not sure. Let me just try rank 20
    # (schoolbook) for M_{2,2,5} and get rank 40 overall.
    #
    # Let me finalize with rank 40.

    # Convert best to lists
    if isinstance(best['U'], np.ndarray):
        best['U'] = best['U'].tolist()
        best['V'] = best['V'].tolist()
        best['W'] = best['W'].tolist()

    return best

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]) if result['U'] else 0)
    U = np.array(result['U'])
    V = np.array(result['V'])
    W = np.array(result['W'])
    n, m, p = 4, 4, 5
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    S = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    print("Max error:", np.max(np.abs(S - 1.0)))
    print("All close:", np.allclose(S, 1.0, atol=1e-9))
