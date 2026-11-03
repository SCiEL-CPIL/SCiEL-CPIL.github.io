import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    deadline = time.time() + 135  # 135s safety margin (5s buffer)

    # Strassen components for 2x2 * 2x2
    # Each component: (U_entries, V_entries, W_entries)
    # Indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),
    ]
    
    def emit_block(a_cs, b_rs, b_cs, c_cs, off, U, V, W, rank):
        for s in range(7):
            col = off + s
            if col >= rank:
                continue
            uc, vc, wc = strassen[s]
            for li, c in uc.items():
                r, cc = li // 2, li % 2
                U[r * m + a_cs + cc][col] += c
            for li, c in vc.items():
                r, cc = li // 2, li % 2
                V[(b_rs + r) * p + b_cs + cc][col] += c
            for li, c in wc.items():
                r, cc = li // 2, li % 2
                W[(c_cs + cc) * n + r][col] += c

    def build_strassen_28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        emit_block(0, 0, 0, 0, 0, U, V, W, rank)
        emit_block(0, 0, 2, 2, 7, U, V, W, rank)
        emit_block(2, 2, 0, 0, 14, U, V, W, rank)
        emit_block(2, 2, 2, 2, 21, U, V, W, rank)
        return U, V, W

    def verify_decomposition_fast(U, V, W, n, m, p):
        """Fast verification using numpy einsum."""
        rank = len(U[0])
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        
        # Compute the tensor: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # U shape: (n*m, rank) -> reshape to (n, m, rank)
        # V shape: (m*p, rank) -> reshape to (m, p, rank)
        # W shape: (p*n, rank) -> reshape to (p, n, rank)
        
        U_3d = U_np.reshape(n, m, rank)
        V_3d = V_np.reshape(m, p, rank)
        W_3d = W_np.reshape(p, n, rank)
        
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # This is a contraction over r
        # Let's compute it step by step or use einsum
        # U[i,j,r] * V[j,k,r] -> intermediate[i,j,k,r]
        # Then sum_r intermediate[i,j,k,r] * W[k,i,r]
        
        # Actually, let's just do it directly with loops for small sizes or einsum
        # T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        # This should give T[i,j,k]
        
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        # For matrix multiplication, T[i,j,k] should be 1 if i,j,k form a valid triple (which they always do in the standard basis)
        # Wait, the tensor for matrix multiplication M_{n,m,p} has entries T[i,j,k] = 1 for all i,j,k.
        # Because C_ik = sum_j A_ij B_jk.
        # The tensor is sum_{i,j,k} e_i (A) e_j (B) e_k (C) * delta_{ij} delta_{jk}? No.
        # The tensor is T = sum_{i,j,k} A_{ij} B_{jk} C_{ik}^*?
        # In the decomposition context:
        # We want sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i in [0,n), j in [0,m), k in [0,p).
        
        if not np.all(T == 1):
            return False
        return True

    def verify_decomposition(U, V, W, n, m, p):
        """Original slow verification, kept as fallback or for small checks."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True

    def is_rank1_tensor(T, nm, mp, pn):
        """Check if tensor T is rank 1."""
        # Find first non-zero
        T000 = None
        i0, j0, k0 = None, None, None
        for i in range(nm):
            if T000 is not None:
                break
            for j in range(mp):
                if T000 is not None:
                    break
                for k in range(pn):
                    if T[i, j, k] != 0:
                        T000 = T[i, j, k]
                        i0, j0, k0 = i, j, k
                        break
        if T000 is None:
            return True, None, None, None
        
        # Use numpy for fast verification
        u_vec = T[:, j0, k0].astype(float) / T000
        v_vec = T[i0, :, k0].astype(float) / T000
        w_vec = T[i0, j0, :].astype(float) / T000
        
        T_recon = np.einsum('i,j,k->ijk', u_vec, v_vec, w_vec)
        
        if not np.allclose(T, T_recon, atol=1e-9):
            return False, None, None, None
            
        # Ensure vectors are integers (or half-integers, but we work with ints here)
        # If T is integer, and rank 1, the vectors should be rational.
        # For the merge to be valid in integer arithmetic, we need to check if we can scale to integers.
        # But here we just want to know if the sum of two terms is a single term.
        # If it is, we replace the two columns with one column representing the sum.
        # The new column values are u_prime, v_prime, w_prime.
        # We need these to be integers (or at least representable).
        # Since U, V, W are integers, the sum tensor T is integer.
        # If T is rank 1, T = u v w^T.
        # u = T[:, j0, k0] / T000.
        # If T000 divides all entries in T[:, j0, k0], then u is integer.
        # Let's check if we can find integer vectors.
        
        # Actually, if T is rank 1 and integer, there exist integer vectors u', v', w' such that T = u' v' w'^T?
        # Not necessarily. Example: T = [[2, 0], [0, 0]]. Rank 1? No, rank 1 means outer product.
        # If T = u v w^T, and T is integer, u, v, w can be chosen as integers if we absorb factors.
        # Let g = gcd of all non-zero entries of u_vec.
        # This is getting complicated. Let's just try to construct the new column with the float values
        # and verify. If verify fails, it's not a valid merge.
        
        # But we need to output integers.
        # If T is rank 1, T = c * (u0 v0 w0^T) where u0, v0, w0 are primitive integer vectors?
        # Let's just return the float vectors and let the caller handle conversion/verification.
        
        return True, u_vec, v_vec, w_vec

    def try_merge_two_terms(U, V, W, n, m, p):
        rank = len(U[0])
        nm, mp, pn = n * m, m * p, p * n
        
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)

        for a in range(rank):
            if time.time() > deadline - 10:
                break
            for b in range(a + 1, rank):
                ua = U_np[:, a]
                ub = U_np[:, b]
                va = V_np[:, a]
                vb = V_np[:, b]
                wa = W_np[:, a]
                wb = W_np[:, b]

                T = np.einsum('i,j,k->ijk', ua, va, wa) + np.einsum('i,j,k->ijk', ub, vb, wb)

                is_r1, u_prime, v_prime, w_prime = is_rank1_tensor(T, nm, mp, pn)
                if not is_r1:
                    continue

                # Try to convert to integers
                # u_prime, v_prime, w_prime are floats.
                # Check if they are close to integers.
                if not (np.allclose(u_prime, np.round(u_prime), atol=1e-9) and
                        np.allclose(v_prime, np.round(v_prime), atol=1e-9) and
                        np.allclose(w_prime, np.round(w_prime), atol=1e-9)):
                    continue
                    
                u_prime = np.round(u_prime).astype(int)
                v_prime = np.round(v_prime).astype(int)
                w_prime = np.round(w_prime).astype(int)

                new_rank = rank - 1
                U_new = [[0] * new_rank for _ in range(nm)]
                V_new = [[0] * new_rank for _ in range(mp)]
                W_new = [[0] * new_rank for _ in range(pn)]

                col_map = [r for r in range(rank) if r != a and r != b]
                for new_c, old_c in enumerate(col_map):
                    for i in range(nm):
                        U_new[i][new_c] = U[i][old_c]
                    for i in range(mp):
                        V_new[i][new_c] = V[i][old_c]
                    for i in range(pn):
                        W_new[i][new_c] = W[i][old_c]

                last_c = new_rank - 1
                for i in range(nm):
                    U_new[i][last_c] = int(u_prime[i])
                for i in range(mp):
                    V_new[i][last_c] = int(v_prime[i])
                for i in range(pn):
                    W_new[i][last_c] = int(w_prime[i])

                if verify_decomposition_fast(U_new, V_new, W_new, n, m, p):
                    return U_new, V_new, W_new

        return None

    def try_merge_three_terms(U, V, W, n, m, p):
        """Try to merge 3 terms into 1 if possible (less likely, but worth checking)."""
        rank = len(U[0])
        nm, mp, pn = n * m, m * p, p * n
        
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)

        # Limit search space for 3-term merges
        # Only check a subset of triples to stay within time budget
        max_triples = 500
        checked = 0
        for a in range(rank):
            if time.time() > deadline - 15:
                break
            for b in range(a + 1, rank):
                if checked >= max_triples:
                    break
                for c in range(b + 1, rank):
                    if checked >= max_triples:
                        break
                    checked += 1
                    ua = U_np[:, a]
                    ub = U_np[:, b]
                    uc = U_np[:, c]
                    va = V_np[:, a]
                    vb = V_np[:, b]
                    vc = V_np[:, c]
                    wa = W_np[:, a]
                    wb = W_np[:, b]
                    wc = W_np[:, c]

                    T = (np.einsum('i,j,k->ijk', ua, va, wa) + 
                         np.einsum('i,j,k->ijk', ub, vb, wb) + 
                         np.einsum('i,j,k->ijk', uc, vc, wc))

                    is_r1, u_prime, v_prime, w_prime = is_rank1_tensor(T, nm, mp, pn)
                    if not is_r1:
                        continue

                    if not (np.allclose(u_prime, np.round(u_prime), atol=1e-9) and
                            np.allclose(v_prime, np.round(v_prime), atol=1e-9) and
                            np.allclose(w_prime, np.round(w_prime), atol=1e-9)):
                        continue
                        
                    u_prime = np.round(u_prime).astype(int)
                    v_prime = np.round(v_prime).astype(int)
                    w_prime = np.round(w_prime).astype(int)

                    new_rank = rank - 2
                    U_new = [[0] * new_rank for _ in range(nm)]
                    V_new = [[0] * new_rank for _ in range(mp)]
                    W_new = [[0] * new_rank for _ in range(pn)]

                    col_map = [r for r in range(rank) if r != a and r != b and r != c]
                    for new_c, old_c in enumerate(col_map):
                        for i in range(nm):
                            U_new[i][new_c] = U[i][old_c]
                        for i in range(mp):
                            V_new[i][new_c] = V[i][old_c]
                        for i in range(pn):
                            W_new[i][new_c] = W[i][old_c]

                    last_c = new_rank - 1
                    for i in range(nm):
                        U_new[i][last_c] = int(u_prime[i])
                    for i in range(mp):
                        V_new[i][last_c] = int(v_prime[i])
                    for i in range(pn):
                        W_new[i][last_c] = int(w_prime[i])

                    if verify_decomposition_fast(U_new, V_new, W_new, n, m, p):
                        return U_new, V_new, W_new

        return None

    U28, V28, W28 = build_strassen_28()

    if not verify_decomposition_fast(U28, V28, W28, n, m, p):
        # Fallback to schoolbook if Strassen construction is buggy (should not happen)
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
        return {"U": U, "V": V, "W": W}

    best_U, best_V, best_W = U28, V28, W28
    current_rank = len(best_U[0])
    
    # Iteratively try to reduce rank
    while time.time() < deadline - 20:
        merged = None
        # Try 2-term merges first (more likely)
        if current_rank > 2:
            merged = try_merge_two_terms(best_U, best_V, best_W, n, m, p)
        
        if merged is None and current_rank > 3:
            # Try 3-term merges if 2-term fails
            merged = try_merge_three_terms(best_U, best_V, best_W, n, m, p)

        if merged is None:
            break
            
        best_U, best_V, best_W = merged
        current_rank = len(best_U[0])

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
