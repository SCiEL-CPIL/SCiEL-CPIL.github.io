"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before the 150 s kill

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    # ------------------------------------------------------------------
    # Build the block-Strassen base decomposition (rank 36).
    # 4 Strassen 2x2x2 blocks (28 terms) + 8 direct terms for column k=4.
    # ------------------------------------------------------------------
    BASE_RANK = 36
    U = np.zeros((n * m, BASE_RANK))
    V = np.zeros((m * p, BASE_RANK))
    W = np.zeros((p * n, BASE_RANK))
    col = [0]

    def strassen_block(y, z, w):
        """y, z, w are tuples of (row,col) indices:
           y00,y01,y10,y11 (A-block), z00,z01,z10,z11 (B-block),
           w00,w01,w10,w11 (C-block, indexed (i,k) -> w_idx(k,i))."""
        y00, y01, y10, y11 = y
        z00, z01, z10, z11 = z
        w00, w01, w10, w11 = w
        c = col[0]
        # M1 = (y00+y11)(z00+z11) -> +w00 +w11
        U[u_idx(*y00)][c] += 1; U[u_idx(*y11)][c] += 1
        V[v_idx(*z00)][c] += 1; V[v_idx(*z11)][c] += 1
        W[w_idx(*w00)][c] += 1; W[w_idx(*w11)][c] += 1
        col[0] += 1
        # M2 = (y10+y11)z00 -> +w10 -w11
        U[u_idx(*y10)][c] += 1; U[u_idx(*y11)][c] += 1
        V[v_idx(*z00)][c] += 1
        W[w_idx(*w10)][c] += 1; W[w_idx(*w11)][c] += -1
        col[0] += 1
        # M3 = y00(z01-z11) -> +w01 +w11
        U[u_idx(*y00)][c] += 1
        V[v_idx(*z01)][c] += 1; V[v_idx(*z11)][c] += -1
        W[w_idx(*w01)][c] += 1; W[w_idx(*w11)][c] += 1
        col[0] += 1
        # M4 = y11(z10-z00) -> +w00 +w10
        U[u_idx(*y11)][c] += 1
        V[v_idx(*z10)][c] += 1; V[v_idx(*z00)][c] += -1
        W[w_idx(*w00)][c] += 1; W[w_idx(*w10)][c] += 1
        col[0] += 1
        # M5 = (y00+y01)z11 -> -w00 +w01
        U[u_idx(*y00)][c] += 1; U[u_idx(*y01)][c] += 1
        V[v_idx(*z11)][c] += 1
        W[w_idx(*w00)][c] += -1; W[w_idx(*w01)][c] += 1
        col[0] += 1
        # M6 = (y10-y00)(z00+z01) -> +w11
        U[u_idx(*y10)][c] += 1; U[u_idx(*y00)][c] += -1
        V[v_idx(*z00)][c] += 1; V[v_idx(*z01)][c] += 1
        W[w_idx(*w11)][c] += 1
        col[0] += 1
        # M7 = (y01-y11)(z10+z11) -> +w00
        U[u_idx(*y01)][c] += 1; U[u_idx(*y11)][c] += -1
        V[v_idx(*z10)][c] += 1; V[v_idx(*z11)][c] += 1
        W[w_idx(*w00)][c] += 1
        col[0] += 1

    # A is 2x4 (i in {0,1}, j in {0..3}); B is 4x5 (j in {0..3}, k in {0..4})
    # Split A into left (j=0,1) and right (j=2,3); B into top (j=0,1) and
    # bottom (j=2,3), and split C's columns k=0,1 / k=2,3 / k=4.
    yL = [(0,0),(0,1),(1,0),(1,1)]
    yR = [(0,2),(0,3),(1,2),(1,3)]
    zTL = [(0,0),(0,1),(1,0),(1,1)]
    zTR = [(0,2),(0,3),(1,2),(1,3)]
    zBL = [(2,0),(2,1),(3,0),(3,1)]
    zBR = [(2,2),(2,3),(3,2),(3,3)]
    w01 = [(0,0),(0,1),(1,0),(1,1)]   # C[0:2, 0:2] as (i,k)
    w23 = [(0,2),(0,3),(1,2),(1,3)]   # C[0:2, 2:4] as (i,k)

    # Four 2x2x2 Strassen blocks
    strassen_block(yL, zTL, w01)  # A_left  x B_top    -> C[0:2,0:2]
    strassen_block(yR, zBL, w01)  # A_right x B_bottom -> C[0:2,0:2]
    strassen_block(yL, zTR, w23)  # A_left  x B_top    -> C[0:2,2:4]
    strassen_block(yR, zBR, w23)  # A_right x B_bottom -> C[0:2,2:4]

    # Last output column k=4: direct (schoolbook) terms.
    for ii in range(2):
        for jj in range(2):
            c = col[0]
            U[u_idx(ii, jj)][c] = 1
            V[v_idx(jj, 4)][c] = 1
            W[w_idx(4, ii)][c] = 1
            col[0] += 1
        for jj in range(2):
            c = col[0]
            U[u_idx(ii, jj + 2)][c] = 1
            V[v_idx(jj + 2, 4)][c] = 1
            W[w_idx(4, ii)][c] = 1
            col[0] += 1

    assert col[0] == BASE_RANK, col[0]

    # ------------------------------------------------------------------
    # Verify the base decomposition exactly:
    #   sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == 1  for all (i,j,k).
    # ------------------------------------------------------------------
    def verify(Um, Vm, Wm):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(Um.shape[1]):
                        s += Um[i*m+j, r] * Vm[j*p+k, r] * Wm[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback: pure schoolbook rank 40 (always valid).
        R = n * m * p
        Us = np.zeros((n*m, R)); Vs = np.zeros((m*p, R)); Ws = np.zeros((p*n, R))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Us[i*m+j, c] = 1; Vs[j*p+k, c] = 1; Ws[k*n+i, c] = 1
                    c += 1
        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

    # ------------------------------------------------------------------
    # Structural rank reduction:
    #  (1) remove terms whose rank-1 tensor is exactly a linear combination
    #      of the others (exact dependency -> drop without changing product);
    #  (2) local search: try to replace a small subset of terms by an equal
    #      number of simpler (lower-support) terms, preserving the tensor.
    # ------------------------------------------------------------------
    def tensor_of(r, Um, Vm, Wm):
        # rank-1 slice: T[i,j,k] = U[i,j,r]*V[j,k,r]*W[k,i,r]
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                uj = Um[i*m+j, r]
                if uj == 0:
                    continue
                for k in range(p):
                    T[i, j, k] = uj * Vm[j*p+k, r] * Wm[k*n+i, r]
        return T

    def full_tensor(Um, Vm, Wm):
        T = np.zeros((n, m, p))
        for r in range(Um.shape[1]):
            T += tensor_of(r, Um, Vm, Wm)
        return T

    def try_remove_dependent(Um, Vm, Wm):
        """Repeatedly drop any term whose tensor lies in the span of the rest."""
        R = Um.shape[1]
        keep = list(range(R))
        changed = True
        while changed:
            changed = False
            R = len(keep)
            if R <= 1:
                break
            # Build the matrix of all terms' tensors (flattened).
            vecs = np.array([tensor_of(keep[r], Um, Vm, Wm).flatten()
                             for r in range(R)])
            # For each term, check if it is in span of the others.
            for r in range(R):
                others = np.delete(vecs, r, axis=0)
                v = vecs[r]
                if np.linalg.norm(v) < 1e-12:
                    keep = [keep[x] for x in range(R) if x != r]
                    changed = True
                    break
                # Is v in colspace of others?
                rank_o = np.linalg.matrix_rank(others)
                rank_all = np.linalg.matrix_rank(np.vstack([others, v]))
                if rank_all == rank_o:
                    keep = [keep[x] for x in range(R) if x != r]
                    changed = True
                    break
        # Rebuild matrices with only kept columns.
        if len(keep) < R:
            Um2 = Um[:, keep]
            Vm2 = Vm[:, keep]
            Wm2 = Wm[:, keep]
            return Um2, Vm2, Wm2
        return Um, Vm, Wm

    # Local search: for each pair of terms, attempt to find an alternative
    # 2-term representation of (T_a + T_b) using a restricted dictionary of
    # simple rank-1 "atoms". If found with fewer nonzero atoms, accept.
    def build_atom_dictionary():
        """A small dictionary of candidate rank-1 tensors built from sparse
           U/V/W patterns (single, double, and Strassen-like products)."""
        atoms = []
        # all single-product atoms  (i,j) x (j,k) -> (i,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T = np.zeros((n, m, p))
                    T[i, j, k] = 1
                    atoms.append(T)
        # Strassen-like atoms for each 2x2x2 sub-block (the 7 patterns)
        def strassen_atoms(y, z, w):
            y00,y01,y10,y11 = y
            z00,z01,z10,z11 = z
            w00,w01,w10,w11 = w
            def mk(u_terms, v_terms, w_terms):
                T = np.zeros((n, m, p))
                for (ii,jj),cu in u_terms:
                    for (jj2,kk),cv in v_terms:
                        if jj == jj2:
                            for (kk2,ii2),cw in w_terms:
                                if kk == kk2 and ii == ii2:
                                    T[ii, jj, kk] += cu*cv*cw
                return T
            return [
                mk([(y00,1),(y11,1)], [(z00,1),(z11,1)], [(w00,1),(w11,1)]),
                mk([(y10,1),(y11,1)], [(z00,1)], [(w10,1),(w11,-1)]),
                mk([(y00,1)], [(z01,1),(z11,-1)], [(w01,1),(w11,1)]),
                mk([(y11,1)], [(z10,1),(z00,-1)], [(w00,1),(w10,1)]),
                mk([(y00,1),(y01,1)], [(z11,1)], [(w00,-1),(w01,1)]),
                mk([(y10,1),(y00,-1)], [(z00,1),(z01,1)], [(w11,1)]),
                mk([(y01,1),(y11,-1)], [(z10,1),(z11,1)], [(w00,1)]),
            ]
        for y in (yL, yR):
            for z in (zTL, zTR, zBL, zBR):
                for w in (w01, w23):
                    atoms.extend(strassen_atoms(y, z, w))
        # de-duplicate
        uniq = []
        seen = set()
        for a in atoms:
            key = a.tobytes()
            if key not in seen:
                seen.add(key)
                uniq.append(a)
        return np.array(uniq)

    ATOMS = build_atom_dictionary()  # shape (A, n*m*p)

    def solve_small(target, max_terms):
        """Find up to max_terms atoms summing to target (exact). Returns list
           of atom indices or None. Uses a greedy + backtracking search over a
           small candidate set of atoms with nonzero overlap with target."""
        target_f = target.flatten()
        nz = np.where(np.abs(target_f) > 1e-12)[0]
        if len(nz) == 0:
            return []
        # Candidate atoms: those that share at least one nonzero coordinate.
        cand = []
        for a in range(len(ATOMS)):
            if np.any(np.abs(ATOMS[a, nz]) > 1e-12):
                cand.append(a)
        # Greedy: repeatedly pick the atom that best reduces residual L2.
        residual = target_f.copy()
        chosen = []
        for _ in range(max_terms):
            if np.linalg.norm(residual) < 1e-9:
                break
            best = -1; best_norm = np.inf
            for a in cand:
                if a in chosen:
                    continue
                r = residual - ATOMS[a]
                nn = np.linalg.norm(r)
                if nn < best_norm:
                    best_norm = nn; best = a
            if best < 0:
                return None
            chosen.append(best)
            residual = residual - ATOMS[best]
        if np.linalg.norm(residual) < 1e-9:
            return chosen
        return None

    def local_search(Um, Vm, Wm, time_budget):
        """Try to reduce rank by replacing pairs of terms with simpler ones."""
        R = Um.shape[1]
        improved = True
        while improved and (time.time() < time_budget):
            improved = False
            R = Um.shape[1]
            for a in range(R):
                if time.time() >= time_budget:
                    return Um, Vm, Wm
                for b in range(a + 1, R):
                    pair = tensor_of(a, Um, Vm, Wm) + tensor_of(b, Um, Vm, Wm)
                    sol = solve_small(pair, 2)
                    if sol is None:
                        continue
                    # Rebuild: drop terms a,b and add the solution atoms as new
                    # rank-1 terms. A new rank-1 term needs U,V,W columns.
                    # We represent each atom by its (i,j),(j,k),(k,i) support.
                    new_terms = []
                    ok = True
                    for idx in sol:
                        A = ATOMS[idx].reshape(n, m, p)
                        # factor into U,V,W: for each (i,j,k) with value v,
                        # we need U[i,j]=u, V[j,k]=v, W[k,i]=w with u*v*w=v.
                        # Use u=1, then V[j,k]*W[k,i]=v.
                        Ucol = np.zeros(n*m); Vcol = np.zeros(m*p); Wcol = np.zeros(p*n)
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    val = A[i, j, k]
                                    if abs(val) > 1e-12:
                                        Ucol[i*m+j] += 1
                                        Vcol[j*p+k] += val
                                        Wcol[k*n+i] += 1
                        new_terms.append((Ucol, Vcol, Wcol))
                    # Assemble new matrices.
                    keep = [x for x in range(R) if x not in (a, b)]
                    n_new = len(new_terms)
                    R2 = len(keep) + n_new
                    if R2 >= R:
                        continue  # no reduction
                    Um2 = np.zeros((n*m, R2)); Vm2 = np.zeros((m*p, R2)); Wm2 = np.zeros((p*n, R2))
                    c = 0
                    for x in keep:
                        Um2[:, c] = Um[:, x]; Vm2[:, c] = Vm[:, x]; Wm2[:, c] = Wm[:, x]
                        c += 1
                    for (Uc, Vc, Wc) in new_terms:
                        Um2[:, c] = Uc; Vm2[:, c] = Vc; Wm2[:, c] = Wc
                        c += 1
                    if verify(Um2, Vm2, Wm2):
                        Um, Vm, Wm = Um2, Vm2, Wm2
                        improved = True
                        break
                if improved:
                    break
        return Um, Vm, Wm

    # ---- Run the reduction pipeline ----
    # Step 1: remove exact linear dependencies.
    U, V, W = try_remove_dependent(U, V, W)
    if not verify(U, V, W):
        # should not happen; guard.
        pass

    # Step 2: local search with a time budget (leave margin).
    search_deadline = start_time + 130.0
    U, V, W = local_search(U, V, W, search_deadline)

    # Final verification; if anything broke, fall back to the verified base.
    if not verify(U, V, W):
        # rebuild base
        U = np.zeros((n*m, BASE_RANK)); V = np.zeros((m*p, BASE_RANK)); W = np.zeros((p*n, BASE_RANK))
        col[0] = 0
        strassen_block(yL, zTL, w01)
        strassen_block(yR, zBL, w01)
        strassen_block(yL, zTR, w23)
        strassen_block(yR, zBR, w23)
        for ii in range(2):
            for jj in range(2):
                U[u_idx(ii, jj)][col[0]] = 1; V[v_idx(jj, 4)][col[0]] = 1; W[w_idx(4, ii)][col[0]] = 1; col[0] += 1
            for jj in range(2):
                U[u_idx(ii, jj+2)][col[0]] = 1; V[v_idx(jj+2, 4)][col[0]] = 1; W[w_idx(4, ii)][col[0]] = 1; col[0] += 1

    # Convert to integer/half-integer entries (values are already integers).
    def to_list(M):
        return [[int(round(float(x))) for x in row] for row in M]

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
