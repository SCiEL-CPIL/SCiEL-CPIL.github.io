import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 6, 6, 6

    # Build the 2x2 Strassen decomposition
    terms_2x2 = []
    terms_2x2.append(([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]))  # M1
    terms_2x2.append(([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]))  # M2
    terms_2x2.append(([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]))  # M3
    terms_2x2.append(([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]))  # M4
    terms_2x2.append(([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]))  # M5
    terms_2x2.append(([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]))  # M6
    terms_2x2.append(([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]))  # M7

    bs2 = 2
    nb2 = n // bs2  # 3

    R_initial = 189  # 27 blocks * 7 Strassen terms

    U = np.zeros((n*m, R_initial), dtype=np.int64)
    V = np.zeros((m*p, R_initial), dtype=np.int64)
    W = np.zeros((p*n, R_initial), dtype=np.int64)

    col = 0
    for ib in range(nb2):
        for jb in range(nb2):
            for kb in range(nb2):
                for (upat, vpat, wpat) in terms_2x2:
                    for (lr, lc, cv) in upat:
                        U[(ib*bs2+lr)*m + (kb*bs2+lc), col] += cv
                    for (lr, lc, cv) in vpat:
                        V[(kb*bs2+lr)*p + (jb*bs2+lc), col] += cv
                    for (lr, lc, cv) in wpat:
                        W[(jb*bs2+lc)*n + (ib*bs2+lr), col] += cv
                    col += 1

    # Fast verification using einsum
    def verify_fast(Um, Vm, Wm):
        r = Um.shape[1]
        U_r = Um.reshape(n, m, -1)
        V_r = Vm.reshape(m, p, -1)
        W_r = Wm.reshape(p, n, -1)
        T = np.einsum('ikt,kjt,jit->ijk', U_r, V_r, W_r)
        return np.all(T == 1)

    assert verify_fast(U, V, W), "Initial decomposition is invalid"

    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_r = R_initial

    # Precompute per-term tensor contributions for fast removal checks
    # Each term t contributes a 6x6x6 tensor: contrib[t,i,j,k] = U[i,k,t]*V[k,j,t]*W[j,i,t]
    # The total tensor should be all ones (for the matmul structure)
    def compute_contribs(Um, Vm, Wm):
        r = Um.shape[1]
        U_r = Um.reshape(n, m, -1)
        V_r = Vm.reshape(m, p, -1)
        W_r = Wm.reshape(p, n, -1)
        # contrib[t, i, j, k] = U[i,k,t] * V[k,j,t] * W[j,i,t]
        contribs = np.zeros((r, n, p, n), dtype=np.int64)
        for t in range(r):
            contribs[t] = np.einsum('ik,kj,ji->ijk', 
                                     U_r[:,:,t], V_r[:,:,t], W_r[:,:,t])
        return contribs

    contribs = compute_contribs(current_U, current_V, current_W)
    total = np.sum(contribs, axis=0)
    assert np.all(total == 1), "Initial tensor check failed"

    def try_remove_fast(contribs, total, t):
        remaining = total - contribs[t]
        return np.all(remaining == 1)

    # Greedy removal with multiple passes and different orderings
    for pass_num in range(30):
        if time.time() > deadline - 20:
            break
        removed_any = False
        # Try different orderings
        if pass_num % 3 == 0:
            order = list(range(current_r - 1, -1, -1))
        elif pass_num % 3 == 1:
            order = list(range(current_r))
        else:
            order = list(range(0, current_r, 2)) + list(range(1, current_r, 2))
        
        for t in order:
            if time.time() > deadline - 20:
                break
            if try_remove_fast(contribs, total, t):
                # Remove term t
                mask = np.ones(current_r, dtype=bool)
                mask[t] = False
                current_U = current_U[:, mask]
                current_V = current_V[:, mask]
                current_W = current_W[:, mask]
                current_r -= 1
                # Update contribs and total
                contribs = np.delete(contribs, t, axis=0)
                total = total - contribs[t]
                removed_any = True
                break
        if not removed_any:
            # Try a few more passes with shuffled orders
            for subpass in range(5):
                if time.time() > deadline - 20:
                    break
                np.random.seed(pass_num * 1000 + subpass * 7 + 42)
                order = np.random.permutation(current_r).tolist()
                for t in order:
                    if time.time() > deadline - 20:
                        break
                    if try_remove_fast(contribs, total, t):
                        mask = np.ones(current_r, dtype=bool)
                        mask[t] = False
                        current_U = current_U[:, mask]
                        current_V = current_V[:, mask]
                        current_W = current_W[:, mask]
                        current_r -= 1
                        contribs = np.delete(contribs, t, axis=0)
                        total = total - contribs[t]
                        removed_any = True
                        break
                if removed_any:
                    break
            if not removed_any:
                break

    # Final verification
    assert verify_fast(current_U, current_V, current_W), "Final decomposition is invalid"

    return {
        "U": current_U.tolist(),
        "V": current_V.tolist(),
        "W": current_W.tolist()
    }
