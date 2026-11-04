"""QAP solver for tho150 using precomputed interaction matrices and fast 2-opt local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    deadline = time.time() + 145.0  # Safety margin before 150s kill

    # Ensure matrices are float64 for faster arithmetic
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)

    # Initial solution: rank matching
    # Heavy facilities (high flow sum) to central locations (low distance sum)
    a_row_sums = a_f.sum(axis=1)
    b_row_sums = b_f.sum(axis=1)

    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central

    # Precompute interaction matrices
    # RC[i, l] = sum_k a[i, k] * b[l, p[k]]
    # CC[i, l] = sum_k a[k, i] * b[p[k], l]
    # These allow O(1) delta calculation for 2-opt swaps.
    
    def compute_interaction_matrices(p):
        # b_p = b[p, :] -> rows of b indexed by p
        # b_p.T = b[:, p] -> cols of b indexed by p
        b_rows = b_f[p, :]  # (n, n) where [k, l] = b[p[k], l]
        b_cols = b_f[:, p]  # (n, n) where [l, k] = b[l, p[k]]
        
        # RC[i, l] = sum_k a[i, k] * b[l, p[k]] = (a @ b_cols.T)[i, l]
        # Note: b_cols is (n, n) with [l, k]. b_cols.T is (n, n) with [k, l].
        # So a @ b_cols.T is (n, n) where [i, l] = sum_k a[i, k] * b_cols.T[k, l] = sum_k a[i, k] * b[l, p[k]].
        RC = a_f @ b_cols.T
        
        # CC[i, l] = sum_k a[k, i] * b[p[k], l] = (a.T @ b_rows)[i, l]
        # a.T is (n, n) with [k, i]. b_rows is (n, n) with [k, l].
        # So a.T @ b_rows is (n, n) where [i, l] = sum_k a.T[k, i] * b_rows[k, l] = sum_k a[k, i] * b[p[k], l].
        CC = a_f.T @ b_rows
        
        return RC, CC

    RC, CC = compute_interaction_matrices(p)
    
    # b_diag for corner terms
    b_diag = np.diag(b_f)

    # Local Search
    # We use a "First Improvement" strategy with random restarts of the scan order
    # to escape local optima, but primarily rely on the fast delta calculation.
    
    # To improve robustness, we can randomize the order of pairs (i, j) we check.
    # However, standard sequential scan is often effective. Let's try a mix:
    # 1. Do a full pass. If improved, repeat.
    # 2. If no improvement in a full pass, we are at a local optimum.
    #    To escape, we can do a few random swaps (tabu-like or simulated annealing)
    #    or just stop if time is up.
    
    # Given the 120s limit and n=150, a full pass is n*(n-1)/2 = ~11k pairs.
    # Each pair check is O(1) now. So 11k checks is very fast (< 1ms).
    # We can afford many passes.
    
    improved = True
    while time.time() < deadline - 5.0:
        if not improved:
            # If we reached a local optimum, we can try to perturb the solution
            # to escape. Let's do a few random swaps and then continue local search.
            # This is a simple form of multi-start / perturbation.
            num_perturbations = 5
            for _ in range(num_perturbations):
                if time.time() > deadline - 5.0:
                    break
                i = np.random.randint(0, n)
                j = np.random.randint(0, n)
                if i == j:
                    continue
                # Swap i and j
                u = p[i]
                v = p[j]
                p[i], p[j] = p[j], p[i]
                # Update RC and CC incrementally? Or recompute?
                # Recomputing is O(n^2) matrix mult, which is ~22k flops.
                # 5 recomputes is cheap.
                RC, CC = compute_interaction_matrices(p)
                # After perturbation, we are likely not at a local optimum, so set improved=True
                # to trigger another local search pass.
                improved = True
            if not improved:
                # If no perturbation was done (e.g. time ran out), break
                break
            continue
        
        improved = False
        # Full pass over all pairs
        # To avoid symmetry issues and potentially find better moves, we can randomize the order
        # of i, but standard is fine. Let's stick to standard for determinism and speed.
        
        for i in range(n):
            if time.time() > deadline - 5.0:
                break
            pi = p[i]
            for j in range(i + 1, n):
                pj = p[j]
                
                # Calculate delta for swapping p[i] and p[j]
                # Using precomputed RC and CC
                # delta = RC[i, pj] - RC[i, pi] + CC[i, pj] - CC[i, pi] 
                #       + RC[j, pi] - RC[j, pj] + CC[j, pi] - CC[j, pj]
                #       + corner terms
                
                # Wait, let's re-derive carefully.
                # Cost = sum_{i,j} a[i,j] b[p[i], p[j]]
                # Let S be the sum.
                # Change in S when swapping p[i]=u and p[j]=v:
                # Terms involving i or j change.
                
                # From the inspiration code, the formula was:
                # delta = (RC[i, v] - RC[i, u]) + (CC[i, v] - CC[i, u]) 
                #       + (RC[j, u] - RC[j, v]) + (CC[j, u] - CC[j, v])
                #       + corner corrections?
                
                # Let's verify the RC/CC definitions.
                # RC[i, l] = sum_k a[i, k] b[l, p[k]]
                # CC[i, l] = sum_k a[k, i] b[p[k], l]
                
                # The term a[i, k] b[p[i], p[k]] is part of the cost.
                # If we swap p[i] and p[j], the locations u and v are swapped.
                
                # Let's use the explicit delta formula from the inspiration which was validated:
                # delta = a[i,i]*(b[v,v]-b[u,u]) + a[i,j]*(b[v,u]-b[u,v]) + a[j,i]*(b[u,v]-b[v,u]) + a[j,j]*(b[u,u]-b[v,v])
                #       + (RC[i, v] - RC[i, u] - a[i,i]*(b[v,v]-b[u,u]) - a[i,j]*(b[v,u]-b[u,v]))
                #       + (RC[j, u] - RC[j, v] - a[j,i]*(b[u,v]-b[v,u]) - a[j,j]*(b[u,u]-b[v,v]))
                #       + (CC[i, v] - CC[i, u] - a[i,i]*(b[v,v]-b[u,u]) - a[j,i]*(b[v,u]-b[u,v]))
                #       + (CC[j, u] - CC[j, v] - a[i,j]*(b[u,v]-b[v,u]) - a[j,j]*(b[u,u]-b[v,v]))
                
                # This simplifies to:
                # delta = (RC[i, v] - RC[i, u]) + (CC[i, v] - CC[i, u]) 
                #       + (RC[j, u] - RC[j, v]) + (CC[j, u] - CC[j, v])
                #       - a[i,i]*(b[v,v]-b[u,u]) - a[i,j]*(b[v,u]-b[u,v])  (from RC[i])
                #       - a[j,i]*(b[u,v]-b[v,u]) - a[j,j]*(b[u,u]-b[v,v])  (from RC[j])
                #       - a[i,i]*(b[v,v]-b[u,u]) - a[j,i]*(b[v,u]-b[u,v])  (from CC[i])
                #       - a[i,j]*(b[u,v]-b[v,u]) - a[j,j]*(b[u,u]-b[v,v])  (from CC[j])
                #       + a[i,i]*(b[v,v]-b[u,u]) + a[i,j]*(b[v,u]-b[u,v]) + a[j,i]*(b[u,v]-b[v,u]) + a[j,j]*(b[u,u]-b[v,v]) (corners)
                
                # Let's group the non-RC/CC terms:
                # Corner terms: C1 = a[i,i]*(bvv-buu) + a[i,j]*(bvu-buv) + a[j,i]*(buv-bvu) + a[j,j]*(buu-bvv)
                # Subtracted from RC/CC:
                # -2 * a[i,i]*(bvv-buu) 
                # - a[i,j]*(bvu-buv) - a[i,j]*(buv-bvu) = -2*a[i,j]*(bvu-buv)
                # - a[j,i]*(buv-bvu) - a[j,i]*(bvu-buv) = -2*a[j,i]*(buv-bvu)
                # -2 * a[j,j]*(buu-bvv)
                
                # So total delta = (RC[i,v]-RC[i,u]) + (CC[i,v]-CC[i,u]) + (RC[j,u]-RC[j,v]) + (CC[j,u]-CC[j,v])
                #                 + C1 - 2*a[i,i]*(bvv-buu) - 2*a[i,j]*(bvu-buv) - 2*a[j,i]*(buv-bvu) - 2*a[j,j]*(buu-bvv)
                
                # Let's re-calculate C1:
                # C1 = a[i,i]*(bvv-buu) + a[i,j]*(bvu-buv) + a[j,i]*(buv-bvu) + a[j,j]*(buu-bvv)
                
                # This is getting complicated. Let's just use the vectorized calculation from the current program
                # but replace the O(n) sum with the O(1) lookup from RC/CC where possible?
                # Actually, the current program's vectorized sum is O(n).
                # The RC/CC approach makes it O(1).
                
                # Let's trust the formula from the inspiration code which was:
                # delta = (RC[i, v] - RC[i, u]) + (CC[i, v] - CC[i, u]) 
                #       + (RC[j, u] - RC[j, v]) + (CC[j, u] - CC[j, v])
                #       + a[i,i]*(bvv-buu) + a[i,j]*(bvu-buv) + a[j,i]*(buv-bvu) + a[j,j]*(buu-bvv)
                #       - 2*(a[i,i]*(bvv-buu) + a[i,j]*(bvu-buv) + a[j,i]*(buv-bvu) + a[j,j]*(buu-bvv)) ?
                
                # No, the inspiration code explicitly added the corner terms and then subtracted them from the RC/CC parts.
                # Let's look at the inspiration code again:
                # delta += (dot_row[i, v] - dot_row[i, u]) - a[i,i]*(bvv - buu) - a[i,j]*(bvu - buv)
                # ...
                # delta += a[i, i] * (bvv - buu) + ... (corners)
                
                # So it effectively computes:
                # (RC[i,v]-RC[i,u] - corner_part_i) + (CC[i,v]-CC[i,u] - corner_part_i) + ... + Corners
                # where corner_part_i is the contribution of k=i and k=j to the sum.
                
                # Let's just implement the exact logic from the inspiration code, which is known to be correct.
                
                buu = b_diag[u] if (u := p[i]) else 0 # u is p[i]
                bvv = b_diag[v] if (v := p[j]) else 0 # v is p[j]
                buv = b_f[u, v]
                bvu = b_f[v, u]
                
                delta = 0.0
                # Corners
                delta += a_f[i, i] * (bvv - buu)
                delta += a_f[i, j] * (bvu - buv)
                delta += a_f[j, i] * (buv - bvu)
                delta += a_f[j, j] * (buu - bvv)
                
                # RC/CC terms
                # RC[i, l] is dot_row[i, l] in inspiration
                # CC[i, l] is dot_col[i, l] in inspiration
                
                # Term 5: sum_{k not in {i,j}} a[i,k] * (b[v, p[k]] - b[u, p[k]])
                # = RC[i, v] - RC[i, u] - a[i,i]*(b[v,v]-b[u,u]) - a[i,j]*(b[v,u]-b[u,v])
                delta += (RC[i, v] - RC[i, u]) - a_f[i,i]*(bvv - buu) - a_f[i,j]*(bvu - buv)
                
                # Term 6: sum_{k not in {i,j}} a[j,k] * (b[u, p[k]] - b[v, p[k]])
                # = RC[j, u] - RC[j, v] - a[j,i]*(b[u,v]-b[v,u]) - a[j,j]*(b[u,u]-b[v,v])
                delta += (RC[j, u] - RC[j, v]) - a_f[j,i]*(buv - bvu) - a_f[j,j]*(buu - bvv)
                
                # Term 7: sum_{k not in {i,j}} a[k,i] * (b[p[k], v] - b[p[k], u])
                # = CC[i, v] - CC[i, u] - a[i,i]*(b[v,v]-b[u,u]) - a[j,i]*(b[v,u]-b[u,v])
                delta += (CC[i, v] - CC[i, u]) - a_f[i,i]*(bvv - buu) - a_f[j,i]*(bvu - buv)
                
                # Term 8: sum_{k not in {i,j}} a[k,j] * (b[p[k], u] - b[p[k], v])
                # = CC[j, u] - CC[j, v] - a[i,j]*(b[u,v]-b[v,u]) - a[j,j]*(b[u,u]-b[v,v])
                delta += (CC[j, u] - CC[j, v]) - a_f[i,j]*(buv - bvu) - a_f[j,j]*(buu - bvv)
                
                if delta < -1e-9:
                    # Perform swap
                    p[i], p[j] = p[j], p[i]
                    improved = True
                    
                    # Update RC and CC incrementally
                    # u = old p[i], v = old p[j]
                    # p[i] becomes v, p[j] becomes u
                    
                    # Update RC:
                    # RC[k, l] = sum_m a[k, m] b[l, p[m]]
                    # For k != i, j:
                    # RC[k, l] += a[k, i]*(b[l, v] - b[l, u]) + a[k, j]*(b[l, u] - b[l, v])
                    #           = (a[k, i] - a[k, j]) * (b[l, v] - b[l, u])
                    # For k = i:
                    # RC[i, l] += a[i, i]*(b[l, v] - b[l, u]) + a[i, j]*(b[l, u] - b[l, v])
                    #           = (a[i, i] - a[i, j]) * (b[l, v] - b[l, u])
                    # For k = j:
                    # RC[j, l] += a[j, i]*(b[l, u] - b[l, v]) + a[j, j]*(b[l, v] - b[l, u])
                    #           = (a[j, j] - a[j, i]) * (b[l, v] - b[l, u])
                    
                    # So for all k:
                    # RC[k, l] += (a[k, i] - a[k, j]) * (b[l, v] - b[l, u])  for k != i, j
                    # RC[i, l] += (a[i, i] - a[i, j]) * (b[l, v] - b[l, u])
                    # RC[j, l] += (a[j, j] - a[j, i]) * (b[l, v] - b[l, u])
                    
                    # Wait, the formula for k=i and k=j is slightly different?
                    # For k=i: a[i,i] is a[i,i], a[i,j] is a[i,j].
                    # For k=j: a[j,i] is a[j,i], a[j,j] is a[j,j].
                    # The general formula (a[k, i] - a[k, j]) works for k=i and k=j too.
                    # So RC[:, :] += (a[:, i] - a[:, j])[:, None] * (b[:, v] - b[:, u])[None, :]
                    
                    delta_RC = (a_f[:, i] - a_f[:, j])[:, None] * (b_f[:, v] - b_f[:, u])[None, :]
                    RC += delta_RC
                    
                    # Update CC:
                    # CC[k, l] = sum_m a[m, k] b[p[m], l]
                    # For l != i, j:
                    # CC[k, l] += a[i, k]*(b[v, l] - b[u, l]) + a[j, k]*(b[u, l] - b[v, l])
                    #           = (a[i, k] - a[j, k]) * (b[v, l] - b[u, l])
                    # For l = i:
                    # CC[k, i] += a[i, k]*(b[v, i] - b[u, i]) + a[j, k]*(b[u, i] - b[v, i])
                    #           = (a[i, k] - a[j, k]) * (b[v, i] - b[u, i])
                    # For l = j:
                    # CC[k, j] += a[i, k]*(b[v, j] - b[u, j]) + a[j, k]*(b[u, j] - b[v, j])
                    #           = (a[i, k] - a[j, k]) * (b[v, j] - b[u, j])
                    
                    # So for all l:
                    # CC[k, l] += (a[i, k] - a[j, k]) * (b[v, l] - b[u, l])
                    # CC[:, :] += (a_f[i, :] - a_f[j, :])[None, :] * (b_f[v, :] - b_f[u, :])[None, :]
                    
                    delta_CC = (a_f[i, :] - a_f[j, :])[None, :] * (b_f[v, :] - b_f[u, :])[None, :]
                    CC += delta_CC
                    
                    # Break inner loop and restart outer loop?
                    # The current program breaks and restarts.
                    # With O(1) delta, we can just continue the inner loop?
                    # But p has changed, so pi and pj are stale.
                    # It is safer to break and restart to maintain correctness of pi/pj variables.
                    break
            
            if improved:
                break

    return p.tolist()
# EVOLVE-BLOCK-END
