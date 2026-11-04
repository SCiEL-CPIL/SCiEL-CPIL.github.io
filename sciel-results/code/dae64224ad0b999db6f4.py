import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    """
    Solve QAP: assign n facilities to n locations minimizing sum a[i,j]*b[p[i],p[j]].
    
    Strategy:
    1. Spectral (eigenvalue) initialization via Procrustes alignment.
    2. Fast vectorized 2-swap local search with precomputed row/col sums.
    3. Multiple restarts with perturbation to escape local minima.
    4. Hungarian linearization as a global move between restarts.
    5. Final polish with tight time budget.
    """
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    # Precompute
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    b_col_sums = b.sum(axis=0)

    def compute_cost(p):
        """Compute exact QAP cost for permutation p."""
        total = 0.0
        for i in range(n):
            total += np.dot(a[i], b[p[i], p])
        return total

    def spectral_init():
        """
        Initialize permutation using spectral embedding / Procrustes alignment.
        Uses the fact that the QAP can be approximated by aligning the 
        eigenvector bases of a and b.
        """
        # Center the matrices for better spectral properties
        a_c = a - a.mean()
        b_c = b - b.mean()
        
        # Get eigen-decompositions
        # For symmetric-ish problems, use eigh; otherwise eig
        try:
            evals_a, evecs_a = np.linalg.eigh(a_c)
            evals_b, evecs_b = np.linalg.eigh(b_c)
        except:
            evals_a, evecs_a = np.linalg.eig(a_c)
            evals_b, evecs_b = np.linalg.eig(b_c)
        
        # Sort by eigenvalue magnitude (descending) for alignment
        idx_a = np.argsort(-np.abs(evals_a))
        idx_b = np.argsort(-np.abs(evals_b))
        evecs_a = evecs_a[:, idx_a]
        evecs_b = evecs_b[:, idx_b]
        
        # Procrustes: find optimal rotation R such that evecs_a @ R ≈ evecs_b
        # SVD of evecs_a^T @ evecs_b
        M = evecs_a.T @ evecs_b
        U, S, Vt = np.linalg.svd(M)
        R = U @ Vt
        
        # Project: X = evecs_a @ R should align with evecs_b
        # Assign facility i to location j where the embedding coordinates match best
        X = evecs_a @ R  # facility embeddings in aligned space
        
        # For each facility, find the location whose embedding is closest
        # Use Hungarian for optimal assignment
        C = -X @ evecs_b.T  # negative distance (we want to minimize)
        row_ind, col_ind = linear_sum_assignment(C)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        
        return p

    def rank_init():
        """Rank-based initialization: match facilities by flow strength to locations by distance centrality."""
        heavy = np.argsort(-a_row_sums, kind="stable")
        central = np.argsort(b_row_sums, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p

    def hungarian_init(p_current):
        """Hungarian linearization given current permutation."""
        b_col_p = b[:, p_current]
        b_row_p = b[p_current, :]
        C = a @ b_col_p.T + a.T @ b_row_p
        row_ind, col_ind = linear_sum_assignment(C)
        p_h = np.empty(n, dtype=np.int64)
        p_h[row_ind] = col_ind
        return p_h

    def local_search_2swap(p, cost, time_limit, batch_size=300):
        """
        Fast 2-swap local search with O(n) delta computation.
        Uses precomputed structures for speed.
        """
        p = p.copy()
        
        # Precompute for fast delta: for each facility i, store a[i,:] and a[:,i]
        # The swap delta for (i,j):
        # delta = sum_k a[i,k]*(b[v,p[k]]-b[u,p[k]]) + sum_k a[j,k]*(b[u,p[k]]-b[v,p[k]])
        #       + sum_k a[k,i]*(b[p[k],v]-b[p[k],u]) + sum_k a[k,j]*(b[p[k],u]-b[p[k],v])
        # where u=p[i], v=p[j]
        # Simplify: delta = (a[i,:]-a[j,:]) dot (b[v,p]-b[u,p]) + (a[:,i]-a[:,j]) dot (b[p,v]-b[p,u])
        #            + correction terms for k=i and k=j
        
        improved = True
        no_improve = 0
        max_no_improve = 5000
        
        while improved and time.time() < time_limit:
            improved = False
            best_delta = 0.0
            best_i, best_j = -1, -1
            
            for _ in range(batch_size):
                if time.time() > time_limit:
                    break
                i, j = np.random.randint(0, n, 2)
                if i == j:
                    continue
                u = p[i]
                v = p[j]
                if u == v:
                    continue
                
                # Fast delta computation
                # Row part: (a[i,:] - a[j,:]) dot (b[v,:] - b[u,:])[p]
                diff_a_row = a[i] - a[j]
                diff_b_row = b[v] - b[u]
                row_part = np.dot(diff_a_row, diff_b_row[p])
                
                # Col part: (a[:,i] - a[:,j]) dot (b[:,v] - b[:,u])[p]
                diff_a_col = a[:, i] - a[:, j]
                diff_b_col = b[:, v] - b[:, u]
                col_part = np.dot(diff_a_col, diff_b_col[p])
                
                # Correction: when k=i or k=j, the above double-counts or miscounts
                # For k=i: in row_part, we have a[i,i]*b[v,u] - a[j,i]*b[v,u] - a[i,i]*b[u,u] + a[j,i]*b[u,u]
                #          but actual change for (i,i) term: a[i,i]*(b[v,v]-b[u,u])
                #          and for (j,i) term: a[j,i]*(b[u,v]-b[v,u])  [j moves from v to u, i stays... wait]
                # Let me recompute more carefully.
                #
                # Original contribution involving i or j:
                #   a[i,i]*b[u,u] + a[i,j]*b[u,v] + sum_{k!=i,j} a[i,k]*b[u,p[k]]
                #   a[j,i]*b[v,u] + a[j,j]*b[v,v] + sum_{k!=i,j} a[j,k]*b[v,p[k]]
                #   a[i,i]*b[u,u] + a[j,i]*b[v,u] + sum_{k!=i,j} a[k,i]*b[p[k],u]
                #   a[i,j]*b[u,v] + a[j,j]*b[v,v] + sum_{k!=i,j} a[k,j]*b[p[k],v]
                #
                # After swap (i gets v, j gets u):
                #   a[i,i]*b[v,v] + a[i,j]*b[v,u] + sum_{k!=i,j} a[i,k]*b[v,p[k]]
                #   a[j,i]*b[u,v] + a[j,j]*b[u,u] + sum_{k!=i,j} a[j,k]*b[u,p[k]]
                #   a[i,i]*b[v,v] + a[j,i]*b[u,v] + sum_{k!=i,j} a[k,i]*b[p[k],v]
                #   a[i,j]*b[v,u] + a[j,j]*b[u,u] + sum_{k!=i,j} a[k,j]*b[p[k],u]
                #
                # Delta for k!=i,j terms:
                #   sum_k a[i,k]*(b[v,p[k]]-b[u,p[k]]) + a[j,k]*(b[u,p[k]]-b[v,p[k]])
                #   + a[k,i]*(b[p[k],v]-b[p[k],u]) + a[k,j]*(b[p[k],u]-b[p[k],v])
                # = (a[i,:]-a[j,:]) dot (b[v,p]-b[u,p])[k!=i,j] + (a[:,i]-a[:,j]) dot (b[:,v]-b[:,u])[p[k] for k!=i,j]
                #
                # The full dot products include k=i and k=j, so we need to subtract those:
                # For k=i in first sum: a[i,i]*(b[v,u]-b[u,u]) + a[j,i]*(b[u,u]-b[v,u])
                #   = (a[i,i]-a[j,i])*(b[v,u]-b[u,u])
                # For k=j in first sum: a[i,j]*(b[v,v]-b[u,v]) + a[j,j]*(b[u,v]-b[v,v])
                #   = (a[i,j]-a[j,j])*(b[v,v]-b[u,v])
                #
                # For k=i in second sum: a[i,i]*(b[v,v]-b[u,u]) + a[j,i]*(b[u,v]-b[v,u])
                #   wait, b[:,v][p[i]] = b[v,u], b[:,u][p[i]] = b[u,u]
                #   so: a[i,i]*(b[v,u]-b[u,u]) + a[j,i]*(b[u,v]-b[v,u])
                #   = a[i,i]*(b[v,u]-b[u,u]) - a[j,i]*(b[v,u]-b[u,v])
                #   Hmm, this is getting complex. Let me just add correction terms directly.
                #
                # Direct correction for the 4 diagonal/cross terms:
                # Old: a[i,i]*b[u,u] + a[i,j]*b[u,v] + a[j,i]*b[v,u] + a[j,j]*b[v,v]
                #      (these appear twice each in the full sum: once in row, once in col)
                # New: a[i,i]*b[v,v] + a[i,j]*b[v,u] + a[j,i]*b[u,v] + a[j,j]*b[u,u]
                #
                # The full dot products row_part + col_part include:
                # row_part has: a[i,i]*b[v,u] - a[j,i]*b[v,u] - a[i,i]*b[u,u] + a[j,i]*b[u,u] (k=i)
                #              a[i,j]*b[v,v] - a[j,j]*b[v,v] - a[i,j]*b[u,v] + a[j,j]*b[u,v] (k=j)
                # col_part has: a[i,i]*b[v,u] - a[j,i]*b[v,u] - a[i,i]*b[u,u] + a[j,i]*b[u,u] (k=i, p[i]=u)
                #              a[i,j]*b[v,v] - a[j,j]*b[v,v] - a[i,j]*b[u,v] + a[j,j]*b[u,v] (k=j, p[j]=v)
                #
                # So row_part + col_part for k=i,j gives:
                # 2*(a[i,i]-a[j,i])*(b[v,u]-b[u,u]) + 2*(a[i,j]-a[j,j])*(b[v,v]-b[u,v])
                #
                # But the TRUE change for the i,j,i,j block is:
                # (a[i,i]*b[v,v] + a[i,j]*b[v,u] + a[j,i]*b[u,v] + a[j,j]*b[u,u])
                # - (a[i,i]*b[u,u] + a[i,j]*b[u,v] + a[j,i]*b[v,u] + a[j,j]*b[v,v])
                # = a[i,i]*(b[v,v]-b[u,u]) + a[i,j]*(b[v,u]-b[u,v]) + a[j,i]*(b[u,v]-b[v,u]) + a[j,j]*(b[u,u]-b[v,v])
                # = a[i,i]*(b[v,v]-b[u,u]) - a[i,j]*(b[v,u]-b[u,v]) - a[j,i]*(b[v,u]-b[u,v]) + a[j,j]*(b[u,u]-b[v,v])
                # = (a[i,i]-a[j,j])*(b[v,v]-b[u,u]) - (a[i,j]+a[j,i])*(b[v,u]-b[u,v])
                #
                # So correction = true_block_change - 2*(a[i,i]-a[j,i])*(b[v,u]-b[u,u]) - 2*(a[i,j]-a[j,j])*(b[v,v]-b[u,v])
                
                buu = b[u, u]
                bvv = b[v, v]
                buv = b[u, v]
                bvu = b[v, u]
                
                true_block = (a[i,i]*(bvv - buu) + a[i,j]*(bvu - buv) 
                            + a[j,i]*(buv - bvu) + a[j,j]*(buu - bvv))
                
                double_counted = (2*(a[i,i]-a[j,i])*(bvu - buu) 
                                + 2*(a[i,j]-a[j,j])*(bvv - buv))
                
                delta = row_part + col_part - double_counted + true_block
                
                if delta < best_delta:
                    best_delta = delta
                    best_i, best_j = i, j
            
            if best_i != -1 and best_delta < -1e-9:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                cost += best_delta
                improved = True
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > max_no_improve:
                    break

        return p, cost

    def perturb(p, num_swaps):
        """Randomly swap num_swaps pairs in permutation."""
        p = p.copy()
        for _ in range(num_swaps):
            i, j = np.random.randint(0, n, 2)
            if i != j:
                p[i], p[j] = p[j], p[i]
        return p

    # --- Generate initial solutions ---
    candidates = []
    
    # Spectral initialization
    p_spec = spectral_init()
    candidates.append(p_spec)
    
    # Rank initialization
    p_rank = rank_init()
    candidates.append(p_rank)
    
    # Hungarian from spectral
    p_hung1 = hungarian_init(p_spec)
    candidates.append(p_hung1)
    
    # Hungarian from rank
    p_hung2 = hungarian_init(p_rank)
    candidates.append(p_hung2)
    
    # Evaluate all candidates
    best_p = None
    best_cost = float('inf')
    for p in candidates:
        c = compute_cost(p)
        if c < best_cost:
            best_cost = c
            best_p = p.copy()
    
    # --- Main search loop with restarts ---
    elapsed = time.time() - start_time
    remaining = deadline - time.time()
    
    # Phase 1: Local search on best initial solution
    phase1_time = min(30.0, remaining * 0.35)
    if phase1_time > 2.0:
        best_p, best_cost = local_search_2swap(best_p, best_cost, time.time() + phase1_time)
    
    # Phase 2: Multiple restarts with perturbation + Hungarian
    num_restarts = 8
    restart_budget = (deadline - time.time()) * 0.55
    per_restart = restart_budget / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Perturb current best
        num_swaps = max(5, n // 10)
        p_pert = perturb(best_p, num_swaps)
        
        # Apply Hungarian linearization
        p_h = hungarian_init(p_pert)
        c_h = compute_cost(p_h)
        
        # Also try local search on perturbed directly
        p_ls, c_ls = local_search_2swap(p_pert, compute_cost(p_pert), time.time() + per_restart * 0.5)
        
        # Choose best
        if c_ls < c_h:
            p_new, c_new = p_ls, c_ls
        else:
            p_new, c_new = p_h, c_h
            # Quick local search on Hungarian result
            if time.time() < deadline - 8.0:
                p_new, c_new = local_search_2swap(p_new, c_new, time.time() + per_restart * 0.3)
        
        if c_new < best_cost:
            best_cost = c_new
            best_p = p_new.copy()
    
    # Phase 3: Final polish
    if time.time() < deadline - 3.0:
        best_p, best_cost = local_search_2swap(best_p, best_cost, deadline - 1.0, batch_size=200)
    
    # Final check
    if time.time() > deadline - 2.0:
        return best_p.tolist()
    
    return best_p.tolist()
