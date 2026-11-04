"""QAP tai256c solver: iterated local search with Hungarian re-optimization and VNS perturbations."""
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin under 120s

    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])

    def fast_swap_delta(p, u, v):
        pu = p[u]
        pv = p[v]

        x, y = pu, pv
        # Compute efficiently using precomputed differences
        # sum_{k!=u,v} (a[u,k]-a[v,k])*(b[y,p[k]]-b[x,p[k]])
        diff_a_row = a[u, :] - a[v, :]  # shape (n,)
        diff_b_col = b[y, p] - b[x, p]  # shape (n,)
        s1 = np.dot(diff_a_row, diff_b_col)
        # Subtract k=u and k=v terms
        s1 -= diff_a_row[u] * diff_b_col[u]
        s1 -= diff_a_row[v] * diff_b_col[v]

        # sum_{k!=u,v} (a[k,u]-a[k,v])*(b[p[k],y]-b[p[k],x])
        diff_a_col = a[:, u] - a[:, v]  # shape (n,)
        diff_b_row = b[p, y] - b[p, x]  # shape (n,)
        s2 = np.dot(diff_a_col, diff_b_row)
        # Subtract k=u and k=v terms
        s2 -= diff_a_col[u] * diff_b_row[u]
        s2 -= diff_a_col[v] * diff_b_row[v]

        # Cross term: 2*(a[u,v]-a[v,u])*(b[y,x]-b[x,y])
        cross = 2.0 * (a[u, v] - a[v, u]) * (b[y, x] - b[x, y])

        # Self term: (a[u,u]-a[v,v])*(b[y,y]-b[x,x])
        self_term = (a[u, u] - a[v, v]) * (b[y, y] - b[x, x])

        return s1 + s2 + cross + self_term

    def local_search(p, cost, max_passes=100):
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            if passes % 3 == 0 and time.time() > deadline:
                return p, cost
            for i in range(n):
                if passes % 10 == 0 and time.time() > deadline:
                    return p, cost
                for j in range(i + 1, n):
                    delta = fast_swap_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
                        break
                if improved:
                    break
        if time.time() > deadline:
            return p, cost
        return p, cost

    def hungarian_reopt(p):
        """Use Hungarian algorithm on a linearized cost matrix to escape local optima."""
        B_p = b[np.ix_(np.arange(n), p)]
        M1 = a @ B_p.T  # M1[i,j] = sum_k a[i,k]*b[j,p[k]]
        
        Bp2 = b[np.ix_(p, np.arange(n))]
        M2 = a.T @ Bp2  # M2[i,j] = sum_k a[k,i]*b[p[k],j]
        
        C = M1 + M2
        
        # Solve assignment
        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        return new_p

    def spectral_init():
        """Generate initial solution using spectral clustering / graph partitioning ideas."""
        a_sym = (a + a.T) / 2.0
        
        try:
            from scipy.sparse.linalg import eigsh
            from scipy.sparse import csc_matrix
            A_sp = csc_matrix(a_sym)
            vals, vecs = eigsh(A_sp, k=3, which='SM')
            idx = np.argsort(vecs[:, 1])
            return idx.astype(np.int64)
        except:
            a_row_sums = a.sum(axis=1)
            b_col_sums = b.sum(axis=1)
            heavy = np.argsort(-a_row_sums, kind="stable")
            central = np.argsort(b_col_sums, kind="stable")
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
            return p

    def eigen_init():
        """Generate initial permutation using eigenvalue-based heuristic."""
        try:
            a_sym = (a + a.T) / 2.0
            b_sym = (b + b.T) / 2.0
            
            eigvals_a, eigvecs_a = np.linalg.eigh(a_sym)
            eigvals_b, eigvecs_b = np.linalg.eigh(b_sym)
            
            k = min(16, n)
            idx_a = np.argsort(np.abs(eigvals_a))[::-1][:k]
            idx_b = np.argsort(np.abs(eigvals_b))[::-1][:k]
            
            score_a = np.zeros(n)
            for idx in idx_a:
                score_a += np.abs(eigvals_a[idx]) * np.abs(eigvecs_a[:, idx])
            
            score_b = np.zeros(n)
            for idx in idx_b:
                score_b += np.abs(eigvals_b[idx]) * np.abs(eigvecs_b[:, idx])
            
            fac_order = np.argsort(score_a)[::-1]
            loc_order = np.argsort(score_b)
            
            p = np.zeros(n, dtype=np.int64)
            for i in range(n):
                p[fac_order[i]] = loc_order[i]
            
            p_rev = np.zeros(n, dtype=np.int64)
            for i in range(n):
                p_rev[fac_order[i]] = loc_order[n - 1 - i]
            
            cost1 = compute_cost(p)
            cost2 = compute_cost(p_rev)
            
            if cost1 <= cost2:
                return p
            else:
                return p_rev
        except:
            return np.random.permutation(n).astype(np.int64)

    def hungarian_init():
        """Generate an initial assignment using the Hungarian algorithm on a surrogate cost matrix."""
        cost_matrix = a @ b + a.T @ b.T
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p

    def perturb_swap(p, k, rng):
        """Randomly swap k pairs of positions."""
        p = p.copy()
        for _ in range(k):
            i, j = rng.choice(n, size=2, replace=False)
            p[i], p[j] = p[j], p[i]
        return p

    def perturb_relocate(p, k, rng):
        """Relocate k facilities to random different positions."""
        p = p.copy()
        for _ in range(k):
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            if i == j:
                j = (j + 1) % n
            val = p[i]
            if i < j:
                p[i:j] = p[i+1:j+1]
            else:
                p[j+1:i+1] = p[j:i]
            p[j] = val
        return p

    def perturb_cycle(p, k, rng):
        """Perform a cyclic shift of a sub-permutation of length k."""
        p = p.copy()
        if k < 3 or k > n:
            return p
        start = rng.integers(0, n - k + 1)
        end = start + k
        segment = p[start:end].copy()
        shift = rng.integers(1, k)
        p[start:end] = np.roll(segment, shift)
        return p

    def vns_perturb(p, level, rng):
        """Variable Neighborhood Search perturbation."""
        if level == 1:
            return perturb_swap(p, 3, rng)
        elif level == 2:
            return perturb_relocate(p, 3, rng)
        elif level == 3:
            return perturb_cycle(p, 5, rng)
        elif level == 4:
            p_temp = perturb_swap(p, 2, rng)
            return perturb_relocate(p_temp, 2, rng)
        else:
            p_temp = perturb_swap(p, 2, rng)
            return perturb_cycle(p_temp, 4, rng)

    # Initial solution: try multiple strategies
    rng = np.random.default_rng(42)
    candidates = []
    
    # 1. Spectral initialization
    p_spectral = spectral_init()
    candidates.append(p_spectral)
    
    # 2. Rank matching
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_col_sums, kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    candidates.append(p_rank)
    
    # 3. Eigenvalue-based initialization
    p_eigen = eigen_init()
    candidates.append(p_eigen)
    
    # 4. Hungarian initialization
    p_hung = hungarian_init()
    candidates.append(p_hung)
    
    # 5. Random permutations
    for _ in range(3):
        candidates.append(rng.permutation(n).astype(np.int64))
    
    # Evaluate all candidates and pick the best
    best_p = None
    best_cost = float('inf')
    for cand in candidates:
        c = compute_cost(cand)
        if c < best_cost:
            best_cost = c
            best_p = cand.copy()
    
    current_p = best_p.copy()
    current_cost = best_cost
    
    # First local search
    current_p, current_cost = local_search(current_p, current_cost)
    if current_cost < best_cost:
        best_p = current_p.copy()
        best_cost = current_cost

    # Iterated Local Search with VNS perturbations and periodic Hungarian re-optimization
    iteration = 0
    vns_level = 1
    consecutive_no_improve = 0
    
    while time.time() < deadline:
        iteration += 1

        # Increase VNS level if stuck
        if consecutive_no_improve > 3:
            vns_level = min(5, vns_level + 1)
            consecutive_no_improve = 0
        elif consecutive_no_improve > 0 and rng.random() < 0.3:
            vns_level = min(5, vns_level + 1)

        # Perturb using VNS neighborhood
        p = vns_perturb(best_p, vns_level, rng)

        # Local search from perturbed
        p, cost = local_search(p, compute_cost(p))

        # Every few iterations, try Hungarian re-optimization
        if iteration % 3 == 0 and time.time() < deadline - 40:
            h_p = hungarian_reopt(p)
            h_cost = compute_cost(h_p)
            h_p, h_cost = local_search(h_p, h_cost)
            if h_cost < cost:
                p, cost = h_p, h_cost

        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1

    return best_p.tolist()
