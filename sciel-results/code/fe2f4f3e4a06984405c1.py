# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    # --- Precompute helpers ---
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)

    def calc_cost_full(perm):
        return int(np.sum(a * b[np.ix_(perm, perm)]))

    def two_opt_delta(p, i, j):
        """Compute delta cost for swapping facilities i and j in permutation p."""
        li = p[i]
        lj = p[j]
        if li == lj:
            return 0
        # d[k] = b[lj, p[k]] - b[li, p[k]] for k != i,j
        d = b[lj, p] - b[li, p]
        d[i] = 0
        d[j] = 0
        # a_diff[k] = a[i,k] - a[j,k] + a[k,i] - a[k,j] for k != i,j
        a_diff = a[i, :] - a[j, :] + a[:, i] - a[:, j]
        a_diff[i] = 0
        a_diff[j] = 0
        return int(np.dot(a_diff, d))

    def linearized_cost_matrix(p):
        """
        Compute the linearized cost matrix for Hungarian algorithm.
        cost[i,l] = sum_j a[i,j] * b[l, p[j]] + sum_j a[j,i] * b[p[j], l]
        """
        C1 = b[:, p]  # C1[l, j] = b[l, p[j]]
        C2 = b[p, :]  # C2[j, l] = b[p[j], l]
        term1 = a @ C1.T  # term1[i,l] = sum_j a[i,j] * b[l, p[j]]
        term2 = a.T @ C2  # term2[i,l] = sum_j a[j,i] * b[p[j], l]
        return term1 + term2

    def mds_init():
        """
        Use MDS on flow and distance matrices to get initial assignment.
        Project both onto 2D Euclidean space, then match nearest neighbors.
        """
        # MDS for symmetric matrices
        def mds_embedding(mat, dim=2):
            # Center the matrix
            n_local = len(mat)
            J = np.eye(n_local) - np.ones((n_local, n_local)) / n_local
            M = -0.5 * J @ (mat ** 2) @ J
            # Eigen decomposition
            evals, evecs = np.linalg.eigh(M)
            # Take top dim eigenvectors
            idx = np.argsort(evals)[::-1][:dim]
            evals_sel = evals[idx]
            evecs_sel = evecs[:, idx]
            # Embedding
            coords = evecs_sel * np.sqrt(np.maximum(evals_sel, 0))
            return coords

        # Get 2D embeddings for flow and distance matrices
        # Note: a and b are not necessarily symmetric, but we can symmetrize for MDS
        a_sym = (a + a.T) / 2.0
        b_sym = (b + b.T) / 2.0

        try:
            fac_coords = mds_embedding(a_sym, dim=2)
            loc_coords = mds_embedding(b_sym, dim=2)
        except Exception:
            # Fallback to random if MDS fails
            p = np.arange(n, dtype=np.int64)
            np.random.shuffle(p)
            return p

        # Match facilities to locations using nearest neighbor in 2D space
        # For each facility, find the closest unassigned location
        p = np.full(n, -1, dtype=np.int64)
        assigned = np.zeros(n, dtype=bool)
        
        # Process facilities in order of "importance" (high flow sum)
        order = np.argsort(-a_row_sums)
        
        for fac_idx in order:
            if time.time() > deadline - 50.0:
                # Fill remaining randomly
                remaining = np.where(p == -1)[0]
                unassigned_locs = np.where(~assigned)[0]
                np.random.shuffle(unassigned_locs)
                p[remaining] = unassigned_locs
                break
            
            # Find closest unassigned location
            fac_point = fac_coords[fac_idx]
            dists = np.sum((loc_coords - fac_point) ** 2, axis=1)
            dists[assigned] = np.inf
            
            best_loc = np.argmin(dists)
            p[fac_idx] = best_loc
            assigned[best_loc] = True
        
        # Ensure all are assigned
        if np.any(p == -1):
            remaining = np.where(p == -1)[0]
            unassigned_locs = np.where(~assigned)[0]
            np.random.shuffle(unassigned_locs)
            p[remaining] = unassigned_locs
        
        return p

    def two_opt_polish(p, cost, max_time=None, max_passes=50):
        """Greedy 2-opt local search. Returns (p, cost)."""
        if max_time is None:
            max_time = deadline
        passes = 0
        while passes < max_passes and time.time() < max_time:
            improved = False
            passes += 1
            for i in range(n):
                if time.time() > max_time:
                    return p, cost
                for j in range(i + 1, n):
                    if time.time() > max_time:
                        return p, cost
                    delta = two_opt_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return p, cost

    # --- Phase 1: Generate multiple diverse initial solutions ---
    def rank_matching_init():
        heavy_order = np.argsort(-a_row_sums, kind='stable')
        central_order = np.argsort(b_row_sums, kind='stable')
        p = np.empty(n, dtype=np.int64)
        p[heavy_order] = central_order
        return p.copy()

    def random_init():
        p = np.arange(n, dtype=np.int64)
        np.random.shuffle(p)
        return p

    def perturbed_init(base_p, num_swaps):
        p = base_p.copy()
        for _ in range(num_swaps):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i != j:
                p[i], p[j] = p[j], p[i]
        return p

    # Generate initial candidates
    candidates = []
    
    # Rank matching
    p_rank = rank_matching_init()
    candidates.append(p_rank)
    
    # MDS-based initialization
    p_mds = mds_init()
    candidates.append(p_mds)
    
    # A few random inits
    for _ in range(2):
        candidates.append(random_init())
    
    # Perturbations of rank matching and MDS
    for ns in [10, 30]:
        candidates.append(perturbed_init(p_rank, ns))
        candidates.append(perturbed_init(p_mds, ns))

    # Polish each candidate briefly and keep the best
    best_p = None
    best_cost = np.inf
    
    for p in candidates:
        if time.time() > deadline - 30.0:
            break
        p, c = two_opt_polish(p, calc_cost_full(p), max_time=min(deadline - 25.0, time.time() + 4.0))
        if c < best_cost:
            best_cost = c
            best_p = p.copy()

    # --- Phase 2: Hungarian Reassignment Loop ---
    hungarian_count = 0
    no_improve_rounds = 0
    max_no_improve = 4
    rng = np.random.default_rng(42)

    while time.time() < deadline - 25.0:
        # Compute linearized cost and solve assignment
        cost_mat = linearized_cost_matrix(best_p)
        
        # Add noise to escape plateaus, scaling with stagnation
        if hungarian_count > 0:
            noise_scale = 1.0 + no_improve_rounds * 1.0
            cost_mat = cost_mat + rng.standard_normal((n, n)) * noise_scale
        
        try:
            row_ind, col_ind = linear_sum_assignment(cost_mat)
        except Exception:
            break
            
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        
        new_cost = calc_cost_full(new_p)
        
        if new_cost < best_cost:
            best_p = new_p
            best_cost = new_cost
            hungarian_count += 1
            no_improve_rounds = 0
            
            # Polish after a good Hungarian step
            best_p, best_cost = two_opt_polish(best_p, best_cost, max_time=min(deadline - 15.0, time.time() + 6.0))
        else:
            no_improve_rounds += 1
            
            # If stuck, try a perturbed restart from best
            if no_improve_rounds >= max_no_improve:
                p_restart = best_p.copy()
                # Randomly swap a few pairs to escape local minimum
                num_swaps = rng.integers(8, 20)
                for _ in range(num_swaps):
                    i = rng.integers(n)
                    j = rng.integers(n)
                    if i != j:
                        p_restart[i], p_restart[j] = p_restart[j], p_restart[i]
                
                p_restart, c_restart = two_opt_polish(p_restart, calc_cost_full(p_restart), max_time=min(deadline - 15.0, time.time() + 4.0))
                if c_restart < best_cost:
                    best_p = p_restart
                    best_cost = c_restart
                no_improve_rounds = 0
                hungarian_count = 0

    # --- Phase 3: Final high-precision polish ---
    # Try multiple polish passes with different random orderings
    final_p = best_p.copy()
    final_cost = best_cost
    
    remaining_time = deadline - time.time() - 3.0
    if remaining_time > 5.0:
        # Do a thorough final polish
        final_p, final_cost = two_opt_polish(final_p, final_cost, max_time=deadline - 3.0, max_passes=100)
    
    # If we still have time, try one more perturbation + polish
    if time.time() < deadline - 8.0 and final_cost > 44759294:  # If not at known best
        p_try = final_p.copy()
        num_swaps = rng.integers(5, 15)
        for _ in range(num_swaps):
            i = rng.integers(n)
            j = rng.integers(n)
            if i != j:
                p_try[i], p_try[j] = p_try[j], p_try[i]
        
        p_try, c_try = two_opt_polish(p_try, calc_cost_full(p_try), max_time=deadline - 3.0, max_passes=80)
        if c_try < final_cost:
            final_p = p_try
            final_cost = c_try

    return final_p.tolist()

# EVOLVE-BLOCK-END
