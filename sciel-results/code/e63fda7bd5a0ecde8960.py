"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def compute_cost(p):
        """Compute exact QAP cost."""
        bp = b[np.ix_(p, p)]
        return int(np.sum(a * bp))

    def compute_swap_delta(i, j, p):
        """Compute cost delta when swapping positions of facilities i and j."""
        li = p[i]
        lj = p[j]
        
        if li == lj:
            return 0
        
        delta = 0
        
        # Diagonal terms
        delta += a[i, i] * (b[lj, lj] - b[li, li])
        delta += a[j, j] * (b[li, li] - b[lj, lj])
        
        # Cross terms between i and j
        delta += a[i, j] * (b[lj, li] - b[li, lj])
        delta += a[j, i] * (b[li, lj] - b[lj, li])
        
        # Terms with other facilities k
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        ks = np.where(mask)[0]
        
        if len(ks) > 0:
            pk = p[ks]
            
            # a[i,k] * (b[lj,pk] - b[li,pk])
            delta += np.sum(a[i, ks] * (b[lj, pk] - b[li, pk]))
            # a[k,i] * (b[pk,lj] - b[pk,li])
            delta += np.sum(a[ks, i] * (b[pk, lj] - b[pk, li]))
            # a[j,k] * (b[li,pk] - b[lj,pk])
            delta += np.sum(a[j, ks] * (b[li, pk] - b[lj, pk]))
            # a[k,j] * (b[pk,li] - b[pk,lj])
            delta += np.sum(a[ks, j] * (b[pk, li] - b[pk, lj]))
        
        return delta

    def linearized_assignment(p):
        """
        Formulate the QAP as a linear assignment problem by linearizing the quadratic
        objective using the current permutation's interaction matrix.
        
        The QAP cost is sum_{i,j} a[i,j] * b[p[i], p[j]].
        
        We can rewrite this as: for each facility i, the cost of assigning it to location l
        depends on where all other facilities are. Specifically:
        
        cost(i -> l) = sum_j a[i,j] * b[l, p[j]] + sum_j a[j,i] * b[p[j], l]
        
        But this double counts and includes j=i terms. Let's be more careful.
        
        Total cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        
        For a fixed assignment of all other facilities, if we consider moving facility i
        to location l (and adjusting others), the linearized cost for facility i at location l
        is:
        
        C[i, l] = sum_{j != i} a[i,j] * b[l, p[j]] + sum_{j != i} a[j,i] * b[p[j], l] + a[i,i]*b[l,l]
        
        This is the exact contribution of facility i to the total cost given the positions of
        all other facilities. If we solve the assignment problem with this cost matrix, we get
        a new permutation that minimizes the linearized objective.
        
        This is essentially a coordinate descent / block coordinate method where we reassign
        all facilities simultaneously based on the current interactions.
        """
        # Build the linearized cost matrix C[i, l]
        # C[i, l] = sum_{j != i} a[i,j] * b[l, p[j]] + sum_{j != i} a[j,i] * b[p[j], l] + a[i,i]*b[l,l]
        
        # Let's compute this efficiently.
        # Let B_p = b[np.ix_(p, p)] so B_p[k, m] = b[p[k], p[m]]
        # Actually, we need b[l, p[j]] for fixed l and varying j.
        # And b[p[j], l] for fixed l and varying j.
        
        # Let's define:
        # For each location l and facility j, we have b[l, p[j]] and b[p[j], l].
        
        # Let's create a matrix M where M[l, j] = b[l, p[j]]
        # Then sum_j a[i,j] * M[l, j] is a matrix product.
        
        # M = b[:, p]  # shape (n, n), M[l, j] = b[l, p[j]]
        M1 = b[:, p]  # M1[l, j] = b[l, p[j]]
        M2 = b[p, :]  # M2[j, l] = b[p[j], l]
        
        # C[i, l] = sum_j a[i,j] * M1[l, j] + sum_j a[j,i] * M2[j, l] + a[i,i] * b[l,l]
        
        # First term: (a @ M1.T)[i, l] = sum_j a[i,j] * M1[l, j]
        # Second term: (a.T @ M2)[i, l] = sum_j a[j,i] * M2[j, l]
        
        term1 = a @ M1.T  # term1[i, l] = sum_j a[i,j] * b[l, p[j]]
        term2 = a.T @ M2  # term2[i, l] = sum_j a[j,i] * b[p[j], l]
        
        # Subtract the j=i terms since we excluded them, but actually the formula above
        # includes j=i. Let's check:
        # When j=i, term1[i, l] includes a[i,i] * b[l, p[i]]
        # term2[i, l] includes a[i,i] * b[p[i], l]
        # We want to exclude j=i from the sums and add a[i,i]*b[l,l] separately.
        
        # So: C[i, l] = (term1[i, l] - a[i,i]*b[l, p[i]]) + (term2[i, l] - a[i,i]*b[p[i], l]) + a[i,i]*b[l,l]
        
        # Let's compute this.
        # b[l, p[i]] is M1[l, i]
        # b[p[i], l] is M2[i, l]
        
        # diag_a = np.diag(a)
        # C = term1 + term2
        # For each i: C[i, :] -= a[i,i] * (M1[:, i] + M2[i, :])
        # For each i: C[i, :] += a[i,i] * b[:, :]  # wait, b[l,l] is just the diagonal of b
        
        # Actually, let's just compute it directly for clarity, n=150 is small.
        C = term1 + term2
        
        # Subtract j=i contributions
        for i in range(n):
            C[i, :] -= a[i, i] * (M1[:, i] + M2[i, :])
            C[i, :] += a[i, i] * np.diag(b)
        
        # Solve the assignment problem
        row_ind, col_ind = linear_sum_assignment(C)
        p_new = np.empty(n, dtype=np.int64)
        p_new[row_ind] = col_ind
        
        return p_new

    def local_search_2swap(p, max_iter=5000, time_limit=None):
        """Perform 2-swap local search."""
        curr_p = p.copy()
        curr_cost = compute_cost(curr_p)
        best_p = curr_p.copy()
        best_cost = curr_cost
        
        rng = np.random.RandomState(42)
        
        if time_limit is None:
            time_limit = deadline - 5.0
        
        for it in range(max_iter):
            if time.time() > time_limit:
                break
            
            # Try all pairs or random pairs? For n=150, all pairs is 11175, which is fast.
            # Let's do a few random pairs and some systematic.
            improved = False
            
            # Random pairs
            for _ in range(200):
                i, j = rng.choice(n, 2, replace=False)
                delta = compute_swap_delta(i, j, curr_p)
                if delta < 0:
                    curr_p[i], curr_p[j] = curr_p[j], curr_p[i]
                    curr_cost += delta
                    improved = True
                    if curr_cost < best_cost:
                        best_cost = curr_cost
                        best_p = curr_p.copy()
            
            if not improved:
                break
        
        return best_p, best_cost

    def simulated_annealing(p, best_cost, end_time):
        """Perform Simulated Annealing."""
        curr_p = p.copy()
        curr_cost = best_cost
        best_p = p.copy()
        best_cost_sa = best_cost
        
        rng = np.random.RandomState(42)
        
        # Estimate initial temperature
        deltas = []
        for _ in range(10):
            i, j = rng.choice(n, 2, replace=False)
            d = compute_swap_delta(i, j, curr_p)
            deltas.append(abs(d))
        avg_delta = np.mean(deltas) if deltas else 1.0
        if avg_delta == 0:
            avg_delta = 1.0
            
        T = avg_delta * 10.0
        cooling_rate = 0.999
        min_T = 1e-5
        
        while time.time() < end_time:
            if T < min_T:
                break
                
            i, j = rng.choice(n, 2, replace=False)
            
            delta = compute_swap_delta(i, j, curr_p)
            
            if delta < 0:
                accept = True
            else:
                if T > 0:
                    prob = np.exp(-delta / T)
                else:
                    prob = 0.0
                accept = rng.random_sample() < prob
            
            if accept:
                curr_p[i], curr_p[j] = curr_p[j], curr_p[i]
                curr_cost += delta
                
                if curr_cost < best_cost_sa:
                    best_cost_sa = curr_cost
                    best_p = curr_p.copy()
            
            T *= cooling_rate
        
        return best_p, best_cost_sa

    # 1. Initializations
    # Try multiple initializations and keep the best
    
    best_p = None
    best_cost = np.inf
    
    # Initialization 1: Hungarian on a @ b.T
    C1 = a @ b.T
    row_ind, col_ind = linear_sum_assignment(-C1)
    p1 = np.empty(n, dtype=np.int64)
    p1[row_ind] = col_ind
    cost1 = compute_cost(p1)
    if cost1 < best_cost:
        best_cost = cost1
        best_p = p1.copy()
    
    # Initialization 2: Rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p2 = np.empty(n, dtype=np.int64)
    p2[heavy] = central
    cost2 = compute_cost(p2)
    if cost2 < best_cost:
        best_cost = cost2
        best_p = p2.copy()
    
    # Initialization 3: Random
    rng_init = np.random.RandomState(123)
    p3 = rng_init.permutation(n)
    cost3 = compute_cost(p3)
    if cost3 < best_cost:
        best_cost = cost3
        best_p = p3.copy()
    
    # 2. Local search from best initialization
    p_ls, cost_ls = local_search_2swap(best_p, max_iter=2000, time_limit=deadline - 30.0)
    if cost_ls < best_cost:
        best_cost = cost_ls
        best_p = p_ls.copy()
    
    # 3. Linearized assignment refinement (block coordinate descent)
    # Iterate: solve linearized assignment, then local search, repeat
    for iteration in range(5):
        if time.time() > deadline - 20.0:
            break
        
        p_lin = linearized_assignment(best_p)
        cost_lin = compute_cost(p_lin)
        
        if cost_lin < best_cost:
            best_cost = cost_lin
            best_p = p_lin.copy()
        
        # Local search from the linearized solution
        p_ls2, cost_ls2 = local_search_2swap(p_lin, max_iter=1000, time_limit=deadline - 15.0)
        if cost_ls2 < best_cost:
            best_cost = cost_ls2
            best_p = p_ls2.copy()
    
    # 4. Simulated Annealing polish
    if time.time() < deadline - 5.0:
        end_time_sa = deadline - 2.0
        p_sa, cost_sa = simulated_annealing(best_p, best_cost, end_time_sa)
        if cost_sa < best_cost:
            best_cost = cost_sa
            best_p = p_sa.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
