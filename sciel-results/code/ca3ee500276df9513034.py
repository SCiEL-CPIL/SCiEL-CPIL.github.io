# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog
from scipy import sparse


def _mobius_upto(n):
    mu = np.ones(n + 1, dtype=np.int64)
    is_prime = np.ones(n + 1, dtype=bool)
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            step = i
            for j in range(i * i, n + 1, step):
                is_prime[j] = False
            mu[i * i:n + 1:step] *= -1
            for j in range(i * i * i, n + 1, i * i):
                mu[j] = 0
    return mu


def _baseline():
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values


def _mu_fallback():
    mu = _mobius_upto(1000)
    domain = np.arange(1, 1001, dtype=np.int64)
    values = mu.astype(np.float64).copy()
    values[0] = -values[1:].sum() / 1000.0
    return domain, values


def _feasible(domain, values, deadline):
    """Check M(x) <= 1.0001 for all x in [1, 10K] using vectorized computation."""
    K = int(domain[-1])
    f = values
    ks = domain.astype(np.int64)
    
    # Build d[j] = sum_{k|j} f(k) for j in [1, K]
    d = np.zeros(K + 1, dtype=np.float64)
    for idx, k in enumerate(ks):
        kk = int(k)
        if kk <= K:
            d[kk:K+1:kk] += f[idx]
    
    # Check x in [1, K]
    if K > 0:
        prefix_d = np.cumsum(d[1:K+1])
        if prefix_d.max() > 1.0001:
            return False
    
    # For x in [K+1, 10K], compute increments
    inc = np.zeros(10*K + 1, dtype=np.float64)
    for idx, k in enumerate(ks):
        kk = int(k)
        if kk <= 10*K:
            inc[kk:10*K+1:kk] += f[idx]
    
    M_K = prefix_d[-1] if K >= 1 else 0.0
    if 10*K > K:
        cum_inc = np.cumsum(inc[K+1:10*K+1])
        M_vals = M_K + cum_inc
        if M_vals.max() > 1.0001:
            return False
    
    return True


def _solve_lp_sparse(c, x_set, K, time_limit):
    """Solve max c^T z s.t. sum_k floor(x/k) z_k <= 1 for x in x_set, z >= 0, using sparse matrix."""
    n_cons = len(x_set)
    if n_cons == 0:
        return None, 0.0
    
    xs = np.array(x_set, dtype=np.int64)
    
    # Build sparse constraint matrix: A[i, k] = floor(x_i / k) for k in 1..K
    # For each k, the entries where floor(x/k) > 0 are x >= k
    # We build column by column for efficiency
    rows = []
    cols = []
    data = []
    
    for k in range(1, K + 1):
        # floor(x/k) for each x in xs
        vals = xs // k
        mask = vals > 0
        if np.any(mask):
            idxs = np.where(mask)[0]
            rows.extend(idxs.tolist())
            cols.extend([k - 1] * len(idxs))
            data.extend(vals[mask].tolist())
    
    if len(data) == 0:
        return None, 0.0
    
    A = sparse.csc_matrix((data, (rows, cols)), shape=(n_cons, K))
    
    res = linprog(-c, A_ub=A, b_ub=np.ones(n_cons),
                  bounds=[(0.0, None)] * K, method="highs",
                  options={"presolve": True, "time_limit": time_limit})
    
    if not res.success:
        return None, 0.0
    
    z = res.x
    obj = float(np.dot(z, c))
    return z, obj


def _find_violated_constraints(z, K, tol=1e-8):
    """Find violated constraints M(x) > 1+tol. Returns list of x values."""
    # Compute d[j] = sum_{k|j} z[k] for j in [1, K]
    d = np.zeros(K + 1, dtype=np.float64)
    nonzero_idx = np.where(np.abs(z) > 1e-15)[0]
    for k_idx in nonzero_idx:
        k_val = int(k_idx + 1)
        d[k_val:K+1:k_val] += z[k_idx]
    
    # Check x in [1, K]
    prefix_d = np.cumsum(d[1:K+1])
    violated_small = np.where(prefix_d > 1.0 + tol)[0]
    violated_xs = [int(idx + 1) for idx in violated_small]
    
    # Check x in [K+1, 10K]
    inc = np.zeros(10*K + 1, dtype=np.float64)
    for k_idx in nonzero_idx:
        k_val = int(k_idx + 1)
        inc[k_val:10*K+1:k_val] += z[k_idx]
    
    M_K = prefix_d[-1] if K >= 1 else 0.0
    if 10*K > K:
        cum_inc = np.cumsum(inc[K+1:10*K+1])
        M_large = M_K + cum_inc
        violated_large = np.where(M_large > 1.0 + tol)[0]
        for idx in violated_large:
            violated_xs.append(int(K + 1 + int(idx)))
    
    return violated_xs


def prime_number_theorem_partial_function():
    """
    Construct a partial function f : subset of positive integers -> R that maximises
    a = -sum_k f(k)*log(k)/k under the step-function constraint
        M(x) = sum_k f(k)*floor(x/k) <= 1   for every integer x in [1, 10K],
    where K = max key in the domain.
    
    Uses iterative constraint generation (cutting-plane) with sparse matrices
    to solve the LP efficiently.
    """
    t0 = time.time()
    DEADLINE = 580.0  # wall-clock safety margin before the 630 s kill
    
    best = _baseline()
    best_a = 1.0
    best_K = 0
    
    # Try a wider range of K values with adaptive time allocation
    K_candidates = [1000, 2000, 3000, 5000, 8000, 12000, 16000, 20000]
    
    for K in K_candidates:
        if time.time() > DEADLINE - 30:
            break
        try:
            ks = np.arange(1, K + 1, dtype=np.int64)
            kf = ks.astype(np.float64)
            c = np.log(kf) / kf  # maximize c^T z
            
            # Initial constraint set: x = 1, 2, ..., min(K, 500)
            init_max = min(K, 500)
            x_set = list(range(1, init_max + 1))
            
            z = None
            max_iter = 150
            tol = 1e-8
            
            for iteration in range(max_iter):
                if time.time() > DEADLINE - 10:
                    break
                
                # Solve LP with sparse matrix
                remaining = DEADLINE - 10 - time.time()
                lp_time_limit = max(5.0, min(40.0, remaining / 3))
                z, obj = _solve_lp_sparse(c, x_set, K, lp_time_limit)
                
                if z is None:
                    break
                
                # Check if all constraints are satisfied
                violated = _find_violated_constraints(z, K, tol)
                
                if len(violated) == 0:
                    break
                
                # Add violated constraints to x_set
                new_xs = set(violated)
                existing = set(x_set)
                to_add = list(new_xs - existing)
                to_add.sort()
                if len(to_add) > 300:
                    step = max(1, len(to_add) // 300)
                    to_add = to_add[::step][:300]
                
                x_set = list(existing | set(to_add))
                x_set.sort()
                
                # Limit total constraint set size to avoid memory issues
                if len(x_set) > 15000:
                    step = max(1, len(x_set) // 15000)
                    x_set = x_set[::step]
                    x_set.sort()
            
            if z is None:
                continue
            
            # Final verification
            values = z.copy()
            values[0] = -values[1:].sum() / K  # enforce sum f(k)/k = 0
            
            # Verify feasibility of the actual returned f
            if _feasible(ks, values, DEADLINE - 5):
                a = float(np.dot(z, c))
                if a > best_a + 1e-9:
                    best = (ks, values)
                    best_a = a
                    best_K = K
                    print(f"K={K}, a={a:.6f}, time={time.time()-t0:.1f}s")
        except Exception as e:
            print(f"K={K} failed: {e}")
            continue
    
    # Polish pass: for the best K found, do additional iterations with tighter tolerance
    if best_K > 0 and time.time() < DEADLINE - 20:
        try:
            K = best_K
            ks = np.arange(1, K + 1, dtype=np.int64)
            kf = ks.astype(np.float64)
            c = np.log(kf) / kf
            
            # Start from a broader initial set
            init_max = min(K, 2000)
            x_set = list(range(1, init_max + 1))
            
            z = None
            max_iter = 200
            tol = 1e-10
            
            for iteration in range(max_iter):
                if time.time() > DEADLINE - 10:
                    break
                
                remaining = DEADLINE - 10 - time.time()
                if remaining < 15:
                    break
                lp_time_limit = max(3.0, min(30.0, remaining / 4))
                z, obj = _solve_lp_sparse(c, x_set, K, lp_time_limit)
                
                if z is None:
                    break
                
                violated = _find_violated_constraints(z, K, tol)
                
                if len(violated) == 0:
                    break
                
                new_xs = set(violated)
                existing = set(x_set)
                to_add = list(new_xs - existing)
                to_add.sort()
                if len(to_add) > 500:
                    step = max(1, len(to_add) // 500)
                    to_add = to_add[::step][:500]
                
                x_set = list(existing | set(to_add))
                x_set.sort()
                
                if len(x_set) > 20000:
                    step = max(1, len(x_set) // 20000)
                    x_set = x_set[::step]
                    x_set.sort()
            
            if z is not None:
                values = z.copy()
                values[0] = -values[1:].sum() / K
                if _feasible(ks, values, DEADLINE - 5):
                    a = float(np.dot(z, c))
                    if a > best_a + 1e-9:
                        best = (ks, values)
                        best_a = a
                        print(f"Polish K={K}, a={a:.6f}, time={time.time()-t0:.1f}s")
        except Exception as e:
            print(f"Polish failed: {e}")
    
    # If LP didn't improve significantly, try mu-based approach for larger K
    if best_a <= 1.05 + 1e-9 and time.time() < DEADLINE - 20:
        for K in (1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000):
            if time.time() > DEADLINE - 10:
                break
            try:
                mu = _mobius_upto(K)
                domain = np.arange(1, K + 1, dtype=np.int64)
                values = mu.astype(np.float64).copy()
                values[0] = -values[1:].sum() / K
                
                if _feasible(domain, values, DEADLINE - 5):
                    a = float(-np.dot(values, np.log(domain.astype(np.float64)) / domain.astype(np.float64)))
                    if a > best_a + 1e-9:
                        best = (domain, values)
                        best_a = a
            except Exception:
                continue
    
    return best


# EVOLVE-BLOCK-END
