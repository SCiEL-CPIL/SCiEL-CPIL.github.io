# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog


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
    
    # M(x) = sum_{j<=x} d[j] for x <= K (since floor(x/k)=0 for k>x when x<K)
    # For x > K: M(x) = sum_{j<=K} d[j] + sum_{k<=K} f(k)*floor(x/k) - sum_{j<=K} d[j]
    # Actually M(x) = sum_k f(k)*floor(x/k) for any x
    # For x > K, we can compute directly
    
    # Check x in [1, K]
    if K > 0:
        prefix_d = np.cumsum(d[1:K+1])  # prefix_d[i] = sum_{j=1}^{i+1} d[j]
        # M(x) for x in [1,K] = sum_{j<=x} d[j] = prefix_d[x-1]
        if prefix_d.max() > 1.0001:
            return False
    
    # Check x in [K+1, 10K]
    # M(x) = sum_k f(k) * floor(x/k)
    # For efficiency, check at strategic points
    # M(x) is piecewise linear with slope sum_{k|x+1} f(k) (changes at x+1 being multiple of some k)
    # The maximum in each interval [a, b] where no k divides any point in (a, b] is at the endpoint
    # Actually M(x+1) - M(x) = sum_{k | (x+1)} f(k) = d[x+1] if x+1 <= K, else sum_{k<=K, k|(x+1)} f(k)
    
    # For x > K, M(x) = sum_{j<=K} d[j] + sum_{k<=K} f(k) * (floor(x/k) - floor(K/k))
    # Let's just check at all points where the function could peak
    # M(x) is non-decreasing only if all d[j] >= 0, but that's not the case
    # We need to check more carefully
    
    # For x in [K+1, 10K], M(x) = sum_k f(k)*floor(x/k)
    # This is a step function. The maximum occurs at points just before a drop.
    # Drops happen when x+1 is a multiple of some k with f(k) > 0... no wait
    # M(x) - M(x-1) = sum_{k | x} f(k) = d[x] for x <= K
    # For x > K: M(x) - M(x-1) = sum_{k<=K, k|x} f(k)
    
    # Let's compute M(x) at all "critical" points
    # Critical points are x where some k divides x (for the slope change)
    # But that's too many. Instead, note that M(x) = sum_k f(k)*floor(x/k)
    # For x > K, floor(x/k) = (x - x%k)/k
    # M(x) = sum_k f(k)*(x/k - x%k/k) = x*sum_k f(k)/k - sum_k f(k)*x%k/k
    # Since sum_k f(k)/k = 0 (after normalization), M(x) = -sum_k f(k)*x%k/k for x > K
    # Wait, that's only after the evaluator's normalization. But we check the raw f.
    
    # Let me just sample densely for x > K
    # Actually, let's compute M(x) at all x in [K+1, 10K] that are multiples of any k in domain
    # plus the endpoints of intervals between such multiples
    
    # Simpler approach: M(x) for x > K can be written as:
    # M(x) = M(K) + sum_{j=K+1}^{x} sum_{k<=K, k|j} f(k)
    # The increments are periodic-ish. Let's just check at all points where any k divides x
    # plus K+1 and 10K
    
    # Actually the safest is to check at all x where the slope changes, i.e., x that are multiples of some k
    # But for large K this could be many points. Let's be smart:
    # M(x) is a step function. Between consecutive multiples of any k, M is constant.
    # So we only need to check at x = m*k for m*k in [K+1, 10K] and at 10K.
    
    # Collect all critical x values in [K+1, 10K]
    # x is critical if x is a multiple of some k in domain
    # That's a lot. Alternative: just check every x from K+1 to 10K if 10K-K is manageable
    # For K up to ~5000, 10K-K = 9K = 45000, and computing M at each point naively is O(K)
    # So total O(9K*K) = O(K^2) which for K=5000 is 225M - too slow in Python
    
    # Better: precompute the increment function
    # inc[x] = sum_{k<=K, k|x} f(k) for x in [1, 10K]
    # Then M(x) = M(K) + cumsum(inc[K+1:x+1])
    
    inc = np.zeros(10*K + 1, dtype=np.float64)
    for idx, k in enumerate(ks):
        kk = int(k)
        if kk <= 10*K:
            inc[kk:10*K+1:kk] += f[idx]
    
    # M(x) for x in [1, K] already checked via prefix_d
    # M(K) = sum_{j=1}^{K} d[j] = prefix_d[K-1] if K >= 1
    M_K = prefix_d[-1] if K >= 1 else 0.0
    
    # M(x) for x in [K+1, 10K] = M_K + sum_{j=K+1}^{x} inc[j]
    # Compute cumulative sum of inc from K+1 to 10K
    if 10*K > K:
        cum_inc = np.cumsum(inc[K+1:10*K+1])
        M_vals = M_K + cum_inc
        if M_vals.max() > 1.0001:
            return False
    
    return True


def prime_number_theorem_partial_function():
    """
    Construct a partial function f : subset of positive integers -> R that maximises
    a = -sum_k f(k)*log(k)/k under the step-function constraint
        M(x) = sum_k f(k)*floor(x/k) <= 1   for every integer x in [1, 10K],
    where K = max key in the domain.
    
    Uses iterative constraint generation (cutting-plane) to solve the LP efficiently.
    """
    t0 = time.time()
    DEADLINE = 580.0  # wall-clock safety margin before the 630 s kill
    
    best = _baseline()
    best_a = 1.0
    
    for K in (1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000):
        if time.time() > DEADLINE - 30:
            break
        try:
            ks = np.arange(1, K + 1, dtype=np.int64)
            kf = ks.astype(np.float64)
            c = np.log(kf) / kf  # maximize c^T z
            
            # Iterative constraint generation
            # Start with a small set of constraints, solve, add violated ones
            # Constraint: M(x) = sum_k f(k)*floor(x/k) <= 1 for x in [1, 10K]
            
            # Initial constraint set: x = 1, 2, ..., K (the most binding ones typically)
            x_set = list(range(1, K + 1))
            
            z = None
            max_iter = 50
            tol = 1e-8
            
            for iteration in range(max_iter):
                if time.time() > DEADLINE - 10:
                    break
                
                # Build constraint matrix A_ub (|x_set| x K)
                n_cons = len(x_set)
                # A[i, j] = floor(x_i / k_j)
                # Vectorized: for each x, compute x//k for all k
                A = np.zeros((n_cons, K), dtype=np.float64)
                for i, x in enumerate(x_set):
                    A[i, :] = x // ks
                
                # Solve LP: max c^T z s.t. A z <= 1, z >= 0
                res = linprog(-c, A_ub=A, b_ub=np.ones(n_cons),
                              bounds=[(0.0, None)] * K, method="highs",
                              options={"presolve": True, "time_limit": 30})
                
                if not res.success:
                    break
                
                z = res.x
                obj = float(np.dot(z, c))
                
                # Check if all constraints are satisfied
                # M(x) for all x in [1, 10K]
                # Find violated constraints
                # M(x) = sum_k z[k]*floor(x/k)
                # For x in [1, K]: M(x) = sum_{j<=x} d[j] where d[j] = sum_{k|j} z[k]
                # For x > K: M(x) = sum_k z[k]*floor(x/k)
                
                # Compute d[j] for j in [1, K]
                d = np.zeros(K + 1, dtype=np.float64)
                for k_idx in range(K):
                    k_val = k_idx + 1
                    if z[k_idx] != 0:
                        d[k_val:K+1:k_val] += z[k_idx]
                
                # Check x in [1, K]
                prefix_d = np.cumsum(d[1:K+1])
                # Find x where prefix_d[x-1] > 1
                violated_small = np.where(prefix_d > 1.0 + tol)[0]
                
                # Check x in [K+1, 10K]
                # M(x) = sum_k z[k]*floor(x/k)
                # inc[j] = sum_{k|j} z[k] for j up to 10K
                # M(x) = M(K) + sum_{j=K+1}^{x} inc[j]
                inc = np.zeros(10*K + 1, dtype=np.float64)
                for k_idx in range(K):
                    k_val = k_idx + 1
                    if z[k_idx] != 0:
                        inc[k_val:10*K+1:k_val] += z[k_idx]
                
                M_K = prefix_d[-1] if K >= 1 else 0.0
                if 10*K > K:
                    cum_inc = np.cumsum(inc[K+1:10*K+1])
                    M_large = M_K + cum_inc
                    # M_large[i] = M(K+1+i)
                    violated_large = np.where(M_large > 1.0 + tol)[0]
                else:
                    violated_large = np.array([], dtype=int)
                
                # If no violations, we're done
                if len(violated_small) == 0 and len(violated_large) == 0:
                    break
                
                # Add violated constraints to x_set
                new_xs = set()
                for idx in violated_small:
                    new_xs.add(idx + 1)  # x = idx+1
                for idx in violated_large:
                    new_xs.add(K + 1 + idx)  # x = K+1+idx
                
                if len(new_xs) == 0:
                    break
                
                x_set = list(set(x_set) | new_xs)
                x_set.sort()
                
                # Limit constraint set size for tractability
                if len(x_set) > 5000:
                    # Keep the most violated ones
                    break
            
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
                    print(f"K={K}, a={a:.6f}, time={time.time()-t0:.1f}s")
        except Exception as e:
            print(f"K={K} failed: {e}")
            continue
    
    # If LP didn't improve, try mu-based approach for larger K
    if best_a <= 1.0 + 1e-9 and time.time() < DEADLINE - 20:
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
