# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    # Deadline guard: evaluator kills at 630s; keep margin.
    t0 = time.time()
    deadline = t0 + 600.0

    # Use a dense set of small integers (up to 200) as the domain.
    # The objective is a = -sum f(k) log(k) / k (after normalization).
    # Constraint: M(x) = sum f(k) floor(x/k) <= 1 for all x in [1, 10K], K=max(domain).
    # And sum f(k)/k = 0 (handled by setting f(1) appropriately).
    
    # Strategy: Use the Möbius function on square-free numbers up to N.
    # The Möbius function satisfies sum_{d|n} mu(d) = 0 for n>1, which implies
    # sum_{k<=x} mu(k) floor(x/k) = 1 for all x >= 1.
    # This makes M(x) = 1 exactly (feasible).
    # The objective for mu is -sum mu(k) log(k)/k, which converges to 1.
    # With a finite cutoff, we get close to 1.
    # But we can do better by optimizing the values.
    
    # Let's use a larger set of square-free numbers up to N=200.
    N = 200
    
    # Compute Möbius function up to N
    mu = np.ones(N + 1, dtype=np.float64)
    is_prime = np.ones(N + 1, dtype=bool)
    primes = []
    for i in range(2, N + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i, N + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            i2 = i * i
            for j in range(i2, N + 1, i2):
                mu[j] = 0.0
    
    # Square-free numbers up to N
    sq_free = np.array([k for k in range(2, N + 1) if mu[k] != 0], dtype=np.int64)
    m = len(sq_free)
    
    # Build the constraint matrix for M(x) <= 1 for x in [1, 10*N]
    # M(x) = sum_{k in domain} f(k) * floor(x/k)
    # We need to include k=1 in the domain for normalization.
    # Let's work with k=2..N square-free, and add k=1.
    
    # Variables: f(2), f(3), ..., f(k) for square-free k in [2, N]
    # Plus f(1) is determined by normalization: sum f(k)/k = 0 => f(1) = -sum_{k>=2} f(k)/k
    
    # M(x) = f(1) * floor(x/1) + sum_{k>=2} f(k) * floor(x/k)
    #       = f(1) * x + sum_{k>=2} f(k) * floor(x/k)
    # Substitute f(1) = -sum_{k>=2} f(k)/k:
    # M(x) = -x * sum_{k>=2} f(k)/k + sum_{k>=2} f(k) * floor(x/k)
    #       = sum_{k>=2} f(k) * (floor(x/k) - x/k)
    
    # Note: floor(x/k) - x/k <= 0 for all x, k.
    # So M(x) = sum_{k>=2} f(k) * (floor(x/k) - x/k)
    
    # Objective: a = -sum_{k} f(k) log(k) / k
    #            = -f(1) log(1)/1 - sum_{k>=2} f(k) log(k)/k
    #            = -sum_{k>=2} f(k) log(k)/k  (since log(1)=0)
    
    # Constraints:
    # 1. M(x) <= 1 for all x in [1, 10N]
    # 2. f(k) <= 10 (clipping upper bound)
    # 3. f(k) >= -10 (clipping lower bound)
    
    # Actually, the problem says values are clipped to [-10, 10] BEFORE the normalization step.
    # So we should set f(k) in [-10, 10] for k >= 2, and f(1) might go outside but gets clipped.
    # Wait, re-reading: "values are clipped to [-10, 10]. Then f(1) is replaced by f(1) - S..."
    # So all values including f(1) are first clipped, then f(1) is adjusted.
    # But if f(1) is clipped first, then adjusted, it might go out of range again?
    # The problem says "f(1) = 0 before this if 1 is not in the domain", suggesting 1 may or may not be in domain.
    # If 1 is in domain, f(1) is clipped, then adjusted. The adjusted f(1) is used in M(x) and objective.
    # But M(x) uses the adjusted f(1). Let me re-read.
    # "values are clipped to [-10, 10]. Then f(1) is replaced by f(1) - S with S = sum_k f(k)/k over the clipped values"
    # So S is computed from clipped values. Then f(1) is updated.
    # "M(x) = sum_k f(k) * floor(x/k)" - this uses the updated f(1).
    # "the metric is a = -sum_k f(k) * log(k) / k" - this also uses updated f(1).
    
    # So the flow is:
    # 1. Start with domain and values.
    # 2. Clip all values to [-10, 10].
    # 3. Compute S = sum_{k in domain} f(k)/k (using clipped values).
    # 4. If 1 in domain: f(1) = f(1) - S. Else: f(1) = 0 - S = -S, and add 1 to domain with value -S.
    # 5. Now sum f(k)/k = 0.
    # 6. Check M(x) <= 1 for all x in [1, 10K].
    # 7. Compute a = -sum f(k) log(k)/k.
    
    # So I should work with the final f after step 4.
    # Let me denote g(k) = final f(k).
    # sum g(k)/k = 0.
    # g(1) = -sum_{k>=2} g(k)/k (assuming 1 is in domain).
    # M(x) = sum_{k} g(k) floor(x/k) = g(1) x + sum_{k>=2} g(k) floor(x/k)
    #       = -x sum_{k>=2} g(k)/k + sum_{k>=2} g(k) floor(x/k)
    #       = sum_{k>=2} g(k) (floor(x/k) - x/k)
    
    # Objective: a = -sum_{k} g(k) log(k)/k = -sum_{k>=2} g(k) log(k)/k (since log(1)=0)
    
    # Constraints on g(k) for k >= 2:
    # - The original f(k) for k >= 2 must be in [-10, 10] before clipping.
    # - But g(k) for k >= 2 equals the clipped f(k), so g(k) in [-10, 10].
    # - g(1) is not constrained by clipping (it's set after clipping).
    
    # So: maximize a = -sum_{k>=2} g(k) log(k)/k
    # subject to:
    #   M(x) = sum_{k>=2} g(k) (floor(x/k) - x/k) <= 1 for x = 1, ..., 10N
    #   -10 <= g(k) <= 10 for k >= 2
    
    # This is a linear program!
    # Let's use scipy.optimize.linprog if available, otherwise do a simple approach.
    
    try:
        from scipy.optimize import linprog
        have_scipy = True
    except ImportError:
        have_scipy = False
    
    # Coefficients for objective: c_k = -log(k) for k in sq_free
    # Maximize -sum g(k) log(k)/k = sum g(k) (-log(k)/k)
    # linprog minimizes c^T x, so c_k = log(k)/k
    
    c = np.array([np.log(k) / k for k in sq_free], dtype=np.float64)
    
    # Constraint matrix A_ub for M(x) <= 1
    # M(x) = sum_{k} g(k) (floor(x/k) - x/k) <= 1
    # A_ub[i, j] = (floor(x_i / k_j) - x_i / k_j)
    # b_ub[i] = 1
    
    x_vals = np.arange(1, 10 * N + 1)
    A_ub = np.zeros((len(x_vals), m), dtype=np.float64)
    for i, x in enumerate(x_vals):
        for j, k in enumerate(sq_free):
            A_ub[i, j] = np.floor(x / k) - x / k
    
    b_ub = np.ones(len(x_vals), dtype=np.float64)
    
    # Bounds: -10 <= g(k) <= 10
    bounds = [(-10.0, 10.0)] * m
    
    if have_scipy:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            g_vals = result.x
            # Compute f(1)
            f1 = -np.sum(g_vals / sq_free)
            
            domain = np.concatenate(([1], sq_free))
            values = np.concatenate(([f1], g_vals))
            
            # Verify
            K = int(domain.max())
            M_max = 0.0
            for x in range(1, 10 * K + 1):
                Mx = f1 * x + np.sum(g_vals * np.floor(x / sq_free))
                if Mx > M_max:
                    M_max = Mx
            
            a = -np.sum(values * np.log(domain) / domain)
            # np.log(1) = 0, so this is fine.
            
            if M_max <= 1.0001:
                return domain.astype(np.int64), values.astype(np.float64)
    
    # Fallback: Use Möbius values directly
    mu_vals = np.array([mu[k] for k in sq_free], dtype=np.float64)
    f1 = -np.sum(mu_vals / sq_free)
    
    domain = np.concatenate(([1], sq_free))
    values = np.concatenate(([f1], mu_vals))
    
    return domain.astype(np.int64), values.astype(np.float64)

# EVOLVE-BLOCK-END
