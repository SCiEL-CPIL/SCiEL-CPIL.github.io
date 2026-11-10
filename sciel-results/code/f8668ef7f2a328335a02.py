# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Uses LP optimization over a dense domain with post-processing to improve score.
    """
    start_time = time.time()
    deadline = start_time + 570  # Leave 60s margin for safety

    def _elapsed():
        return time.time() - start_time

    def _evaluator_score(domain, values):
        """Compute the actual evaluator score for a candidate (domain, values)."""
        m = len(domain)
        if m < 1:
            return -np.inf
        k = domain.astype(np.float64)
        f = values.astype(np.float64).copy()
        # Clip to [-10, 10]
        f = np.clip(f, -10.0, 10.0)
        # Compute S = sum f(k)/k, then f(1) -= S
        S = np.sum(f / k)
        # f(1) is at index where domain == 1
        idx1 = np.where(domain == 1)[0]
        if len(idx1) > 0:
            f[idx1[0]] -= S
        else:
            # f(1) = 0 - S = -S, but 1 not in domain so we add it? No, evaluator says f(1)=0 before this if 1 not in domain
            # Actually re-reading: "f(1) is replaced by f(1) - S ... (f(1) = 0 before this if 1 is not in the domain)"
            # This means if 1 not in domain, f(1) starts at 0, so after: f(1) = -S
            # But 1 is not in domain, so this adjustment doesn't apply to any stored value
            # Wait, this is confusing. Let me re-read.
            # "f(1) is replaced by f(1) - S with S = sum_k f(k)/k over the clipped values (f(1) = 0 before this if 1 is not in the domain), so that afterwards sum_k f(k)/k = 0."
            # I think this means: the function value at 1 is adjusted. If 1 is in domain, adjust it. If not, conceptually f(1)=-S but since 1 not in domain, it doesn't contribute to M(x) or the metric.
            # Actually wait - if 1 is not in domain, then sum_k f(k)/k doesn't include k=1, so S = sum_{k in domain} f(k)/k.
            # The adjustment is f(1) = f(1) - S. If 1 not in domain, f(1) was 0, becomes -S.
            # But since 1 is not in domain, this doesn't affect M(x) or the metric computation.
            # Hmm, but it says "so that afterwards sum_k f(k)/k = 0" - this must mean the sum over the domain equals 0 after adjustment.
            # If 1 is in domain: new_f(1) = old_f(1) - S, and sum new_f(k)/k = sum old_f(k)/k - S/1 = S - S = 0. ✓
            # If 1 not in domain: the sum over domain is still S (unchanged), which is not 0. Contradiction.
            # I think the intent is: we must include 1 in domain, or the formula only works when 1 is in domain.
            # Let me just ensure 1 is always in domain for safety.
            pass
        # M(x) check
        K_max_val = int(np.max(domain))
        x_max = 10 * K_max_val
        x_range = np.arange(1, x_max + 1, dtype=np.int64)
        M = np.zeros(x_max, dtype=np.float64)
        for i in range(m):
            M += f[i] * (x_range // int(domain[i]))
        if np.max(M) > 1.0001:
            return -np.inf
        # Metric
        metric = -np.sum(f * np.log(k) / k)
        return metric

    best_domain = None
    best_values = None
    best_score = -np.inf

    # Try several K values - keep manageable for LP
    K_candidates = [200, 300, 400, 500, 700, 1000]

    for K in K_candidates:
        if _elapsed() > deadline - 80:
            break

        k_vals = np.arange(1, K + 1, dtype=np.int64)
        x_max = 10 * K

        # Build constraint matrix A: M(x) = sum_k f(k) * floor(x/k) <= 1 for x in [1, x_max]
        x_range = np.arange(1, x_max + 1, dtype=np.int64)
        A = (x_range[:, None] // k_vals[None, :]).astype(np.float64)

        # The evaluator adjusts f(1) by -S where S = sum f(k)/k.
        # After adjustment, sum f(k)/k = 0.
        # The metric is -sum f_adjusted(k) * log(k) / k.
        # f_adjusted(1) = f(1) - S, others unchanged.
        # So metric = -[(f(1)-S)*log(1)/1 + sum_{k>=2} f(k)*log(k)/k]
        #           = -[0 + sum_{k>=2} f(k)*log(k)/k]  since log(1)=0
        # So the metric doesn't depend on f(1) at all!
        # But f(1) affects M(x) through floor(x/1) = x.
        # And the constraint sum f(k)/k = 0 after adjustment is automatically satisfied.
        # 
        # For the LP: we need M(x) <= 1 for all x, where M(x) uses adjusted values.
        # M(x) = sum_k f_adj(k) * floor(x/k) = f_adj(1)*x + sum_{k>=2} f(k)*floor(x/k)
        #      = (f(1) - S)*x + sum_{k>=2} f(k)*floor(x/k)
        #      = f(1)*x - S*x + sum_{k>=2} f(k)*floor(x/k)
        #
        # This is getting complex. Let's use a simpler approach:
        # Set up LP with variables f(1),...,f(K) and constraint sum f(k)/k = 0 (pre-adjustment).
        # Then the adjusted values have sum = 0 automatically.
        # M(x) with adjusted values: M_adj(x) = sum_k f_adj(k)*floor(x/k)
        # f_adj(1) = f(1) - S where S = sum f(k)/k = 0 (by our constraint)
        # So f_adj(1) = f(1) when sum f(k)/k = 0!
        # Therefore M_adj(x) = M(x) when the equality constraint is satisfied.
        # 
        # So the LP with equality constraint sum f(k)/k = 0 and M(x) <= 1 is correct.
        # And the metric is -sum f(k)*log(k)/k (f(1) term vanishes).

        # Objective: maximize -sum f(k) * log(k) / k
        # In linprog (minimize), minimize sum f(k) * log(k) / k
        log_k = np.log(k_vals.astype(np.float64))
        c_obj = log_k / k_vals.astype(np.float64)  # minimize this dot f

        # Equality constraint: sum f(k) / k = 0
        A_eq = (1.0 / k_vals.astype(np.float64)).reshape(1, -1)
        b_eq = np.array([0.0])

        # Inequality constraints: A @ f <= 1
        b_ub = np.ones(x_max, dtype=np.float64)

        # Bounds
        bounds = [(-10.0, 10.0)] * K

        try:
            time_limit = 40 if K <= 500 else 60
            result = linprog(c_obj, A_ub=A, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                             bounds=bounds, method='highs',
                             options={'presolve': True, 'time_limit': time_limit,
                                      'mip_rel_gap': 0.01})
            if _elapsed() > deadline - 30:
                break

            if result.success:
                f_vals = result.x.copy()
                # Post-processing: try to improve by pushing toward objective-optimal
                # Direction: decrease f(k) for k>=2 where log(k)/k > 0 (i.e., k>1)
                # But must maintain M(x) <= 1 and sum f(k)/k = 0
                # Simple approach: compute slack in M(x) constraints, then adjust
                score = -np.sum(f_vals * log_k / k_vals.astype(np.float64))
                if score > best_score:
                    best_score = score
                    best_domain = k_vals.copy()
                    best_values = f_vals.copy()
        except Exception:
            continue

    # Post-processing: improve the best solution found
    if best_domain is not None:
        # Try to push values to improve objective while maintaining feasibility
        # The objective is -sum f(k)*log(k)/k, so we want f(k) negative for k>1
        # and f(1) can be anything (doesn't affect objective directly)
        # But f(1) affects M(x) via x term
        k_vals = best_domain.astype(np.int64)
        f_vals = best_values.astype(np.float64).copy()
        K = len(k_vals)
        x_max = 10 * int(np.max(k_vals))
        x_range = np.arange(1, x_max + 1, dtype=np.int64)
        
        # Compute current M(x)
        M = np.zeros(x_max, dtype=np.float64)
        for i in range(K):
            M += f_vals[i] * (x_range // k_vals[i])
        slack = 1.0 - M  # slack >= 0 for feasible
        
        # Try small perturbations: decrease f(k) for k>1 (improves objective)
        # and adjust f(1) to maintain sum f(k)/k = 0
        # d(f(k))/dk for k>1: -delta, d(f(1))/d1 = delta * sum_{k>1} 1/k * k = delta * (K-1)
        # Wait: constraint sum f(k)/k = 0 => f(1)/1 + sum_{k>1} f(k)/k = 0
        # If we change f(k) by delta_k for k>1, then f(1) must change by -sum_{k>1} delta_k/k * 1
        # Actually: f(1) = -sum_{k>1} f(k)/k
        # If f(k) -> f(k) + delta_k, then f(1) -> f(1) - sum_{k>1} delta_k/k
        
        # Try uniform decrease for k>1
        for trial_delta in [0.01, 0.05, 0.1, 0.2, 0.5]:
            if _elapsed() > deadline - 10:
                break
            f_trial = f_vals.copy()
            # Decrease f(k) for k > 1
            for i in range(1, K):
                f_trial[i] -= trial_delta
            # Adjust f(1) to maintain constraint
            f_trial[0] = -np.sum(f_trial[1:] / k_vals[1:].astype(np.float64))
            # Check bounds
            if np.any(f_trial < -10.0) or np.any(f_trial > 10.0):
                continue
            # Check M(x)
            M_trial = np.zeros(x_max, dtype=np.float64)
            for i in range(K):
                M_trial += f_trial[i] * (x_range // k_vals[i])
            if np.max(M_trial) <= 1.0001:
                score_trial = -np.sum(f_trial * np.log(k_vals.astype(np.float64)) / k_vals.astype(np.float64))
                if score_trial > best_score:
                    best_score = score_trial
                    best_values = f_trial.copy()

    # Fallback: use Möbius function if LP didn't produce anything
    if best_domain is None:
        K_max = 50000
        mu = np.zeros(K_max + 1, dtype=np.int8)
        mu[1] = 1
        is_prime = np.ones(K_max + 1, dtype=bool)
        primes = []
        for i in range(2, K_max + 1):
            if is_prime[i]:
                primes.append(i)
                mu[i] = -1
            for p in primes:
                if i * p > K_max:
                    break
                is_prime[i * p] = False
                if i % p == 0:
                    mu[i * p] = 0
                    break
                else:
                    mu[i * p] = -mu[i]

        k_vals = np.arange(1, K_max + 1)
        mu_over_k = mu[1:] / k_vals.astype(np.float64)

        best_domain = None
        best_values = None
        best_score = -np.inf

        for K in range(5000, 999, -1000):
            if _elapsed() > deadline - 10:
                break
            S1 = np.sum(mu_over_k[:K])
            if S1 < 0:
                continue
            domain = k_vals[:K].copy()
            values = mu[1:K+1].astype(np.float64).copy()
            values[0] = 1.0 - S1
            x_max = 10 * K
            M_prime = np.zeros(x_max, dtype=np.float64)
            x_range = np.arange(1, x_max + 1)
            for k in range(1, K + 1):
                M_prime += values[k-1] * (x_range // k)
            if np.max(M_prime) <= 1.0001:
                log_k = np.log(k_vals[:K])
                score = -np.sum(values * log_k / k_vals[:K])
                if score > best_score:
                    best_score = score
                    best_domain = domain
                    best_values = values

        if best_domain is None:
            domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
            values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        else:
            domain = best_domain
            values = best_values
    else:
        domain = best_domain
        values = best_values

    return domain, values

# EVOLVE-BLOCK-END
