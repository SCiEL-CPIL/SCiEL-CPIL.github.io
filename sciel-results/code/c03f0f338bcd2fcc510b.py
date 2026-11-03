import numpy as np
import time

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max)) / (DELTA**2)

def _find_max_sidon(n):
    """Find a maximum-size Sidon set in {0, ..., n-1} using greedy + local search."""
    # Greedy construction
    S = [0]
    used_sums = set()
    for i in range(1, n):
        # Check if i can be added
        ok = True
        for s in S:
            if s + i in used_sums:
                ok = False
                break
        if ok:
            for s in S:
                used_sums.add(s + i)
            used_sums.add(2 * i)
            S.append(i)
    
    S = sorted(S)
    
    # Try to improve with local search: try swapping elements
    improved = True
    while improved and len(S) < n:
        improved = False
        S_set = set(S)
        # Try adding each missing element
        for cand in range(n):
            if cand in S_set:
                continue
            test_S = S + [cand]
            # Check Sidon property
            sums = set()
            valid = True
            for i in range(len(test_S)):
                for j in range(i, len(test_S)):
                    s = test_S[i] + test_S[j]
                    if s in sums:
                        valid = False
                        break
                    sums.add(s)
                if not valid:
                    break
            if valid:
                S = sorted(test_S)
                S_set = set(S)
                improved = True
                break
    
    return S

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Find maximum Sidon set in {0, ..., N-1}
    sidon = _find_max_sidon(N)
    K = len(sidon)
    
    # For a Sidon set with uniform weights, the autoconvolution peak is h_max^2
    # (since all i+j are unique, no collisions at any lag except lag 0 which has only h_i^2)
    # Actually at lag 0: sum of h_i^2. Wait no - autoconvolution at lag 0 is sum_i h_i * h_i = sum h_i^2
    # At lag k: sum_i h_i * h_{i+k}. For Sidon set, each pair (i,j) with i!=j gives a unique sum i+j,
    # so at each nonzero lag, there's at most one term h_i*h_j.
    # The peak is max(max_i h_i^2, max_{i!=j, i+j=k} h_i*h_j)
    # With uniform weights h_i = 1/K: peak = max(1/K^2, 1/K^2) = 1/K^2
    # Ratio = (K * (1/K)^2) / (1 * 1/K^2) = (1/K) / (1/K^2) = K
    # So ratio = K for uniform Sidon.
    
    # But we can do better! The score is L2^2 / (L1 * Sup(autoconv))
    # With L1 = 1: L2^2 / Sup(autoconv)
    # For uniform on K Sidon positions: L2^2 = K*(1/K)^2 = 1/K
    # Sup(autoconv) = 1/K^2 (peak of convolution)
    # Ratio = (1/K) / (1/K^2) = K
    # Then divided by DELTA^2: K / DELTA^2
    
    # Can we beat K? What if we use non-uniform weights?
    # L2^2 = sum h_i^2, Sup(autoconv) >= max_i h_i^2 (lag 0 contribution from single element)
    # Actually Sup(autoconv) = max over all lags. At lag 0: sum h_i^2. At other lags: at most h_i*h_j.
    # So Sup = max(sum h_i^2, max_{i<j, i+j unique} h_i*h_j)
    # Since sum h_i^2 >= h_i^2 for all i, and h_i*h_j <= (h_i^2+h_j^2)/2 <= sum h_i^2,
    # the peak is always at lag 0: Sup = sum h_i^2 = L2^2!
    # Wait that's wrong. Let me recompute.
    
    # Autoconvolution (h*h)[k] = sum_i h[i] * h[i+k] for valid i
    # At k=0: sum_i h[i]^2 = L2^2
    # At k!=0: for Sidon set, at most one pair contributes, so (h*h)[k] = h[i]*h[j] for some i,j
    # Since h[i]*h[j] <= (h[i]^2 + h[j]^2)/2 <= L2^2/2 < L2^2 (for K>=2)
    # So Sup(autoconv) = L2^2 always!
    # Then ratio = L2^2 / (L1 * L2^2) = 1/L1
    # With L1=1: ratio = 1. That can't be right since we're getting 1.148.
    
    # I think I'm confusing discrete convolution with the actual autoconvolution of the step function.
    # The step function f(x) = h[i] * DELTA for x in [i*DELTA, (i+1)*DELTA)
    # The autoconvolution (f*f)(t) = integral f(s)*f(t-s) ds
    # This is a continuous convolution, not just the discrete sum.
    # The discrete np.convolve gives the coefficients, but the actual continuous autoconvolution
    # is a piecewise linear function whose peak depends on the overlap structure.
    
    # For a step function, the autoconvolution is a trapezoid-like shape.
    # The peak of the continuous autoconvolution of a step function with heights h[i]*DELTA:
    # (f*f)(t) at t=0 is sum_i (h[i]*DELTA)^2 * DELTA = DELTA^2 * sum h[i]^2 * DELTA... 
    # Actually let me think more carefully.
    
    # f(x) = h[i] * DELTA for x in [i*DELTA, (i+1)*DELTA)
    # (f*f)(t) = ∫ f(s) f(t-s) ds
    # At t=0: ∫ f(s) f(-s) ds. Since f is on [0, (N-1)*DELTA] roughly...
    # Actually the domain is [-1/4, 1/4], so positions are at -1/4 + i*DELTA for i=0..N-1
    # Each step has width DELTA and height h[i].
    
    # The autoconvolution peak for a single step of height h and width DELTA is h^2 * DELTA (at t=0).
    # For multiple non-overlapping steps (Sidon-like), the peak is max_i h[i]^2 * DELTA.
    # But if steps are close, their autoconvolutions overlap.
    
    # For the discrete approximation: np.convolve(h, h, 'full') * DELTA^2 approximates the autoconv
    # The peak of the continuous autoconv ≈ max of the trapezoid peaks.
    
    # Let me just use the empirical formula from the scoring function.
    # _score_raw computes: (sum h^2) / (sum h * max(convolve(h,h,'full'))) / DELTA^2
    
    # For uniform h on K Sidon positions (h[i]=1/K on those, 0 elsewhere):
    # sum h^2 = K * (1/K)^2 = 1/K
    # sum h = 1
    # convolve(h,h,'full'): at lag 0, value = sum h[i]^2 = 1/K
    # At other lags, for Sidon: at most one h[i]*h[j] = 1/K^2
    # So max = 1/K
    # Score = (1/K) / (1 * 1/K) / DELTA^2 = 1/DELTA^2
    
    # Wait, that gives 1/DELTA^2 regardless of K! That's the same as a single delta.
    # Let me recheck with actual computation.
    
    # Actually I think the issue is that for the Sidon set, the autoconvolution at lag 0 
    # equals sum h[i]^2, and that IS the maximum. So the ratio is always 1/L1 = 1.
    # But the current program gets 1.148, so something is different.
    
    # Let me just test empirically what works best.
    
    # Strategy: Use exponentially weighted Sidon set
    # h[i] = r^(rank) for i in Sidon set, normalized so sum = 1
    # As r -> 1, approaches uniform -> ratio -> 1/DELTA^2
    # As r -> 0, approaches delta -> ratio -> 1/DELTA^2
    # Hmm, both limits give the same?
    
    # Wait, I need to reconsider. The _score_raw function divides by DELTA^2.
    # For a single delta (h[25]=1, rest 0):
    # sum h^2 = 1, sum h = 1, convolve peak = 1
    # score = 1/(1*1)/DELTA^2 = 1/DELTA^2 = 1/(0.01^2) = 10000
    
    # That seems too high. Let me re-read the problem.
    # "maximize the ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution"
    # Score = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    # The _score_raw divides by DELTA^2, which accounts for the discrete approximation.
    
    # The "lower bound on C2" is what's being measured. The raw value is the score.
    # Higher is better. Current best is 1.1484 (the score value, not divided by DELTA^2).
    
    # Wait, looking at _score_raw: it returns (l2_sq / (l1 * autoconv_max)) / (DELTA**2)
    # For uniform h (all 1/50): l2_sq = 50*(1/50)^2 = 1/50, l1=1
    # convolve peak at lag 0 = 1/50, so score = (1/50)/(1*1/50)/0.0001 = 1/0.0001 = 10000
    # That's way more than 1.148.
    
    # I think the score reported (1.1484) is value/seed, where seed is the baseline.
    # The raw value might be around 1.1484 * seed_value.
    # Or the _score_raw in the current program is computing something specific.
    
    # Let me just focus on maximizing _score_raw as defined.
    # The key question: what h maximizes sum(h^2)/(sum(h) * max(convolve(h,h)))?
    
    # Since it's scale invariant, fix sum(h)=1. Maximize sum(h^2) / max(convolve(h,h)).
    # max(convolve(h,h)) >= sum(h^2) (at lag 0, it equals sum h[i]^2)
    # So the ratio is always <= 1!
    # But we're getting 1.148... 
    # 
    # OH WAIT. The convolve in numpy for 'full' mode: the peak might NOT be at lag 0!
    # For h with support on a Sidon set, at lag 0: sum h[i]^2
    # But the actual maximum of the convolution array might be elsewhere if the
    # continuous autoconvolution has a higher peak due to overlapping trapezoids.
    # 
    # No wait, np.convolve(h,h,'full') at index N-1 (lag 0) = sum h[i]*h[i] = sum h[i]^2
    # This is always >= any other lag value for nonnegative h (by Cauchy-Schwarz or direct:
    # (h*h)[k] = sum h[i]h[i+k] <= sqrt(sum h[i]^2) * sqrt(sum h[i+k]^2) <= sum h[i]^2)
    # 
    # So max(convolve) = sum h[i]^2 always, and the ratio is always 1/sum(h) = 1 when sum=1.
    # 
    # This means _score_raw always returns 1/DELTA^2 for any valid h!
    # Unless... the heights can be zero and the "nonnegative" allows that.
    # 
    # I'm confused. Let me just run the current program's logic and see what it produces,
    # then try to improve from there. The key insight from the findings is that Sidon sets
    # with exponentially skewed weights break the barrier. Let me trust that and optimize.
    
    # Given the empirical finding that Sidon + exponential skew works, let me:
    # 1. Find max Sidon set
    # 2. Optimize the exponential rate r to maximize the score
    # 3. Also try local search on the weights
    
    # For now, use the Sidon set with optimized weights
    
    # Let me try a direct optimization approach
    from scipy.optimize import minimize
    
    def neg_score(h):
        h = np.asarray(h)
        l1 = np.sum(h)
        if l1 < 1e-15:
            return 0.0
        l2_sq = np.sum(h**2)
        hc = np.convolve(h, h, mode='full')
        mc = np.max(hc)
        if mc < 1e-15:
            return 0.0
        return -(l2_sq / (l1 * mc))
    
    # Initialize with Sidon set uniform
    h0 = np.zeros(N)
    for i in sidon:
        h0[i] = 1.0 / K
    
    # Normalize
    h0 = h0 / np.sum(h0)
    
    best_h = h0.copy()
    best_val = -neg_score(h0)
    
    # Try exponential weighting on Sidon set
    for r in np.linspace(0.5, 0.999, 100):
        h_test = np.zeros(N)
        for j, idx in enumerate(sorted(sidon)):
            h_test[idx] = r**j
        s = np.sum(h_test)
        if s > 0:
            h_test /= s
        val = -neg_score(h_test)
        if val > best_val:
            best_val = val
            best_h = h_test.copy()
    
    # Local search: perturb weights on Sidon set
    if time.time() < deadline - 20:
        for _ in range(200):
            if time.time() > deadline - 10:
                break
            h_test = best_h.copy()
            # Random perturbation
            mask = np.random.random(N) < 0.3
            h_test[mask] *= (1 + 0.1 * np.random.randn(mask.sum()))
            h_test = np.clip(h_test, 0, None)
            s = np.sum(h_test)
            if s > 0:
                h_test /= s
            val = -neg_score(h_test)
            if val > best_val:
                best_val = val
                best_h = h_test.copy()
    
    # SLSQP optimization
    if time.time() < deadline - 15:
        constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0}]
        bounds = [(0.0, 1.0)] * N
        try:
            res = minimize(neg_score, best_h, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 300, 'ftol': 1e-12})
            h_cand = np.clip(res.x, 0, None)
            s = np.sum(h_cand)
            if s > 0:
                h_cand /= s
            val = -neg_score(h_cand)
            if val > best_val:
                best_val = val
                best_h = h_cand
        except:
            pass
    
    # Final normalization and return
    best_h = np.clip(best_h, 0, None)
    s = np.sum(best_h)
    if s > 0:
        best_h /= s
    
    return list(best_h)
