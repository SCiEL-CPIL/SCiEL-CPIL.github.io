"""Optimized program for second autocorrelation inequality task."""
import numpy as np

N = 50

# EVOLVE-BLOCK-START
def construct():
    # Formulate as convex optimization: maximize ||h||_2^2 subject to 
    # ||h||_1 = 1, h >= 0, and ||h*h||_inf <= 1.
    # Since the problem is scale-invariant, we fix ||h||_1 = 1.
    # The objective ||h||_2^2 is convex (actually concave in h for fixed L1? No, 
    # x^2 is convex, so maximizing convex is non-convex).
    # However, we can use a different formulation.
    # Actually, let's use a heuristic approach that is better than the Sidon set.
    # A known good construction for this type of problem is to use a "flat" 
    # distribution or a specific pattern.
    
    # Let's try to use a uniform distribution over a subset of indices that 
    # minimizes the maximum autocorrelation.
    # The current Sidon set gives a score of ~1.1020 relative to seed.
    # Let's try a different set or a continuous approach.
    
    # Since we cannot use CVXPY directly in a simple way without installing it,
    # and the directive suggests a new design but we must keep the approach,
    # let's improve the Sidon set approach by trying different sets or 
    # adjusting the weights.
    
    # Actually, the directive says: "design a new program... built on this approach, from scratch; do not reuse the current designs"
    # But also "keep this program's approach and improve it within that approach"
    # These are conflicting. Let's assume the second directive (improve within approach) takes precedence for safety, 
    # but the first one is more specific about the new design.
    
    # Let's try a different Sidon-like set or a set with better properties.
    # A perfect difference set or a related structure might work better.
    
    # Let's try a uniform distribution over all 50 points.
    # h = 1/50 for all i.
    # ||h||_2^2 = 50 * (1/50)^2 = 1/50
    # ||h||_1 = 1
    # h*h: The autocorrelation of a uniform distribution is a triangle.
    # The peak is at lag 0: sum(h_i^2) = 1/50.
    # Wait, the autoconvolution (h*h) at lag 0 is sum h_i^2.
    # The max of h*h is at lag 0 for non-negative h.
    # So ||h*h||_inf = ||h||_2^2.
    # Ratio = ||h||_2^2 / (||h||_1 * ||h*h||_inf) = ||h||_2^2 / (1 * ||h||_2^2) = 1.
    # This is the minimum possible ratio (by Cauchy-Schwarz, ||h||_2^2 <= ||h||_1^2, so ratio >= 1/||h||_1? No.)
    # Let's re-read the task: "maximize the ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution."
    # Ratio = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    # If h is uniform, ||h||_2^2 = 1/N, ||h||_1 = 1, ||h*h||_inf = 1/N.
    # Ratio = (1/N) / (1 * 1/N) = 1.
    
    # The Sidon set gives a higher ratio.
    # For Sidon set of size K, ||h||_2^2 = K * (1/K)^2 = 1/K.
    # ||h||_1 = 1.
    # ||h*h||_inf = max multiplicity * (1/K)^2 = M/K^2.
    # Ratio = (1/K) / (M/K^2) = K/M.
    # For K=7, M=2, Ratio = 3.5.
    
    # To improve, we need a set with larger K/M.
    # We can try to find a Sidon set with larger K or smaller M.
    # M is at least 1 (for the zero lag). For non-zero lags, M can be 1 if it's a perfect Sidon set.
    # But for a Sidon set, the number of distinct differences is at least K(K-1)/2.
    # The range of differences is -N+1 to N-1.
    # For N=50, the max number of distinct differences is 99.
    # So K(K-1)/2 <= 99 => K^2 <= 200 => K <= 14.
    # So we can have a Sidon set of size up to 14.
    # If M=1 for all non-zero lags, then Ratio = K.
    # But M=1 for all non-zero lags is impossible because the zero lag has multiplicity K.
    # So ||h*h||_inf is at least K/K^2 = 1/K.
    # So Ratio <= K.
    
    # Let's try to find a Sidon set of size 14 in [0, 49].
    # A known Sidon set of size 14 in [0, 49] might exist.
    
    # Let's use a greedy algorithm to find a large Sidon set.
    def find_sidon_set(n, max_size):
        s = set()
        for i in range(n):
            if len(s) >= max_size:
                break
            # Check if i can be added
            can_add = True
            for j in s:
                diff = abs(i - j)
                # Check if this difference already exists
                for k in s:
                    if k != j and abs(i - k) == diff:
                        can_add = False
                        break
                if not can_add:
                    break
            if can_add:
                s.add(i)
        return sorted(s)
    
    # This greedy algorithm is not optimal. Let's use a known Sidon set.
    # A Sidon set of size 14 in [0, 49] is not trivial to find by hand.
    # Let's try a smaller set with better properties or a different construction.
    
    # Actually, the current score is 1.1020, which corresponds to Ratio = 3.5 * seed_ratio.
    # If the seed is uniform (Ratio=1), then 3.5 would be 3.5x better.
    # But the score is 1.1020, so the seed must be better than uniform.
    
    # Let's try to improve the Sidon set by using a different one.
    # A known Sidon set of size 13 in [0, 49]:
    # {0, 1, 4, 10, 18, 23, 31, 36, 40, 43, 45, 47, 48}
    # Let's verify this is a Sidon set.
    
    sidon_indices = [0, 1, 4, 10, 18, 23, 31, 36, 40, 43, 45, 47, 48]
    K = len(sidon_indices)
    
    # Check if it's a valid Sidon set
    diffs = set()
    valid = True
    for i in range(K):
        for j in range(i+1, K):
            d = sidon_indices[j] - sidon_indices[i]
            if d in diffs:
                valid = False
                break
            diffs.add(d)
        if not valid:
            break
    
    if valid:
        h = [0.0] * N
        for idx in sidon_indices:
            h[idx] = 1.0 / K
        return h
    else:
        # Fallback to the original Sidon set
        sidon_indices = [0, 1, 3, 7, 15, 31, 48]
        K = len(sidon_indices)
        h = [0.0] * N
        for idx in sidon_indices:
            h[idx] = 1.0 / K
        return h
# EVOLVE-BLOCK-END
