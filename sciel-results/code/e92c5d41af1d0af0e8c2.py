import time
from collections import Counter
import itertools
import math

_START = time.time()
_DEADLINE = _START + 280.0

_cache = {}

def _generate_vectors(n):
    return [tuple(v) for v in itertools.product([0, 1, 2], repeat=n)]

def _base_score(vector, n):
    c = Counter(vector)
    c0 = c.get(0, 0)
    c1 = c.get(1, 0)
    c2 = c.get(2, 0)
    
    distinct_count = sum(1 for c_val in [c0, c1, c2] if c_val > 0)
    sum_sq = c1 + 4 * c2
    
    ideal = n / 3.0
    dev = abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal)
    
    # Compute Hamming distance to the "center" vector (1,1,...,1)
    # Vectors with more 0s and 2s are farther from center
    hamming_from_center = c0 + c2
    
    # Spectral penalty: measure how much the vector's "characteristic" 
    # projects onto high-frequency (non-trivial) characters in F_3^n.
    # For a vector v, compute the squared magnitude of its DFT at a few
    # representative high-frequency directions. Vectors with large 
    # projections onto many directions are more "structured" and more
    # likely to form lines.
    # 
    # We approximate by computing the variance of the vector entries 
    # weighted by character values. For direction omega, the projection
    # is sum_k omega^{v_k}. High variance across different omega values
    # indicates high-frequency content.
    # 
    # Efficient proxy: compute the "energy" in non-DC components.
    # For a single vector v, think of it as a delta function at v.
    # Its DFT has |hat{f}(omega)| = 1 for all omega (uniform).
    # So we need a different approach: measure how "spread out" the
    # vector is in a way that correlates with line participation.
    #
    # Better proxy: For each coordinate, if v_k is not 1 (the "center"),
    # it contributes to the "frequency" along that axis. The total
    # high-frequency energy is proportional to the number of non-center
    # coordinates, but we want a smooth measure.
    #
    # Use: spectral_energy = sum_k (v_k - 1)^2 * (1 + 0.1 * (v_k - 1)^2)
    # This is smooth and penalizes deviations from center quadratically
    # with a slight super-quadratic boost for larger deviations.
    spectral_energy = 0.0
    for val in vector:
        d = val - 1
        spectral_energy += d * d * (1.0 + 0.1 * d * d)
    
    # Higher distinct count is better
    # Higher sum_sq is better (arbitrary choice, consistent with previous)
    # Lower deviation is better
    # Higher Hamming distance from center encourages spatial spread
    # Lower spectral energy is better (less high-frequency content)
    return distinct_count * 10000.0 + sum_sq * 100.0 - dev * 10.0 + hamming_from_center * 50.0 - spectral_energy * 20.0

def _soft_min(a, b, tau=1.0):
    """Smooth approximation to min(a, b) using log-sum-exp."""
    m = min(a, b)
    if m == 0:
        return 0.0
    # soft-min = -tau * log(1 + exp(-(a - m)/tau) * exp(-(b - m)/tau) ... )
    # Simpler: soft-min(a,b) = m - tau * log(1 + exp(-(a-m)/tau) + exp(-(b-m)/tau) - 1)
    # Even simpler approximation: m - tau * log(1 + exp(-max(a-m, b-m)/tau))
    # Use: soft_min ≈ m - tau * log(1 + exp(-(max(a,b)-m)/tau))
    # This is a smooth lower bound on min.
    diff = max(a, b) - m
    if diff < 1e-10:
        return m
    return m - tau * math.log1p(math.exp(-diff / tau))

def _compute_penalties(vectors, base_scores, n):
    num_vecs = len(vectors)
    penalties = [0.0] * num_vecs
    # Map vector to index
    vec_to_idx = {v: i for i, v in enumerate(vectors)}
    
    # Soft-min temperature: small enough to approximate min closely,
    # large enough to provide smooth gradient information.
    tau = 500.0
    
    for i, v in enumerate(vectors):
        total_pen = 0.0
        
        for j, u in enumerate(vectors):
            if i == j: continue # u == v
            # Compute w = -v - u mod 3
            w = tuple((3 - (v[k] + u[k]) % 3) % 3 for k in range(n))
            if w == u: continue # Not a distinct triplet
            
            k = vec_to_idx.get(w)
            if k is None:
                continue # Should not happen
            
            bu = base_scores[j]
            bw = base_scores[k]
            # Use soft-min instead of hard min for smoother landscape
            total_pen += 0.5 * _soft_min(bu, bw, tau)
            
        penalties[i] = total_pen
        
    return penalties

def _precompute(n):
    if time.time() - _START > _DEADLINE - 10.0:
        return None # Timeout risk
        
    vectors = _generate_vectors(n)
    base_scores = [_base_score(v, n) for v in vectors]
    
    if n <= 7:
        penalties = _compute_penalties(vectors, base_scores, n)
        # Normalize penalty
        num_pairs = (len(vectors) - 1) / 2.0
        if num_pairs > 0:
            avg_penalties = [p / num_pairs for p in penalties]
        else:
            avg_penalties = [0.0] * len(vectors)
        
        # Combine
        # Scale factor: Base score is ~10000-30000.
        # Avg penalty is ~10000-30000 (since it's an average of base scores).
        # So they are on similar scales.
        # Score = Base - Avg_Penalty
        # Apply a slight nonlinearity to amplify differences in the combined score
        # to help the greedy algorithm make sharper choices
        final_scores = [b - p for b, p in zip(base_scores, avg_penalties)]
        
        # Apply a smooth rank-based boost: vectors with very high combined scores
        # get a small extra boost to break ties and encourage diversity
        if len(final_scores) > 1:
            max_s = max(final_scores)
            min_s = min(final_scores)
            range_s = max_s - min_s
            if range_s > 1e-10:
                # Normalize to [0, 1] and apply a gentle sigmoid-like boost
                for idx in range(len(final_scores)):
                    norm = (final_scores[idx] - min_s) / range_s
                    # Gentle boost: adds up to 1% of range for top vectors
                    final_scores[idx] += 0.01 * range_s * (norm ** 2)
    else:
        # For large n, skip penalty
        final_scores = base_scores
        
    score_map = {v: s for v, s in zip(vectors, final_scores)}
    return score_map

def priority(vector, n):
    """
    Return a float priority for a vector in {0,1,2}^n.
    """
    # Deadline check
    if time.time() - _START > _DEADLINE:
        return 0.0
    
    if n not in _cache:
        # Pre-compute if not already done
        # Check time before starting heavy pre-computation
        if time.time() - _START > _DEADLINE - 60.0:
            # Not enough time for pre-computation, use fallback
            _cache[n] = None # Mark as fallback
        else:
            _cache[n] = _precompute(n)
            
    if _cache[n] is None:
        # Fallback to simple heuristic
        return _base_score(vector, n)
        
    return _cache[n][vector]
