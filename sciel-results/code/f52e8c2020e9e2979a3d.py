"""
Cap set task: greedy insertion with a learned, conflict-aware priority surrogate.

Strategy:
  1. On first call (or per n), build a small high-quality reference cap set via
     greedy + local search.
  2. From the reference set, compute for every candidate vector a "marginal conflict
     cost" feature vector (counts of 3-term arithmetic progressions it would form
     with pairs from the reference set, plus structural features).
  3. Train a fast feed-forward / linear surrogate (with regularization and higher-order
     features) to predict a priority score that favors vectors with low conflict cost.
  4. Use that surrogate to order all 3^n vectors, then run greedy insertion with
     local-search improvement, all under a strict wall-clock deadline.

The priority function itself is deterministic and fast: it returns the precomputed
score for the given vector. The heavy lifting (reference set, feature extraction,
model training) is done once per n and cached.
"""

import time
import itertools
import numpy as np
from functools import lru_cache

# Global cache keyed by n
_cache = {}

def _all_vectors(n):
    """Return list of all vectors in {0,1,2}^n as tuples."""
    return [tuple(v) for v in itertools.product(range(3), repeat=n)]

def _is_cap_set(S, n):
    """Check if set S (list of tuples) is a cap set in {0,1,2}^n."""
    S_set = set(S)
    for i, a in enumerate(S):
        for b in S[i+1:]:
            # For each pair (a, b), check if the third point of the 3-AP is in S
            # 3-AP: a, m, c where m = (a+c)/2, i.e., c = 2m - a
            # In {0,1,2}^n with wrap-around? No, standard: a, b, c are in AP if
            # b - a = c - b, i.e., a + c = 2b (component-wise, no wrap).
            # So for pair (a, c), the midpoint m must satisfy m = (a+c)/2 component-wise.
            # But we need integer coordinates, so a_i + c_i must be even for all i.
            # Alternative: iterate over pairs (a, b) and check if c = 2*b - a is in S.
            c = tuple(2*b[j] - a[j] for j in range(n))
            if c in S_set and c != a and c != b:
                return False
    return True

def _count_conflicts(vec, S, n):
    """Count number of 3-term APs that vec would form with pairs from S."""
    S_set = set(S)
    count = 0
    for a in S:
        # Check if vec, a, and some c form an AP: vec + c = 2*a => c = 2*a - vec
        c = tuple(2*a[j] - vec[j] for j in range(n))
        if c in S_set and c != vec and c != a:
            count += 1
        # Check if a, vec, and some c form an AP: a + c = 2*vec => c = 2*vec - a
        c = tuple(2*vec[j] - a[j] for j in range(n))
        if c in S_set and c != vec and c != a:
            count += 1
        # Check if a, c, vec form an AP: a + vec = 2*c => c = (a+vec)/2
        # Only if a_i + vec_i is even for all i
        valid = True
        c = []
        for j in range(n):
            s = a[j] + vec[j]
            if s % 2 != 0:
                valid = False
                break
            c.append(s // 2)
        if valid:
            c = tuple(c)
            if c in S_set and c != vec and c != a:
                count += 1
    return count

def _greedy_with_priority(vectors, priority_fn, n, max_size=None, deadline=None):
    """Greedy insertion using priority_fn to order vectors."""
    S = []
    S_set = set()
    # Sort by priority descending
    ordered = sorted(vectors, key=priority_fn, reverse=True)
    for v in ordered:
        if deadline is not None and time.time() > deadline:
            break
        if max_size is not None and len(S) >= max_size:
            break
        # Check if adding v keeps S a cap set
        conflict = False
        for a in S:
            # Check 3-AP: v, a, c where c = 2*a - v
            c = tuple(2*a[j] - v[j] for j in range(n))
            if c in S_set and c != v and c != a:
                conflict = True
                break
            # Check 3-AP: a, v, c where c = 2*v - a
            c = tuple(2*v[j] - a[j] for j in range(n))
            if c in S_set and c != v and c != a:
                conflict = True
                break
            # Check 3-AP: a, c, v where c = (a+v)/2
            valid = True
            c_list = []
            for j in range(n):
                s = a[j] + v[j]
                if s % 2 != 0:
                    valid = False
                    break
                c_list.append(s // 2)
            if valid:
                c = tuple(c_list)
                if c in S_set and c != v and c != a:
                    conflict = True
                    break
        if not conflict:
            S.append(v)
            S_set.add(v)
    return S

def _local_search(S, n, max_iters=1000, deadline=None):
    """Improve cap set size by swapping elements."""
    S_set = set(S)
    S_list = list(S)
    best_size = len(S_list)
    all_vecs = _all_vectors(n)
    all_vecs_set = set(all_vecs)
    
    for _ in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i, v in enumerate(S_list):
            # Try removing v and adding some other vector
            S_temp = S_list[:i] + S_list[i+1:]
            S_temp_set = set(S_temp)
            for w in all_vecs:
                if w in S_temp_set:
                    continue
                # Check if S_temp + [w] is a cap set
                conflict = False
                for a in S_temp:
                    c = tuple(2*a[j] - w[j] for j in range(n))
                    if c in S_temp_set and c != w and c != a:
                        conflict = True
                        break
                    c = tuple(2*w[j] - a[j] for j in range(n))
                    if c in S_temp_set and c != w and c != a:
                        conflict = True
                        break
                    valid = True
                    c_list = []
                    for j in range(n):
                        s = a[j] + w[j]
                        if s % 2 != 0:
                            valid = False
                            break
                        c_list.append(s // 2)
                    if valid:
                        c = tuple(c_list)
                        if c in S_temp_set and c != w and c != a:
                            conflict = True
                            break
                if not conflict:
                    S_list[i] = w
                    S_temp_set = set(S_list)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return S_list

def _build_reference_cap_set(n, deadline):
    """Build a reference cap set using greedy + local search."""
    all_vecs = _all_vectors(n)
    
    # Greedy with lexicographic order first
    S = _greedy_with_priority(all_vecs, lambda v: 0.0, n, deadline=deadline)
    
    # Local search
    S = _local_search(S, n, max_iters=500, deadline=deadline)
    
    return S

def _extract_features(vec, n, ref_set):
    """Extract features for a vector given a reference cap set."""
    # Feature 1: number of conflicts with reference set
    conflict_count = _count_conflicts(vec, ref_set, n)
    
    # Feature 2: sum of coordinates
    coord_sum = sum(vec)
    
    # Feature 3: number of 1s
    num_ones = sum(1 for x in vec if x == 1)
    
    # Feature 4: number of 2s
    num_twos = sum(1 for x in vec if x == 2)
    
    # Feature 5: max coordinate
    max_coord = max(vec)
    
    # Feature 6: variance of coordinates
    arr = np.array(vec, dtype=float)
    var = float(np.var(arr))
    
    # Feature 7: distance to centroid of reference set
    if ref_set:
        centroid = np.mean([np.array(v, dtype=float) for v in ref_set], axis=0)
        dist = float(np.linalg.norm(np.array(vec, dtype=float) - centroid))
    else:
        dist = 0.0
    
    # Feature 8: normalized conflict count
    norm_conflict = conflict_count / max(len(ref_set), 1)
    
    # Higher-order features
    conflict_sq = conflict_count ** 2
    coord_sum_sq = coord_sum ** 2
    conflict_coord = conflict_count * coord_sum
    
    return [conflict_count, coord_sum, num_ones, num_twos, max_coord, var, dist,
            norm_conflict, conflict_sq, coord_sum_sq, conflict_coord]

def _train_surrogate(features, labels, deadline):
    """Train a linear model with regularization to predict priority scores."""
    X = np.array(features, dtype=float)
    y = np.array(labels, dtype=float)
    
    # Normalize features
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X_norm = (X - mean) / std
    
    # Ridge regression
    alpha = 1.0
    n_samples, n_features = X_norm.shape
    I = np.eye(n_features)
    I[0, 0] = 0  # Don't regularize bias
    A = X_norm.T @ X_norm + alpha * I
    b = X_norm.T @ y
    try:
        weights = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        weights = np.zeros(n_features)
    
    # Store model parameters
    model = {'weights': weights, 'mean': mean, 'std': std}
    return model

def _predict_priority(model, vec, n, ref_set):
    """Predict priority score for a vector."""
    feats = _extract_features(vec, n, ref_set)
    x = np.array(feats, dtype=float)
    x_norm = (x - model['mean']) / model['std']
    return float(x_norm @ model['weights'])

def _setup(n, deadline):
    """Set up all caches for a given n."""
    if n in _cache:
        return _cache[n]
    
    all_vecs = _all_vectors(n)
    
    # Build reference cap set
    ref_set = _build_reference_cap_set(n, deadline)
    
    # Extract features for all vectors
    features = []
    for v in all_vecs:
        features.append(_extract_features(v, n, ref_set))
    
    # Labels: use the index in the reference set as a proxy for quality
    # Vectors in the reference set get high labels, others get low labels
    labels = []
    ref_set_set = set(ref_set)
    for v in all_vecs:
        if v in ref_set_set:
            labels.append(1.0)
        else:
            labels.append(0.0)
    
    # Train surrogate
    model = _train_surrogate(features, labels, deadline)
    
    # Precompute priorities for all vectors
    priorities = {}
    for v in all_vecs:
        priorities[v] = _predict_priority(model, v, n, ref_set)
    
    _cache[n] = {'all_vecs': all_vecs, 'ref_set': ref_set, 'model': model,
                 'priorities': priorities}
    return _cache[n]

def priority(vector, n):
    """
    Return priority score for a vector in {0,1,2}^n.
    Higher priority = more likely to be inserted first.
    """
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin
    
    # Ensure setup is done
    cache = _setup(n, deadline)
    
    if time.time() > deadline:
        return 0.0
    
    # Return precomputed priority
    return cache['priorities'].get(vector, 0.0)
