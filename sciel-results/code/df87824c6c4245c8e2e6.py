"""Rewritten program: fast vectorized greedy + local improvement for covering codes."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np

def _all_words(q, n):
    """Return array of shape (q**n, n) with all words in lex order."""
    return np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)

def _word_to_idx(w, q, n):
    """Convert a word (sequence of ints) to its lex index."""
    idx = 0
    for i in range(n):
        idx = idx * q + int(w[i])
    return idx

def _build_spheres(q, n, radius):
    """Precompute neighbors (indices within Hamming radius) for each word."""
    all_w = _all_words(q, n)
    total = q ** n
    neighbors = [None] * total
    
    if radius == 1:
        offsets = []
        for i in range(n):
            for d in range(1, q):
                off = np.zeros(n, dtype=np.int32)
                off[i] = d
                offsets.append(off)
        for idx in range(total):
            w = all_w[idx]
            nbs = [idx]
            for off in offsets:
                nw = (w + off) % q
                nbs.append(_word_to_idx(nw, q, n))
            neighbors[idx] = np.array(nbs, dtype=np.int32)
    elif radius == 2:
        offsets = []
        for i in range(n):
            for d in range(1, q):
                off = np.zeros(n, dtype=np.int32)
                off[i] = d
                offsets.append(off)
        for i in range(n):
            for j in range(i + 1, n):
                for d1 in range(1, q):
                    for d2 in range(1, q):
                        off = np.zeros(n, dtype=np.int32)
                        off[i] = d1
                        off[j] = d2
                        offsets.append(off)
        for idx in range(total):
            w = all_w[idx]
            nbs = [idx]
            for off in offsets:
                nw = (w + off) % q
                nbs.append(_word_to_idx(nw, q, n))
            neighbors[idx] = np.array(nbs, dtype=np.int32)
    else:
        # Generic: brute force
        for idx in range(total):
            w = all_w[idx]
            nbs = []
            for cand in range(total):
                if np.count_nonzero(all_w[cand] != w) <= radius:
                    nbs.append(cand)
            neighbors[idx] = np.array(nbs, dtype=np.int32)
    return neighbors

def _is_valid_cover(q, n, radius, neighbors, centers_idx):
    """Verify that centers cover all words."""
    total = q ** n
    covered = np.zeros(total, dtype=bool)
    for c in centers_idx:
        covered[neighbors[c]] = True
    return covered.all()

def _greedy_cover(q, n, radius, neighbors, seed=0, time_limit=3.0):
    """Greedy set-cover with random tie-breaking."""
    rng = random.Random(seed)
    start = time.time()
    deadline = start + time_limit
    total = q ** n
    uncovered = np.ones(total, dtype=bool)
    centers = []
    order = list(range(total))
    rng.shuffle(order)
    
    while uncovered.any() and time.time() < deadline:
        best_idx = -1
        best_count = -1
        unc_total = uncovered.sum()
        for idx in order:
            nbs = neighbors[idx]
            cnt = uncovered[nbs].sum()
            if cnt > best_count:
                best_count = cnt
                best_idx = idx
            if cnt == unc_total:
                break
        if best_idx < 0 or best_count <= 0:
            break
        centers.append(best_idx)
        uncovered[neighbors[best_idx]] = False
    return centers

def _local_improve(q, n, radius, neighbors, centers_idx, time_limit=10.0, seed=42):
    """Local search: iteratively try to remove or swap codewords."""
    rng = random.Random(seed)
    start = time.time()
    deadline = start + time_limit
    total = q ** n
    centers = list(centers_idx)
    
    def build_coverage():
        cov = np.zeros(total, dtype=np.int32)
        for c in centers:
            cov[neighbors[c]] += 1
        return cov
    
    cov = build_coverage()
    
    while time.time() < deadline:
        improved = False
        # Phase 1: try to remove any center whose removal keeps coverage
        for i in range(len(centers)):
            if time.time() > deadline:
                break
            c = centers[i]
            if cov[neighbors[c]].min() >= 2:
                cov[neighbors[c]] -= 1
                centers.pop(i)
                improved = True
                break
        if improved:
            continue
        # Phase 2: try swaps - remove one center, find best replacement
        tried = 0
        max_tries = 50
        while tried < max_tries and time.time() < deadline:
            tried += 1
            if len(centers) < 2:
                break
            i = rng.randrange(len(centers))
            c = centers[i]
            cov[neighbors[c]] -= 1
            unc = np.where(cov == 0)[0]
            if len(unc) == 0:
                centers.pop(i)
                improved = True
                break
            # Find candidate that covers most uncovered
            best_new = -1
            best_cnt = 0
            # Sample candidates from uncovered words and random
            candidates = list(set(list(unc[:30]) + [rng.randrange(total) for _ in range(30)]))
            for cand in candidates:
                if cand in centers:
                    continue
                in_sphere = neighbors[cand]
                cnt = np.isin(unc, in_sphere).sum()
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_new = cand
            if best_new >= 0:
                centers[i] = best_new
                cov[neighbors[best_new]] += 1
                improved = True
                break
            else:
                cov[neighbors[c]] += 1
        if not improved:
            break
    
    return centers

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140
    
    neighbors = _build_spheres(q, n, radius)
    total = q ** n
    
    best_centers = None
    best_count = float('inf')
    
    # Multiple greedy runs
    seed = 0
    while time.time() < deadline - 30:
        run_time = min(2.0, max(0.3, deadline - 30 - time.time()))
        centers_idx = _greedy_cover(q, n, radius, neighbors, seed=seed, time_limit=run_time)
        if centers_idx is not None and len(centers_idx) < best_count:
            best_count = len(centers_idx)
            best_centers = list(centers_idx)
        seed += 1
        if best_count <= 72:
            break
    
    # Local improvement
    if best_centers is not None and time.time() < deadline - 10:
        improve_time = min(40.0, deadline - 10 - time.time())
        improved = _local_improve(q, n, radius, neighbors, best_centers, time_limit=improve_time, seed=1)
        if len(improved) < best_count:
            best_count = len(improved)
            best_centers = improved
    
    # More local improvement rounds
    if best_centers is not None and time.time() < deadline - 5:
        for rnd in range(5):
            if time.time() > deadline - 5:
                break
            improve_time = min(8.0, deadline - 5 - time.time())
            improved = _local_improve(q, n, radius, neighbors, best_centers, time_limit=improve_time, seed=100 + rnd)
            if len(improved) < best_count:
                best_count = len(improved)
                best_centers = improved
    
    if best_centers is None:
        return [list(w) for w in itertools.product(range(q), repeat=n)]
    
    all_w = _all_words(q, n)
    return [list(all_w[i]) for i in best_centers]
# EVOLVE-BLOCK-END
