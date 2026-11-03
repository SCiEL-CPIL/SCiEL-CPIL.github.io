"""Covering code K_3(6,1) - spectral seeding + vectorized greedy + local search."""
import time
import random
import numpy as np
import itertools


def _generic_greedy(q, n, radius):
    """Simple greedy set cover for the general (q, n, radius) case."""
    total = q ** n
    powers = [q ** i for i in range(n)]

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx += w[i] * powers[i]
        return idx

    # Build balls for all words
    balls = []
    for w_idx in range(total):
        w = idx_to_word(w_idx)
        ball = {w_idx}
        for deltas in itertools.product(range(q), repeat=n):
            d = sum(1 for a, b in zip(w, deltas) if a != b)
            if d <= radius:
                ball.add(word_to_idx(deltas))
        balls.append(ball)

    uncovered = set(range(total))
    selected = []
    while uncovered:
        best_cw, best_cnt = -1, -1
        for cw in range(total):
            if cw in selected:
                continue
            cnt = len(balls[cw] & uncovered)
            if cnt > best_cnt:
                best_cnt = cnt
                best_cw = cw
        if best_cw < 0 or best_cnt == 0:
            break
        selected.append(best_cw)
        uncovered -= balls[best_cw]
    return [idx_to_word(cw) for cw in selected]


def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        return _generic_greedy(q, n, radius)

    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s limit

    total = q ** n
    powers = [q ** i for i in range(n)]

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx += w[i] * powers[i]
        return idx

    # --- Precompute balls as numpy arrays ---
    ball_size = 1 + n * (q - 1)
    balls_flat = np.zeros((total, ball_size), dtype=np.int32)
    for w_idx in range(total):
        w = idx_to_word(w_idx)
        ball = [w_idx]
        for pos in range(n):
            for delta in range(1, q):
                new_w = w[:]
                new_w[pos] = (w[pos] + delta) % q
                ball.append(word_to_idx(new_w))
        balls_flat[w_idx, :] = ball

    # --- Spectral seeding: compute eigenvectors of distance-2 graph Laplacian ---
    # Build adjacency matrix of distance-2 graph
    dist2_adj = np.zeros((total, total), dtype=np.float64)
    for i in range(total):
        wi = idx_to_word(i)
        for j in range(i + 1, total):
            wj = idx_to_word(j)
            d = 0
            for k in range(n):
                if wi[k] != wj[k]:
                    d += 1
                    if d > 2:
                        break
            if d == 2:
                dist2_adj[i, j] = 1.0
                dist2_adj[j, i] = 1.0

    # Compute Laplacian
    degree = dist2_adj.sum(axis=1)
    L = degree[:, None] + degree[None, :] - 2 * dist2_adj

    # Compute eigenvectors (only need a few)
    # Use numpy's eigh for symmetric matrices
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Select eigenvectors corresponding to small non-zero eigenvalues
        # These represent "smooth" modes on the graph
        # We want vertices that are "central" in these modes
        # Use the first few non-trivial eigenvectors
        num_modes = min(10, total)
        # Skip the first eigenvector (constant, eigenvalue 0)
        selected_eigvecs = eigvecs[:, 1:1+num_modes]
        
        # Compute spectral centrality: weighted sum of absolute values
        weights = 1.0 / (1.0 + eigvals[1:1+num_modes])
        spectral_score = np.abs(selected_eigvecs) @ weights
    except Exception:
        # Fallback: use degree-based centrality
        spectral_score = degree.copy()

    # Normalize spectral score
    spectral_score = (spectral_score - spectral_score.min()) / (spectral_score.max() - spectral_score.min() + 1e-10)

    def make_spectral_seed(seed, num_words=80):
        """Select vertices with high spectral centrality, with random perturbation."""
        rng = random.Random(seed)
        # Perturb scores slightly
        perturbed = spectral_score + rng.random() * 0.1
        top_idx = np.argsort(perturbed)[::-1][:num_words]
        return [int(i) for i in top_idx]

    def make_independent_set(seed):
        """Independent set via randomized greedy (low-degree-first)."""
        rng = random.Random(seed)
        remaining = set(range(total))
        indep = []
        while remaining:
            k = min(60, len(remaining))
            cands = rng.sample(list(remaining), k)
            best_v, best_deg = None, float('inf')
            for v in cands:
                deg = int(dist2_adj[v].sum())  # approximate degree in remaining
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
            if best_v is None:
                break
            indep.append(best_v)
            remaining.discard(best_v)
            # Remove neighbors
            neighbors = np.where(dist2_adj[best_v] > 0)[0]
            remaining -= set(neighbors.tolist())
        return indep

    # --- Greedy set cover (vectorized) ---
    def make_greedy(seed=None):
        rng = random.Random(seed) if seed is not None else random.Random()
        cov_count = np.zeros(total, dtype=np.int32)
        uncovered = np.ones(total, dtype=bool)
        remaining = total
        selected = []
        while remaining > 0:
            if time.time() > deadline:
                break
            scores = np.zeros(total, dtype=np.int32)
            for cw in range(total):
                scores[cw] = np.sum(uncovered[balls_flat[cw]])
            top_k = min(5, total)
            top_idx = np.argsort(scores)[::-1][:top_k]
            if rng.random() < 0.7:
                best_cw = int(top_idx[0])
            else:
                best_cw = int(top_idx[rng.randint(0, top_k - 1)])
            if scores[best_cw] == 0:
                break
            selected.append(best_cw)
            ball = balls_flat[best_cw]
            cov_count[ball] += 1
            uncovered[ball] = False
            remaining -= int(scores[best_cw])
        return selected, cov_count

    # --- Repair: add codewords until every word is covered ---
    def repair(codeword_set, cov_count):
        uncovered_mask = cov_count < 1
        if not np.any(uncovered_mask):
            return codeword_set, cov_count
        uncovered_idx = set(np.where(uncovered_mask)[0].tolist())
        while uncovered_idx:
            if time.time() > deadline:
                break
            best_cw, best_cnt = -1, 0
            for cw in range(total):
                if cw in codeword_set:
                    continue
                cnt = len(set(balls_flat[cw].tolist()) & uncovered_idx)
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_cw = cw
            if best_cw < 0 or best_cnt == 0:
                break
            codeword_set.add(best_cw)
            cov_count[balls_flat[best_cw]] += 1
            for w in balls_flat[best_cw]:
                if cov_count[w] == 1:
                    uncovered_idx.discard(int(w))
        return codeword_set, cov_count

    def is_redundant(cw_idx, cov_count):
        return bool(np.all(cov_count[balls_flat[cw_idx]] >= 2))

    def try_remove_redundant(codeword_set, cov_count):
        changed = True
        removed = 0
        while changed:
            changed = False
            for cw in list(codeword_set):
                if time.time() > deadline:
                    break
                if is_redundant(cw, cov_count):
                    cov_count[balls_flat[cw]] -= 1
                    codeword_set.discard(cw)
                    changed = True
                    removed += 1
        return codeword_set, cov_count, removed

    def try_swap(old_cw, new_cw, codeword_set, cov_count):
        if new_cw in codeword_set:
            return False, cov_count
        ball_old = balls_flat[old_cw]
        ball_new = balls_flat[new_cw]
        cov_count[ball_old] -= 1
        cov_count[ball_new] += 1
        if np.min(cov_count) >= 1:
            codeword_set.discard(old_cw)
            codeword_set.add(new_cw)
            return True, cov_count
        else:
            cov_count[ball_old] += 1
            cov_count[ball_new] -= 1
            return False, cov_count

    def try_double_swap(c1, c2, c3, codeword_set, cov_count):
        """Remove c1 and c2, add c3. Valid iff after the move every word is covered."""
        if c1 == c2 or c3 in (c1, c2) or c3 in codeword_set:
            return False, cov_count
        b1 = balls_flat[c1]
        b2 = balls_flat[c2]
        b3 = balls_flat[c3]
        cov_count[b1] -= 1
        cov_count[b2] -= 1
        cov_count[b3] += 1
        if np.min(cov_count) >= 1:
            codeword_set.discard(c1)
            codeword_set.discard(c2)
            codeword_set.add(c3)
            return True, cov_count
        else:
            cov_count[b1] += 1
            cov_count[b2] += 1
            cov_count[b3] -= 1
            return False, cov_count

    def local_optimize(codeword_set, cov_count, do_double=True):
        codeword_set, cov_count, _ = try_remove_redundant(codeword_set, cov_count)
        size = len(codeword_set)
        for it in range(400):
            if time.time() > deadline:
                break
            if size == 0:
                break
            cw_list = list(codeword_set)
            if do_double and len(cw_list) >= 2 and random.random() < 0.5:
                # 2-for-1: remove two, add one
                c1, c2 = random.sample(cw_list, 2)
                c3 = random.randint(0, total - 1)
                success, cov_count = try_double_swap(c1, c2, c3, codeword_set, cov_count)
                if success:
                    codeword_set, cov_count, _ = try_remove_redundant(codeword_set, cov_count)
                    size = len(codeword_set)
            else:
                old_cw = random.choice(cw_list)
                new_cw = random.randint(0, total - 1)
                success, cov_count = try_swap(old_cw, new_cw, codeword_set, cov_count)
                if success:
                    codeword_set, cov_count, _ = try_remove_redundant(codeword_set, cov_count)
                    size = len(codeword_set)
        return codeword_set, cov_count

    # --- Tetracode direct-sum construction (known size 81) ---
    def tetracode_construction():
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        s = set()
        for h in tetracode:
            for tail in itertools.product(range(q), repeat=n - 4):
                s.add(word_to_idx(h + list(tail)))
        return s

    # --- Main: combine seeds, repair, optimize, keep best ---
    best_set = None
    best_size = total + 1

    # Seed 0: tetracode (guaranteed valid, size 81)
    tset = tetracode_construction()
    tcov = np.zeros(total, dtype=np.int32)
    for cw in tset:
        tcov[balls_flat[cw]] += 1
    tset, tcov = local_optimize(set(tset), tcov)
    if len(tset) < best_size:
        best_size = len(tset)
        best_set = set(tset)

    # Seeds 1..: spectral seeds
    for r in range(1, 6):
        if time.time() > deadline - 25:
            break
        sset = make_spectral_seed(r * 999 + 7, num_words=80)
        sset = set(sset)
        scov = np.zeros(total, dtype=np.int32)
        for cw in sset:
            scov[balls_flat[cw]] += 1
        sset, scov = repair(sset, scov)
        sset, scov = local_optimize(sset, scov)
        if len(sset) < best_size:
            best_size = len(sset)
            best_set = set(sset)

    # Seeds: independent sets (repair to valid)
    for r in range(1, 4):
        if time.time() > deadline - 20:
            break
        indep = make_independent_set(r * 999 + 7)
        if not indep:
            continue
        sset = set(indep)
        scov = np.zeros(total, dtype=np.int32)
        for cw in sset:
            scov[balls_flat[cw]] += 1
        sset, scov = repair(sset, scov)
        sset, scov = local_optimize(sset, scov)
        if len(sset) < best_size:
            best_size = len(sset)
            best_set = set(sset)

    # Seeds: perturbed greedy
    for r in range(1, 5):
        if time.time() > deadline - 10:
            break
        sel, cov = make_greedy(seed=r * 12345 + 67890)
        if not sel:
            continue
        sset = set(sel)
        scov = np.zeros(total, dtype=np.int32)
        for cw in sset:
            scov[balls_flat[cw]] += 1
        sset, scov = repair(sset, scov)
        sset, scov = local_optimize(sset, scov)
        if len(sset) < best_size:
            best_size = len(sset)
            best_set = set(sset)

    # Final polish on the best
    if best_set is not None and time.time() < deadline - 5:
        bcov = np.zeros(total, dtype=np.int32)
        for cw in best_set:
            bcov[balls_flat[cw]] += 1
        best_set, _ = local_optimize(set(best_set), bcov)

    if best_set is None:
        best_set = tetracode_construction()

    result = [idx_to_word(cw) for cw in best_set]

    # --- Verify validity; fall back to tetracode if broken ---
    verify_cov = np.zeros(total, dtype=np.int32)
    for cw in result:
        verify_cov[balls_flat[word_to_idx(cw)]] += 1
    if np.min(verify_cov) < 1:
        tset = tetracode_construction()
        result = [idx_to_word(cw) for cw in tset]

    return result
