import time
import math
import random

def construct():
    start_time = time.time()
    deadline = 145.0

    def elapsed():
        return time.time() - start_time

    def line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        g = math.gcd(A, math.gcd(B, C))
        if g == 0:
            return None
        A //= g
        B //= g
        C //= g
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A, B, C = -A, -B, -C
        return (A, B, C)

    def has_4_collinear(pts):
        n = len(pts)
        if n < 4:
            return False
        seen = set()
        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(pts[i], pts[j])
                if key is None or key in seen:
                    continue
                seen.add(key)
                A, B, C = key
                c = 0
                for k in range(n):
                    x3, y3 = pts[k]
                    if A * x3 + B * y3 + C == 0:
                        c += 1
                        if c >= 4:
                            return True
        return False

    def fast_greedy_independent_set_size(pts):
        n = len(pts)
        conflicts = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                for k in range(j + 1, n):
                    x3, y3 = pts[k]
                    if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                        conflicts[i].append((1 << j) | (1 << k))
                        conflicts[j].append((1 << i) | (1 << k))
                        conflicts[k].append((1 << i) | (1 << j))
        
        order = sorted(range(n), key=lambda i: len(conflicts[i]))
        
        mask = 0
        count = 0
        for idx in order:
            valid = True
            for cm in conflicts[idx]:
                if (mask & cm) == cm:
                    valid = False
                    break
            if valid:
                count += 1
                mask |= (1 << idx)
        return count

    def calculate_score(pts):
        """Exact B&B for largest subset with no three collinear."""
        n = len(pts)
        if n < 3:
            return n

        conflicts = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                for k in range(j + 1, n):
                    x3, y3 = pts[k]
                    if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                        conflicts[i].append((1 << j) | (1 << k))
                        conflicts[j].append((1 << i) | (1 << k))
                        conflicts[k].append((1 << i) | (1 << j))

        order = sorted(range(n), key=lambda i: len(conflicts[i]), reverse=True)

        def greedy_upper_bound(start_step, current_mask):
            remaining = [order[s] for s in range(start_step, n) if not (current_mask >> order[s]) & 1]
            if not remaining:
                return 0
            count = 0
            mask = current_mask
            for idx in remaining:
                valid = True
                for cm in conflicts[idx]:
                    if (mask & cm) == cm:
                        valid = False
                        break
                if valid:
                    count += 1
                    mask |= (1 << idx)
            return count

        best = [fast_greedy_independent_set_size(pts)]

        def bbb(step, current_mask, current_size):
            if elapsed() > deadline - 3:
                return
            if current_size > best[0]:
                best[0] = current_size
            if step == n:
                return

            if current_size + (n - step) <= best[0]:
                return

            ub = greedy_upper_bound(step, current_mask)
            if current_size + ub <= best[0]:
                return

            idx = order[step]

            valid = True
            for cm in conflicts[idx]:
                if (current_mask & cm) == cm:
                    valid = False
                    break
            if valid:
                bbb(step + 1, current_mask | (1 << idx), current_size + 1)
                if elapsed() > deadline - 3:
                    return

            bbb(step + 1, current_mask, current_size)

        bbb(0, 0, 0)
        return best[0]

    def greedy_construct(seed):
        """
        Greedy construction that maximizes perfect triples while
        ensuring no 4 collinear.
        """
        rng = random.Random(seed)
        pts = []

        for _ in range(4):
            while True:
                p = (rng.randint(-80, 80), rng.randint(-80, 80))
                if p not in pts:
                    pts.append(p)
                    break

        while len(pts) < 20:
            candidates = set()
            for _ in range(200):
                candidates.add((rng.randint(-100, 100), rng.randint(-100, 100)))
            for p in pts:
                for dx in range(-8, 9):
                    for dy in range(-8, 9):
                        np_ = (p[0] + dx, p[1] + dy)
                        if -100 <= np_[0] <= 100 and -100 <= np_[1] <= 100:
                            candidates.add(np_)

            best_cand = None
            best_triples = -1
            best_est_score = float('inf')
            best_valid = False

            for cand in candidates:
                if cand in pts:
                    continue
                if not (-100 <= cand[0] <= 100 and -100 <= cand[1] <= 100):
                    continue

                line_counts = {}
                for p1 in pts:
                    key = line_key(cand, p1)
                    if key:
                        line_counts[key] = line_counts.get(key, 0) + 1

                valid = True
                for cnt in line_counts.values():
                    if cnt >= 3:
                        valid = False
                        break
                if not valid:
                    continue

                trial_pts = pts + [cand]
                est_score = fast_greedy_independent_set_size(trial_pts)
                
                new_perfect = 0
                for key, cnt in line_counts.items():
                    if cnt == 2:
                        new_perfect += 1

                if est_score < best_est_score or (est_score == best_est_score and new_perfect > best_triples):
                    best_est_score = est_score
                    best_triples = new_perfect
                    best_cand = cand
                    best_valid = True

            if best_cand is None:
                while True:
                    cand = (rng.randint(-100, 100), rng.randint(-100, 100))
                    if cand in pts:
                        continue
                    line_counts = {}
                    for p1 in pts:
                        key = line_key(cand, p1)
                        if key:
                            line_counts[key] = line_counts.get(key, 0) + 1
                    valid = all(cnt < 3 for cnt in line_counts.values())
                    if valid:
                        best_cand = cand
                        break

            pts.append(best_cand)

        return pts

    def local_search(pts, max_iter=3000, seed=0):
        """
        Local search to minimize score while maintaining no-4-collinear.
        Uses a hybrid move strategy: single-point moves and 2-point swaps.
        """
        rng = random.Random(seed)
        current = [tuple(p) for p in pts]
        current_score = calculate_score(current)
        best = current[:]
        best_score = current_score
        no_improve = 0
        max_no_improve = 100

        for it in range(max_iter):
            if elapsed() > deadline - 10:
                break

            # Choose move type: 70% single, 30% swap
            if rng.random() < 0.7:
                idx = rng.randint(0, 19)
                old_p = current[idx]
                if rng.random() < 0.6:
                    new_p = (old_p[0] + rng.randint(-4, 4), old_p[1] + rng.randint(-4, 4))
                else:
                    new_p = (rng.randint(-100, 100), rng.randint(-100, 100))

                if new_p in current:
                    continue
                if not (-100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100):
                    continue

                trial = current[:]
                trial[idx] = new_p
                if has_4_collinear(trial):
                    continue

                s_est = fast_greedy_independent_set_size(trial)
                if s_est >= best_score:
                    continue

                s = calculate_score(trial)
                if s < best_score:
                    best_score = s
                    best = trial
                    current = trial
                    current_score = s
                    no_improve = 0
                elif s == current_score and rng.random() < 0.15:
                    current = trial
                    current_score = s
                    no_improve += 1
                else:
                    no_improve += 1
            else:
                # 2-point swap
                i1, i2 = rng.sample(range(20), 2)
                p1, p2 = current[i1], current[i2]
                
                # Generate new positions
                if rng.random() < 0.5:
                    # Swap with small perturbation
                    new_p1 = (p1[0] + rng.randint(-3, 3), p1[1] + rng.randint(-3, 3))
                    new_p2 = (p2[0] + rng.randint(-3, 3), p2[1] + rng.randint(-3, 3))
                else:
                    # Random new positions
                    new_p1 = (rng.randint(-100, 100), rng.randint(-100, 100))
                    new_p2 = (rng.randint(-100, 100), rng.randint(-100, 100))

                # Check validity
                if new_p1 in current or new_p2 in current or new_p1 == new_p2:
                    continue
                if not (-100 <= new_p1[0] <= 100 and -100 <= new_p1[1] <= 100):
                    continue
                if not (-100 <= new_p2[0] <= 100 and -100 <= new_p2[1] <= 100):
                    continue

                trial = current[:]
                trial[i1] = new_p1
                trial[i2] = new_p2
                if has_4_collinear(trial):
                    continue

                s_est = fast_greedy_independent_set_size(trial)
                if s_est >= best_score:
                    continue

                s = calculate_score(trial)
                if s < best_score:
                    best_score = s
                    best = trial
                    current = trial
                    current_score = s
                    no_improve = 0
                elif s == current_score and rng.random() < 0.15:
                    current = trial
                    current_score = s
                    no_improve += 1
                else:
                    no_improve += 1

            # Restart if stuck
            if no_improve > max_no_improve:
                # Try a more aggressive restart
                for _ in range(20):
                    idx = rng.randint(0, 19)
                    old_p = current[idx]
                    new_p = (rng.randint(-100, 100), rng.randint(-100, 100))
                    if new_p in current:
                        continue
                    if not (-100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100):
                        continue
                    trial = current[:]
                    trial[idx] = new_p
                    if has_4_collinear(trial):
                        continue
                    s = calculate_score(trial)
                    if s <= current_score:
                        current = trial
                        current_score = s
                        break
                no_improve = 0

        return best, best_score

    best_points = None
    best_val = float('inf')

    # Multiple greedy restarts with different seeds
    for attempt in range(15):
        if elapsed() > deadline - 25:
            break

        cand = greedy_construct(seed=42 + attempt)
        if has_4_collinear(cand):
            continue

        # Local search
        pts, score = local_search(cand, max_iter=1500, seed=100 + attempt)

        if pts is not None and score < best_val:
            best_val = score
            best_points = pts

        # Additional local search with more iterations if time permits
        if elapsed() < deadline - 40 and best_points is not None:
            pts2, score2 = local_search(best_points, max_iter=2000, seed=200 + attempt)
            if pts2 is not None and score2 < best_val:
                best_val = score2
                best_points = pts2

    # Fallback
    if best_points is None:
        while True:
            if elapsed() > deadline - 5:
                break
            pts = []
            while len(pts) < 20:
                p = (random.randint(-100, 100), random.randint(-100, 100))
                if p not in pts:
                    pts.append(p)
            if not has_4_collinear(pts):
                best_points = pts
                break

    if best_points is None:
        # Sparse fallback
        pts = [(i * 11, (i * 13) % 199 - 99) for i in range(20)]
        pts = [(min(max(p[0], -100), 100), min(max(p[1], -100), 100)) for p in pts]
        seen = set()
        final = []
        for p in pts:
            if p not in seen:
                seen.add(p)
                final.append(p)
        while len(final) < 20:
            p = (random.randint(-100, 100), random.randint(-100, 100))
            if p not in final:
                final.append(p)
        best_points = final[:20]

    return [[x, y] for x, y in best_points]


if __name__ == "__main__":
    pts = construct()
    print(pts)
