# EVOLVE-BLOCK-START
import numpy as np
import time


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place n=10 squares with pairwise disjoint
    interiors inside the unit square [0,1]^2 so as to MAXIMIZE the sum of their
    side lengths.

    Uses differential evolution with a large overlap penalty, followed by
    local polishing (grow-to-max + coordinate ascent).
    """
    n = 10
    deadline = time.time() + 580  # safety margin before 630s kill

    def _overlap_penalty(centers, sides, tol=1e-7):
        pen = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2.0 - tol
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                ox = max(0.0, t - dx)
                oy = max(0.0, t - dy)
                if ox > 0.0 and oy > 0.0:
                    pen += ox * oy
        for i in range(n):
            h = sides[i] / 2.0
            if centers[i, 0] - h < -tol:
                pen += (-tol - (centers[i, 0] - h)) ** 2
            if centers[i, 0] + h > 1.0 + tol:
                pen += ((centers[i, 0] + h) - (1.0 + tol)) ** 2
            if centers[i, 1] - h < -tol:
                pen += (-tol - (centers[i, 1] - h)) ** 2
            if centers[i, 1] + h > 1.0 + tol:
                pen += ((centers[i, 1] + h) - (1.0 + tol)) ** 2
        return pen

    def _is_valid(centers, sides, tol=1e-7):
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
        for i in range(n):
            h = sides[i] / 2.0
            if centers[i, 0] - h < -tol or centers[i, 0] + h > 1.0 + tol:
                return False
            if centers[i, 1] - h < -tol or centers[i, 1] + h > 1.0 + tol:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2.0 - tol
                if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                    return False
        return True

    def _objective(centers, sides):
        s = np.sum(sides)
        pen = _overlap_penalty(centers, sides)
        return s - 1000.0 * pen

    def _clamp(centers, sides):
        sides = np.clip(sides, 1e-6, 1.0)
        centers = centers.copy()
        for i in range(n):
            h = sides[i] / 2.0
            centers[i, 0] = np.clip(centers[i, 0], h, 1.0 - h)
            centers[i, 1] = np.clip(centers[i, 1], h, 1.0 - h)
        return centers, sides

    def _grow_square_max(i, centers, sides, tol=1e-7):
        cx, cy = centers[i]
        s_max = min(2.0 * cx, 2.0 * (1.0 - cx), 2.0 * cy, 2.0 * (1.0 - cy))
        s_max = max(s_max, 1e-6)
        for j in range(n):
            if j == i:
                continue
            ox, oy = centers[j]
            oj = sides[j]
            dx = abs(cx - ox)
            dy = abs(cy - oy)
            s_limit = max(2.0 * dx - oj + 2.0 * tol, 2.0 * dy - oj + 2.0 * tol)
            if s_limit < s_max:
                s_max = max(s_limit, 1e-6)
        return s_max

    def _local_polish(centers, sides, max_iters=300):
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            for i in range(n):
                new_s = _grow_square_max(i, centers, sides)
                if new_s > sides[i] + 1e-10:
                    sides[i] = new_s
                    improved = True
            if not improved:
                break
        return centers, sides

    def _coordinate_ascent(centers, sides, steps=80):
        best_c = centers.copy()
        best_s = sides.copy()
        best_obj = _objective(best_c, best_s)
        lr = 0.05
        for _ in range(steps):
            if time.time() > deadline:
                break
            for i in range(n):
                for coord in range(2):
                    for delta in [lr, -lr]:
                        trial_c = best_c.copy()
                        trial_c[i, coord] += delta
                        trial_c, trial_s = _clamp(trial_c, best_s)
                        obj = _objective(trial_c, trial_s)
                        if obj > best_obj:
                            best_c = trial_c
                            best_s = trial_s
                            best_obj = obj
            for i in range(n):
                new_s = _grow_square_max(i, best_c, best_s)
                if new_s > best_s[i] + 1e-10:
                    trial_s = best_s.copy()
                    trial_s[i] = new_s
                    obj = _objective(best_c, trial_s)
                    if obj > best_obj:
                        best_s = trial_s
                        best_obj = obj
            lr *= 0.95
        return best_c, best_s

    def _de(pop_size=35, max_gen=250, F=0.8, CR=0.9):
        dim = 3 * n
        pop = np.zeros((pop_size, dim))
        k = int(np.ceil(np.sqrt(n)))
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        for i in range(n):
            pop[0, 3*i] = (cols[i] + 0.5) * side
            pop[0, 3*i+1] = (rows[i] + 0.5) * side
            pop[0, 3*i+2] = side
        pop[1, 0] = 0.25; pop[1, 1] = 0.25; pop[1, 2] = 0.5
        positions = [(0.75,0.25,0.25),(0.25,0.75,0.25),(0.75,0.75,0.25),
                     (0.5,0.1,0.2),(0.9,0.5,0.2),(0.5,0.9,0.2),
                     (0.1,0.5,0.2),(0.9,0.9,0.2),(0.1,0.9,0.2)]
        for i in range(1, n):
            x, y, s = positions[i-1]
            pop[1, 3*i] = x; pop[1, 3*i+1] = y; pop[1, 3*i+2] = s
        for p in range(2, pop_size):
            for i in range(n):
                s = np.random.uniform(0.1, 0.5)
                x = np.random.uniform(s/2, 1-s/2)
                y = np.random.uniform(s/2, 1-s/2)
                pop[p, 3*i] = x; pop[p, 3*i+1] = y; pop[p, 3*i+2] = s

        objs = np.zeros(pop_size)
        for p in range(pop_size):
            c = np.column_stack([pop[p, 0::3], pop[p, 1::3]])
            s = pop[p, 2::3].copy()
            c, s = _clamp(c, s)
            objs[p] = _objective(c, s)

        best_p = np.argmax(objs)
        best_c = np.column_stack([pop[best_p, 0::3], pop[best_p, 1::3]]).copy()
        best_s = pop[best_p, 2::3].copy()
        best_c, best_s = _clamp(best_c, best_s)
        best_obj = objs[best_p]

        for gen in range(max_gen):
            if time.time() > deadline:
                break
            for p in range(pop_size):
                if time.time() > deadline:
                    break
                a, b, c = np.random.choice(pop_size, 3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])
                trial = pop[p].copy()
                for d in range(dim):
                    if np.random.random() < CR or d == np.random.randint(dim):
                        trial[d] = mutant[d]
                tc = np.column_stack([trial[0::3], trial[1::3]])
                ts = trial[2::3].copy()
                tc, ts = _clamp(tc, ts)
                t_obj = _objective(tc, ts)
                if t_obj >= objs[p]:
                    pop[p, 0::3] = tc[:, 0]
                    pop[p, 1::3] = tc[:, 1]
                    pop[p, 2::3] = ts
                    objs[p] = t_obj
                    if t_obj > best_obj:
                        best_obj = t_obj
                        best_c = tc.copy()
                        best_s = ts.copy()

        return best_c, best_s

    best_centers = None
    best_sides = None
    best_score = -np.inf

    for restart in range(4):
        if time.time() > deadline:
            break
        c, s = _de(pop_size=30, max_gen=200)
        c, s = _local_polish(c, s)
        c, s = _coordinate_ascent(c, s)
        c, s = _local_polish(c, s)

        if _is_valid(c, s):
            score = np.sum(s)
            if score > best_score:
                best_score = score
                best_centers = c.copy()
                best_sides = s.copy()

    if best_centers is None:
        k = int(np.ceil(np.sqrt(n)))
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(n, side, dtype=np.float64)

    return best_centers, best_sides


# EVOLVE-BLOCK-END
