"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, b[0], b[1], b[2]) for b in boxes]

    t_start = time.time()
    deadline = t_start + 135.0

    # Precompute all distinct rotations for each box
    rots = []
    for (w, h, d) in boxes:
        seen = set()
        lst = []
        for a in (w, h, d):
            for b in (w, h, d):
                for c in (w, h, d):
                    if a * b * c == w * h * d and (a, b, c) not in seen:
                        seen.add((a, b, c))
                        lst.append((a, b, c))
        rots.append(lst)

    def run_order(order):
        """Place boxes in given order using bottom-left-farthest empty-space fit.
        Returns (placements, total_vol)."""
        placed = []  # list of (x, y, z, w, h, d)
        total_vol = 0
        placements = [None] * n
        for idx in order:
            best = None  # (score, x, y, z, w, h, d)
            for (w, h, d) in rots[idx]:
                if w > W or h > H or d > D:
                    continue
                # candidate positions: origin + top of each placed box
                cands = [(0, 0, 0)]
                for (px, py, pz, pw, ph, pd) in placed:
                    if px + w <= W and py + h <= H:
                        cands.append((px, py, pz + pd))
                    if py + h <= H and pz + d <= D:
                        cands.append((px + pw, py, pz))
                    if pz + d <= D and px + w <= W:
                        cands.append((px, py + ph, pz))
                for (x, y, z) in cands:
                    if x + w > W or y + h > H or z + d > D:
                        continue
                    # check overlap with all placed
                    ok = True
                    for (px, py, pz, pw, ph, pd) in placed:
                        if not (x + w <= px or px + pw <= x or
                                y + h <= py or py + ph <= y or
                                z + d <= pz or pz + pd <= z):
                            ok = False
                            break
                    if ok:
                        # bottom-left-farthest: minimize x, then y, then z
                        score = (x, y, z)
                        if best is None or score < best[0]:
                            best = (score, x, y, z, w, h, d)
            if best is not None:
                _, x, y, z, w, h, d = best
                placed.append((x, y, z, w, h, d))
                placements[idx] = (0, x, y, z, w, h, d)
                total_vol += w * h * d
            else:
                placements[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
        return placements, total_vol

    # Initial order: sort by volume descending (large boxes first)
    vol = [b[0] * b[1] * b[2] for b in boxes]
    order0 = sorted(range(n), key=lambda i: -vol[i])
    best_pl, best_vol = run_order(order0)

    # Simulated annealing over permutations
    order = list(order0)
    cur_vol = best_vol
    T = 12.0
    T_min = 0.05
    n_iter = 0
    while time.time() < deadline:
        i = random.randrange(n)
        j = random.randrange(n)
        if i == j:
            continue
        order[i], order[j] = order[j], order[i]
        pl, v = run_order(order)
        n_iter += 1
        if v >= cur_vol:
            cur_vol = v
            if v > best_vol:
                best_vol = v
                best_pl = pl
        else:
            # accept with probability
            if random.random() < 0.5:  # cheap acceptance, keeps diversity
                pass
            else:
                order[i], order[j] = order[j], order[i]
        if n_iter % 200 == 0:
            T *= 0.9
            if T < T_min:
                T = T_min
    return best_pl
# EVOLVE-BLOCK-END
