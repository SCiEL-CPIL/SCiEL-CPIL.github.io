"""No-isosceles set: greedy construction with efficient equidistance checking."""
import time
import random

N = 64


def construct():
    deadline = time.time() + 320.0  # safety margin before 330s kill

    # Precompute squared distances between all grid points
    all_pts = [(x, y) for x in range(N) for y in range(N)]
    npts = len(all_pts)
    dist2 = [[0] * npts for _ in range(npts)]
    for i in range(npts):
        xi, yi = all_pts[i]
        for j in range(i + 1, npts):
            xj, yj = all_pts[j]
            dx = xi - xj
            dy = yi - yj
            d = dx * dx + dy * dy
            dist2[i][j] = d
            dist2[j][i] = d

    def can_add(pidx, point_set, dist_sets):
        """Check whether adding pidx creates an isosceles triangle."""
        for i, p in enumerate(point_set):
            d = dist2[pidx][p]
            if d in dist_sets[i]:
                return False
        return True

    def add_point(pidx, point_set, dist_sets):
        for i, p in enumerate(point_set):
            dist_sets[i].add(dist2[pidx][p])
        point_set.append(pidx)
        dist_sets.append(set())

    # Greedy with randomized ordering, keep best solution
    best_count = 0
    best_set = []

    for trial in range(20000):
        if time.time() > deadline:
            break
        # Shuffle indices for diversity
        order = list(range(npts))
        random.shuffle(order)
        # Greedy pass
        point_set = []
        dist_sets = []
        for pidx in order:
            if can_add(pidx, point_set, dist_sets):
                add_point(pidx, point_set, dist_sets)
        if len(point_set) > best_count:
            best_count = len(point_set)
            best_set = list(point_set)
        if best_count >= 200:  # unlikely but stop early if great
            break

    return [all_pts[i] for i in best_set]
