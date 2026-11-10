"""Rewritten program: local search to maximize max triangle-area multiplicity."""
import time
import random
from collections import Counter
import numpy as np

_DEADLINE = 80.0  # safety margin before 90s kill
_START = time.time()

def _elapsed():
    return time.time() - _START

def _is_valid(points):
    if len(points) != 30:
        return False
    s = set(points)
    if len(s) != 30:
        return False
    for x, y in points:
        if not (isinstance(x, int) and isinstance(y, int)):
            return False
        if x < -30 or x > 30 or y < -30 or y > 30:
            return False
    return True

def _doubled_areas(points):
    n = len(points)
    areas = []
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            for k in range(j + 1, n):
                xk, yk = points[k]
                d = abs(xi*(yj - yk) + xj*(yk - yi) + xk*(yi - yj))
                if d > 0:
                    areas.append(d)
    return areas

def _score(points):
    if not _is_valid(points):
        return 0
    areas = _doubled_areas(points)
    if not areas:
        return 0
    return max(Counter(areas).values())

def _random_points():
    # Random 30 distinct integer points in [-30,30]^2
    candidates = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    return random.sample(candidates, 30)

def _neighborhood_move(points, rng):
    """Return a new list of 30 points after one random perturbation."""
    pts = list(points)
    idx = rng.randrange(30)
    x, y = pts[idx]
    # Random small move
    dx = rng.randint(-3, 3)
    dy = rng.randint(-3, 3)
    if dx == 0 and dy == 0:
        dx = 1
    nx, ny = x + dx, y + dy
    if -30 <= nx <= 30 and -30 <= ny <= 30:
        # Ensure distinct
        if (nx, ny) not in pts:
            pts[idx] = (nx, ny)
        else:
            # pick a different random point
            cand = rng.choice([(xx, yy) for xx in range(-30, 31) for yy in range(-30, 31) if (xx, yy) not in pts])
            pts[idx] = cand
    return pts

def _swap_move(points, rng):
    """Swap two points with random new positions."""
    pts = list(points)
    i, j = rng.sample(range(30), 2)
    for idx in (i, j):
        x, y = pts[idx]
        dx = rng.randint(-4, 4)
        dy = rng.randint(-4, 4)
        if dx == 0 and dy == 0:
            dx = 1
        nx, ny = x + dx, y + dy
        if -30 <= nx <= 30 and -30 <= ny <= 30:
            if (nx, ny) not in pts:
                pts[idx] = (nx, ny)
    return pts

def _greedy_construct():
    """Build points greedily to maximize area multiplicity."""
    # Start with a structured grid that gives many equal areas
    # A 6x5 grid gives many triangles of the same area
    base = [(x, y) for x in range(6) for y in range(5)]
    # Center it
    base = [(x - 2, y - 2) for x, y in base]
    return base

def _local_search(points, rng, time_limit):
    """Simulated annealing style local search."""
    best = list(points)
    best_score = _score(best)
    current = list(points)
    current_score = best_score
    temp = 5.0
    cool = 0.999
    no_improve = 0
    max_no_improve = 500

    while _elapsed() < time_limit:
        # Choose move type
        if rng.random() < 0.6:
            candidate = _neighborhood_move(current, rng)
        else:
            candidate = _swap_move(current, rng)

        if not _is_valid(candidate):
            continue

        cand_score = _score(candidate)
        diff = cand_score - current_score

        if diff > 0 or rng.random() < np.exp(diff / max(temp, 0.01)):
            current = candidate
            current_score = cand_score
            no_improve = 0
            if current_score > best_score:
                best = list(current)
                best_score = current_score
                temp = 5.0
        else:
            no_improve += 1

        temp *= cool
        if no_improve > max_no_improve:
            no_improve = 0
            # Restart with perturbation
            for _ in range(5):
                current = _neighborhood_move(current, rng)
            current_score = _score(current)

    return best, best_score

def construct():
    start = time.time()
    deadline = start + _DEADLINE - 5  # leave 5s margin

    rng = random.Random(42)

    # Start with greedy structured grid
    best_points = _greedy_construct()
    best_score = _score(best_points)

    # Local search from multiple restarts
    restarts = 0
    while time.time() < deadline:
        if restarts == 0:
            start_pts = best_points
        elif time.time() < deadline - 30:
            # Occasionally restart from random
            if rng.random() < 0.3:
                start_pts = _random_points()
            else:
                start_pts = list(best_points)
                for _ in range(3):
                    start_pts = _neighborhood_move(start_pts, rng)
        else:
            # Near deadline: polish around best
            start_pts = list(best_points)
            for _ in range(2):
                start_pts = _neighborhood_move(start_pts, rng)

        local_best, local_score = _local_search(start_pts, rng, min(deadline, time.time() + 15))

        if local_score > best_score:
            best_points = local_best
            best_score = local_score

        restarts += 1
        # Check time
        if time.time() > deadline:
            break

    # Final polish: try to improve best with more iterations
    if time.time() < deadline:
        final_best, final_score = _local_search(best_points, rng, deadline)
        if final_score > best_score:
            best_points = final_best
            best_score = final_score

    # Ensure valid output
    if not _is_valid(best_points):
        best_points = _greedy_construct()

    return best_points
