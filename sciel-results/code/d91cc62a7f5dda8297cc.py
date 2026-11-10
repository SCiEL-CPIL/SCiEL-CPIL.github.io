import time
import itertools
import numpy as np

DEADLINE = 82.0  # safety margin before 90s kill

# Precompute all C(30,3) triples (i,j,k) with i<j<k
_TRIPLES = np.array(list(itertools.combinations(range(30), 3)), dtype=np.int64)
_N_TRIPLES = _TRIPLES.shape[0]

def _doubled_areas(pts):
    """pts: (30,2) int array. Returns (N,2) array of [area, count] for each positive doubled area,
    where area is the doubled area value (2*|det|/2 = |det|) and count is multiplicity."""
    # pts shape (30,2); index into triples
    a = pts[_TRIPLES[:, 0]]
    b = pts[_TRIPLES[:, 1]]
    c = pts[_TRIPLES[:, 2]]
    # doubled area = |det| = |(b-a) x (c-a)|
    # det = (b1-a1)*(c2-a2) - (b2-a2)*(c1-a1)
    dx1 = b[:, 0] - a[:, 0]
    dy1 = b[:, 1] - a[:, 1]
    dx2 = c[:, 0] - a[:, 0]
    dy2 = c[:, 1] - a[:, 1]
    det = dx1 * dy2 - dy1 * dx2
    adet = np.abs(det)
    adet = adet[adet > 0]
    if adet.size == 0:
        return np.array([0, 0])
    vals, cnts = np.unique(adet, return_counts=True)
    # find max count
    maxc = cnts.max()
    return np.array([vals[cnts == maxc].min(), int(maxc)], dtype=np.int64)

def _score(pts):
    """Return largest multiplicity of a single positive doubled area."""
    r = _doubled_areas(pts)
    return int(r[1])

def _generate_grid():
    """Generate a reasonable initial 30-point configuration in [-30,30]^2."""
    # Use a small grid pattern to get many equal areas
    # 6x5 grid from 0..5 x 0..4
    pts = np.array([(x, y) for x in range(6) for y in range(5)], dtype=np.int64)
    return pts

def construct():
    start = time.time()
    best_pts = _generate_grid()
    best_score = _score(best_pts)

    # Local search: move one point to a random nearby location
    rng = np.random.default_rng(42)
    coords = best_pts.copy()

    # Precompute all possible positions in [-30,30]^2
    xs = np.arange(-30, 31)
    ys = np.arange(-30, 31)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing='ij')
    all_positions = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # Neighborhood: for each point, candidate positions within radius
    radius = 4
    candidate_offsets = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx != 0 or dy != 0:
                candidate_offsets.append((dx, dy))
    candidate_offsets = np.array(candidate_offsets, dtype=np.int64)

    iter_count = 0
    while time.time() < DEADLINE:
        # Pick a random point to move
        idx = rng.integers(0, 30)
        cur = coords[idx]

        # Generate candidate positions
        candidates = cur + candidate_offsets
        # Filter to valid range
        mask = (candidates[:, 0] >= -30) & (candidates[:, 0] <= 30) & \
               (candidates[:, 1] >= -30) & (candidates[:, 1] <= 30)
        candidates = candidates[mask]

        if candidates.shape[0] == 0:
            continue

        # Try each candidate, keep best
        best_candidate = None
        best_candidate_score = best_score

        for cand in candidates:
            # Check not colliding with another point
            # coords is (30,2); check if cand equals any other point
            diff = coords - cand
            # exclude self
            diff[idx] = [999, 999]
            if np.any(np.all(diff == 0, axis=1)):
                continue

            coords[idx] = cand
            s = _score(coords)
            if s > best_candidate_score:
                best_candidate_score = s
                best_candidate = cand.copy()
            coords[idx] = cur

        if best_candidate is not None:
            coords[idx] = best_candidate
            best_score = best_candidate_score
            best_pts = coords.copy()

        iter_count += 1
        # If we've been stuck for a while, try a random restart
        if iter_count % 500 == 0 and time.time() > start + 30:
            # Random perturbation
            for _ in range(5):
                idx2 = rng.integers(0, 30)
                cand = rng.integers(-30, 31, size=2)
                diff = coords - cand
                diff[idx2] = [999, 999]
                if not np.any(np.all(diff == 0, axis=1)):
                    coords[idx2] = cand

    # Return as list of tuples
    return [(int(x), int(y)) for x, y in best_pts]

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_score(np.array(pts))}")
    print(f"Points: {len(pts)}")
