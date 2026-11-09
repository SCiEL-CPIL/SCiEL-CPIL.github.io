"""Improved program for minimum maximal no-isosceles set."""
import itertools
import time
import random

N = 16
DEADLINE_MARGIN = 30  # seconds to leave before the 330s kill

# Precompute all squared distances
# dist[i][j] = squared distance between point i and point j
_ALL_POINTS = list(itertools.product(range(N), repeat=2))
_NUM_POINTS = N * N
_point_index = {p: i for i, p in enumerate(_ALL_POINTS)}
_dist = [[0] * _NUM_POINTS for _ in range(_NUM_POINTS)]
for i in range(_NUM_POINTS):
    for j in range(i + 1, _NUM_POINTS):
        dx = _ALL_POINTS[i][0] - _ALL_POINTS[j][0]
        dy = _ALL_POINTS[i][1] - _ALL_POINTS[j][1]
        d = dx * dx + dy * dy
        _dist[i][j] = d
        _dist[j][i] = d


def _can_add(p_idx, chosen_indices):
    """Check if adding p_idx creates any isosceles triangle with pairs in chosen_indices."""
    n = len(chosen_indices)
    for i in range(n):
        a = chosen_indices[i]
        da = _dist[p_idx][a]
        for j in range(i + 1, n):
            b = chosen_indices[j]
            db = _dist[p_idx][b]
            ab = _dist[a][b]
            if da == db or da == ab or db == ab:
                return False
    return True


def _is_maximal(chosen_indices):
    """Check if the set is maximal: no point can be added."""
    chosen_set = set(chosen_indices)
    for p_idx in range(_NUM_POINTS):
        if p_idx in chosen_set:
            continue
        if _can_add(p_idx, chosen_indices):
            return False
    return True


def _make_maximal(chosen_indices, deadline):
    """Ensure maximality by adding any point that can be added."""
    chosen_set = set(chosen_indices)
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        for p_idx in range(_NUM_POINTS):
            if time.time() > deadline:
                break
            if p_idx in chosen_set:
                continue
            if _can_add(p_idx, chosen_indices):
                chosen_indices.append(p_idx)
                chosen_set.add(p_idx)
                changed = True
    return chosen_indices


def _greedy_construction(order, deadline):
    """Greedy construction following a given order of points."""
    chosen = []
    for p_idx in order:
        if time.time() > deadline:
            break
        if _can_add(p_idx, chosen):
            chosen.append(p_idx)
    return chosen


def _local_search_remove(chosen_indices, deadline):
    """Try to remove points and see if the set remains valid (no isosceles)."""
    # We want to minimize size, so try removing each point
    # After removal, the set must still be maximal (or we'll repair it)
    # Actually, we just need it to remain a valid no-isosceles set
    # Then we'll make it maximal again
    for i in range(len(chosen_indices)):
        if time.time() > deadline:
            break
        # Try removing point i
        new_chosen = chosen_indices[:i] + chosen_indices[i+1:]
        # Check if still valid (no isosceles)
        valid = True
        n = len(new_chosen)
        for a_i in range(n):
            a = new_chosen[a_i]
            for b_i in range(a_i + 1, n):
                b = new_chosen[b_i]
                for c_i in range(b_i + 1, n):
                    c = new_chosen[c_i]
                    da = _dist[a][b]
                    db = _dist[a][c]
                    dc = _dist[b][c]
                    if da == db or da == dc or db == dc:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            # Check if it's still maximal
            if _is_maximal(new_chosen):
                chosen_indices = new_chosen
            else:
                # Make it maximal, then check size
                made_max = _make_maximal(new_chosen, deadline)
                if len(made_max) < len(chosen_indices):
                    chosen_indices = made_max
    return chosen_indices


def construct():
    start_time = time.time()
    deadline = start_time + (330 - DEADLINE_MARGIN)
    
    best_chosen = None
    best_size = _NUM_POINTS + 1
    
    # First, do a row-major greedy to get a baseline
    row_major_order = list(range(_NUM_POINTS))
    chosen = _greedy_construction(row_major_order, deadline)
    chosen = _make_maximal(chosen, deadline)
    if len(chosen) < best_size:
        best_size = len(chosen)
        best_chosen = chosen[:]
    
    # Try multiple random orders
    random.seed(42)
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Create a random order with some structure
        # Mix of random and sorted-by-row, sorted-by-col, sorted-by-diagonal
        strategy = attempts % 4
        if strategy == 0:
            order = list(range(_NUM_POINTS))
            random.shuffle(order)
        elif strategy == 1:
            # Sort by row
            order = sorted(range(_NUM_POINTS), key=lambda i: (_ALL_POINTS[i][1], _ALL_POINTS[i][0]))
            random.shuffle(order)
        elif strategy == 2:
            # Sort by column
            order = sorted(range(_NUM_POINTS), key=lambda i: (_ALL_POINTS[i][0], _ALL_POINTS[i][1]))
            random.shuffle(order)
        else:
            # Sort by distance from center
            cx, cy = (N - 1) / 2.0, (N - 1) / 2.0
            order = sorted(range(_NUM_POINTS), key=lambda i: (_ALL_POINTS[i][0] - cx)**2 + (_ALL_POINTS[i][1] - cy)**2)
            random.shuffle(order)
        
        chosen = _greedy_construction(order, deadline)
        chosen = _make_maximal(chosen, deadline)
        if len(chosen) < best_size:
            best_size = len(chosen)
            best_chosen = chosen[:]
        
        # Try local search removal on the best
        if time.time() < deadline:
            improved = _local_search_remove(best_chosen[:], deadline)
            if len(improved) < best_size:
                best_size = len(improved)
                best_chosen = improved[:]
    
    return [_ALL_POINTS[i] for i in best_chosen]
