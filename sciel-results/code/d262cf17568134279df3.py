"""Edge-colourings of K16 in which every K4 sees five colours."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

# Precompute edge index to (i,j) mapping
_edge_to_vertices = []
for _i in range(16):
    for _j in range(_i + 1, 16):
        _edge_to_vertices.append((_i, _j))

# Precompute all K4s and their edge indices
_all_k4s = []
for _quad in combinations(range(16), 4):
    _edges = []
    for _a, _b in combinations(_quad, 2):
        if _a > _b:
            _a, _b = _b, _a
        # Compute edge index
        _idx = _a * 15 - _a * (_a - 1) // 2 + (_b - _a - 1)
        _edges.append(_idx)
    _all_k4s.append(_edges)

# Precompute: for each edge index, which K4 indices contain it
_edge_to_k4s = [[] for _ in range(120)]
for _k4_idx, _edges in enumerate(_all_k4s):
    for _e in _edges:
        _edge_to_k4s[_e].append(_k4_idx)


def _count_used(flat_arr):
    return len(set(flat_arr))


def _check_valid_affected(flat_arr, changed_edges):
    """Check only K4s affected by changed edges."""
    affected = set()
    for eidx in changed_edges:
        for k4_idx in _edge_to_k4s[eidx]:
            affected.add(k4_idx)
    for k4_idx in affected:
        colors = set()
        for e in _all_k4s[k4_idx]:
            colors.add(flat_arr[e])
        if len(colors) < 5:
            return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # Initialize with seed coloring: (i+j) % 17
    flat = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    best_flat = flat[:]
    best_colors = _count_used(flat)
    current_colors = best_colors

    rng = random.Random(12345)

    while time.time() < deadline:
        used_colors = sorted(set(flat))
        improved = False

        # Try to eliminate each color by recoloring its edges
        for target_color in used_colors:
            if time.time() > deadline:
                break

            edges_of_color = [i for i in range(120) if flat[i] == target_color]
            if not edges_of_color:
                continue

            for eidx in edges_of_color:
                if time.time() > deadline:
                    break

                i, j = _edge_to_vertices[eidx]

                for new_color in used_colors:
                    if new_color == target_color:
                        continue

                    old_val = flat[eidx]
                    flat[eidx] = new_color

                    if _check_valid_affected(flat, [eidx]):
                        if target_color not in flat:
                            current_colors -= 1
                            best_flat = flat[:]
                            best_colors = current_colors
                            improved = True
                            break
                        else:
                            flat[eidx] = old_val
                    else:
                        flat[eidx] = old_val

                if improved:
                    break

            if improved:
                break

        if not improved:
            # Random perturbations to escape local optima
            for _ in range(500):
                if time.time() > deadline:
                    break

                eidx = rng.randint(0, 119)
                old_val = flat[eidx]
                new_color = rng.randint(0, max(used_colors) + 2)

                flat[eidx] = new_color
                if _check_valid_affected(flat, [eidx]):
                    new_count = _count_used(flat)
                    if new_count < current_colors:
                        current_colors = new_count
                        best_flat = flat[:]
                        best_colors = current_colors
                        break
                else:
                    flat[eidx] = old_val

            if time.time() > deadline:
                break

            # If stuck for a while, perturb the best solution
            if current_colors >= best_colors:
                flat = best_flat[:]
                for _ in range(20):
                    if time.time() > deadline:
                        break
                    eidx = rng.randint(0, 119)
                    old_val = flat[eidx]
                    new_color = rng.randint(0, best_colors + 3)
                    flat[eidx] = new_color
                    if _check_valid_affected(flat, [eidx]):
                        nc = _count_used(flat)
                        if nc < current_colors:
                            current_colors = nc
                            best_flat = flat[:]
                            best_colors = current_colors
                    else:
                        flat[eidx] = old_val

    return best_flat

# EVOLVE-BLOCK-END
