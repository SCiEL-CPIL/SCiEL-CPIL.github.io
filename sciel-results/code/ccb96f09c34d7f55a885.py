"""Rewritten 3D container loading with stronger heuristics and time guard."""
import time
import itertools

# EVOLVE-BLOCK-START
def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s wall-clock

    W, H, D = container
    container_vol = W * H * D
    if container_vol <= 0:
        return [(-1, 0, 0, 0, w, h, d) for (w, h, d) in boxes]

    n = len(boxes)
    # Box dimensions as list of (w,h,d)
    dims = [(w, h, d) for (w, h, d) in boxes]

    # Precompute all 6 rotations for each box
    def rotations(w, h, d):
        # Generate all unique permutations
        perms = set(itertools.permutations((w, h, d)))
        return list(perms)

    box_rots = [rotations(*dims[i]) for i in range(n)]
    box_vols = [w * h * d for (w, h, d) in dims]

    # Sorting strategies: index -> key
    # 0: volume desc
    # 1: max_dim desc
    # 2: surface area desc
    # 3: volume / max_dim desc
    # 4: min_dim desc
    # 5: volume desc then max_dim desc
    # 6: surface area / max_dim desc

    def sort_key_volume(i):
        return -box_vols[i]

    def sort_key_maxdim(i):
        return -max(dims[i])

    def sort_key_surfarea(i):
        w, h, d = dims[i]
        return -(2 * (w * h + h * d + w * d))

    def sort_key_vol_over_maxdim(i):
        return -(box_vols[i] / max(dims[i]))

    def sort_key_mindim(i):
        return -min(dims[i])

    def sort_key_vol_then_maxdim(i):
        return (-box_vols[i], -max(dims[i]))

    def sort_key_surfarea_over_maxdim(i):
        w, h, d = dims[i]
        sa = 2 * (w * h + h * d + w * d)
        return -(sa / max(dims[i]))

    sort_strategies = [
        sort_key_volume,
        sort_key_maxdim,
        sort_key_surfarea,
        sort_key_vol_over_maxdim,
        sort_key_mindim,
        sort_key_vol_then_maxdim,
        sort_key_surfarea_over_maxdim,
    ]

    def place_box(box_idx, order, placements, used):
        """Try to place box_idx in container using corner-based search.
        placements: list of (x, y, z, w, h, d) for already placed boxes.
        used: boolean array or set of placed indices.
        Returns (x, y, z, rw, rh, rd) or None."""
        w, h, d = dims[box_idx]
        best = None
        best_vol = -1

        # Candidate positions: origin + each corner of existing boxes (6 faces)
        candidates = [(0, 0, 0)]
        for (px, py, pz, pw, ph, pd) in placements:
            # 6 face corners
            candidates.append((px + pw, py, pz))
            candidates.append((px, py + ph, pz))
            candidates.append((px, py, pz + pd))
            candidates.append((px + pw, py + ph, pz))
            candidates.append((px + pw, py, pz + pd))
            candidates.append((px, py + ph, pz + pd))
            candidates.append((px + pw, py + ph, pz + pd))

        # Deduplicate
        seen = set()
        for (cx, cy, cz) in candidates:
            if (cx, cy, cz) in seen:
                continue
            seen.add((cx, cy, cz))

            # Try each rotation
            for (rw, rh, rd) in box_rots[box_idx]:
                if cx + rw > W or cy + rh > H or cz + rd > D:
                    continue
                # Check overlap
                overlap = False
                for (px, py, pz, pw, ph, pd) in placements:
                    if not (cx + rw <= px or px + pw <= cx or
                            cy + rh <= py or py + ph <= cy or
                            cz + rd <= pz or pz + pd <= cz):
                        overlap = True
                        break
                if overlap:
                    continue
                # Valid placement
                if best is None:
                    best = (cx, cy, cz, rw, rh, rd)
                # Prefer placements that are "tighter" - use heuristic: minimize max coord
                # Actually just take first valid? Let's use a simple heuristic: prefer lower z, then y, then x
                # For now, take the first valid placement found
                break
            if best is not None:
                break

        return best

    def solve_with_order(order):
        """Solve using a specific box ordering. Returns (filled_vol, result_list)."""
        placements = []  # (x, y, z, w, h, d)
        result = [None] * n
        filled_vol = 0

        for idx in order:
            if time.time() > deadline:
                break
            pos = place_box(idx, order, placements, None)
            if pos is not None:
                x, y, z, rw, rh, rd = pos
                placements.append((x, y, z, rw, rh, rd))
                result[idx] = (0, x, y, z, rw, rh, rd)
                filled_vol += box_vols[idx]
            else:
                w, h, d = dims[idx]
                result[idx] = (-1, 0, 0, 0, w, h, d)

        # Fill in any None entries (shouldn't happen if we break on deadline)
        for i in range(n):
            if result[i] is None:
                w, h, d = dims[i]
                result[i] = (-1, 0, 0, 0, w, h, d)

        return filled_vol, result

    best_filled = 0
    best_result = [(-1, 0, 0, 0, w, h, d) for (w, h, d) in dims]

    for strat_idx, key_func in enumerate(sort_strategies):
        if time.time() > deadline:
            break
        order = sorted(range(n), key=key_func)
        filled, result = solve_with_order(order)
        if filled > best_filled:
            best_filled = filled
            best_result = result

    return best_result
# EVOLVE-BLOCK-END
