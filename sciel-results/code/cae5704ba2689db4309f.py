import time
import itertools

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0

    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    n = len(boxes)
    container_vol = W * H * D

    # Precompute volumes and max dims
    box_vols = [w * h * d for w, h, d in boxes]
    box_max_dims = [max(w, h, d) for w, h, d in boxes]
    box_sums = [w + h + d for w, h, d in boxes]
    box_aspect = [max(w, h, d) / min(w, h, d) if min(w, h, d) > 0 else 0 for w, h, d in boxes]

    # Generate multiple sort orders
    def make_order(key_func, reverse=True):
        return sorted(range(n), key=key_func, reverse=reverse)

    orders = [
        make_order(lambda i: box_vols[i]),
        make_order(lambda i: box_max_dims[i]),
        make_order(lambda i: box_sums[i]),
        make_order(lambda i: box_aspect[i]),
        # Mixed: volume descending, but break ties by max dim descending
        make_order(lambda i: (box_vols[i], box_max_dims[i])),
    ]

    def get_orientations(w, h, d):
        """Unique orientations, sorted by z-height ascending for layer packing."""
        orients = list(set(itertools.permutations([w, h, d])))
        orients.sort(key=lambda o: (o[2], o[1], o[0]))
        return orients

    def _try_pack(order):
        """Try to pack boxes in given order using best-fit decreasing with candidate pruning."""
        placed = []  # List of (x, y, z, w, h, d)
        results = [None] * n
        candidate_set = set()
        candidates = [(0, 0, 0)]
        candidate_set.add((0, 0, 0))

        MAX_CANDIDATES = 800

        def fits(x, y, z, w, h, d):
            if x + w > W or y + h > H or z + d > D:
                return False
            for px, py, pz, pw, ph, pd in placed:
                if not (x + w <= px or px + pw <= x or
                        y + h <= py or py + ph <= y or
                        z + d <= pz or pz + pd <= z):
                    return False
            return True

        def prune_candidates():
            """Remove dominated candidates: if A has x<=x', y<=y', z<=z' and at least one strict, keep A."""
            nonlocal candidates, candidate_set
            if len(candidates) <= MAX_CANDIDATES:
                return
            candidates.sort(key=lambda c: (c[2], c[1], c[0]))
            kept = []
            for c in candidates:
                dominated = False
                for kc in kept:
                    if kc[0] <= c[0] and kc[1] <= c[1] and kc[2] <= c[2] and (kc[0], kc[1], kc[2]) != c:
                        dominated = True
                        break
                if not dominated:
                    kept.append(c)
                if len(kept) >= MAX_CANDIDATES:
                    break
            candidates = kept
            candidate_set = set(candidates)

        def cost_function(x, y, z, w, h, d):
            """Cost: prioritize low z, then compact in y and x."""
            return z * (H + 1) * (W + 1) + y * (W + 1) + x

        for idx in order:
            if time.time() > deadline:
                break

            w, h, d = boxes[idx]
            best_placement = None
            best_cost = float('inf')

            orientations = get_orientations(w, h, d)

            for cx, cy, cz in candidates[:MAX_CANDIDATES]:
                for ow, oh, od in orientations:
                    if not fits(cx, cy, cz, ow, oh, od):
                        continue
                    cost = cost_function(cx, cy, cz, ow, oh, od)
                    if cost < best_cost:
                        best_cost = cost
                        best_placement = (cx, cy, cz, ow, oh, od)

            if best_placement:
                bx, by, bz, bw, bh, bd = best_placement
                placed.append((bx, by, bz, bw, bh, bd))
                results[idx] = (0, bx, by, bz, bw, bh, bd)

                # Generate new candidates from faces of the placed box
                new_cands = [
                    (bx + bw, by, bz),
                    (bx, by + bh, bz),
                    (bx, by, bz + bd),
                    (bx + bw, by + bh, bz),
                    (bx + bw, by, bz + bd),
                    (bx, by + bh, bz + bd),
                    (bx + bw, by + bh, bz + bd),
                ]
                for nc in new_cands:
                    if 0 <= nc[0] <= W and 0 <= nc[1] <= H and 0 <= nc[2] <= D:
                        if nc not in candidate_set:
                            candidates.append(nc)
                            candidate_set.add(nc)

                prune_candidates()
            else:
                results[idx] = (-1, 0, 0, 0, w, h, d)

        # Fill remaining
        for i in range(n):
            if results[i] is None:
                w, h, d = boxes[i]
                results[i] = (-1, 0, 0, 0, w, h, d)

        # Compute filled volume
        filled = 0.0
        for r in results:
            if r[0] == 0:
                filled += r[4] * r[5] * r[6]
        return results, filled

    best_result = None
    best_filled = 0.0

    for order in orders:
        if time.time() > deadline:
            break
        result, filled = _try_pack(order)
        if filled > best_filled:
            best_filled = filled
            best_result = result

    if best_result is None:
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    return best_result
