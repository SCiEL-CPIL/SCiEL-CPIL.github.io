"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    start = time.time()
    deadline = start + 110.0

    # Pre-compute all 6 rotations for each box
    rots = []
    for w, h, d in boxes:
        seen = set()
        orients = []
        for a, b, c in [(w,h,d),(w,d,h),(h,w,d),(h,d,w),(d,w,h),(d,h,w)]:
            key = (a,b,c)
            if key not in seen:
                seen.add(key)
                orients.append((a,b,c))
        rots.append(orients)

    # Sort boxes by volume descending
    order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)

    def shelf_pack(order, max_time=110.0):
        """
        Multi-layer shelf packing.
        Each layer is at a fixed z. Within a layer, shelves are horizontal strips.
        """
        placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]

        # layers: list of dicts with 'z', 'd', 'shelves'
        # shelves: list of [y, h, x_cursor, items]
        layers = []
        placed_count = 0

        for idx in order:
            if time.time() > max_time:
                break
            w, h, d = boxes[idx]
            placed = False

            # Try existing layers
            for li, layer in enumerate(layers):
                if placed:
                    break
                lz, ld = layer['z'], layer['d']
                for ow, oh, od in rots[idx]:
                    if od > ld:
                        continue
                    # Try existing shelves in this layer
                    for si, shelf in enumerate(layer['shelves']):
                        sy, sh, sx, items = shelf
                        if oh > sh:
                            continue
                        if sx + ow <= W:
                            # Place it
                            placements[idx] = (0, sx, sy, lz, ow, oh, od)
                            shelf[2] = sx + ow
                            shelf[3].append((idx, sx, sy, lz, ow, oh, od))
                            placed = True
                            placed_count += 1
                            break
                    if placed:
                        break
                    # Try creating a new shelf in this layer
                    # Find max y used so far in this layer
                    max_y = 0
                    for s in layer['shelves']:
                        max_y = max(max_y, s[0] + s[1])
                    if max_y + oh <= H:
                        new_shelf = [max_y, oh, ow, [(idx, 0, max_y, lz, ow, oh, od)]]
                        layer['shelves'].append(new_shelf)
                        placements[idx] = (0, 0, max_y, lz, ow, oh, od)
                        placed = True
                        placed_count += 1
                        break

            # Try creating a new layer
            if not placed:
                for ow, oh, od in rots[idx]:
                    if od > D:
                        continue
                    # Find max z used
                    max_z = 0
                    for l in layers:
                        max_z = max(max_z, l['z'] + l['d'])
                    if max_z + od <= D:
                        new_layer = {'z': max_z, 'd': od, 'shelves': [[0, oh, ow, [(idx, 0, 0, max_z, ow, oh, od)]]]}
                        layers.append(new_layer)
                        placements[idx] = (0, 0, 0, max_z, ow, oh, od)
                        placed = True
                        placed_count += 1
                        break

        return placements

    # Try multiple orderings and pick the best
    best_result = None
    best_score = -1

    # Ordering 1: volume descending
    order1 = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)

    # Ordering 2: max dimension descending
    order2 = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)

    # Ordering 3: volume / max_dim (prefer compact boxes first)
    order3 = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2] / max(boxes[i]) if max(boxes[i]) > 0 else 0, reverse=True)

    for order in [order1, order2, order3]:
        if time.time() > deadline:
            break
        result = shelf_pack(order, max_time=deadline - time.time())
        total_vol = sum(p[4]*p[5]*p[6] for p in result if p[0] == 0)
        score = total_vol / (W*H*D) if W*H*D > 0 else 0
        if score > best_score:
            best_score = score
            best_result = result

    if best_result is None:
        best_result = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]

    return best_result
# EVOLVE-BLOCK-END
