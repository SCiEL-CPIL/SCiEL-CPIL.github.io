"""3D Container Loading: multi-strategy skyline + local-search polish."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def _unique_orients(w, h, d):
    seen = set()
    out = []
    for p in ((w, h, d), (w, d, h), (h, w, d), (h, d, w), (d, w, h), (d, h, w)):
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def load(boxes, container):
    t0 = time.time()
    deadline = t0 + 115.0

    W, H, D = container
    n = len(boxes)
    if W <= 0 or H <= 0 or D <= 0 or n == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    cvol = float(W) * float(H) * float(D)

    # ---- Precompute orientations and volumes ----
    orient_list = []
    vols = []
    for (w, h, d) in boxes:
        ors = _unique_orients(w, h, d)
        ors = [o for o in ors if o[0] <= W and o[1] <= H and o[2] <= D]
        if not ors:
            ors = [(w, h, d)]
        # prefer flatter (smaller max dim) orientations first
        ors.sort(key=lambda o: (max(o), o[0], o[1], o[2]))
        orient_list.append(ors)
        vols.append(float(w) * float(h) * float(d))

    idx = list(range(n))
    idx.sort(key=lambda i: vols[i], reverse=True)

    # ---- Fast skyline placement over an occupancy grid ----
    def place(order, cand_order):
        occ = np.zeros((W + 1, H + 1, D + 1), dtype=np.uint8)
        res = [None] * n
        placed = []
        tot = 0.0
        for i in order:
            if time.time() > deadline:
                break
            olist = orient_list[i]
            # candidate coordinates
            cx = [0]
            cy = [0]
            cz = [0]
            for (px, py, pz, pw, ph, pd) in placed:
                cx.append(px)
                cx.append(px + pw)
                cy.append(py)
                cy.append(py + ph)
                cz.append(pz)
                cz.append(pz + pd)
            cx = sorted(set(cx))
            cy = sorted(set(cy))
            cz = sorted(set(cz))
            if cand_order == 0:
                order_c = [(z, y, x) for z in cz for y in cy for x in cx]
                order_c.sort(key=lambda t: (t[0], t[1], t[2]))
            else:
                order_c = [(x, y, z) for x in cx for y in cy for z in cz]
                order_c.sort(key=lambda t: (t[0], t[1], t[2]))

            found = False
            for (cx_, cy_, cz_) in order_c:
                for (bw, bh, bd) in olist:
                    if cx_ + bw > W or cy_ + bh > H or cz_ + bd > D:
                        continue
                    if np.any(occ[cx_:cx_ + bw, cy_:cy_ + bh, cz_:cz_ + bd]):
                        continue
                    occ[cx_:cx_ + bw, cy_:cy_ + bh, cz_:cz_ + bd] = 1
                    placed.append((cx_, cy_, cz_, bw, bh, bd))
                    res[i] = (0, cx_, cy_, cz_, bw, bh, bd)
                    tot += vols[i]
                    found = True
                    break
                if found:
                    break
            if not found:
                w, h, d = boxes[i]
                res[i] = (-1, 0, 0, 0, w, h, d)
        for i in range(n):
            if res[i] is None:
                w, h, d = boxes[i]
                res[i] = (-1, 0, 0, 0, w, h, d)
        return res, tot

    # ---- Local search: try re-orienting / re-placing individual boxes ----
    def polish(res, tot):
        # rebuild occupancy and placement info from res
        occ = np.zeros((W + 1, H + 1, D + 1), dtype=np.uint8)
        info = {}
        for i in range(n):
            b, x, y, z, bw, bh, bd = res[i]
            if b == 0:
                occ[x:x + bw, y:y + bh, z:z + bd] = 1
                info[i] = (x, y, z, bw, bh, bd)
        cur = tot
        improved = True
        guard = 0
        while improved and time.time() < deadline - 3.0:
            improved = False
            guard += 1
            if guard > 40:
                break
            for i in range(n):
                if i not in info:
                    continue
                # remove box i
                x, y, z, bw, bh, bd = info[i]
                occ[x:x + bw, y:y + bh, z:z + bd] = 0
                # candidate coords from all other boxes
                cx = [0]
                cy = [0]
                cz = [0]
                for j, (px, py, pz, pw, ph, pd) in info.items():
                    if j == i:
                        continue
                    cx.append(px)
                    cx.append(px + pw)
                    cy.append(py)
                    cy.append(py + ph)
                    cz.append(pz)
                    cz.append(pz + pd)
                cx = sorted(set(cx))
                cy = sorted(set(cy))
                cz = sorted(set(cz))
                cand = [(z, y, x) for z in cz for y in cy for x in cx]
                cand.sort(key=lambda t: (t[0], t[1], t[2]))
                best = None
                for (cx_, cy_, cz_) in cand:
                    for (ow, oh, od) in orient_list[i]:
                        if cx_ + ow > W or cy_ + oh > H or cz_ + od > D:
                            continue
                        if np.any(occ[cx_:cx_ + ow, cy_:cy_ + oh, cz_:cz_ + od]):
                            continue
                        # score: prefer lower, then earlier x/y
                        score = (cz_, cy_, cx_)
                        if best is None or score < best[0]:
                            best = (score, cx_, cy_, cz_, ow, oh, od)
                    if best is not None:
                        break
                if best is not None:
                    _, cx_, cy_, cz_, ow, oh, od = best
                    occ[cx_:cx_ + ow, cy_:cy_ + oh, cz_:cz_ + od] = 1
                    info[i] = (cx_, cy_, cz_, ow, oh, od)
                    res[i] = (0, cx_, cy_, cz_, ow, oh, od)
                    improved = True
        return res, cur

    # ---- Run strategies ----
    strategies = [
        (idx, 0),
        (idx, 1),
    ]
    # extra orderings: by max dim, by volume*min dim
    idx2 = list(range(n))
    idx2.sort(key=lambda i: max(boxes[i]), reverse=True)
    strategies.append((idx2, 0))
    idx3 = list(range(n))
    idx3.sort(key=lambda i: vols[i] * min(boxes[i]), reverse=True)
    strategies.append((idx3, 0))

    best_res = None
    best_tot = -1.0
    for order, co in strategies:
        if time.time() > deadline - 5.0:
            break
        res, tot = place(order, co)
        if tot > best_tot:
            best_tot = tot
            best_res = res

    # polish the best
    if best_res is not None:
        best_res, best_tot = polish(best_res, best_tot)

    if best_res is None:
        best_res = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    return best_res
# EVOLVE-BLOCK-END
