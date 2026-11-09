import time
import random
import numpy as np

def construct():
    d = 7
    n = 2 ** d
    deadline = time.time() + 112.0

    # Build edge list
    edges = []
    edge_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_idx[(x, y)] = idx
                edge_idx[(y, x)] = idx

    m = len(edges)  # 448

    # Build face-edge incidence
    face_edges = []
    edge_faces = [[] for _ in range(m)]

    for b in range(d):
        for c in range(b + 1, d):
            for a in range(n):
                if (a & (1 << b)) or (a & (1 << c)):
                    continue
                v0 = a
                v1 = a ^ (1 << b)
                v2 = a ^ (1 << c)
                v3 = a ^ (1 << b) ^ (1 << c)
                e0 = edge_idx[(min(v0, v1), max(v0, v1))]
                e1 = edge_idx[(min(v0, v2), max(v0, v2))]
                e2 = edge_idx[(min(v1, v3), max(v1, v3))]
                e3 = edge_idx[(min(v2, v3), max(v2, v3))]
                fid = len(face_edges)
                face_edges.append((e0, e1, e2, e3))
                for ei in (e0, e1, e2, e3):
                    edge_faces[ei].append(fid)

    num_faces = len(face_edges)  # 288

    # Precompute edge_faces as numpy arrays for fast access
    edge_faces_arr = [np.array(f, dtype=np.int32) for f in edge_faces]

    def greedy_random(seed):
        rng = random.Random(seed)
        order = list(range(m))
        rng.shuffle(order)
        selected = np.zeros(m, dtype=bool)
        face_count = np.zeros(num_faces, dtype=np.int32)
        for ei in order:
            fids = edge_faces_arr[ei]
            if np.all(face_count[fids] < 3):
                selected[ei] = True
                face_count[fids] += 1
        return selected, face_count

    def complete(selected, face_count):
        """Add all edges that can be safely added (iterate until fixed point)."""
        changed = True
        while changed:
            changed = False
            for ei in range(m):
                if selected[ei]:
                    continue
                fids = edge_faces_arr[ei]
                if np.all(face_count[fids] < 3):
                    selected[ei] = True
                    face_count[fids] += 1
                    changed = True
        return selected, face_count

    def count_sel(selected):
        return int(np.sum(selected))

    # Multi-start greedy
    best_selected = None
    best_face_count = None
    best_count = 0

    for seed in range(20):
        sel, fc = greedy_random(seed)
        sel, fc = complete(sel, fc)
        cnt = count_sel(sel)
        if cnt > best_count:
            best_count = cnt
            best_selected = sel
            best_face_count = fc

    current_selected = best_selected.copy()
    current_face_count = best_face_count.copy()
    current_count = best_count

    # Simulated annealing with various move types
    T = 20.0
    T_min = 0.05
    alpha = 0.997

    rng = random.Random(123)
    stuck_count = 0
    no_improve = 0

    while time.time() < deadline - 8.0 and T > T_min:
        r = rng.random()

        if r < 0.35:
            # 1-elimination + refill
            sel_list = np.where(current_selected)[0]
            if len(sel_list) == 0:
                continue
            ri = int(rng.choice(sel_list))

            old_selected = current_selected.copy()
            old_face_count = current_face_count.copy()
            old_count = current_count

            current_selected[ri] = False
            current_face_count[edge_faces_arr[ri]] -= 1
            current_selected, current_face_count = complete(current_selected, current_face_count)
            new_count = count_sel(current_selected)

            delta = new_count - old_count
            if delta > 0 or (T > 0.01 and rng.random() < np.exp(delta / T)):
                current_count = new_count
                if new_count > best_count:
                    best_count = new_count
                    best_selected = current_selected.copy()
                    best_face_count = current_face_count.copy()
                    stuck_count = 0
                    no_improve = 0
                else:
                    stuck_count += 1
                    no_improve += 1
            else:
                current_selected = old_selected
                current_face_count = old_face_count
                current_count = old_count
                stuck_count += 1
                no_improve += 1

        elif r < 0.65:
            # 2-elimination + refill
            sel_list = np.where(current_selected)[0]
            if len(sel_list) < 2:
                continue
            r1, r2 = rng.sample(sel_list.tolist(), 2)

            old_selected = current_selected.copy()
            old_face_count = current_face_count.copy()
            old_count = current_count

            current_selected[r1] = False
            current_face_count[edge_faces_arr[r1]] -= 1
            current_selected[r2] = False
            current_face_count[edge_faces_arr[r2]] -= 1
            current_selected, current_face_count = complete(current_selected, current_face_count)
            new_count = count_sel(current_selected)

            delta = new_count - old_count
            if delta > 0 or (T > 0.01 and rng.random() < np.exp(delta / T)):
                current_count = new_count
                if new_count > best_count:
                    best_count = new_count
                    best_selected = current_selected.copy()
                    best_face_count = current_face_count.copy()
                    stuck_count = 0
                    no_improve = 0
                else:
                    stuck_count += 1
                    no_improve += 1
            else:
                current_selected = old_selected
                current_face_count = old_face_count
                current_count = old_count
                stuck_count += 1
                no_improve += 1

        elif r < 0.85:
            # 3-5 elimination + refill
            sel_list = np.where(current_selected)[0]
            if len(sel_list) < 3:
                continue
            k = rng.randint(3, 5)
            if len(sel_list) < k:
                k = len(sel_list)
            to_remove = rng.sample(sel_list.tolist(), k)

            old_selected = current_selected.copy()
            old_face_count = current_face_count.copy()
            old_count = current_count

            for ri in to_remove:
                current_selected[ri] = False
                current_face_count[edge_faces_arr[ri]] -= 1
            current_selected, current_face_count = complete(current_selected, current_face_count)
            new_count = count_sel(current_selected)

            delta = new_count - old_count
            if delta > 0 or (T > 0.01 and rng.random() < np.exp(delta / T)):
                current_count = new_count
                if new_count > best_count:
                    best_count = new_count
                    best_selected = current_selected.copy()
                    best_face_count = current_face_count.copy()
                    stuck_count = 0
                    no_improve = 0
                else:
                    stuck_count += 1
                    no_improve += 1
            else:
                current_selected = old_selected
                current_face_count = old_face_count
                current_count = old_count
                stuck_count += 1
                no_improve += 1

        else:
            # 1-swap: remove one selected, try to add one unselected
            sel_list = np.where(current_selected)[0].tolist()
            unsel_list = np.where(~current_selected)[0].tolist()
            if not sel_list or not unsel_list:
                continue
            ri = rng.choice(sel_list)
            ai = rng.choice(unsel_list)

            old_selected = current_selected.copy()
            old_face_count = current_face_count.copy()
            old_count = current_count

            current_selected[ri] = False
            current_face_count[edge_faces_arr[ri]] -= 1

            fids_ai = edge_faces_arr[ai]
            if np.all(current_face_count[fids_ai] < 3):
                current_selected[ai] = True
                current_face_count[fids_ai] += 1

            current_selected, current_face_count = complete(current_selected, current_face_count)
            new_count = count_sel(current_selected)

            delta = new_count - old_count
            if delta > 0 or (T > 0.01 and rng.random() < np.exp(delta / T)):
                current_count = new_count
                if new_count > best_count:
                    best_count = new_count
                    best_selected = current_selected.copy()
                    best_face_count = current_face_count.copy()
                    stuck_count = 0
                    no_improve = 0
                else:
                    stuck_count += 1
                    no_improve += 1
            else:
                current_selected = old_selected
                current_face_count = old_face_count
                current_count = old_count
                stuck_count += 1
                no_improve += 1

        T *= alpha

        # If stuck for a while, restart from best with perturbation
        if stuck_count > 80:
            current_selected = best_selected.copy()
            current_face_count = best_face_count.copy()
            current_count = best_count
            # Perturb: remove k random edges from best
            k = rng.randint(2, 6)
            sel_list = np.where(current_selected)[0].tolist()
            to_remove = rng.sample(sel_list, min(k, len(sel_list)))
            for ri in to_remove:
                current_selected[ri] = False
                current_face_count[edge_faces_arr[ri]] -= 1
            current_selected, current_face_count = complete(current_selected, current_face_count)
            current_count = count_sel(current_selected)
            stuck_count = 0

    # Final polish: try all single-edge swaps to improve
    if time.time() < deadline - 3.0:
        for _ in range(3):
            improved = False
            sel_list = np.where(best_selected)[0].tolist()
            unsel_list = np.where(~best_selected)[0].tolist()
            for ai in unsel_list:
                fids_ai = edge_faces_arr[ai]
                if np.any(best_face_count[fids_ai] >= 3):
                    continue
                # Try swapping: remove each selected edge that shares a face with ai
                for ri in sel_list:
                    fids_ri = edge_faces_arr[ri]
                    common = np.intersect1d(fids_ai, fids_ri)
                    if len(common) == 0:
                        continue
                    # Check if removing ri makes ai addable
                    test_fc = best_face_count.copy()
                    test_fc[fids_ri] -= 1
                    if np.all(test_fc[fids_ai] < 3):
                        # This swap is feasible
                        new_sel = best_selected.copy()
                        new_sel[ri] = False
                        new_sel[ai] = True
                        new_fc = test_fc
                        new_fc[fids_ai] += 1
                        # Complete
                        new_sel, new_fc = complete(new_sel, new_fc)
                        new_cnt = count_sel(new_sel)
                        if new_cnt > best_count:
                            best_count = new_cnt
                            best_selected = new_sel
                            best_face_count = new_fc
                            improved = True
                            break
                if improved:
                    sel_list = np.where(best_selected)[0].tolist()
                    unsel_list = np.where(~best_selected)[0].tolist()
                    break
            if not improved:
                break

    out = []
    for ei in range(m):
        if best_selected[ei]:
            x, y = edges[ei]
            out.append([x, y])

    return out
