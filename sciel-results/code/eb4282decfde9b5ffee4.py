import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    d = 7
    n = 2 ** d
    deadline = time.time() + 110.0

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

    m = len(edges)

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

    num_faces = len(face_edges)
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
        """Vectorized completion: add all edges that can be safely added."""
        changed = True
        while changed:
            changed = False
            # Find all unselected edges whose faces are all < 3
            candidates = []
            for ei in range(m):
                if selected[ei]:
                    continue
                fids = edge_faces_arr[ei]
                if np.all(face_count[fids] < 3):
                    candidates.append(ei)
            if not candidates:
                break
            # Add all candidates at once (they don't conflict since we check all at start)
            # But adding one might make another invalid, so we add in batches
            # For safety, add one at a time but in a single pass
            for ei in candidates:
                fids = edge_faces_arr[ei]
                if np.all(face_count[fids] < 3):
                    selected[ei] = True
                    face_count[fids] += 1
                    changed = True
        return selected, face_count

    def count_sel(selected):
        return int(np.sum(selected))

    best_selected, best_face_count = greedy_random(0)
    best_selected, best_face_count = complete(best_selected, best_face_count)
    best_count = count_sel(best_selected)

    current_selected, current_face_count = best_selected.copy(), best_face_count.copy()
    current_count = best_count

    T = 15.0
    T_min = 0.1
    alpha = 0.995

    rng = random.Random(42)
    iter_count = 0
    stuck_count = 0

    while time.time() < deadline - 5.0 and T > T_min:
        iter_count += 1

        # Choose move type
        r = rng.random()
        if r < 0.5:
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
                else:
                    stuck_count += 1
            else:
                current_selected = old_selected
                current_face_count = old_face_count
                current_count = old_count
                stuck_count += 1

        elif r < 0.8:
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
                else:
                    stuck_count += 1
            else:
                current_selected = old_selected
                current_face_count = old_face_count
                current_count = old_count
                stuck_count += 1

        else:
            # Multi-elimination (remove 3-5 edges) + refill
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
                else:
                    stuck_count += 1
            else:
                current_selected = old_selected
                current_face_count = old_face_count
                current_count = old_count
                stuck_count += 1

        T *= alpha

        # Periodic 2-swap polish when stuck
        if stuck_count > 50 and time.time() < deadline - 15.0:
            sel_list = np.where(current_selected)[0].tolist()
            unsel_list = np.where(~current_selected)[0].tolist()
            if sel_list and unsel_list:
                best_gain = 0
                best_swap = None
                # Try a limited number of 2-swaps
                attempts = min(20, len(sel_list))
                for _ in range(attempts):
                    ri = rng.choice(sel_list)
                    ai = rng.choice(unsel_list)
                    if ai == ri:
                        continue

                    # Check if removing ri and adding ai is feasible
                    # Remove ri
                    old_fc = current_face_count.copy()
                    old_sel = current_selected.copy()

                    current_selected[ri] = False
                    current_face_count[edge_faces_arr[ri]] -= 1

                    # Try to add ai
                    fids_ai = edge_faces_arr[ai]
                    if np.all(current_face_count[fids_ai] < 3):
                        current_selected[ai] = True
                        current_face_count[fids_ai] += 1
                        gain = 0  # 1 out, 1 in
                        if best_gain < gain:
                            best_gain = gain
                            best_swap = (ri, ai)
                    else:
                        # Try to add ai after refill
                        pass

                    # Restore
                    current_selected = old_sel
                    current_face_count = old_fc

                if best_swap is not None:
                    ri, ai = best_swap
                    current_selected[ri] = False
                    current_face_count[edge_faces_arr[ri]] -= 1
                    current_selected[ai] = True
                    current_face_count[edge_faces_arr[ai]] += 1
                    current_selected, current_face_count = complete(current_selected, current_face_count)
                    current_count = count_sel(current_selected)
                    if current_count > best_count:
                        best_count = current_count
                        best_selected = current_selected.copy()
                        best_face_count = current_face_count.copy()
                    stuck_count = 0

    out = []
    for ei in range(m):
        if best_selected[ei]:
            x, y = edges[ei]
            out.append([x, y])

    return out
# EVOLVE-BLOCK-END
