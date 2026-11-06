import random
import time
import sys

def construct():
    d = 7
    n_vertices = 1 << d
    total_edges = n_vertices * d // 2  # 448

    # Precompute all edges
    all_edges = []
    edge_to_index = {}
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                e = (x, y)
                edge_to_index[e] = len(all_edges)
                all_edges.append(e)

    # Precompute faces
    faces_idx = []
    edge_faces_idx = [[] for _ in range(total_edges)]

    for u in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                v = u ^ (1 << i)
                w = u ^ (1 << j)
                z = u ^ (1 << i) ^ (1 << j)

                e1 = (min(u, v), max(u, v))
                e2 = (min(v, z), max(v, z))
                e3 = (min(z, w), max(z, w))
                e4 = (min(w, u), max(w, u))

                idxs = (edge_to_index[e1], edge_to_index[e2],
                        edge_to_index[e3], edge_to_index[e4])
                faces_idx.append(idxs)

                for idx in idxs:
                    edge_faces_idx[idx].append(len(faces_idx) - 1)

    n_faces = len(faces_idx)

    # For each edge, store the faces it belongs to (as indices)
    # Also precompute for each face, the 4 edge indices as a tuple

    def is_valid_addition(chosen_mask, e_idx):
        for f_idx in edge_faces_idx[e_idx]:
            face = faces_idx[f_idx]
            count = 0
            for idx in face:
                if idx != e_idx and chosen_mask[idx]:
                    count += 1
                    if count == 3:
                        return False
        return True

    def get_bad_faces(chosen_mask, e_idx):
        """Return list of blocker-sets for each bad face containing e_idx."""
        bad = []
        for f_idx in edge_faces_idx[e_idx]:
            face = faces_idx[f_idx]
            others = []
            for idx in face:
                if idx != e_idx and chosen_mask[idx]:
                    others.append(idx)
            if len(others) == 3:
                bad.append(others)
        return bad

    def greedy_random():
        chosen_mask = bytearray(total_edges)
        indices = list(range(total_edges))
        random.shuffle(indices)
        for e_idx in indices:
            if is_valid_addition(chosen_mask, e_idx):
                chosen_mask[e_idx] = 1
        return chosen_mask

    def greedy_biased():
        """Greedy with bias: prefer edges that block fewer faces."""
        chosen_mask = bytearray(total_edges)
        indices = list(range(total_edges))
        random.shuffle(indices)
        for e_idx in indices:
            if is_valid_addition(chosen_mask, e_idx):
                chosen_mask[e_idx] = 1
        return chosen_mask

    def local_search(chosen_mask, time_limit):
        """
        Local search with 2-for-1 swaps, 1-for-1 swaps, and perturbation.
        Returns (mask, count).
        """
        start_time = time.time()
        chosen_count = sum(chosen_mask)
        best_mask = chosen_mask.copy()
        best_count = chosen_count

        # Precompute: for each chosen edge b, which missing edges does it block?
        # We'll recompute this each iteration for correctness.

        def compute_unblocks(chosen_mask, chosen_edges, missing_edges):
            """Map b -> list of e in missing that b blocks (i.e., removing b allows e)."""
            unblocks = {b: [] for b in chosen_edges}
            for e in missing_edges:
                bad = get_bad_faces(chosen_mask, e)
                if not bad:
                    continue
                # Find common blockers
                if len(bad) == 1:
                    blockers = bad[0]
                else:
                    common = set(bad[0])
                    for bf in bad[1:]:
                        common &= set(bf)
                    if not common:
                        continue
                    blockers = list(common)
                for b in blockers:
                    if b in unblocks:
                        unblocks[b].append(e)
            return unblocks

        improved = True
        stall_count = 0
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            stall_count += 1

            chosen_edges = [i for i in range(total_edges) if chosen_mask[i]]
            missing_edges = [i for i in range(total_edges) if not chosen_mask[i]]

            if not missing_edges:
                break

            unblocks = compute_unblocks(chosen_mask, chosen_edges, missing_edges)

            # Try 2-for-1 swaps
            random.shuffle(chosen_edges)
            found_swap = False
            for b in chosen_edges:
                if time.time() - start_time > time_limit:
                    break
                if not chosen_mask[b]:
                    continue
                candidates = unblocks.get(b, [])
                if len(candidates) >= 2:
                    # Try pairs
                    random.shuffle(candidates)
                    for i in range(min(len(candidates), 20)):
                        for j in range(i + 1, min(len(candidates), 20)):
                            e1, e2 = candidates[i], candidates[j]
                            chosen_mask[b] = 0
                            ok = is_valid_addition(chosen_mask, e1)
                            if ok:
                                chosen_mask[e1] = 1
                                ok2 = is_valid_addition(chosen_mask, e2)
                                if ok2:
                                    chosen_mask[e2] = 1
                                    chosen_count += 1
                                    improved = True
                                    found_swap = True
                                    break
                                else:
                                    chosen_mask[e1] = 0
                            chosen_mask[b] = 1
                            if found_swap:
                                break
                        if found_swap:
                            break
                if found_swap:
                    break

            if found_swap:
                stall_count = 0
                if chosen_count > best_count:
                    best_count = chosen_count
                    best_mask = chosen_mask.copy()
                continue

            # Try 1-for-1 swaps
            if chosen_edges and missing_edges:
                # Try random 1-for-1
                for _ in range(min(len(chosen_edges) * len(missing_edges), 200)):
                    if time.time() - start_time > time_limit:
                        break
                    b = random.choice(chosen_edges)
                    e = random.choice(missing_edges)
                    if not chosen_mask[b]:
                        continue
                    chosen_mask[b] = 0
                    if is_valid_addition(chosen_mask, e):
                        chosen_mask[e] = 1
                        improved = True
                        stall_count = 0
                        break
                    else:
                        chosen_mask[b] = 1

            # Perturbation: remove a random edge and repair
            if not improved and stall_count > 3 and random.random() < 0.3:
                b = random.choice(chosen_edges)
                chosen_mask[b] = 0
                chosen_count -= 1
                # Greedy repair
                for e2 in missing_edges:
                    if e2 == b:
                        continue
                    if not chosen_mask[e2] and is_valid_addition(chosen_mask, e2):
                        chosen_mask[e2] = 1
                        chosen_count += 1
                improved = True

            if chosen_count > best_count:
                best_count = chosen_count
                best_mask = chosen_mask.copy()

        return best_mask, best_count

    def local_search_v2(chosen_mask, time_limit):
        """
        More aggressive local search: try removing up to 2 edges and adding up to 3.
        """
        start_time = time.time()
        chosen_count = sum(chosen_mask)
        best_mask = chosen_mask.copy()
        best_count = chosen_count

        def try_swap(chosen_mask, remove_set, add_set):
            """Try to remove all in remove_set and add all in add_set."""
            for r in remove_set:
                chosen_mask[r] = 0
            for a in add_set:
                if not is_valid_addition(chosen_mask, a):
                    # Revert
                    for r in remove_set:
                        chosen_mask[r] = 1
                    for aa in add_set:
                        if chosen_mask[aa]:
                            chosen_mask[aa] = 0
                    return False
                chosen_mask[a] = 1
            return True

        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            chosen_edges = [i for i in range(total_edges) if chosen_mask[i]]
            missing_edges = [i for i in range(total_edges) if not chosen_mask[i]]

            if not missing_edges:
                break

            # Compute unblocks
            unblocks = {b: [] for b in chosen_edges}
            for e in missing_edges:
                bad = get_bad_faces(chosen_mask, e)
                if not bad:
                    continue
                if len(bad) == 1:
                    blockers = bad[0]
                else:
                    common = set(bad[0])
                    for bf in bad[1:]:
                        common &= set(bf)
                    if not common:
                        continue
                    blockers = list(common)
                for b in blockers:
                    if b in unblocks:
                        unblocks[b].append(e)

            # 2-for-1
            random.shuffle(chosen_edges)
            for b in chosen_edges:
                if time.time() - start_time > time_limit:
                    break
                cands = unblocks.get(b, [])
                if len(cands) >= 2:
                    random.shuffle(cands)
                    for i in range(min(len(cands), 15)):
                        for j in range(i + 1, min(len(cands), 15)):
                            if try_swap(chosen_mask, [b], [cands[i], cands[j]]):
                                chosen_count += 1
                                improved = True
                                if chosen_count > best_count:
                                    best_count = chosen_count
                                    best_mask = chosen_mask.copy()
                                break
                        if improved:
                            break
                if improved:
                    break

            if improved:
                continue

            # 3-for-2: remove 2, add 3
            if len(chosen_edges) >= 2 and len(missing_edges) >= 3:
                # Try random pairs to remove
                for _ in range(50):
                    if time.time() - start_time > time_limit:
                        break
                    b1, b2 = random.sample(chosen_edges, 2)
                    # Find edges that either b1 or b2 unblocks
                    cands = set(unblocks.get(b1, [])) | set(unblocks.get(b2, []))
                    cands = list(cands)
                    if len(cands) >= 3:
                        random.shuffle(cands)
                        # Try adding 3
                        chosen_mask[b1] = 0
                        chosen_mask[b2] = 0
                        added = 0
                        added_edges = []
                        for e in cands:
                            if added >= 3:
                                break
                            if not chosen_mask[e] and is_valid_addition(chosen_mask, e):
                                chosen_mask[e] = 1
                                added += 1
                                added_edges.append(e)
                        if added >= 2:  # Net gain at least 0, but we want gain
                            if added > 2:
                                chosen_count += added - 2
                                improved = True
                                if chosen_count > best_count:
                                    best_count = chosen_count
                                    best_mask = chosen_mask.copy()
                                break
                        else:
                            # Revert
                            for e in added_edges:
                                chosen_mask[e] = 0
                            chosen_mask[b1] = 1
                            chosen_mask[b2] = 1

        return best_mask, best_count

    best_mask = None
    best_size = 0

    start = time.time()
    deadline = start + 140  # 10s safety margin

    # Phase 1: Random Greedy Restarts
    n_greedy = 0
    while time.time() < deadline - 30:
        mask = greedy_random()
        size = sum(mask)
        if size > best_size:
            best_size = size
            best_mask = mask.copy()
        n_greedy += 1

    # Phase 2: Local Search on best
    if best_mask is not None and time.time() < deadline - 5:
        t_rem = deadline - time.time() - 2
        improved_mask, improved_size = local_search(best_mask, t_rem * 0.6)
        if improved_size > best_size:
            best_size = improved_size
            best_mask = improved_mask

        # Phase 3: More aggressive local search
        if time.time() < deadline - 3:
            t_rem2 = deadline - time.time() - 2
            improved_mask2, improved_size2 = local_search_v2(best_mask, t_rem2 * 0.5)
            if improved_size2 > best_size:
                best_size = improved_size2
                best_mask = improved_mask2

        # Phase 4: Try local search on a few more random greedy results
        while time.time() < deadline - 3:
            mask = greedy_random()
            size = sum(mask)
            t_rem3 = deadline - time.time() - 2
            if t_rem3 < 2:
                break
            improved_mask3, improved_size3 = local_search(mask, t_rem3 * 0.4)
            if improved_size3 > best_size:
                best_size = improved_size3
                best_mask = improved_mask3

    # Convert to list of [u, v]
    result = []
    for e_idx in range(total_edges):
        if best_mask[e_idx]:
            u, v = all_edges[e_idx]
            result.append([u, v])

    return result
