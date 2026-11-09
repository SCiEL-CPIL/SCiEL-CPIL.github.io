<<<<<<< SEARCH
def _polish(points, rng, time_budget):
    """Final polish with wider search window."""
    n = len(points)
    cur = [list(p) for p in points]
    cur_n4, cur_n5, cur_n3, cur_n2 = _score(cur)
    best_key = (cur_n4, -cur_n5, cur_n3, cur_n2)
    best_pts = [list(p) for p in cur]

    for i in range(n):
        if time.time() >= time_budget:
            break
        x0, y0 = cur[i]
        w = 35
        best_local = None
        best_local_key = None

        for dx in range(-w, w + 1):
            if time.time() >= time_budget:
                break
            for dy in range(-w, w + 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x0 + dx, y0 + dy
                if abs(nx) > _BOUND or abs(ny) > _BOUND:
                    continue
                np_ = [nx, ny]
                if any(np_ == p for p in cur):
                    continue

                # Fast validity
                valid = True
                for j in range(n):
                    if j == i:
                        continue
                    k = _line_key(np_, cur[j])
                    cnt = 1
                    for m in range(n):
                        if m == i or m == j:
                            continue
                        if _line_key(np_, cur[m]) == k:
                            cnt += 1
                    if cnt >= 5:
                        valid = False
                        break
                if not valid:
                    continue

                cand = [list(p) for p in cur]
                cand[i] = np_
                n4, n5, n3, n2 = _score(cand)
                key = (n4, -n5, n3, n2)
                if best_local_key is None or key > best_local_key:
                    best_local_key = key
                    best_local = np_

        if best_local is not None:
            cur_key = (cur_n4, -cur_n5, cur_n3, cur_n2)
            if best_local_key > cur_key:
                cur[i] = best_local
                cur_n4, cur_n5, cur_n3, cur_n2 = best_local_key[0], -best_local_key[1], best_local_key[2], best_local_key[3]
                if (cur_n4, -cur_n5, cur_n3, cur_n2) > best_key:
                    best_key = (cur_n4, -cur_n5, cur_n3, cur_n2)
                    best_pts = [list(p) for p in cur]

    return (best_key[0], -best_key[1], best_key[2], best_key[3], best_pts)
=======
def _polish(points, rng, time_budget):
    """Final polish with wider search window and targeted candidates."""
    n = len(points)
    cur = [list(p) for p in points]
    cur_counts = _build_line_counts(cur)
    cur_n4, cur_n5, cur_n3, cur_n2 = _score_from_counts(cur_counts)
    best_key = (cur_n4, -cur_n5, cur_n3, cur_n2)
    best_pts = [list(p) for p in cur]

    # Multiple passes for convergence
    for pass_num in range(5):
        if time.time() >= time_budget:
            break
        for i in range(n):
            if time.time() >= time_budget:
                break
            x0, y0 = cur[i]
            w = 40
            best_local = None
            best_local_key = None
            seen_cands = set()

            # Grid candidates
            for dx in range(-w, w + 1):
                if time.time() >= time_budget:
                    break
                for dy in range(-w, w + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x0 + dx, y0 + dy
                    if abs(nx) > _BOUND or abs(ny) > _BOUND:
                        continue
                    np_ = [nx, ny]
                    cand_key = (nx, ny)
                    if cand_key in seen_cands:
                        continue
                    if any(np_ == p for p in cur):
                        continue
                    seen_cands.add(cand_key)

                    # Fast validity
                    valid = True
                    for j in range(n):
                        if j == i:
                            continue
                        k = _line_key(np_, cur[j])
                        cnt = 1
                        for m in range(n):
                            if m == i or m == j:
                                continue
                            if _line_key(np_, cur[m]) == k:
                                cnt += 1
                        if cnt >= 5:
                            valid = False
                            break
                    if not valid:
                        continue

                    # Score using incremental update
                    cand_counts = dict(cur_counts)
                    for j in range(n):
                        if j == i:
                            continue
                        key = (min(i, j), max(i, j))
                        k = _line_key(cur[i], cur[j])
                        cand_counts[k] = cand_counts.get(k, 0) - 1
                        if cand_counts[k] == 0:
                            del cand_counts[k]
                    
                    for j in range(n):
                        if j == i:
                            continue
                        k = _line_key(np_, cur[j])
                        cand_counts[k] = cand_counts.get(k, 0) + 1
                    
                    n4, n5, n3, n2 = _score_from_counts(cand_counts)
                    key = (n4, -n5, n3, n2)
                    if best_local_key is None or key > best_local_key:
                        best_local_key = key
                        best_local = np_

            # Targeted candidates for creating 4-lines
            for j in range(n):
                if j == i:
                    continue
                for m in range(n):
                    if m == i or m == j:
                        continue
                    k_jm = _line_key(cur[j], cur[m])
                    line_pts = [cur[j], cur[m]]
                    for l in range(n):
                        if l != i and l != j and l != m:
                            if _line_key(cur[j], cur[l]) == k_jm:
                                line_pts.append(cur[l])
                    
                    if len(line_pts) == 3:
                        x1, y1 = cur[j]
                        x2, y2 = cur[m]
                        dx = x2 - x1
                        dy = y2 - y1
                        g = _gcd(dx, dy)
                        if g > 0:
                            dx //= g
                            dy //= g
                            for t in range(1, 12):
                                for sign in [1, -1]:
                                    nx = x1 + sign * t * dx
                                    ny = y1 + sign * t * dy
                                    if abs(nx) > _BOUND or abs(ny) > _BOUND:
                                        continue
                                    np_ = [nx, ny]
                                    cand_key = (nx, ny)
                                    if cand_key in seen_cands:
                                        continue
                                    if any(np_ == p for p in cur):
                                        continue
                                    seen_cands.add(cand_key)

                                    valid = True
                                    for jj in range(n):
                                        if jj == i:
                                            continue
                                        k = _line_key(np_, cur[jj])
                                        cnt = 1
                                        for mm in range(n):
                                            if mm == i or mm == jj:
                                                continue
                                            if _line_key(np_, cur[mm]) == k:
                                                cnt += 1
                                        if cnt >= 5:
                                            valid = False
                                            break
                                    if not valid:
                                        continue

                                    cand_counts = dict(cur_counts)
                                    for jj in range(n):
                                        if jj == i:
                                            continue
                                        key = (min(i, jj), max(i, jj))
                                        k = _line_key(cur[i], cur[jj])
                                        cand_counts[k] = cand_counts.get(k, 0) - 1
                                        if cand_counts[k] == 0:
                                            del cand_counts[k]
                                    
                                    for jj in range(n):
                                        if jj == i:
                                            continue
                                        k = _line_key(np_, cur[jj])
                                        cand_counts[k] = cand_counts.get(k, 0) + 1
                                    
                                    n4, n5, n3, n2 = _score_from_counts(cand_counts)
                                    key = (n4, -n5, n3, n2)
                                    if best_local_key is None or key > best_local_key:
                                        best_local_key = key
                                        best_local = np_

            if best_local is not None:
                cur_key = (cur_n4, -cur_n5, cur_n3, cur_n2)
                if best_local_key > cur_key:
                    cur[i] = best_local
                    cur_counts = _build_line_counts(cur)
                    cur_n4, cur_n5, cur_n3, cur_n2 = _score_from_counts(cur_counts)
                    if (cur_n4, -cur_n5, cur_n3, cur_n2) > best_key:
                        best_key = (cur_n4, -cur_n5, cur_n3, cur_n2)
                        best_pts = [list(p) for p in cur]

    return (best_key[0], -best_key[1], best_key[2], best_key[3], best_pts)
>>>>>>> REPLACE

