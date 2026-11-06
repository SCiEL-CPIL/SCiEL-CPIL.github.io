import time
import random
import math

def construct():
    d = 7
    n = 1 << d  # 128 vertices
    deadline = time.time() + 130  # Safety margin under 150s kill

    # Precompute all edges (u < v, differ in one bit)
    all_edges = []
    edge_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y))
                edge_idx[(x, y)] = idx
    m = len(all_edges)  # 448

    # Precompute all 2-faces (672) and their 4 edges
    face_edges = []
    edge_faces = [[] for _ in range(m)]
    for i in range(d):
        for j in range(i + 1, d):
            for x in range(n):
                if ((x >> i) & 1) == 0 and ((x >> j) & 1) == 0:
                    a = x
                    b = x ^ (1 << i)
                    c = x ^ (1 << j)
                    dv = x ^ (1 << i) ^ (1 << j)
                    e1 = edge_idx[(min(a, b), max(a, b))]
                    e2 = edge_idx[(min(b, dv), max(b, dv))]
                    e3 = edge_idx[(min(dv, c), max(dv, c))]
                    e4 = edge_idx[(min(c, a), max(c, a))]
                    fid = len(face_edges)
                    face_edges.append([e1, e2, e3, e4])
                    for ei in (e1, e2, e3, e4):
                        edge_faces[ei].append(fid)
    n_faces = len(face_edges)  # 672

    ef = edge_faces  # alias

    def recompute_fc(R):
        fc = [0] * n_faces
        for e in range(m):
            if R[e]:
                for f in ef[e]:
                    fc[f] += 1
        return fc

    def greedy_hit_set():
        """Greedy: pick edge covering most uncovered faces."""
        R = [False] * m
        fc = [0] * n_faces
        unc = [True] * n_faces
        n_unc = n_faces
        while n_unc > 0:
            best_e = -1
            best_g = -1
            for e in range(m):
                if R[e]:
                    continue
                g = 0
                for f in ef[e]:
                    if unc[f]:
                        g += 1
                if g > best_g:
                    best_g = g
                    best_e = e
            if best_e < 0:
                break
            R[best_e] = True
            for f in ef[best_e]:
                if unc[f]:
                    unc[f] = False
                    n_unc -= 1
                fc[f] += 1
        return R, fc

    def greedy_reduce(R):
        """Remove redundant edges from hitting set."""
        changed = True
        while changed:
            changed = False
            fc = recompute_fc(R)
            for e in range(m):
                if R[e]:
                    ok = True
                    for f in ef[e]:
                        if fc[f] <= 1:
                            ok = False
                            break
                    if ok:
                        R[e] = False
                        changed = True
                        break
        return R

    # Improved smooth surrogate using log-sum-exp
    def smooth_penalty(n_unc, scale=1.0):
        """Log-sum-exp style smooth approximation of uncovered face penalty."""
        if n_unc <= 0:
            return 0.0
        x = n_unc / scale
        if x > 50:
            return scale * x
        return scale * math.log1p(math.exp(x))

    def sa_search(R_init, fc_init, budget_sec, T0=2.0, T_min=0.01, cool=0.9997, lam=8.0):
        """Simulated annealing to minimize hitting set size with smooth penalty."""
        R = R_init[:]
        fc = fc_init[:]
        cur_size = sum(R)
        best_size = cur_size
        best_R = R[:]

        n_unc = sum(1 for c in fc if c == 0)
        best_unc = n_unc

        T = T0
        t_end = time.time() + budget_sec
        steps = 0

        while time.time() < t_end:
            e = random.randint(0, m - 1)

            if R[e]:
                # Try removing edge e from hitting set (add to selection)
                newly_unc = 0
                for f in ef[e]:
                    if fc[f] == 1:
                        newly_unc += 1
                new_n_unc = n_unc + newly_unc
                new_size = cur_size - 1

                # Apply
                for f in ef[e]:
                    fc[f] -= 1
                R[e] = False
                cur_size = new_size
                n_unc = new_n_unc

                if n_unc == 0 and cur_size < best_size:
                    best_size = cur_size
                    best_R = R[:]

                # Acceptance with smooth surrogate
                if n_unc == 0:
                    accept = True
                else:
                    old_cost = (cur_size + 1) + lam * smooth_penalty(n_unc - newly_unc)
                    new_cost = cur_size + lam * smooth_penalty(n_unc)
                    delta = new_cost - old_cost
                    if delta < 0:
                        accept = True
                    else:
                        accept = (random.random() < math.exp(-delta / T))

                if not accept:
                    R[e] = True
                    for f in ef[e]:
                        fc[f] += 1
                    cur_size += 1
                    n_unc -= newly_unc
            else:
                # Try adding edge e to hitting set (remove from selection)
                dec_unc = 0
                for f in ef[e]:
                    if fc[f] == 0:
                        dec_unc += 1
                new_n_unc = n_unc - dec_unc
                new_size = cur_size + 1

                # Apply
                R[e] = True
                for f in ef[e]:
                    fc[f] += 1
                cur_size = new_size
                n_unc = new_n_unc

                if n_unc == 0 and cur_size < best_size:
                    best_size = cur_size
                    best_R = R[:]

                # Acceptance
                if dec_unc > 0:
                    accept = True
                else:
                    old_cost = (cur_size - 1) + lam * smooth_penalty(n_unc + dec_unc)
                    new_cost = cur_size + lam * smooth_penalty(n_unc)
                    delta = new_cost - old_cost
                    if delta < 0:
                        accept = True
                    else:
                        accept = (random.random() < math.exp(-delta / T))

                if not accept:
                    R[e] = False
                    for f in ef[e]:
                        fc[f] -= 1
                    cur_size -= 1
                    n_unc += dec_unc

            steps += 1
            T *= cool
            if T < T_min:
                T = T_min

        return best_R, best_size

    def two_swap_polish(R, budget_sec):
        """Systematically try to remove pairs of edges from hitting set."""
        fc = recompute_fc(R)
        n_unc = sum(1 for c in fc if c == 0)
        if n_unc > 0:
            return R
        cur_size = sum(R)
        best_size = cur_size
        best_R = R[:]
        t_end = time.time() + budget_sec

        in_hs = [e for e in range(m) if R[e]]
        random.shuffle(in_hs)

        while time.time() < t_end:
            if len(in_hs) < 2:
                break
            i = random.randint(0, len(in_hs) - 1)
            j = random.randint(0, len(in_hs) - 1)
            if i == j:
                continue
            e1 = in_hs[i]
            e2 = in_hs[j]

            # Check if removing both keeps all faces covered
            ok = True
            faces_e1 = ef[e1]
            faces_e2 = ef[e2]

            for f in faces_e1:
                if fc[f] <= 1:
                    if f not in faces_e2:
                        ok = False
                        break
            if ok:
                for f in faces_e2:
                    if fc[f] <= 1:
                        if f not in faces_e1:
                            ok = False
                            break

            if ok:
                R[e1] = False
                R[e2] = False
                for f in faces_e1:
                    fc[f] -= 1
                for f in faces_e2:
                    fc[f] -= 1

                # Update in_hs
                in_hs = [e for e in in_hs if e != e1 and e != e2]
                cur_size -= 2

                # Greedy reduce
                R = greedy_reduce(R)
                fc = recompute_fc(R)
                cur_size = sum(R)

                if cur_size < best_size:
                    best_size = cur_size
                    best_R = R[:]

                in_hs = [e for e in range(m) if R[e]]
                random.shuffle(in_hs)

        return best_R

    def repair_phase(R, budget_sec):
        """Focus on fixing any uncovered faces by adding edges strategically."""
        fc = recompute_fc(R)
        n_unc = sum(1 for c in fc if c == 0)
        if n_unc == 0:
            return R, n_unc

        t_end = time.time() + budget_sec
        cur_size = sum(R)

        while time.time() < t_end and n_unc > 0:
            # Find uncovered faces
            unc_faces = [f for f in range(n_faces) if fc[f] == 0]
            if not unc_faces:
                break

            # Pick a random uncovered face
            f = random.choice(unc_faces)
            face_e = face_edges[f]

            # Try adding one of the face's edges to hitting set
            best_e = -1
            best_gain = -1
            for e in face_e:
                if not R[e]:
                    gain = 0
                    for ff in ef[e]:
                        if fc[ff] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_e = e

            if best_e >= 0:
                R[best_e] = True
                for ff in ef[best_e]:
                    if fc[ff] == 0:
                        n_unc -= 1
                    fc[ff] += 1
                cur_size += 1

        return R, n_unc

    # --- Main search ---
    R0, fc0 = greedy_hit_set()
    initial_size = sum(R0)

    best_R = R0[:]
    best_size = initial_size

    # Greedy reduce the initial solution
    best_R = greedy_reduce(best_R)
    best_size = sum(best_R)

    # Multi-restart SA with varied parameters
    num_restarts = 8
    base_time = 12.0
    for r in range(num_restarts):
        if time.time() > deadline - 30:
            break

        if r == 0:
            R_r = best_R[:]
            fc_r = recompute_fc(R_r)
        else:
            # Perturb best solution
            R_r = best_R[:]
            k = random.randint(3, 8)
            for _ in range(k):
                e = random.randint(0, m - 1)
                R_r[e] = not R_r[e]
            fc_r = recompute_fc(R_r)

        # Vary parameters
        T0 = 1.5 + 0.5 * r
        T_min = 0.005 + 0.002 * r
        cool = 0.9996 + 0.0001 * r
        lam = 6.0 + 1.0 * r

        per_time = min(base_time, max(3.0, deadline - 30 - time.time()))
        R_new, size_new = sa_search(R_r, fc_r, per_time, T0=T0, T_min=T_min, cool=cool, lam=lam)

        # Repair if needed
        R_new, unc_after = repair_phase(R_new, min(2.0, deadline - 30 - time.time()))
        R_new = greedy_reduce(R_new)
        size_new = sum(R_new)

        if size_new < best_size:
            best_size = size_new
            best_R = R_new[:]

    # Final polish: low-temperature SA
    if time.time() < deadline - 20:
        fc_cur = recompute_fc(best_R)
        R_pol, size_pol = sa_search(best_R, fc_cur, min(15.0, deadline - 20 - time.time()),
                                     T0=0.3, T_min=0.002, cool=0.999, lam=10.0)
        R_pol = greedy_reduce(R_pol)
        size_pol = sum(R_pol)
        if size_pol < best_size:
            best_size = size_pol
            best_R = R_pol[:]

    # Two-swap polish
    if time.time() < deadline - 10:
        R_sw = two_swap_polish(best_R, min(10.0, deadline - 10 - time.time()))
        if sum(R_sw) < best_size:
            best_size = sum(R_sw)
            best_R = R_sw[:]

    # Final greedy reduce
    best_R = greedy_reduce(best_R)
    best_size = sum(best_R)

    # Validate
    fc_final = recompute_fc(best_R)
    is_valid = all(c > 0 for c in fc_final)
    if not is_valid:
        R_g, fc_g = greedy_hit_set()
        R_g = greedy_reduce(R_g)
        best_R = R_g
        best_size = sum(R_g)

    # Build output: selected edges are those NOT in hitting set
    out = []
    for e in range(m):
        if not best_R[e]:
            x, y = all_edges[e]
            out.append([x, y])

    return out

if __name__ == "__main__":
    t0 = time.time()
    edges = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s, Edges: {len(edges)}")

    d = 7
    n = 1 << d
    all_edges_set = set()
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges_set.add((x, y))

    selected_set = set()
    for u, v in edges:
        selected_set.add((min(u, v), max(u, v)))

    valid = True
    for i in range(d):
        for j in range(i + 1, d):
            for x in range(n):
                if (x >> i) & 1 == 0 and (x >> j) & 1 == 0:
                    a = x
                    b = x ^ (1 << i)
                    c = x ^ (1 << j)
                    d2 = x ^ (1 << i) ^ (1 << j)

                    e1 = (min(a, b), max(a, b))
                    e2 = (min(b, d2), max(b, d2))
                    e3 = (min(d2, c), max(d2, c))
                    e4 = (min(c, a), max(c, a))

                    cnt = 0
                    if e1 in selected_set: cnt += 1
                    if e2 in selected_set: cnt += 1
                    if e3 in selected_set: cnt += 1
                    if e4 in selected_set: cnt += 1

                    if cnt == 4:
                        print(f"Invalid face found at {x}, dims {i},{j}")
                        valid = False
                        break
            if not valid:
                break
        if not valid:
            break

    if valid:
        print("Valid!")
    else:
        print("Invalid!")
