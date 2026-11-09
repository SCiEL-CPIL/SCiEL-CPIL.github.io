"""Sparse pancyclic graphs on 40 vertices: Hamiltonian cycle + minimal chords."""
import time

N = 40


def _verify_cyc(cyc, edge_set):
    """Check a cycle: distinct vertices, all edges present."""
    k = len(cyc)
    if len(set(cyc)) != k:
        return False
    for i in range(k):
        u, v = cyc[i], cyc[(i + 1) % k]
        if (min(u, v), max(u, v)) not in edge_set:
            return False
    return True


def _build_fan_cycles(edge_set):
    """Build 38 cycles (lengths 3..40) from the 58-edge fan graph."""
    cycs = []
    for k in range(3, N + 1):
        if k == N:
            c = list(range(N))
        elif k <= 19:
            c = list(range(k))
        elif k == 20:
            c = list(range(2, 22))
        elif k == 21:
            c = [0, 2] + list(range(21, 40))
        elif k == 22:
            c = [2, 1, 0] + list(range(39, 20, -1))
        else:
            dp = 41 - k
            c = [0] + list(range(39, dp - 1, -1))
        if not _verify_cyc(c, edge_set):
            return None
        cycs.append(c)
    return cycs


def construct():
    t0 = time.time()
    DL = t0 + 70.0

    # ---- Proven 58-edge construction (40 rim + 17 fan chords + 1 bridge) ----
    rim = set()
    for i in range(N):
        a, b = i, (i + 1) % N
        rim.add((min(a, b), max(a, b)))

    fan_chords = set()
    for d in range(2, 19):
        fan_chords.add((0, d))

    bridge = (2, 21)
    e58 = frozenset(rim | fan_chords | {bridge})

    # Verify the known-good construction
    cycs58 = _build_fan_cycles(set(e58))
    if cycs58 is not None:
        best_edges = sorted(e58)
        best_cycs = cycs58
    else:
        # Should never happen, but fallback
        best_edges = sorted(rim | fan_chords | {bridge})
        best_cycs = []
        for k in range(3, N + 1):
            best_cycs.append(list(range(k)))

    # ---- Attempt: remove chords in order, test with 21-cycle variants ----
    remove_order = list(range(18, 1, -1))  # try d=18,17,...,2 first
    improved = False
    for d_rem in remove_order:
        if time.time() > DL:
            break

        cand = set(rim | {bridge})
        for d in range(2, 19):
            if d != d_rem:
                cand.add((0, d))

        es = frozenset(cand)

        # Try several 21-cycle candidates
        found_21 = None
        try_candidates = [
            [0, 2] + list(range(21, 40)),          # original
            list(range(20, 41)),                   # 20..39,0
            list(range(19, 40)),                   # 19..38,0
            list(range(21, 42)),                   # 21..40→0 (40=0)
            list(range(1, 22)),                    # 1..21
            list(range(19, 41)),                   # 19..39,0
            list(range(18, 40)),                   # 18..38,0
        ]
        cset = set(cand)
        for tc in try_candidates:
            if _verify_cyc(tc, cset):
                found_21 = tc
                break

        if found_21 is None:
            continue

        # Build full cycle set with this 21-cycle
        cycs = []
        ok = True
        for k in range(3, N + 1):
            if k == N:
                c = list(range(N))
            elif k <= 19:
                c = list(range(k))
            elif k == 20:
                c = list(range(2, 22))
            elif k == 21:
                c = found_21
            elif k == 22:
                c = [2, 1, 0] + list(range(39, 20, -1))
            else:
                dp = 41 - k
                c = [0] + list(range(39, dp - 1, -1))
            if not _verify_cyc(c, cset):
                ok = False
                break
            cycs.append(c)

        if ok and len(cand) < len(best_edges):
            best_edges = sorted(cand)
            best_cycs = cycs
            improved = True

        if time.time() > DL:
            break

    # ---- Attempt: remove chord d, add different bridge ----
    if not improved and time.time() < DL - 5:
        alt_bridges = [(i, j) for i in range(3, 15) for j in range(22, 40)
                       if abs(i - j) >= 5 and (i, j) != (2, 21)]
        for d_rem in [18, 17, 16, 15]:
            if time.time() > DL - 3:
                break
            for br in alt_bridges[:20]:
                if time.time() > DL:
                    break
                cand = set(rim | {br})
                for d in range(2, 19):
                    if d != d_rem:
                        cand.add((0, d))
                cset = set(cand)

                # Try 21-cycle: 0 -> d_rem is missing, try alternative
                # Need path from 0 to some point using available edges
                found_21 = None
                # Try: 0, a, ..., b, cset check
                for a in range(1, 15):
                    if (0, a) not in cset or a > 20:
                        continue
                    # Try going a, a+1, ..., 39, 0
                    cand21 = [0] + list(range(a, 40))
                    if len(set(cand21)) == 21 and _verify_cyc(cand21, cset):
                        found_21 = cand21
                        break
                    if found_21:
                        break
                    # Try: 0, a, b, ..., 39, 0 where b connects
                if found_21 is None:
                    continue

                cycs = []
                ok = True
                for k in range(3, N + 1):
                    if k == N:
                        c = list(range(N))
                    elif k <= 19:
                        c = list(range(k))
                    elif k == 20:
                        c = list(range(2, 22))
                    elif k == 21:
                        c = found_21
                    elif k == 22:
                        c = [2, 1, 0] + list(range(39, 20, -1))
                    else:
                        dp = 41 - k
                        c = [0] + list(range(39, dp - 1, -1))
                    if not _verify_cyc(c, cset):
                        ok = False
                        break
                    cycs.append(c)

                if ok and len(cand) < len(best_edges):
                    best_edges = sorted(cand)
                    best_cycs = cycs
                    break
            if time.time() > DL:
                break

    # Final validation
    final_es = set(best_edges)
    assert len(final_es) == len(best_edges), "Duplicate edges"
    for c in best_cycs:
        assert _verify_cyc(c, final_es), f"Invalid cycle: {c}"

    return {'edges': best_edges, 'cycles': best_cycs}
