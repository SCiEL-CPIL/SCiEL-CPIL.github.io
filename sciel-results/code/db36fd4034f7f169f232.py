"""HP lattice protein folding: grow + regrow-SA local optimization with deadline guard."""
import random
import time

_DEADLINE = time.time() + 168.0  # safety margin before the 180s hard limit


def _nb(p):
    x, y = p
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _nb(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _grow(sequence, dimension, rng):
    """Randomized greedy growth: prefer sites with new H-H contacts, then free space."""
    n = len(sequence)
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    for i in range(1, n):
        options = []
        last = chain[-1]
        for site in _nb(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _nb(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = 0
            for q in _nb(site):
                if q not in occupied and q != last:
                    free += 1
            score = gain * 2.5 + 0.15 * free + rng.random() * 0.5
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _optimize(sequence, chain, occupied, contacts, max_iters=4000):
    """SA with single-residue, kink and regrow moves; returns best chain + energy."""
    n = len(chain)
    if n < 4:
        return chain, contacts

    best_energy = -contacts
    current = -contacts
    best_chain = list(chain)
    best_occupied = dict(occupied)
    best_contacts = contacts

    T = max(0.5, n * 0.05)
    T_min = 0.01
    cooling = 0.9985
    rng = random.Random()

    # precompute H flags
    H = [c == "H" for c in sequence]

    iters = 0
    while iters < max_iters and time.time() < _DEADLINE:
        iters += 1
        r = rng.random()

        if r < 0.55:
            # ---- single-residue move ----
            i = rng.randint(1, n - 2)
            x, y = chain[i]
            # pick a valid neighbour candidate
            cands = [s for s in _nb((x, y)) if s != (x, y)]
            if not cands:
                continue
            new_site = rng.choice(cands)
            if new_site in occupied and occupied[new_site] != i:
                continue
            # connectivity
            if i > 0:
                px, py = chain[i - 1]
                if abs(new_site[0] - px) + abs(new_site[1] - py) != 1:
                    continue
            if i < n - 1:
                px, py = chain[i + 1]
                if abs(new_site[0] - px) + abs(new_site[1] - py) != 1:
                    continue

            old_site = (x, y)
            delta = 0
            if H[i]:
                for q in _nb(old_site):
                    j = occupied.get(q)
                    if j is not None and j != i and H[j]:
                        delta += 1
                for q in _nb(new_site):
                    j = occupied.get(q)
                    if j is not None and j != i and H[j]:
                        delta -= 1

            if delta < 0 or rng.random() < 2.718281828459045 ** (-delta / T):
                del occupied[old_site]
                occupied[new_site] = i
                chain[i] = new_site
                contacts += delta
                current = -contacts
                if current < best_energy:
                    best_energy = current
                    best_chain = list(chain)
                    best_occupied = dict(occupied)
                    best_contacts = contacts

        elif r < 0.80:
            # ---- kink move (rotate consecutive pair i, i+1) ----
            i = rng.randint(1, n - 3)
            p1, p2 = chain[i], chain[i + 1]
            x1, y1 = p1
            x2, y2 = p2
            dx, dy = x2 - x1, y2 - y1
            if dx == 0 and dy == 0:
                continue
            if dx in (1, -1):
                moves = [((x1, y1 + 1), (x2, y2 + 1)), ((x1, y1 - 1), (x2, y2 - 1))]
            elif dy in (1, -1):
                moves = [((x1 + 1, y1), (x2 + 1, y2)), ((x1 - 1, y1), (x2 - 1, y2))]
            else:
                continue
            for new_p1, new_p2 in moves:
                if new_p1 in occupied and occupied[new_p1] not in (i, i + 1):
                    continue
                if new_p2 in occupied and occupied[new_p2] not in (i, i + 1):
                    continue
                if i > 0:
                    px, py = chain[i - 1]
                    if abs(new_p1[0] - px) + abs(new_p1[1] - py) != 1:
                        continue
                if i + 2 < n:
                    px, py = chain[i + 2]
                    if abs(new_p2[0] - px) + abs(new_p2[1] - py) != 1:
                        continue
                if abs(new_p1[0] - new_p2[0]) + abs(new_p1[1] - new_p2[1]) != 1:
                    continue

                delta = 0
                for idx, old_s, new_s in ((i, p1, new_p1), (i + 1, p2, new_p2)):
                    if H[idx]:
                        for q in _nb(old_s):
                            j = occupied.get(q)
                            if j is not None and j not in (i, i + 1) and H[j]:
                                delta += 1
                        for q in _nb(new_s):
                            j = occupied.get(q)
                            if j is not None and j not in (i, i + 1) and H[j]:
                                delta -= 1

                if delta < 0 or rng.random() < 2.718281828459045 ** (-delta / T):
                    del occupied[p1]
                    del occupied[p2]
                    occupied[new_p1] = i
                    occupied[new_p2] = i + 1
                    chain[i] = new_p1
                    chain[i + 1] = new_p2
                    contacts += delta
                    current = -contacts
                    if current < best_energy:
                        best_energy = current
                        best_chain = list(chain)
                        best_occupied = dict(occupied)
                        best_contacts = contacts
                    break

        else:
            # ---- regrow move: remove a window [i, i+len), regrow via SA ----
            seglen = rng.choice([3, 4, 5, 6, 8, 10])
            if n - 4 < 1:
                continue
            i = rng.randint(1, n - seglen - 1)
            j_end = i + seglen - 1
            if j_end > n - 2:
                continue
            # save and remove the window
            saved = list(chain[i:j_end + 1])
            for k in range(i, j_end + 1):
                del occupied[chain[k]]
            # prefix ends at i-1, suffix starts at j_end+1
            # regrow one residue at a time from i to j_end
            ok = True
            for k in range(i, j_end + 1):
                last = chain[k - 1]
                nxt = chain[k + 1] if k + 1 <= j_end else chain[j_end + 1]
                options = []
                for site in _nb(last):
                    if site in occupied and occupied[site] != k:
                        continue
                    # must be adjacent to next residue
                    if abs(site[0] - nxt[0]) + abs(site[1] - nxt[1]) != 1:
                        continue
                    gain = 0
                    if H[k]:
                        for q in _nb(site):
                            jj = occupied.get(q)
                            if jj is not None and jj != k and H[jj]:
                                gain += 1
                    free = 0
                    for q in _nb(site):
                        if q not in occupied and q != last:
                            free += 1
                    sc = gain * 2.0 + 0.2 * free + rng.random() * 1.5
                    options.append((sc, site))
                if not options:
                    ok = False
                    break
                # SA-ish choice: softmax over top scores
                options.sort(reverse=True)
                # pick with a temperature-weighted probability among top few
                top = options[:min(3, len(options))]
                if len(top) == 1 or rng.random() < 0.7:
                    site = top[0][1]
                else:
                    site = rng.choice(top)[1]
                occupied[site] = k
                chain[k] = site
            if not ok:
                # revert: reinsert saved (they were removed; need to re-add)
                for k in range(i, j_end + 1):
                    occupied[saved[k - i]] = k
                    chain[k] = saved[k - i]
                continue
            # recompute current energy delta by full recompute (cheap enough for a window)
            # recompute contacts from scratch for correctness
            contacts = 0
            where = {p: idx for idx, p in enumerate(chain)}
            for idx, p in enumerate(chain):
                if H[idx]:
                    for q in _nb(p):
                        jj = where.get(q)
                        if jj is not None and jj > idx + 1 and H[jj]:
                            contacts += 1
            current = -contacts
            if current < best_energy:
                best_energy = current
                best_chain = list(chain)
                best_occupied = dict(occupied)
                best_contacts = contacts

        T = max(T_min, T * cooling)

    return best_chain, best_contacts


def _polish(sequence, chain, occupied, contacts, max_iters=2000):
    """Deterministic hill-climb polish: only accept improving moves."""
    n = len(chain)
    if n < 4:
        return chain, contacts
    H = [c == "H" for c in sequence]
    best_energy = -contacts
    iters = 0
    while iters < max_iters and time.time() < _DEADLINE:
        iters += 1
        improved = False
        # single-residue improving moves
        for i in range(1, n - 1):
            x, y = chain[i]
            for new_site in _nb((x, y)):
                if new_site == (x, y):
                    continue
                if new_site in occupied and occupied[new_site] != i:
                    continue
                if i > 0:
                    px, py = chain[i - 1]
                    if abs(new_site[0] - px) + abs(new_site[1] - py) != 1:
                        continue
                if i < n - 1:
                    px, py = chain[i + 1]
                    if abs(new_site[0] - px) + abs(new_site[1] - py) != 1:
                        continue
                old_site = (x, y)
                delta = 0
                if H[i]:
                    for q in _nb(old_site):
                        j = occupied.get(q)
                        if j is not None and j != i and H[j]:
                            delta += 1
                    for q in _nb(new_site):
                        j = occupied.get(q)
                        if j is not None and j != i and H[j]:
                            delta -= 1
                if delta < 0:
                    del occupied[old_site]
                    occupied[new_site] = i
                    chain[i] = new_site
                    contacts += delta
                    best_energy = -contacts
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return chain, contacts


def fold(sequence, dimension):
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 1
    restarts = 0
    while time.time() < _DEADLINE:
        restarts += 1
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        occupied = {p: i for i, p in enumerate(chain)}
        # initial contact count
        contacts = 0
        where = {p: i for i, p in enumerate(chain)}
        H = [c == "H" for c in sequence]
        for i, p in enumerate(chain):
            if H[i]:
                for q in _nb(p):
                    j = where.get(q)
                    if j is not None and j > i + 1 and H[j]:
                        contacts += 1
        chain, contacts = _optimize(sequence, chain, occupied, contacts)
        chain, contacts = _polish(sequence, chain, occupied, contacts)
        value = -contacts
        if value < best_energy:
            best, best_energy = chain, value
        if restarts > 400:
            break
    if best is None:
        best = [(i, 0) for i in range(n)]
    return [list(p) for p in best]
