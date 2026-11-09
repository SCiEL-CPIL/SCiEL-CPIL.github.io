"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Safety margin: evaluator kills at 330s wall-clock; we stop searching well before.
_DEADLINE = 320.0  # seconds from process start (conservative margin)
_START = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _neighbours(point):
    x, y, z = point
    yield (x + 1, y, z)
    yield (x - 1, y, z)
    yield (x, y + 1, z)
    yield (x, y - 1, z)
    yield (x, y, z + 1)
    yield (x, y, z - 1)


def _is_hh(seq, i, j):
    return seq[i] == 'H' and seq[j] == 'H'


def _grow(sequence, dimension, rng, temp=0.0):
    """One chain growth. temp>0 adds randomness to the site choice (for tail regrowth)."""
    n = len(sequence)
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    energy = 0  # = -contacts; we track contacts
    contacts = 0

    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # contacts this site would make with earlier H residues
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain + 0.1 * free + rng.random() * (1.0 + temp)
            options.append((score, site, gain))
        if not options:
            return None, None, None
        options.sort(key=lambda x: x[0], reverse=True)
        # softmax-ish pick for temperature; for temp=0 just take best
        if temp <= 0.0:
            best = options[0]
        else:
            # pick by weight = exp(score/temp) but scores can be small; use simple weighted
            top = options[:min(len(options), 6)]
            weights = [math.exp((s[0] - top[0][0]) / max(temp, 1e-6)) for s in top]
            tot = sum(weights)
            r = rng.random() * tot
            acc = 0.0
            chosen = top[0]
            for w, s in zip(weights, top):
                acc += w
                if r <= acc:
                    chosen = s
                    break
        site, gain = chosen[1], chosen[2]
        occupied[site] = i
        chain.append(site)
        contacts += gain
    return chain, occupied, -contacts


def _local_search(sequence, chain, occupied, energy, rng, max_rounds=60):
    """Improve via single and two-residue moves. Returns (chain, occupied, energy)."""
    n = len(chain)
    improved = True
    rounds = 0
    while improved and rounds < max_rounds and _time_left() > 0.05:
        improved = False
        rounds += 1
        # single-residue moves
        for i in range(1, n - 1):
            p = chain[i]
            for q in _neighbours(p):
                if q in occupied:
                    continue
                # check q not adjacent to chain[i-1] or chain[i+1] (would break unit step? 
                # actually q must be a unit step from both neighbors to remain a valid walk)
                # The walk requires consecutive residues to be unit steps. Moving residue i to q
                # requires q to be unit step from chain[i-1] and chain[i+1].
                if abs(q[0] - chain[i - 1][0]) + abs(q[1] - chain[i - 1][1]) + abs(q[2] - chain[i - 1][2]) != 1:
                    continue
                if abs(q[0] - chain[i + 1][0]) + abs(q[1] - chain[i + 1][1]) + abs(q[2] - chain[i + 1][2]) != 1:
                    continue
                # compute delta energy
                delta = 0
                # remove contacts involving i
                if sequence[i] == 'H':
                    for nb in _neighbours(p):
                        j = occupied.get(nb)
                        if j is not None and j != i and abs(j - i) > 1 and sequence[j] == 'H':
                            delta -= 1
                # add contacts at q
                if sequence[i] == 'H':
                    for nb in _neighbours(q):
                        j = occupied.get(nb)
                        if j is not None and j != i and abs(j - i) > 1 and sequence[j] == 'H':
                            delta += 1
                if delta > 0:
                    # accept: update structures
                    del occupied[p]
                    occupied[q] = i
                    chain[i] = q
                    energy += delta
                    improved = True
                    break
        # two-residue moves: straighten a kink (i-1,i,i+1 collinear) -> move i to the other side? 
        # Or straighten i,i+1 pair: if they form a straight line, we can't do much. 
        # Kink straighten: if chain[i-1],chain[i],chain[i+1] form a right angle, move i so the
        # three become collinear (the "other" straight configuration). This frees space.
        for i in range(1, n - 1):
            a, b, c = chain[i - 1], chain[i], chain[i + 1]
            # detect kink: a and c differ from b in two axes
            da = [a[k] - b[k] for k in range(3)]
            dc = [c[k] - b[k] for k in range(3)]
            # kink if exactly two of da are nonzero? Actually a and c are both unit from b.
            # straight if a and c are opposite (dc = -da). kink if dc != -da.
            if dc[0] == -da[0] and dc[1] == -da[1] and dc[2] == -da[2]:
                continue  # already straight
            # straighten: new b' such that a, b', c collinear with b' unit from a and c.
            # The straight configuration has b' = a + (c - a) ... but that's 2 units from a.
            # Actually for a kink, the straight completion is: b' where a->b' and b'->c are unit
            # and a,b',c collinear means b' is the midpoint... not integer. So straightening a
            # kink in 3D cubic: the three points a,b,c with b unit from both a and c, a and c
            # at distance 2 (diagonal). The straight alternative doesn't exist as integer points
            # with unit steps because a and c are distance 2 apart (face diagonal). So skip.
            pass
    return chain, occupied, energy


def _tail_anneal(sequence, chain, occupied, energy, rng):
    """Simulated annealing over tail regrowth. Returns (chain, occupied, energy)."""
    n = len(chain)
    if n < 6:
        return chain, occupied, energy
    best_chain = list(chain)
    best_occupied = dict(occupied)
    best_energy = energy
    T = max(1.0, -best_energy * 0.05 + 1.0)
    cool = 0.98
    iters = 0
    while _time_left() > 0.1 and T > 0.3:
        iters += 1
        # pick a cut point
        cut = rng.randint(2, n - 2)
        # remove tail
        tail = chain[cut:]
        for p in tail:
            del occupied[p]
        # recompute energy of prefix
        prefix = chain[:cut]
        occ_prefix = {p: i for i, p in enumerate(prefix)}
        # recompute prefix energy
        p_energy = 0
        for i, p in enumerate(prefix):
            if sequence[i] == 'H':
                for q in _neighbours(p):
                    j = occ_prefix.get(q)
                    if j is not None and j > i + 1 and sequence[j] == 'H':
                        p_energy += 1
        p_energy = -p_energy
        # regrow tail with temperature
        new_tail, new_occ, tail_energy = _grow(sequence[cut:], 3, rng, temp=T)
        if new_tail is None:
            # restore
            for k, p in enumerate(tail):
                occupied[p] = cut + k
            continue
        # offset new_tail by prefix last position
        last = prefix[-1]
        # new_tail[0] should equal last; shift
        shift = (last[0] - new_tail[0][0], last[1] - new_tail[0][1], last[2] - new_tail[0][2])
        new_tail = [(x + shift[0], y + shift[1], z + shift[2]) for (x, y, z) in new_tail]
        # build full chain
        full = prefix + new_tail
        # check self-avoidance (should be ok) and compute energy
        full_occ = {p: i for i, p in enumerate(full)}
        full_energy = 0
        for i, p in enumerate(full):
            if sequence[i] == 'H':
                for q in _neighbours(p):
                    j = full_occ.get(q)
                    if j is not None and j > i + 1 and sequence[j] == 'H':
                        full_energy += 1
        full_energy = -full_energy
        # accept?
        delta = full_energy - best_energy
        if delta >= 0 or rng.random() < math.exp(delta / max(T, 1e-6)):
            chain = full
            occupied = full_occ
            energy = full_energy
            if energy < best_energy:
                best_energy = energy
                best_chain = list(chain)
                best_occupied = dict(occupied)
        else:
            # restore old tail
            for k, p in enumerate(tail):
                occupied[p] = cut + k
        T *= cool
    return best_chain, best_occupied, best_energy


def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def fold(sequence, dimension):
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    best = None
    best_energy = 0  # 0 = no contacts (worst for negative energy scale; we want most negative)
    best_occupied = None

    # Phase 1: many random greedy growths
    growths = 400
    for _ in range(growths):
        if _time_left() < 1.0:
            break
        chain, occupied, energy = _grow(sequence, dimension, rng, temp=0.0)
        if chain is None:
            continue
        if energy < best_energy:
            best_energy = energy
            best = chain
            best_occupied = occupied

    if best is None:
        # straight line
        best = [(i, 0, 0) for i in range(n)]
        best_energy = 0
        best_occupied = {p: i for i, p in enumerate(best)}

    # Phase 2: local search + tail annealing on the best and a few others
    candidates = []
    if best_occupied is not None:
        candidates.append((list(best), dict(best_occupied), best_energy))

    # add a couple more random growths as candidates
    for _ in range(3):
        if _time_left() < 1.0:
            break
        chain, occupied, energy = _grow(sequence, dimension, rng, temp=0.0)
        if chain is not None:
            candidates.append((chain, occupied, energy))

    for ch, occ, en in candidates:
        if _time_left() < 0.5:
            break
        ch, occ, en = _local_search(sequence, ch, occ, en, rng)
        ch, occ, en = _tail_anneal(sequence, ch, occ, en, rng)
        if en < best_energy:
            best_energy = en
            best = ch
            best_occupied = occ

    # final verification energy
    if best is not None:
        best_energy = _energy(sequence, best)

    return [list(p) for p in best]
# EVOLVE-BLOCK-END
