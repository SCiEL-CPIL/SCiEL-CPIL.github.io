# EVOLVE-BLOCK-START
import random
import time

_DEADLINE = time.time() + 320.0          # safety margin before 330 s kill
_EPS = 1e-9


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _grow(sequence, dimension, rng):
    """Greedy self-avoiding walk: prefer sites with new H-H contacts, then free space."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _mc_refine(sequence, chain, rng, seed):
    """Metropolis local search over pivot + slide moves. Returns (chain, energy)."""
    n = len(chain)
    if n < 3:
        return chain, _energy(sequence, chain)

    is_h = [c == "H" for c in sequence]
    chain = list(chain)
    occ = {p: i for i, p in enumerate(chain)}
    energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = energy

    def gain(pos):
        """New H-H contacts created if residue at pos moves to a free site."""
        if not is_h[pos]:
            return 0
        p = chain[pos]
        g = 0
        for q in _neighbours(p):
            j = occ.get(q)
            if j is not None and j != pos - 1 and j != pos + 1 and is_h[j]:
                g += 1
        return g

    def set_pos(pos, site):
        old = chain[pos]
        if old != site:
            del occ[old]
            chain[pos] = site
            occ[site] = pos

    def pivot(p, k, sign):
        """Rotate suffix chain[p:] about the axis through chain[p] and chain[p+1]."""
        pivot_pt = chain[p]
        axis = [chain[p + 1][d] - pivot_pt[d] for d in range(dimension)]
        if all(a == 0 for a in axis):
            return
        if sign == 0:
            return
        suffix = chain[p:]
        new_suffix = [pivot_pt]
        for pt in suffix[1:]:
            v = [pt[d] - pivot_pt[d] for d in range(dimension)]
            if sign == 1:
                # rotate about axis by +90 deg
                a2 = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]
                cross = [
                    axis[1] * v[2] - axis[2] * v[1],
                    axis[2] * v[0] - axis[0] * v[2],
                    axis[0] * v[1] - axis[1] * v[0],
                ]
                dot = axis[0] * v[0] + axis[1] * v[1] + axis[2] * v[2]
                v = [axis[d] * dot // a2 + cross[d] for d in range(dimension)]
            else:
                # rotate about axis by -90 deg
                a2 = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]
                cross = [
                    axis[1] * v[2] - axis[2] * v[1],
                    axis[2] * v[0] - axis[0] * v[2],
                    axis[0] * v[1] - axis[1] * v[0],
                ]
                dot = axis[0] * v[0] + axis[1] * v[1] + axis[2] * v[2]
                v = [axis[d] * dot // a2 - cross[d] for d in range(dimension)]
            new_suffix.append(tuple(pivot_pt[d] + v[d] for d in range(dimension)))
        return new_suffix

    def apply_new_suffix(p, new_suffix):
        """Replace chain[p:] with new_suffix if it is self-avoiding. Returns new energy delta or None."""
        # check no collisions with prefix and within suffix
        for i, pt in enumerate(new_suffix):
            idx = p + i
            if pt in occ:
                j = occ[pt]
                if j < p:      # collides with prefix
                    return None
                if j != idx:   # collides with another suffix position
                    return None
        # remove old suffix from occ
        for i in range(p, n):
            del occ[chain[i]]
        # energy change from contacts involving suffix
        # recompute by removing old suffix contacts then adding new
        old_energy = energy
        # contacts lost: any H-H contact where at least one endpoint in [p, n)
        lost = 0
        for i in range(p, n):
            if is_h[i]:
                pt = chain[i]
                for q in _neighbours(pt):
                    j = occ.get(q)
                    if j is not None and j > i + 1 and is_h[j]:
                        lost += 1
        # set new suffix
        for i in range(p, n):
            occ[new_suffix[i - p]] = i
            chain[i] = new_suffix[i - p]
        # contacts gained
        gained = 0
        for i in range(p, n):
            if is_h[i]:
                pt = chain[i]
                for q in _neighbours(pt):
                    j = occ.get(q)
                    if j is not None and j > i + 1 and is_h[j]:
                        gained += 1
        new_energy = old_energy - lost + gained
        return new_energy

    def slide(i):
        """Move residue i to a free neighbour site (if not breaking bonds). Returns (new_site, delta) or None."""
        p = chain[i]
        for q in _neighbours(p):
            if q in occ:
                continue
            # ensure moving i to q does not create a bond to a non-adjacent site
            # and does not break adjacency: i's neighbours i-1, i+1 must stay adjacent to q
            if i > 0 and q == chain[i - 1]:
                continue
            if i < n - 1 and q == chain[i + 1]:
                continue
            # moving i: remove from p, add at q. Energy change = new contacts at q - old contacts at p
            # old contacts at p (H-H, non-adjacent)
            old_c = 0
            if is_h[i]:
                for r in _neighbours(p):
                    j = occ.get(r)
                    if j is not None and j != i - 1 and j != i + 1 and is_h[j]:
                        old_c += 1
            # new contacts at q
            new_c = 0
            if is_h[i]:
                for r in _neighbours(q):
                    j = occ.get(r)
                    if j is not None and j != i - 1 and j != i + 1 and is_h[j]:
                        new_c += 1
            # but also moving i may break a contact between i-1 and i+1? No, they are not neighbours of i's site necessarily.
            # However, if i-1 and i+1 were both contacting i, that's fine (they contact i, not each other).
            # Also, if i was contacting some j, and i moves, that contact is lost (counted in old_c).
            # The neighbours i-1 and i+1 now contact q; if i-1 is H and adjacent to q that's a contact (counted in new_c).
            delta = new_c - old_c
            return q, delta
        return None

    T0 = 0.5
    T = T0
    step = 0
    total_steps = max(2000, 40 * n)
    cooling = 0.9995

    while step < total_steps and time.time() < _DEADLINE - 2.0:
        step += 1
        if rng.random() < 0.6:
            # pivot
            p = rng.randint(0, n - 2)
            k = rng.randint(p + 1, n - 1)
            sign = rng.choice([-1, 0, 1])
            if sign == 0:
                continue
            new_suffix = pivot(p, k, sign)
            if new_suffix is None:
                continue
            delta = apply_new_suffix(p, new_suffix)
            if delta is None:
                continue
            if delta < 0 or rng.random() < pow(2.718281828459045, -delta / T):
                energy += delta
                if energy < best_energy:
                    best_energy = energy
                    best_chain = list(chain)
            else:
                # revert
                # recompute: we already applied; revert by re-applying old suffix
                # simplest: recompute chain from best? No. Revert manually.
                # We'll just recompute energy and rebuild occ from chain (cheap enough rarely)
                # Actually we applied new suffix; to revert, set back.
                # Rebuild occ from chain to be safe.
                occ = {pt: i for i, pt in enumerate(chain)}
                energy = _energy(sequence, chain)
        else:
            i = rng.randint(1, n - 2)
            res = slide(i)
            if res is None:
                continue
            q, delta = res
            if delta < 0 or rng.random() < pow(2.718281828459045, -delta / T):
                old = chain[i]
                del occ[old]
                chain[i] = q
                occ[q] = i
                energy += delta
                if energy < best_energy:
                    best_energy = energy
                    best_chain = list(chain)
            else:
                pass
        if step % 50 == 0:
            T *= cooling
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation via greedy seeds + Monte Carlo local search, with a deadline guard."""
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))

    def straight_line():
        return [tuple([i] + [0] * (dimension - 1)) for i in range(n)]

    best, best_energy = None, 0

    # collect candidate seeds
    seeds = []
    for _ in range(3):
        c = _grow(sequence, dimension, rng)
        if c is not None:
            seeds.append(c)
    if not seeds:
        seeds.append(straight_line())

    # refine each seed with MC, respecting deadline
    for c in seeds:
        if time.time() > _DEADLINE - 5.0:
            break
        e = _energy(sequence, c)
        if best is None or e < best_energy:
            best, best_energy = c, e
        # refine
        if time.time() > _DEADLINE - 8.0:
            break
        chain2, e2 = _mc_refine(sequence, c, rng, len(sequence))
        if e2 < best_energy:
            best, best_energy = chain2, e2

    if best is None:
        best = straight_line()
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
