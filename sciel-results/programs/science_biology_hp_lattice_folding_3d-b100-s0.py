"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import time
import random
import math

# Precompute neighbor offsets for 3D cubic lattice
_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

# Global deadline guard (safety margin before the 330s kill)
_START_TIME = time.time()
_DEADLINE = _START_TIME + 298.0


def _time_left():
    return _DEADLINE - time.time()


def _energy_full(sequence, chain):
    """Compute energy: -number of H-H contacts on neighbouring non-consecutive sites."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] != 'H':
            continue
        for dx, dy, dz in _OFFSETS:
            q = (p[0] + dx, p[1] + dy, p[2] + dz)
            j = where.get(q)
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


def _beam_search(sequence, dimension, beam_width=300):
    """
    Beam search for HP protein folding.
    Maintains a beam of the best partial conformations at each step.
    Score combines energy (lower better) and a free-space bonus so the
    chain does not trap itself; a little noise adds diversity.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]

    start = (0, 0, 0)
    beam = [(0.0, [start], {start}, {start: 0}, 0)]

    for i in range(1, n):
        if _time_left() < 8.0:
            break

        candidates = []
        seq_i = sequence[i]

        for score, chain, occ, pos_idx, energy in beam:
            last = chain[-1]
            lx, ly, lz = last

            for dx, dy, dz in _OFFSETS:
                site = (lx + dx, ly + dy, lz + dz)
                if site in occ:
                    continue

                # incremental energy gain from new H-H contacts
                gain = 0
                if seq_i == 'H':
                    for dx2, dy2, dz2 in _OFFSETS:
                        q = (site[0] + dx2, site[1] + dy2, site[2] + dz2)
                        j = pos_idx.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            gain += 1

                # free neighbours of the new site (room to grow)
                free = 0
                for dx2, dy2, dz2 in _OFFSETS:
                    q = (site[0] + dx2, site[1] + dy2, site[2] + dz2)
                    if q not in occ:
                        free += 1

                new_energy = energy - gain
                new_score = new_energy - 0.05 * free + random.random() * 0.01

                new_chain = chain + [site]
                new_occ = set(occ)
                new_occ.add(site)
                new_pos = dict(pos_idx)
                new_pos[site] = i

                candidates.append((new_score, new_chain, new_occ, new_pos, new_energy))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[0])
        beam = candidates[:beam_width]

    if not beam:
        return None
    best = min(beam, key=lambda x: x[0])
    return list(best[1])


# ---------------------------------------------------------------------------
# Local search with incremental energy: segment reversals + pivot rotations
# ---------------------------------------------------------------------------

def _rot(p, axis, angle):
    """Rotate point p about the given axis by 90/180/270 degrees (angle 1/2/3)."""
    x, y, z = p
    if axis == 0:  # x-axis
        if angle == 1:
            return (x, -z, y)
        if angle == 2:
            return (x, -y, -z)
        return (x, z, -y)
    if axis == 1:  # y-axis
        if angle == 1:
            return (z, y, -x)
        if angle == 2:
            return (-x, y, -z)
        return (-z, y, x)
    # z-axis
    if angle == 1:
        return (y, -x, z)
    if angle == 2:
        return (-x, -y, z)
    return (-y, x, z)


def _local_search(sequence, dimension, chain, rng, max_iterations=5000):
    """
    Local search combining segment-reversal moves and pivot-rotation moves,
    both using O(1) incremental energy deltas. Simulated-annealing acceptance.
    """
    n = len(chain)
    if n < 5:
        return chain

    chain = [tuple(p) for p in chain]
    occ = set(chain)
    pos_idx = {p: i for i, p in enumerate(chain)}

    current_energy = _energy_full(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy

    T = 1.5
    T_min = 0.004
    alpha = 0.9997

    for it in range(max_iterations):
        if _time_left() < 4.0:
            break

        if rng.random() < 0.5:
            # ---- segment reversal ----
            max_seg = min(n - 2, 14)
            if max_seg < 2:
                continue
            r = rng.random()
            if r < 0.5:
                seg_len = rng.randint(2, 4)
            elif r < 0.8:
                seg_len = rng.randint(4, 8)
            else:
                seg_len = rng.randint(8, max_seg)
            seg_len = min(seg_len, max_seg)
            if seg_len < 2:
                continue
            s = rng.randint(1, n - seg_len - 1)
            e = s + seg_len - 1

            # reverse the segment in-place and track moved positions
            moved = []
            for k in range(seg_len // 2):
                a = s + k
                b = e - k
                pa, pb = chain[a], chain[b]
                moved.append((a, pa))
                moved.append((b, pb))
                chain[a], chain[b] = pb, pa
                occ.discard(pa)
                occ.discard(pb)
                occ.add(pa)
                occ.add(pb)
                pos_idx[pa] = a
                pos_idx[pb] = b

            # check boundary unit-steps + self-avoidance of moved positions
            valid = True
            # boundary at s-1 <-> s (now holds old chain[e])
            if s > 0:
                p1 = chain[s - 1]
                p2 = chain[s]
                if abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) + abs(p1[2]-p2[2]) != 1:
                    valid = False
            # boundary at e <-> e+1 (now holds old chain[s])
            if e < n - 1:
                p1 = chain[e]
                p2 = chain[e + 1]
                if abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) + abs(p1[2]-p2[2]) != 1:
                    valid = False
            # self-avoidance: no two moved positions equal
            seen = set()
            for a, p in moved:
                if p in seen:
                    valid = False
                    break
                seen.add(p)

            if valid:
                delta = 0
                for a, _ in moved:
                    if sequence[a] == 'H':
                        p = chain[a]
                        for dx, dy, dz in _OFFSETS:
                            q = (p[0]+dx, p[1]+dy, p[2]+dz)
                            j = pos_idx.get(q)
                            if j is not None and abs(a - j) > 1 and sequence[j] == 'H':
                                delta -= 1
                new_energy = current_energy + delta
                if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = chain[:]
                else:
                    # revert
                    for a, p in moved:
                        chain[a] = p
                        pos_idx[p] = a
            else:
                # revert
                for a, p in moved:
                    chain[a] = p
                    pos_idx[p] = a
        else:
            # ---- pivot rotation of subchain [pivot+1:] ----
            pivot = rng.randint(1, n - 2)
            axis = rng.randint(0, 2)
            angle = rng.choice([1, 2, 3])

            sub = chain[pivot+1:]
            if not sub:
                continue
            new_sub = [_rot(p, axis, angle) for p in sub]

            old_set = set(sub)
            new_set = set(new_sub)
            # self-avoidance: new positions must not collide with the fixed part
            conflict = False
            for p in new_set:
                if p in occ and p not in old_set:
                    conflict = True
                    break
            # pivot connection must be a unit step
            pp = chain[pivot]
            f = new_sub[0]
            if abs(pp[0]-f[0]) + abs(pp[1]-f[1]) + abs(pp[2]-f[2]) != 1:
                conflict = True

            if conflict:
                continue

            # apply
            for p in old_set:
                occ.discard(p)
                pos_idx.pop(p, None)
            for k, p in enumerate(new_sub):
                idx = pivot + 1 + k
                occ.add(p)
                pos_idx[p] = idx
                chain[idx] = p

            delta = 0
            for k, p in enumerate(new_sub):
                idx = pivot + 1 + k
                if sequence[idx] == 'H':
                    for dx, dy, dz in _OFFSETS:
                        q = (p[0]+dx, p[1]+dy, p[2]+dz)
                        j = pos_idx.get(q)
                        if j is not None and abs(idx - j) > 1 and sequence[j] == 'H':
                            delta -= 1
            new_energy = current_energy + delta
            if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain[:]
            else:
                # revert
                for k, p in enumerate(sub):
                    idx = pivot + 1 + k
                    occ.discard(new_sub[k])
                    occ.add(p)
                    pos_idx[new_sub[k]] = idx if False else pos_idx.pop(new_sub[k], None)
                    pos_idx[p] = idx
                    chain[idx] = p

        T = max(T * alpha, T_min)

    return best_chain


def _quantum_tunneling(sequence, dimension, chain, rng, max_jumps=40):
    """
    Basin jumping: large-scale perturbations (big reversal + several rotations)
    followed by a short local-search recovery to escape deep local minima.
    """
    n = len(chain)
    if n < 10:
        return chain

    best_chain = list(chain)
    best_energy = _energy_full(sequence, chain)

    for _ in range(max_jumps):
        if _time_left() < 12.0:
            break

        cand = [tuple(p) for p in chain]

        # large reversal
        if n > 12:
            max_seg = min(n - 2, max(10, n // 3))
            for _try in range(6):
                seg_len = rng.randint(max(5, n // 4), max_seg)
                s = rng.randint(1, n - seg_len - 1)
                e = s + seg_len - 1
                cand[s:e+1] = cand[s:e+1][::-1]
                # quick self-avoidance check
                if len(set(cand)) == n:
                    break
                else:
                    cand[s:e+1] = cand[s:e+1][::-1]  # revert

        # a few extra rotations to diversify
        for _ in range(rng.randint(2, 4)):
            pivot = rng.randint(1, n - 2)
            axis = rng.randint(0, 2)
            angle = rng.choice([1, 2, 3])
            sub = cand[pivot+1:]
            new_sub = [_rot(p, axis, angle) for p in sub]
            old_set = set(sub)
            occ = set(cand)
            new_set = set(new_sub)
            ok = all((p not in occ or p in old_set) for p in new_set)
            pp = cand[pivot]
            f = new_sub[0]
            if ok and abs(pp[0]-f[0]) + abs(pp[1]-f[1]) + abs(pp[2]-f[2]) == 1:
                cand[pivot+1:] = new_sub

        local_rng = random.Random(rng.randint(0, 2**31))
        improved = _local_search(sequence, dimension, cand, local_rng, max_iterations=300)
        improved_energy = _energy_full(sequence, improved)
        if improved_energy < best_energy:
            best_chain = improved
            best_energy = improved_energy

    return best_chain


def fold(sequence, dimension):
    """
    Fold a protein sequence on the cubic lattice.
    Returns a list of len(sequence) integer lattice points (lists of length dimension).
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]

    best_chain = None
    best_energy = 0

    seq_start = time.time()
    # Time budget: shorter for small chains, longer for large ones
    time_budget = min(110.0, max(12.0, n * 0.7))

    # Phase 1: Beam search with multiple configurations for diversity
    beam_configs = [
        (200, 12345),
        (300, 54321),
        (400, 98765),
        (250, 11111),
        (350, 22222),
        (500, 33333),
    ]

    for beam_width, seed in beam_configs:
        if time.time() - seq_start > time_budget * 0.45:
            break
        if _time_left() < 18.0:
            break

        random.seed(seed)
        chain = _beam_search(sequence, dimension, beam_width=beam_width)
        if chain is None or len(chain) != n:
            continue

        energy = _energy_full(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy

    # Phase 2: Local search refinement (incremental, reversal + pivot)
    if best_chain is not None and _time_left() > 18.0:
        for ls_seed in [42, 123, 456, 789]:
            if _time_left() < 12.0:
                break
            rng = random.Random(ls_seed)
            improved = _local_search(sequence, dimension, best_chain, rng, max_iterations=4000)
            improved_energy = _energy_full(sequence, improved)
            if improved_energy < best_energy:
                best_chain = improved
                best_energy = improved_energy

    # Phase 3: Quantum-inspired tunneling for basin jumping
    if best_chain is not None and _time_left() > 22.0:
        tunnel_rng = random.Random(999)
        tunneled = _quantum_tunneling(sequence, dimension, best_chain, tunnel_rng, max_jumps=35)
        tunneled_energy = _energy_full(sequence, tunneled)
        if tunneled_energy < best_energy:
            best_chain = tunneled
            best_energy = tunneled_energy

    # Phase 4: Final polish if time remains
    if best_chain is not None and _time_left() > 15.0:
        polish_rng = random.Random(7777)
        polished = _local_search(sequence, dimension, best_chain, polish_rng, max_iterations=6000)
        polished_energy = _energy_full(sequence, polished)
        if polished_energy < best_energy:
            best_chain = polished
            best_energy = polished_energy

    # Fallback: straight line if no valid chain found
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
        best_energy = _energy_full(sequence, best_chain)

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
