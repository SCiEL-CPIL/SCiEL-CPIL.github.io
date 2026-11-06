"""HP lattice protein folding: beam search with simulated-annealing refinement."""
# EVOLVE-BLOCK-START
import time
import random
import math

_T0 = time.time()
_DEADLINE = 195.0  # 210s wall-clock limit, 15s safety margin


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _energy_full(seq, chain):
    """Compute energy: -number of H-H non-bonded contacts."""
    n = len(chain)
    if n < 3:
        return 0
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    h_set = set(i for i, c in enumerate(seq) if c == "H")
    contacts = 0
    for i in h_set:
        x, y = chain[i]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = pos_to_idx.get((x + dx, y + dy))
            if j is not None and j > i + 1 and seq[j] == "H":
                contacts += 1
    return -contacts


def _beam_search(seq, beam_width=50):
    """Beam search to find a good initial conformation."""
    n = len(seq)
    if n <= 2:
        return [(0, 0), (1, 0)] if n == 2 else [(0, 0)]

    is_h = [c == "H" for c in seq]
    beams = [(-1, [(0, 0)], {(0, 0): 0})]

    for i in range(1, n):
        new_beams = []
        for energy, chain, occupied in beams:
            last = chain[-1]
            x, y = last
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                new_pos = (x + dx, y + dy)
                if new_pos in occupied:
                    continue
                new_chain = chain + [new_pos]
                new_occupied = occupied.copy()
                new_occupied[new_pos] = i

                new_energy = energy
                if is_h[i]:
                    for dx2, dy2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        j = new_occupied.get((new_pos[0] + dx2, new_pos[1] + dy2))
                        if j is not None and j < i - 1 and is_h[j]:
                            new_energy -= 1

                new_beams.append((new_energy, new_chain, new_occupied))

        if not new_beams:
            return beams[0][1]

        new_beams.sort(key=lambda x: x[0])
        beams = new_beams[:beam_width]

    return beams[0][1]


def _count_contacts(seq, chain):
    """Fast contact count for a chain (positive number of H-H contacts)."""
    n = len(chain)
    if n < 3:
        return 0
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    is_h = [c == "H" for c in seq]
    contacts = 0
    for i in range(n):
        if not is_h[i]:
            continue
        x, y = chain[i]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = pos_to_idx.get((x + dx, y + dy))
            if j is not None and j > i + 1 and is_h[j]:
                contacts += 1
    return contacts


def _kink_candidates(chain, i, occupied):
    """Yield valid new positions for a kink move at index i (interior)."""
    n = len(chain)
    if i < 1 or i > n - 2:
        return
    prev = chain[i - 1]
    nxt = chain[i + 1]
    curr = chain[i]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        new_pos = (prev[0] + dx, prev[1] + dy)
        if new_pos == curr:
            continue
        # Must be adjacent to next
        if abs(new_pos[0] - nxt[0]) + abs(new_pos[1] - nxt[1]) != 1:
            continue
        if new_pos in occupied:
            continue
        yield new_pos


def _simulated_anneal(seq, chain, max_steps=2000, rng=None):
    """Simulated annealing with kink moves and segment relocation moves."""
    if rng is None:
        rng = random.Random()
    n = len(chain)
    if n < 3:
        return chain, _energy_full(seq, chain)

    is_h = [c == "H" for c in seq]
    chain = list(chain)
    occupied = {p: i for i, p in enumerate(chain)}
    current_contacts = _count_contacts(seq, chain)
    current_energy = -current_contacts

    start_temp = 3.0
    end_temp = 0.05

    for step in range(max_steps):
        if _remaining() < 8.0:
            break
        progress = step / max(1, max_steps)
        temp = start_temp * (end_temp / start_temp) ** progress

        move_type = rng.random()

        if move_type < 0.7:
            # Kink move
            i = rng.randint(1, n - 2)
            candidates = list(_kink_candidates(chain, i, occupied))
            if not candidates:
                continue
            new_pos = rng.choice(candidates)

            # Apply move
            old_pos = chain[i]
            del occupied[old_pos]
            occupied[new_pos] = i
            chain[i] = new_pos

            new_contacts = _count_contacts(seq, chain)
            new_energy = -new_contacts
            delta = new_energy - current_energy

            if delta < 0 or (temp > 0.01 and rng.random() < math.exp(-delta / temp)):
                current_energy = new_energy
                current_contacts = new_contacts
            else:
                # Revert
                chain[i] = old_pos
                del occupied[new_pos]
                occupied[old_pos] = i

        else:
            # Segment relocation move (length 2 or 3)
            seg_len = rng.choice([2, 3])
            if n < seg_len + 2:
                continue
            start_idx = rng.randint(1, n - seg_len - 1)
            end_idx = start_idx + seg_len - 1

            anchor_start = chain[start_idx - 1]
            anchor_end = chain[end_idx + 1]

            old_seg = [chain[i] for i in range(start_idx, end_idx + 1)]

            # Try to find a valid new placement via random walk
            found = False
            for _attempt in range(30):
                pos = anchor_start
                path = [pos]
                valid = True
                for _ in range(seg_len):
                    neighbors = [(pos[0] + dx, pos[1] + dy) for dx, dy in
                                 ((1, 0), (-1, 0), (0, 1), (0, -1))]
                    pos = rng.choice(neighbors)
                    path.append(pos)

                if path[-1] != anchor_end:
                    continue

                new_seg = path[1:]

                # Check self-intersection of new segment
                if len(set(new_seg)) != len(new_seg):
                    continue

                # Check collision with non-segment residues
                collision = False
                for p in new_seg:
                    if p in occupied:
                        idx = occupied[p]
                        if not (start_idx <= idx <= end_idx):
                            collision = True
                            break
                if collision:
                    continue

                # Apply move
                for idx in range(start_idx, end_idx + 1):
                    del occupied[chain[idx]]
                for k, idx in enumerate(range(start_idx, end_idx + 1)):
                    new_pos = new_seg[k]
                    occupied[new_pos] = idx
                    chain[idx] = new_pos

                new_contacts = _count_contacts(seq, chain)
                new_energy = -new_contacts
                delta = new_energy - current_energy

                if delta < 0 or (temp > 0.01 and rng.random() < math.exp(-delta / temp)):
                    current_energy = new_energy
                    current_contacts = new_contacts
                    found = True
                else:
                    # Revert
                    for idx in range(start_idx, end_idx + 1):
                        del occupied[chain[idx]]
                    for idx, old_p in zip(range(start_idx, end_idx + 1), old_seg):
                        occupied[old_p] = idx
                        chain[idx] = old_p
                break  # Only try one placement per segment move

    return chain, current_energy


def _polish(seq, chain, max_rounds=200):
    """Deterministic polish: exhaustive kink + small segment moves until no improvement."""
    n = len(chain)
    if n < 3:
        return chain, _energy_full(seq, chain)

    is_h = [c == "H" for c in seq]
    chain = list(chain)
    occupied = {p: i for i, p in enumerate(chain)}
    best_energy = -_count_contacts(seq, chain)

    for _round in range(max_rounds):
        if _remaining() < 6.0:
            break
        improved = False

        # Exhaustive kink moves
        for i in range(1, n - 1):
            for new_pos in _kink_candidates(chain, i, occupied):
                old_pos = chain[i]
                del occupied[old_pos]
                occupied[new_pos] = i
                chain[i] = new_pos

                new_energy = -_count_contacts(seq, chain)
                if new_energy < best_energy:
                    best_energy = new_energy
                    improved = True
                else:
                    # Revert
                    chain[i] = old_pos
                    del occupied[new_pos]
                    occupied[old_pos] = i

        # Small segment moves (length 2)
        for start_idx in range(1, n - 3):
            end_idx = start_idx + 1
            anchor_start = chain[start_idx - 1]
            anchor_end = chain[end_idx + 1]
            old_seg = [chain[start_idx], chain[end_idx]]

            # Enumerate all length-2 paths from anchor_start to anchor_end
            for dx1, dy1 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                mid = (anchor_start[0] + dx1, anchor_start[1] + dy1)
                if mid == anchor_start:
                    continue
                if abs(mid[0] - anchor_end[0]) + abs(mid[1] - anchor_end[1]) != 1:
                    continue
                new_seg = [mid, anchor_end]
                # Wait, new_seg should be the 2 positions for indices start_idx, end_idx
                # path: anchor_start -> mid -> anchor_end, so new_seg = [mid, anchor_end]? 
                # No: the segment is residues at start_idx and end_idx.
                # Path is anchor_start (start_idx-1), then start_idx, end_idx, then anchor_end (end_idx+1).
                # So new_seg = [pos_start, pos_end] where pos_start adjacent to anchor_start,
                # pos_end adjacent to pos_start and to anchor_end.
                # Let me re-derive: we need pos_a (for start_idx) and pos_b (for end_idx)
                # such that pos_a adjacent to anchor_start, pos_b adjacent to pos_a and anchor_end.
                pass

            # Simpler: try all possible (pos_a, pos_b) pairs
            for pos_a in [(anchor_start[0] + dx, anchor_start[1] + dy) for dx, dy in
                          ((1, 0), (-1, 0), (0, 1), (0, -1))]:
                if pos_a == anchor_start:
                    continue
                for pos_b in [(pos_a[0] + dx, pos_a[1] + dy) for dx, dy in
                              ((1, 0), (-1, 0), (0, 1), (0, -1))]:
                    if pos_b == pos_a:
                        continue
                    if abs(pos_b[0] - anchor_end[0]) + abs(pos_b[1] - anchor_end[1]) != 1:
                        continue
                    new_seg = [pos_a, pos_b]
                    if len(set(new_seg)) != 2:
                        continue
                    collision = False
                    for p in new_seg:
                        if p in occupied:
                            idx = occupied[p]
                            if not (start_idx <= idx <= end_idx):
                                collision = True
                                break
                    if collision:
                        continue

                    for idx in (start_idx, end_idx):
                        del occupied[chain[idx]]
                    for k, idx in enumerate((start_idx, end_idx)):
                        occupied[new_seg[k]] = idx
                        chain[idx] = new_seg[k]

                    new_energy = -_count_contacts(seq, chain)
                    if new_energy < best_energy:
                        best_energy = new_energy
                        improved = True
                    else:
                        for idx in (start_idx, end_idx):
                            del occupied[chain[idx]]
                        for idx, old_p in zip((start_idx, end_idx), old_seg):
                            occupied[old_p] = idx
                            chain[idx] = old_p

        if not improved:
            break

    return chain, best_energy


def fold(sequence, dimension):
    """Fold a protein sequence into a low-energy conformation."""
    n = len(sequence)
    if n <= 2:
        return [[i, 0] for i in range(n)]

    best_chain = None
    best_energy = 0

    # Generate initial conformations via beam search with different widths
    initial_chains = []
    for beam_width in [30, 60, 100]:
        if _remaining() < 15.0:
            break
        chain = _beam_search(sequence, beam_width)
        if chain is not None and len(chain) == n:
            initial_chains.append(chain)

    if not initial_chains:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
        return [list(p) for p in best_chain]

    # Refine each initial conformation with simulated annealing + polish
    rng = random.Random(42)
    for chain in initial_chains:
        if _remaining() < 12.0:
            break

        # Simulated annealing
        refined, energy = _simulated_anneal(sequence, chain, max_steps=1500, rng=rng)

        # Deterministic polish
        polished, energy = _polish(sequence, refined, max_rounds=100)

        if best_chain is None or energy < best_energy:
            best_chain = polished
            best_energy = energy

    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
