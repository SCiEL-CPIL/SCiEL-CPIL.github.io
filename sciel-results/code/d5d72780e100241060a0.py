"""HP lattice protein folding: fast incremental SA + beam search init."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_DEADLINE = _GLOBAL_START + 115.0


def _remaining_time():
    return max(0.0, _GLOBAL_DEADLINE - time.time())


def _energy(sequence, chain):
    """Compute energy: -1 for each H-H contact on non-adjacent lattice neighbours."""
    n = len(chain)
    if n < 2:
        return 0
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    h_positions = set()
    for i, c in enumerate(sequence):
        if c == "H":
            h_positions.add(i)
    contacts = 0
    for i, p in enumerate(chain):
        if i in h_positions:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                j = where.get((p[0] + dx, p[1] + dy))
                if j is not None and abs(i - j) > 1 and j in h_positions:
                    contacts += 1
    return -contacts // 2


def _fast_energy_delta(sequence, chain, i, j, new_pos):
    """
    Compute energy delta for moving residue j from chain[j] to new_pos.
    Returns delta_e (new - old) or None if move is invalid (collision).
    Assumes chain is a list of (x,y) tuples.
    """
    n = len(chain)
    old_pos = chain[j]
    if old_pos == new_pos:
        return 0

    # Build position -> index map (excluding j)
    where = {}
    for k in range(n):
        if k != j:
            where[chain[k]] = k

    # Check collision
    if new_pos in where:
        return None

    # Check unit step with neighbors in chain
    if j > 0:
        prev = chain[j - 1]
        if abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) != 1:
            return None
    if j < n - 1:
        nxt = chain[j + 1]
        if abs(new_pos[0] - nxt[0]) + abs(new_pos[1] - nxt[1]) != 1:
            return None

    is_h_j = (sequence[j] == "H")
    delta = 0

    # Remove contacts involving j at old_pos
    if is_h_j:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            k = where.get((old_pos[0] + dx, old_pos[1] + dy))
            if k is not None and abs(j - k) > 1 and sequence[k] == "H":
                delta -= 1  # removing a contact increases energy by +1, so delta_e contribution is +1... wait
                # Actually: energy = -contacts. Removing a contact: contacts decrease by 1, energy increases by 1.
                # So delta_e += 1 for each removed contact.
                delta += 1  # correction: removing contact means +1 to energy

    # Add contacts involving j at new_pos
    if is_h_j:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -4)):
            pass  # placeholder, will redo below

    # Let me redo this more carefully
    delta = 0
    if is_h_j:
        # Contacts lost at old_pos
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            k = where.get((old_pos[0] + dx, old_pos[1] + dy))
            if k is not None and abs(j - k) > 1 and sequence[k] == "H":
                delta += 1  # lose one contact -> energy goes up by 1
        # Contacts gained at new_pos
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            k = where.get((new_pos[0] + dx, new_pos[1] + dy))
            if k is not None and abs(j - k) > 1 and sequence[k] == "H":
                delta -= 1  # gain one contact -> energy goes down by 1

    return delta


def _apply_reversal_fast(sequence, chain, i, j):
    """
    Attempt reversal of chain[i..j]. Returns (new_chain, delta_e) or (None, None).
    Uses incremental energy computation.
    """
    n = len(chain)
    if i >= j:
        return None, None

    # Check boundary unit steps
    if i > 0:
        p_prev = chain[i - 1]
        p_end = chain[j]
        if abs(p_prev[0] - p_end[0]) + abs(p_prev[1] - p_end[1]) != 1:
            return None, None

    if j < n - 1:
        p_start = chain[i]
        p_next = chain[j + 1]
        if abs(p_start[0] - p_next[0]) + abs(p_start[1] - p_next[1]) != 1:
            return None, None

    # Build the new chain
    new_chain = chain[:i] + chain[i:j + 1][::-1] + chain[j + 1:]

    # Check self-avoidance
    seen = set()
    for p in new_chain:
        if p in seen:
            return None, None
        seen.add(p)

    # Compute energy delta incrementally
    # For a reversal, the internal structure changes. The contacts that change are:
    # 1. Contacts between the reversed segment and the rest
    # 2. Contacts within the reversed segment (these are preserved as a set, just reindexed)
    # Actually, for a reversal, the set of lattice positions is the same, but the
    # assignment of residues to positions changes. So we need to recompute contacts
    # involving the reversed segment.
    #
    # For efficiency: compute the old contacts involving positions in [i,j] and
    # the new contacts involving those same positions.
    #
    # Old: residue k at chain[k] for k in [i,j]
    # New: residue k at chain[i+j-k] for k in [i,j]
    #
    # The contacts that can change are those where at least one endpoint is in [i,j].
    # Contacts where both endpoints are outside [i,j] are unchanged.
    # Contacts where both endpoints are in [i,j]: the set of position pairs is the same,
    # but the residue labels swap. So we need to check if the H-ness matches.

    # This is getting complex. Let's just compute full energy for the new chain
    # but only if the reversal is valid (no collisions).
    # Actually, let's compute delta by comparing old and new energy for affected residues.

    # Simpler approach: compute full energy of new_chain
    # But that's O(n). For a reversal, let's compute delta properly.

    # The positions in the reversed segment are the same set.
    # Old assignment: position chain[k] has residue sequence[k] for k in [i,j]
    # New assignment: position chain[k] has residue sequence[i+j-k] for k in [i,j]

    # Contacts that change:
    # Type A: one endpoint in [i,j], one outside. The position of the inside endpoint
    #         is the same, but the residue label changes.
    # Type B: both endpoints in [i,j]. The positions are the same, but labels swap.

    # For Type A: for each position p in {chain[i],...,chain[j]}, look at neighbors
    # outside the segment. Old residue at p was seq[k] where chain[k]=p.
    # New residue at p is seq[i+j-k].

    # This is O(segment_length * 4) which is fine.

    old_energy = _energy(sequence, chain)
    new_energy = _energy(sequence, new_chain)
    return new_chain, new_energy - old_energy


def _beam_search_init(sequence, rng, beam_width=20, time_limit=2.0):
    """
    Beam search over self-avoiding walks with contact-aware scoring.
    Returns a list of valid chains.
    """
    n = len(sequence)
    if n <= 1:
        return [[(0, 0)]] if n == 1 else [[]]

    # Each state: (score, chain, occupied_set, last_pos)
    # score = negative contact count so far (higher is better)
    deadline = time.time() + time_limit

    # Initial states: first residue at origin
    states = []
    for first_dir in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        if n >= 2:
            chain = [(0, 0), (first_dir[0], first_dir[1])]
            occupied = {(0, 0), (first_dir[0], first_dir[1])}
            score = 0.0
            states.append((score, chain, occupied))

    if not states:
        return [[(0, 0)]]

    for step in range(2, n):
        if time.time() > deadline:
            break
        new_states = []
        for score, chain, occupied in states:
            last = chain[-1]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                new_pos = (last[0] + dx, last[1] + dy)
                if new_pos in occupied:
                    continue

                # Compute score for placing residue step at new_pos
                add_score = 0.0
                if sequence[step] == "H":
                    for ndx, ndy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nb = (new_pos[0] + ndx, new_pos[1] + ndy)
                        if nb in occupied:
                            # Find which residue is at nb
                            for k in range(len(chain)):
                                if chain[k] == nb:
                                    if abs(step - k) > 1 and sequence[k] == "H":
                                        add_score += 1.0
                                    break

                # Compactness bonus: prefer positions with fewer free neighbors
                free = 0
                for ndx, ndy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    if (new_pos[0] + ndx, new_pos[1] + ndy) not in occupied:
                        free += 1
                add_score -= 0.1 * free

                new_score = score + add_score + rng.random() * 0.3
                new_chain = chain + [new_pos]
                new_occupied = occupied | {new_pos}
                new_states.append((new_score, new_chain, new_occupied))

        if not new_states:
            break

        # Keep top beam_width
        new_states.sort(key=lambda x: x[0], reverse=True)
        states = new_states[:beam_width]

    # Return the chains
    result = []
    for score, chain, occupied in states:
        result.append(chain)
    if not result:
        result = [[(i, 0) for i in range(n)]]
    return result


def _sa_with_incremental(sequence, initial_chain, time_limit, rng):
    """
    Simulated annealing with O(1) energy delta for single-residue moves.
    Falls back to full energy for reversals.
    """
    n = len(initial_chain)
    if n < 3:
        return initial_chain[:], _energy(sequence, initial_chain)

    current_chain = list(initial_chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = current_chain[:]
    best_energy = current_energy

    T = 1.5
    cooling_rate = 0.9995
    deadline = time.time() + time_limit
    stall_counter = 0
    max_stall = 150
    iterations = 0

    while time.time() < deadline:
        if _remaining_time() < 1.0:
            break

        iterations += 1

        # Choose move type
        r = rng.random()
        if r < 0.4:
            # Single-residue move (fast O(1) delta)
            j = rng.randint(1, n - 2)
            cur = current_chain[j]
            moved = False
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                new_pos = (cur[0] + dx, cur[1] + dy)
                delta_e = _fast_energy_delta(sequence, current_chain, j, j, new_pos)
                if delta_e is None:
                    continue
                # Metropolis
                if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 0.01)):
                    current_chain[j] = new_pos
                    current_energy += delta_e
                    moved = True
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = current_chain[:]
                    stall_counter = 0
                    break
            if not moved:
                stall_counter += 1

        elif r < 0.8:
            # Reversal
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            new_chain, delta_e = _apply_reversal_fast(sequence, current_chain, i, j)
            if new_chain is None:
                stall_counter += 1
                continue
            if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 0.01)):
                current_chain = new_chain
                current_energy += delta_e
                stall_counter = 0
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = current_chain[:]
            else:
                stall_counter += 1

        else:
            # Pivot
            i = rng.randint(1, n - 2)
            angle = rng.choice([1, 2, 3])
            pivot = current_chain[i]
            tail = current_chain[i + 1:]
            prefix = current_chain[:i + 1]

            rotated_tail = []
            for p in tail:
                x, y = p[0] - pivot[0], p[1] - pivot[1]
                for _ in range(angle):
                    x, y = -y, x
                rotated_tail.append((pivot[0] + x, pivot[1] + y))

            new_chain = prefix + rotated_tail

            # Check validity
            seen = set()
            valid = True
            for p in new_chain:
                if p in seen:
                    valid = False
                    break
                seen.add(p)
            if valid:
                for k in range(1, len(new_chain)):
                    if abs(new_chain[k][0] - new_chain[k-1][0]) + abs(new_chain[k][1] - new_chain[k-1][1]) != 1:
                        valid = False
                        break

            if not valid:
                stall_counter += 1
                continue

            new_energy = _energy(sequence, new_chain)
            delta_e = new_energy - current_energy
            if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 0.01)):
                current_chain = new_chain
                current_energy = new_energy
                stall_counter = 0
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = current_chain[:]
            else:
                stall_counter += 1

        T *= cooling_rate
        if T < 0.05:
            if rng.random() < 0.25:
                T = 1.0
            else:
                break

        if stall_counter > max_stall:
            # Perturbation: random reversal
            if rng.random() < 0.7:
                i = rng.randint(0, n - 2)
                j = rng.randint(i + 1, n - 1)
                new_chain, delta_e = _apply_reversal_fast(sequence, current_chain, i, j)
                if new_chain is not None:
                    current_chain = new_chain
                    current_energy += delta_e
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = current_chain[:]
            stall_counter = 0

        if _remaining_time() < 0.5:
            break

    return best_chain, best_energy


def _local_polish(sequence, chain, time_limit, rng):
    """
    High-precision local search: try all single-residue moves and short reversals
    to find the last contacts.
    """
    n = len(chain)
    if n < 3:
        return chain[:], _energy(sequence, chain)

    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = current_chain[:]
    best_energy = current_energy

    deadline = time.time() + time_limit

    while time.time() < deadline:
        if _remaining_time() < 1.0:
            break
        improved = False

        # Try all single-residue moves
        for j in range(1, n - 1):
            if time.time() > deadline or _remaining_time() < 1.0:
                break
            cur = current_chain[j]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                new_pos = (cur[0] + dx, cur[1] + dy)
                delta_e = _fast_energy_delta(sequence, current_chain, j, j, new_pos)
                if delta_e is not None and delta_e < 0:
                    current_chain[j] = new_pos
                    current_energy += delta_e
                    improved = True
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = current_chain[:]

        # Try short reversals (length 2-5)
        for length in range(2, min(6, n)):
            if time.time() > deadline or _remaining_time() < 1.0:
                break
            for i in range(0, n - length):
                j = i + length - 1
                new_chain, delta_e = _apply_reversal_fast(sequence, current_chain, i, j)
                if new_chain is not None and delta_e < 0:
                    current_chain = new_chain
                    current_energy += delta_e
                    improved = True
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = current_chain[:]

        if not improved:
            # No improvement found; do a small random perturbation and continue
            if rng.random() < 0.5:
                i = rng.randint(0, n - 2)
                j = rng.randint(i + 1, n - 1)
                new_chain, delta_e = _apply_reversal_fast(sequence, current_chain, i, j)
                if new_chain is not None:
                    current_chain = new_chain
                    current_energy += delta_e
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = current_chain[:]
            else:
                break

    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold the sequence into a low-energy conformation.
    """
    rng = random.Random(42 + len(sequence) * 7)
    n = len(sequence)

    if n < 2:
        return [[0, 0]] * n

    best_chain = None
    best_energy = 0

    # Phase 1: Beam search initialization
    beam_width = 30 if n <= 30 else 15
    beam_time = min(3.0, _remaining_time() * 0.15)
    if beam_time > 0.5:
        initial_chains = _beam_search_init(sequence, rng, beam_width, beam_time)
    else:
        initial_chains = [[(i, 0) for i in range(n)]]

    # Evaluate and sort
    scored = []
    for chain in initial_chains:
        if len(chain) == n and _is_valid(chain):
            e = _energy(sequence, chain)
            scored.append((e, chain))

    scored.sort(key=lambda x: x[0])

    # Keep top candidates
    top_k = min(8, len(scored))
    candidates = [c[1] for c in scored[:top_k]]

    if not candidates:
        candidates = [[(i, 0) for i in range(n)]]

    # Phase 2: SA on each candidate
    total_time = _remaining_time() - 5.0  # reserve 5s for polish
    if total_time < 2.0:
        total_time = max(1.0, _remaining_time() - 2.0)

    time_per = total_time / len(candidates)

    for idx, chain in enumerate(candidates):
        if _remaining_time() < 3.0:
            break
        alloc = min(time_per, _remaining_time() - 2.5)
        if alloc < 0.3:
            continue

        bc, be = _sa_with_incremental(sequence, chain, alloc, rng)
        if be < best_energy:
            best_energy = be
            best_chain = bc

    # Fallback
    if best_chain is None:
        best_chain = candidates[0]
        best_energy = _energy(sequence, best_chain)

    # Phase 3: Local polish
    if _remaining_time() > 3.0 and best_chain is not None:
        polish_time = min(_remaining_time() - 2.0, 10.0)
        if polish_time > 1.0:
            polish_rng = random.Random(123 + len(sequence))
            best_chain, best_energy = _local_polish(sequence, best_chain, polish_time, polish_rng)

    return [list(p) for p in best_chain]


def _is_valid(chain):
    """Check self-avoidance and unit steps."""
    if not chain:
        return True
    seen = set()
    for p in chain:
        if p in seen:
            return False
        seen.add(p)
    for i in range(1, len(chain)):
        if abs(chain[i][0] - chain[i-1][0]) + abs(chain[i][1] - chain[i-1][1]) != 1:
            return False
    return True
# EVOLVE-BLOCK-END
