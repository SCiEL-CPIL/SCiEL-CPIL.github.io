"""Combined program: High-volume greedy restarts with optional fast hill-climb for short chains."""
# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE = 315.0  # safety margin before 330s wall-clock kill
_START_TIME = None

def _get_deadline():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME + _DEADLINE

def _time_left():
    return _get_deadline() - time.time()

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _lattice_adjacent(p1, p2):
    return sum(abs(a - b) for a, b in zip(p1, p2)) == 1

def _grow(sequence, dimension, rng):
    """Randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
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
            # Weighting: Prioritize contacts heavily, then space, then random
            options.append((gain * 10.0 + free * 0.1 + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

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

def _compute_contacts(sequence, chain):
    """Return set of (i,j) H-H contact pairs (i<j, not chain-adjacent)."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    contacts = set()
    for i, p in enumerate(chain):
        if sequence[i] != "H":
            continue
        for q in _neighbours(p):
            j = where.get(q)
            if j is not None and j > i + 1 and sequence[j] == "H":
                contacts.add((i, j))
    return contacts

class _FoldingState:
    """State for incremental local search with optimized contact tracking."""
    def __init__(self, sequence, chain):
        self.seq = sequence
        self.n = len(chain)
        self.chain = list(chain)
        self.occupied = {p: i for i, p in enumerate(chain)}
        self.is_h = [c == 'H' for c in sequence]
        self.contacts = _compute_contacts(sequence, chain)
        self.contact_map = {i: set() for i in range(self.n)}
        for i, j in self.contacts:
            self.contact_map[i].add(j)
            self.contact_map[j].add(i)
        self.energy = -len(self.contacts)
        self.best_chain = list(chain)
        self.best_energy = self.energy

    def _accept_if_better(self):
        if self.energy < self.best_energy:
            self.best_energy = self.energy
            self.best_chain = list(self.chain)

    def _remove_contacts_involving(self, indices):
        indices_set = set(indices)
        for i in indices_set:
            partners = list(self.contact_map[i])
            for j in partners:
                pair = (i, j) if i < j else (j, i)
                if pair in self.contacts:
                    self.contacts.discard(pair)
                    self.energy += 1
                    self.contact_map[i].discard(j)
                    self.contact_map[j].discard(i)

    def _add_new_contacts(self, indices, new_positions):
        indices_set = set(indices)
        for i in indices_set:
            if not self.is_h[i]:
                continue
            pos = new_positions[i]
            for q in _neighbours(pos):
                j = self.occupied.get(q)
                if j is None or j == i:
                    continue
                if abs(j - i) == 1:
                    continue
                if self.is_h[j]:
                    pair = (i, j) if i < j else (j, i)
                    if pair not in self.contacts:
                        self.contacts.add(pair)
                        self.energy -= 1
                        self.contact_map[i].add(j)
                        self.contact_map[j].add(i)

    def _update_energy_for_move(self, moved_indices, new_positions):
        self._remove_contacts_involving(moved_indices)
        self._add_new_contacts(moved_indices, new_positions)

    def _pivot_move_translate(self, rng):
        """Translation-based pivot: shift subchain by changing first step direction."""
        n = self.n
        if n < 4:
            return False
        pivot_idx = rng.randint(1, n - 3)
        orig_dir = tuple(self.chain[pivot_idx + 1][k] - self.chain[pivot_idx][k] for k in range(3))

        possible_dirs = []
        for ax in range(3):
            for s in (-1, 1):
                d = [0, 0, 0]
                d[ax] = s
                d = tuple(d)
                if d != orig_dir and d != tuple(-x for x in orig_dir):
                    possible_dirs.append(d)

        if not possible_dirs:
            return False

        new_dir = possible_dirs[rng.randint(0, len(possible_dirs) - 1)]
        subchain_len = n - pivot_idx - 1
        subchain_old = list(self.chain[pivot_idx + 1:])

        new_subchain = [None] * subchain_len
        new_subchain[0] = tuple(self.chain[pivot_idx][k] + new_dir[k] for k in range(3))

        if new_subchain[0] in self.occupied:
            return False

        valid = True
        for k in range(1, subchain_len):
            old_step = tuple(subchain_old[k][m] - subchain_old[k-1][m] for m in range(3))
            new_pos = tuple(new_subchain[k-1][m] + old_step[m] for m in range(3))
            if new_pos in self.occupied:
                valid = False
                break
            new_subchain[k] = new_pos

        if not valid:
            return False

        seen = set()
        for pos in new_subchain:
            if pos in seen:
                return False
            seen.add(pos)

        if pivot_idx + subchain_len < n:
            if not _lattice_adjacent(new_subchain[-1], self.chain[pivot_idx + subchain_len]):
                return False

        moved_indices = list(range(pivot_idx + 1, n))
        
        for pos in subchain_old:
            del self.occupied[pos]
        for k, pos in enumerate(new_subchain):
            self.chain[pivot_idx + 1 + k] = pos
            self.occupied[pos] = pivot_idx + 1 + k

        new_positions = {i: self.chain[i] for i in moved_indices}
        self._update_energy_for_move(moved_indices, new_positions)
        self._accept_if_better()
        return True

def _fast_hill_climb(sequence, chain, dimension, rng, max_iters=20):
    """Very fast hill climbing: accept only improvements, short duration."""
    if chain is None:
        return None
    n = len(chain)
    if n < 4:
        return chain

    state = _FoldingState(sequence, chain)

    for it in range(max_iters):
        if _time_left() < 0.5:
            break
        state._pivot_move_translate(rng)

    return state.best_chain

def fold(sequence, dimension):
    """High-volume greedy restarts with optional fast hill-climb for short chains."""
    n = len(sequence)
    rng = random.Random(12345 + n)
    best, best_energy = None, 0  # 0 is neutral, we want negative energy (lower is better)
    # Actually, energy is negative. So we want to minimize energy (make it more negative).
    # best_energy should start at a high value (e.g., 0) to be beaten by negative values.
    
    # Determine if we can afford hill climbing
    # For n > 60, hill climbing is too slow for high volume.
    use_hc = n <= 60
    hc_iters = 10 if n <= 40 else 5

    while _time_left() > 2.0:
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue

        value = _energy(sequence, chain)
        
        if use_hc and _time_left() > 1.0:
            # Try to improve slightly
            improved_chain = _fast_hill_climb(sequence, chain, dimension, rng, max_iters=hc_iters)
            if improved_chain is not None:
                improved_val = _energy(sequence, improved_chain)
                if improved_val < value:
                    chain = improved_chain
                    value = improved_val

        if value < best_energy:
            best, best_energy = chain, value

    if best is None:
        # Fallback: linear chain
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]

    return [list(p) for p in best]
# EVOLVE-BLOCK-END
