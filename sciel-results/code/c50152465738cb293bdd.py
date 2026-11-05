"""Combined program: optimized contact tracking + proper simulated annealing."""
# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE = 320.0  # safety margin before 330s wall-clock kill
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
            options.append((gain + 0.1 * free + rng.random(), site))
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

def _rotation_matrices():
    """Return list of 24 rotation matrices for the cubic group (as 3x3 tuples)."""
    mats = []
    e = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    for x in e:
        for y in e:
            cx = x[1]*y[2] - x[2]*y[1]
            cy = x[2]*y[0] - x[0]*y[2]
            cz = x[0]*y[1] - x[1]*y[0]
            z = (cx, cy, cz)
            if sum(a*b for a,b in zip(x,y)) != 0:
                continue
            if sum(a*b for a,b in zip(x,z)) != 0:
                continue
            if sum(a*b for a,b in zip(y,z)) != 0:
                continue
            m = (x, y, z)
            if m not in mats:
                mats.append(m)
    return mats

_ROT_MATS = _rotation_matrices()

def _apply_rotation(mat, vec):
    r0, r1, r2 = mat
    return (
        r0[0]*vec[0] + r0[1]*vec[1] + r0[2]*vec[2],
        r1[0]*vec[0] + r1[1]*vec[1] + r1[2]*vec[2],
        r2[0]*vec[0] + r2[1]*vec[1] + r2[2]*vec[2],
    )

class _FoldingState:
    """State for incremental local search with optimized contact tracking."""
    def __init__(self, sequence, chain):
        self.seq = sequence
        self.n = len(chain)
        self.chain = list(chain)
        self.occupied = {p: i for i, p in enumerate(chain)}
        self.is_h = [c == 'H' for c in sequence]
        self.contacts = _compute_contacts(sequence, chain)
        # contact_map[i] = set of j such that (min(i,j), max(i,j)) is in contacts
        self.contact_map = {i: set() for i in range(self.n)}
        for i, j in self.contacts:
            self.contact_map[i].add(j)
            self.contact_map[j].add(i)
        self.energy = -len(self.contacts)
        self.best_chain = list(chain)
        self.best_energy = self.energy
        self.best_contacts = set(self.contacts)
        self.best_contact_map = {i: set(s) for i, s in self.contact_map.items()}

    def _accept_if_better(self):
        if self.energy < self.best_energy:
            self.best_energy = self.energy
            self.best_chain = list(self.chain)
            self.best_contacts = set(self.contacts)
            self.best_contact_map = {i: set(s) for i, s in self.contact_map.items()}

    def _remove_contacts_involving(self, indices):
        """Remove all contacts involving any index in the set. Returns list of removed (i,j) pairs."""
        indices_set = set(indices)
        removed = []
        # We only need to iterate over the indices in the set, not all contacts
        for i in indices_set:
            # Copy the set because we are modifying it
            partners = list(self.contact_map[i])
            for j in partners:
                if j in indices_set:
                    # Both i and j are in the moved set. 
                    # We must ensure we don't remove the same pair twice.
                    # If i < j, we process it here. If i > j, it will be processed when we hit j? 
                    # No, if we iterate over all i in indices_set, and for each i we look at its partners,
                    # we might see (i,j) and (j,i). 
                    # Let's just remove (min(i,j), max(i,j)) if it exists in self.contacts.
                    pass
                # Simplify: just remove all pairs (i, j) where i is in indices_set and j is in contact_map[i]
                # and the pair is in self.contacts.
                pair = (i, j) if i < j else (j, i)
                if pair in self.contacts:
                    self.contacts.discard(pair)
                    self.energy += 1
                    self.contact_map[i].discard(j)
                    self.contact_map[j].discard(i)
                    removed.append(pair)
        return removed

    def _add_new_contacts(self, indices, new_positions):
        """Add new contacts for moved indices at their new positions."""
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

    def _update_energy_for_move(self, moved_indices, old_positions, new_positions):
        """
        Incrementally update energy after moving residues.
        moved_indices: list of residue indices that moved
        old_positions: dict index -> old position (unused for removal, just for context)
        new_positions: dict index -> new position
        """
        # 1. Remove all contacts involving moved residues
        self._remove_contacts_involving(moved_indices)
        
        # 2. Add new contacts involving moved residues
        self._add_new_contacts(moved_indices, new_positions)

    def _snapshot(self):
        """Save state for potential undo."""
        return {
            'chain': list(self.chain),
            'occupied': dict(self.occupied),
            'contacts': set(self.contacts),
            'contact_map': {i: set(s) for i, s in self.contact_map.items()},
            'energy': self.energy
        }

    def _restore(self, snap):
        """Restore state from snapshot."""
        self.chain = snap['chain']
        self.occupied = snap['occupied']
        self.contacts = snap['contacts']
        self.contact_map = snap['contact_map']
        self.energy = snap['energy']

    def _pivot_move(self, rng):
        """True pivot: rotate subchain around a bond using a rotation matrix."""
        n = self.n
        if n < 4:
            return False
        pivot_idx = rng.randint(1, n - 3)
        orig_dir = tuple(self.chain[pivot_idx + 1][k] - self.chain[pivot_idx][k] for k in range(3))

        valid_mats = []
        for mat in _ROT_MATS:
            new_dir = _apply_rotation(mat, orig_dir)
            if new_dir != orig_dir and new_dir != tuple(-x for x in orig_dir):
                valid_mats.append((mat, new_dir))

        if not valid_mats:
            return False

        mat, new_dir = valid_mats[rng.randint(0, len(valid_mats) - 1)]
        subchain_len = n - pivot_idx - 1
        subchain_old = list(self.chain[pivot_idx + 1:])

        new_subchain = [None] * subchain_len
        new_subchain[0] = tuple(self.chain[pivot_idx][k] + new_dir[k] for k in range(3))

        if new_subchain[0] in self.occupied:
            return False

        valid = True
        for k in range(1, subchain_len):
            old_step = tuple(subchain_old[k][m] - subchain_old[k-1][m] for m in range(3))
            new_step = _apply_rotation(mat, old_step)
            new_pos = tuple(new_subchain[k-1][m] + new_step[m] for m in range(3))
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
        old_positions = {i: self.chain[i] for i in moved_indices}

        for pos in subchain_old:
            del self.occupied[pos]
        for k, pos in enumerate(new_subchain):
            self.chain[pivot_idx + 1 + k] = pos
            self.occupied[pos] = pivot_idx + 1 + k

        new_positions = {i: self.chain[i] for i in moved_indices}
        self._update_energy_for_move(moved_indices, old_positions, new_positions)
        self._accept_if_better()
        return True

    def _pivot_move_translate(self, rng):
        """Translation-based pivot (faster): shift subchain by changing first step direction."""
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
        old_positions = {i: self.chain[i] for i in moved_indices}

        for pos in subchain_old:
            del self.occupied[pos]
        for k, pos in enumerate(new_subchain):
            self.chain[pivot_idx + 1 + k] = pos
            self.occupied[pos] = pivot_idx + 1 + k

        new_positions = {i: self.chain[i] for i in moved_indices}
        self._update_energy_for_move(moved_indices, old_positions, new_positions)
        self._accept_if_better()
        return True

    def _crank_move(self, rng):
        """Crank: translate a subchain by one unit in a random direction."""
        n = self.n
        if n < 5:
            return False
        start_idx = rng.randint(1, n - 3)
        end_idx = rng.randint(start_idx + 1, n - 1)
        subchain = list(self.chain[start_idx:end_idx + 1])

        ax = rng.randint(0, 2)
        s = rng.choice((-1, 1))
        new_subchain = []
        for p in subchain:
            np = list(p)
            np[ax] += s
            new_subchain.append(tuple(np))

        seen = set()
        for np in new_subchain:
            if np in seen:
                return False
            seen.add(np)

        for p in subchain:
            del self.occupied[p]

        for np in new_subchain:
            if np in self.occupied:
                for i, p in enumerate(subchain):
                    self.occupied[p] = start_idx + i
                return False

        if start_idx > 0:
            if not _lattice_adjacent(self.chain[start_idx - 1], new_subchain[0]):
                for i, p in enumerate(subchain):
                    self.occupied[p] = start_idx + i
                return False

        if end_idx < n - 1:
            if not _lattice_adjacent(new_subchain[-1], self.chain[end_idx + 1]):
                for i, p in enumerate(subchain):
                    self.occupied[p] = start_idx + i
                return False

        moved_indices = list(range(start_idx, end_idx + 1))
        old_positions = {i: self.chain[i] for i in moved_indices}

        for i, np in enumerate(new_subchain):
            self.chain[start_idx + i] = np
            self.occupied[np] = start_idx + i

        new_positions = {i: self.chain[i] for i in moved_indices}
        self._update_energy_for_move(moved_indices, old_positions, new_positions)
        self._accept_if_better()
        return True

def _local_search(sequence, chain, dimension, rng, max_iters=500):
    """Local search with incremental energy updates."""
    if chain is None:
        return None
    n = len(chain)
    if n < 4:
        return chain

    state = _FoldingState(sequence, chain)

    for it in range(max_iters):
        if _time_left() < 1.0:
            break
        r = rng.random()
        if r < 0.35:
            state._pivot_move(rng)
        elif r < 0.70:
            state._pivot_move_translate(rng)
        else:
            state._crank_move(rng)

    return state.best_chain

def _simulated_annealing(sequence, chain, dimension, rng, max_iters=1000):
    """Simulated annealing with proper accept/reject using snapshot-based undo."""
    if chain is None:
        return None
    n = len(chain)
    if n < 4:
        return chain

    state = _FoldingState(sequence, chain)
    T_start = 5.0
    T_end = 0.05
    cooling_rate = (T_end / T_start) ** (1.0 / max_iters)
    T = T_start

    for it in range(max_iters):
        if _time_left() < 1.0:
            break
        T *= cooling_rate

        old_energy = state.energy
        snap = state._snapshot()

        r = rng.random()
        if r < 0.35:
            accepted = state._pivot_move(rng)
        elif r < 0.70:
            accepted = state._pivot_move_translate(rng)
        else:
            accepted = state._crank_move(rng)

        if not accepted:
            continue

        delta = state.energy - old_energy
        if delta < 0 or (T > 0.01 and rng.random() < math.exp(-delta / T)):
            state._accept_if_better()
        else:
            state._restore(snap)

    return state.best_chain

def fold(sequence, dimension):
    """Best conformation using greedy growth + incremental local search + simulated annealing."""
    n = len(sequence)
    rng = random.Random(12345 + n)
    best, best_energy = None, 1

    # Number of restarts based on sequence length and time budget
    time_per_restart = 0.15 if n <= 50 else (0.3 if n <= 80 else 0.6)
    max_restarts = int(_time_left() / time_per_restart)
    max_restarts = max(30, min(max_restarts, 3000))

    for r in range(max_restarts):
        if _time_left() < 3.0:
            break

        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue

        if _time_left() > 2.0:
            # Alternate between local search and simulated annealing
            if r % 3 == 0:
                max_ls = 400 if n <= 50 else (250 if n <= 80 else 120)
                chain = _local_search(sequence, chain, dimension, rng, max_iters=max_ls)
            elif r % 3 == 1:
                max_sa = 500 if n <= 50 else (300 if n <= 80 else 150)
                chain = _simulated_annealing(sequence, chain, dimension, rng, max_iters=max_sa)
            else:
                # Combined: SA then LS
                max_sa = 250 if n <= 50 else (180 if n <= 80 else 100)
                chain = _simulated_annealing(sequence, chain, dimension, rng, max_iters=max_sa)
                if chain is not None and _time_left() > 2.0:
                    max_ls = 200 if n <= 50 else (120 if n <= 80 else 80)
                    chain = _local_search(sequence, chain, dimension, rng, max_iters=max_ls)

        if chain is None:
            continue

        value = _energy(sequence, chain)
        if value < best_energy:
            best, best_energy = chain, value

    if best is None:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]

    return [list(p) for p in best]
# EVOLVE-BLOCK-END
