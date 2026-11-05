"""HP lattice protein folding with lazy-copy beam search and incremental local search."""
# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE = 320.0
_START_TIME = None

def _get_deadline():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME + _DEADLINE

def _time_left():
    return _get_deadline() - time.time()

def _neighbours(point):
    x, y, z = point
    yield (x+1, y, z)
    yield (x-1, y, z)
    yield (x, y+1, z)
    yield (x, y-1, z)
    yield (x, y, z+1)
    yield (x, y, z-1)

def _lattice_adjacent(p1, p2):
    return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) + abs(p1[2]-p2[2]) == 1

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    is_h = [c == 'H' for c in sequence]
    for i, p in enumerate(chain):
        if is_h[i]:
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and is_h[j]:
                    contacts += 1
    return -contacts

def _compute_contacts(sequence, chain):
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    contacts = set()
    is_h = [c == 'H' for c in sequence]
    for i, p in enumerate(chain):
        if not is_h[i]:
            continue
        for q in _neighbours(p):
            j = where.get(q)
            if j is not None and j > i + 1 and is_h[j]:
                contacts.add((i, j))
    return contacts

def _beam_grow_lazy(sequence, dimension, rng, beam_width=60):
    """Beam search growth with lazy state copying for speed."""
    n = len(sequence)
    if n < 4:
        start = (0, 0, 0)
        return [start] + [(i, 0, 0) for i in range(1, n)]
    
    is_h = [c == 'H' for c in sequence]
    start = (0, 0, 0)
    
    # Each beam entry: [score, chain_list, occupied_dict]
    # Use lazy copying - only copy when actually keeping a beam
    beams = [[0.0, [start], {start: 0}]]
    
    for i in range(1, n):
        if _time_left() < 2.0:
            break
        new_beams = []
        seen_keys = set()
        
        for score, chain, occupied in beams:
            last = chain[-1]
            for site in _neighbours(last):
                if site in occupied:
                    continue
                
                # Diversity key: last 3 positions + new site
                if len(chain) >= 3:
                    key = (chain[-1], site)
                else:
                    key = tuple(chain[-1:]) + (site,)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                
                gain = 0
                if is_h[i]:
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and is_h[j]:
                            gain += 1
                
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                new_score = score + gain * 4.0 + 0.02 * free + rng.random() * 0.2
                
                # Lazy copy: only create new lists/dicts if we might keep this
                new_chain = chain + [site]
                new_occ = dict(occupied)
                new_occ[site] = i
                new_beams.append([new_score, new_chain, new_occ])
        
        if not new_beams:
            break
        
        # Keep top beam_width beams by score
        new_beams.sort(key=lambda x: -x[0])
        beams = new_beams[:beam_width]
    
    if not beams:
        return None
    
    best = max(beams, key=lambda x: x[0])
    return best[1]

def _grow(sequence, dimension, rng):
    """Randomized greedy chain growth with contact-aware scoring."""
    start = (0, 0, 0)
    chain, occupied = [start], {start: 0}
    is_h = [c == 'H' for c in sequence]
    
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if is_h[i]:
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and is_h[j]:
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 4.0 + 0.03 * free + rng.random() * 0.3
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

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
        self.stagnation = 0

    def _accept_if_better(self):
        if self.energy < self.best_energy:
            self.best_energy = self.energy
            self.best_chain = list(self.chain)
            self.stagnation = 0
        else:
            self.stagnation += 1

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

    def _snapshot(self):
        return {
            'chain': list(self.chain),
            'occupied': dict(self.occupied),
            'contacts': set(self.contacts),
            'contact_map': {i: set(s) for i, s in self.contact_map.items()},
            'energy': self.energy
        }

    def _restore(self, snap):
        self.chain = snap['chain']
        self.occupied = snap['occupied']
        self.contacts = snap['contacts']
        self.contact_map = snap['contact_map']
        self.energy = snap['energy']

    def _pivot_move(self, rng):
        """Pivot: rotate a subchain around a pivot point."""
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

        for i, np in enumerate(new_subchain):
            self.chain[start_idx + i] = np
            self.occupied[np] = start_idx + i

        new_positions = {i: self.chain[i] for i in moved_indices}
        self._update_energy_for_move(moved_indices, new_positions)
        self._accept_if_better()
        return True

    def _single_residue_move(self, rng):
        """Move a single residue to an adjacent free site."""
        n = self.n
        if n < 3:
            return False
        idx = rng.randint(1, n - 2)
        pos = self.chain[idx]
        
        for new_pos in _neighbours(pos):
            if new_pos in self.occupied:
                continue
            prev_pos = self.chain[idx - 1]
            next_pos = self.chain[idx + 1]
            if not _lattice_adjacent(prev_pos, new_pos):
                continue
            if not _lattice_adjacent(new_pos, next_pos):
                continue
            
            del self.occupied[pos]
            self.chain[idx] = new_pos
            self.occupied[new_pos] = idx
            
            self._update_energy_for_move([idx], {idx: new_pos})
            self._accept_if_better()
            return True
        return False

    def _tunneling_move(self, rng):
        """Tunneling: jump subchain to escape local minima."""
        n = self.n
        if n < 8:
            return False
        
        pivot_idx = rng.randint(2, n - 4)
        subchain_len = min(rng.randint(3, max(4, n // 4)), n - pivot_idx - 1)
        
        if subchain_len < 2:
            return False
        
        end_idx = pivot_idx + subchain_len
        subchain = list(self.chain[pivot_idx:end_idx + 1])
        anchor_pos = self.chain[pivot_idx - 1]
        
        candidates = []
        for site in _neighbours(anchor_pos):
            if site in self.occupied:
                continue
            min_dist = min(sum(abs(a - b) for a, b in zip(site, sp)) for sp in subchain)
            if min_dist >= 2:
                candidates.append((min_dist, site))
        
        if not candidates:
            return False
        
        candidates.sort(reverse=True)
        new_start = candidates[0][1]
        
        for pos in subchain:
            del self.occupied[pos]
        
        new_subchain = [new_start]
        new_occupied = dict(self.occupied)
        new_occupied[new_start] = pivot_idx
        
        is_h = self.is_h
        for k in range(1, subchain_len + 1):
            i = pivot_idx + k
            options = []
            for site in _neighbours(new_subchain[-1]):
                if site in new_occupied:
                    continue
                gain = 0
                if is_h[i]:
                    for q in _neighbours(site):
                        j = new_occupied.get(q)
                        if j is not None and j != i - 1 and is_h[j]:
                            gain += 1
                free = sum(1 for q in _neighbours(site) if q not in new_occupied)
                score = gain * 4.0 + 0.03 * free + rng.random() * 0.3
                options.append((score, site))
            
            if not options:
                for pos in subchain:
                    self.occupied[pos] = pivot_idx + subchain.index(pos)
                return False
            
            site = max(options)[1]
            new_subchain.append(site)
            new_occupied[site] = i
        
        if end_idx < n - 1:
            last_pos = new_subchain[-1]
            next_fixed = self.chain[end_idx + 1]
            if not _lattice_adjacent(last_pos, next_fixed):
                for pos in subchain:
                    self.occupied[pos] = pivot_idx + subchain.index(pos)
                return False
        
        moved_indices = list(range(pivot_idx, end_idx + 1))
        
        for i, pos in enumerate(new_subchain):
            self.chain[pivot_idx + i] = pos
            self.occupied[pos] = pivot_idx + i
        
        new_positions = {i: self.chain[i] for i in moved_indices}
        self._update_energy_for_move(moved_indices, new_positions)
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
        
        if state.stagnation > 50 and r < 0.25:
            state._tunneling_move(rng)
        elif r < 0.35:
            state._pivot_move(rng)
        elif r < 0.70:
            state._crank_move(rng)
        else:
            state._single_residue_move(rng)

    return state.best_chain

def _simulated_annealing(sequence, chain, dimension, rng, max_iters=1000):
    """Simulated annealing with proper accept/reject."""
    if chain is None:
        return None
    n = len(chain)
    if n < 4:
        return chain

    state = _FoldingState(sequence, chain)
    T_start = 5.0
    T_end = 0.02
    cooling_rate = (T_end / T_start) ** (1.0 / max_iters)
    T = T_start

    for it in range(max_iters):
        if _time_left() < 1.0:
            break
        T *= cooling_rate

        old_energy = state.energy
        snap = state._snapshot()

        r = rng.random()
        
        use_tunneling = (state.stagnation > 40 and r < 0.15) or (T > 2.5 and r < 0.08)
        
        if use_tunneling:
            accepted = state._tunneling_move(rng)
        elif r < 0.35:
            accepted = state._pivot_move(rng)
        elif r < 0.70:
            accepted = state._crank_move(rng)
        else:
            accepted = state._single_residue_move(rng)

        if not accepted:
            continue

        delta = state.energy - old_energy
        if delta < 0 or (T > 0.01 and rng.random() < math.exp(-delta / T)):
            state._accept_if_better()
        else:
            state._restore(snap)

    return state.best_chain

def fold(sequence, dimension):
    """Best conformation using beam search initialization and multi-strategy local search."""
    n = len(sequence)
    rng = random.Random(12345 + n)
    best, best_energy = None, 0

    # Time budget allocation
    if n <= 50:
        time_per_restart = 0.08
        max_restarts = int(_time_left() / time_per_restart)
        max_restarts = max(50, min(max_restarts, 4000))
        beam_w = 80
    elif n <= 80:
        time_per_restart = 0.15
        max_restarts = int(_time_left() / time_per_restart)
        max_restarts = max(30, min(max_restarts, 2000))
        beam_w = 50
    else:
        time_per_restart = 0.30
        max_restarts = int(_time_left() / time_per_restart)
        max_restarts = max(15, min(max_restarts, 800))
        beam_w = 30

    for r in range(max_restarts):
        if _time_left() < 3.0:
            break

        strategy = r % 5
        
        if strategy <= 1:
            # Beam search + local search (primary)
            bw = beam_w + (r % 5)
            chain = _beam_grow_lazy(sequence, dimension, rng, beam_width=bw)
            if chain is None:
                chain = _grow(sequence, dimension, rng)
            if chain is None:
                continue
            if _time_left() > 2.0:
                max_ls = 800 if n <= 50 else (400 if n <= 80 else 200)
                chain = _local_search(sequence, chain, dimension, rng, max_iters=max_ls)
                
        elif strategy == 2:
            # Beam search + simulated annealing
            chain = _beam_grow_lazy(sequence, dimension, rng, beam_width=beam_w)
            if chain is None:
                chain = _grow(sequence, dimension, rng)
            if chain is None:
                continue
            if _time_left() > 2.0:
                max_sa = 900 if n <= 50 else (450 if n <= 80 else 250)
                chain = _simulated_annealing(sequence, chain, dimension, rng, max_iters=max_sa)
                
        elif strategy == 3:
            # Greedy growth + extended LS
            chain = _grow(sequence, dimension, rng)
            if chain is None:
                continue
            if _time_left() > 2.0:
                max_ls = 1200 if n <= 50 else (600 if n <= 80 else 300)
                chain = _local_search(sequence, chain, dimension, rng, max_iters=max_ls)
                
        elif strategy == 4:
            # Beam search + SA + LS
            chain = _beam_grow_lazy(sequence, dimension, rng, beam_width=beam_w)
            if chain is None:
                chain = _grow(sequence, dimension, rng)
            if chain is None:
                continue
            if _time_left() > 2.0:
                max_sa = 400 if n <= 50 else (200 if n <= 80 else 100)
                chain = _simulated_annealing(sequence, chain, dimension, rng, max_iters=max_sa)
                if chain is not None and _time_left() > 2.0:
                    max_ls = 400 if n <= 50 else (200 if n <= 80 else 100)
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
