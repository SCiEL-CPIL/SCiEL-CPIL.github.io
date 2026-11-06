"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

_START_TIME = None
_DEADLINE = 185.0

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _init_time()
    return _DEADLINE - (time.time() - _START_TIME)

def _neighbours(point):
    x, y = point
    return ((x+1, y), (x-1, y), (x, y+1), (x, y-1))

def _compute_energy(sequence, chain):
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = pos_to_idx.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _grow_greedy(sequence, dimension, rng):
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 10.0 + free * 0.5 + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options, key=lambda x: x[0])[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _grow_random_walk(sequence, dimension, rng):
    for _ in range(30):
        chain = [(0, 0)]
        occupied = {(0, 0): 0}
        success = True
        for i in range(1, len(sequence)):
            neighbors = list(_neighbours(chain[-1]))
            rng.shuffle(neighbors)
            placed = False
            for site in neighbors:
                if site not in occupied:
                    occupied[site] = i
                    chain.append(site)
                    placed = True
                    break
            if not placed:
                success = False
                break
        if success:
            return chain
    return None

def _grow_compact(sequence, dimension, rng):
    """Grow trying to stay compact by preferring inward directions."""
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    for i in range(1, len(sequence)):
        cx, cy = chain[-1]
        # Prefer directions toward center of mass
        avg_x = sum(p[0] for p in chain) / len(chain)
        avg_y = sum(p[1] for p in chain) / len(chain)
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            sx, sy = site
            dist_to_center = abs(sx - avg_x) + abs(sy - avg_y)
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            score = gain * 10.0 - dist_to_center * 0.5 + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options, key=lambda x: x[0])[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _local_search(sequence, chain):
    """
    Local search with delta-energy. Tries kink, reversal, and pivot moves.
    """
    n = len(chain)
    if n < 4:
        return chain, _compute_energy(sequence, chain)
    
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    rng = random.Random(hash(tuple(chain)) % (2**31))
    
    temperature = 1.0
    no_improve = 0
    
    def _kink_delta(i):
        """Compute delta for kink move at position i (moving chain[i+1] to new pos adjacent to chain[i])."""
        if i >= n - 1:
            return None
        i2 = i + 1
        p1 = chain[i]
        p2 = chain[i2]
        for new_p2 in _neighbours(p1):
            if new_p2 == p2:
                continue
            if new_p2 in pos_to_idx and pos_to_idx[new_p2] != i2:
                continue
            # Check that new_p2 doesn't conflict with i2's other constraints
            delta = 0
            if sequence[i2] == 'H':
                for q in _neighbours(p2):
                    j = pos_to_idx.get(q)
                    if j is not None and j != i2 and abs(i2 - j) > 1 and sequence[j] == 'H':
                        if i2 < j:
                            delta += 1
                for q in _neighbours(new_p2):
                    j = pos_to_idx.get(q)
                    if j is not None and j != i2 and abs(i2 - j) > 1 and sequence[j] == 'H':
                        if i2 < j:
                            delta -= 1
            return delta, new_p2
        return None
    
    def _reversal_delta(start, end):
        """Compute delta for reversing chain[start:end+1]."""
        if end - start < 2:
            return None
        seg = chain[start:end+1]
        rev = list(reversed(seg))
        # Check validity
        occ = set(chain)
        for p in seg:
            occ.discard(p)
        seen = set()
        for p in rev:
            if p in occ or p in seen:
                return None
            seen.add(p)
        if start > 0:
            pp = chain[start-1]
            pn = rev[0]
            if abs(pp[0]-pn[0]) + abs(pp[1]-pn[1]) != 1:
                return None
        if end < n-1:
            pn = chain[end+1]
            pr = rev[-1]
            if abs(pn[0]-pr[0]) + abs(pn[1]-pr[1]) != 1:
                return None
        delta = 0
        for idx in range(start, end+1):
            if sequence[idx] != 'H':
                continue
            old_p = chain[idx]
            new_p = rev[idx - start]
            for q in _neighbours(old_p):
                j = pos_to_idx.get(q)
                if j is None or (start <= j <= end) or j == idx:
                    continue
                if abs(idx - j) > 1 and sequence[j] == 'H':
                    if idx < j:
                        delta += 1
            for q in _neighbours(new_p):
                j = pos_to_idx.get(q)
                if j is None or (start <= j <= end) or j == idx:
                    continue
                if abs(idx - j) > 1 and sequence[j] == 'H':
                    if idx < j:
                        delta -= 1
        return delta, rev
    
    def _pivot_delta(pivot):
        """Compute delta for rotating tail after pivot."""
        if pivot < 1 or pivot >= n - 1:
            return None
        pp = chain[pivot]
        tail = chain[pivot+1:]
        for rot in range(4):
            translated = [(p[0]-pp[0], p[1]-pp[1]) for p in tail]
            rotated = []
            for x, y in translated:
                for _ in range(rot):
                    x, y = y, -x
                rotated.append((x+pp[0], y+pp[1]))
            # Check first point adjacent to pivot
            r0 = rotated[0]
            if abs(r0[0]-pp[0]) + abs(r0[1]-pp[1]) != 1:
                continue
            # Check self-avoidance
            occ = set(chain[:pivot+1])
            seen = set()
            valid = True
            for p in rotated:
                if p in occ or p in seen:
                    valid = False
                    break
                seen.add(p)
            if not valid:
                continue
            delta = 0
            for offset, idx in enumerate(range(pivot+1, n)):
                if sequence[idx] != 'H':
                    continue
                old_p = chain[idx]
                new_p = rotated[offset]
                for q in _neighbours(old_p):
                    j = pos_to_idx.get(q)
                    if j is None or j == idx or (pivot+1 <= j <= n-1):
                        continue
                    if abs(idx - j) > 1 and sequence[j] == 'H':
                        if idx < j:
                            delta += 1
                for q in _neighbours(new_p):
                    j = pos_to_idx.get(q)
                    if j is None or j == idx or (pivot+1 <= j <= n-1):
                        continue
                    if abs(idx - j) > 1 and sequence[j] == 'H':
                        if idx < j:
                            delta -= 1
            return delta, rotated
        return None
    
    def _apply_kink(i, new_p2):
        nonlocal pos_to_idx
        i2 = i + 1
        old_p2 = chain[i2]
        del pos_to_idx[old_p2]
        chain[i2] = new_p2
        pos_to_idx[new_p2] = i2
    
    def _apply_reversal(start, end, rev):
        nonlocal pos_to_idx
        for idx in range(start, end+1):
            del pos_to_idx[chain[idx]]
        for idx in range(start, end+1):
            chain[idx] = rev[idx - start]
            pos_to_idx[chain[idx]] = idx
    
    def _apply_pivot(pivot, rotated):
        nonlocal pos_to_idx
        for idx in range(pivot+1, n):
            del pos_to_idx[chain[idx]]
        for offset, idx in enumerate(range(pivot+1, n)):
            chain[idx] = rotated[offset]
            pos_to_idx[chain[idx]] = idx
    
    while _time_remaining() > 1.0:
        improved = False
        # Kink moves
        for _ in range(30):
            if _time_remaining() < 1.0:
                break
            i = rng.randint(0, n-2)
            result = _kink_delta(i)
            if result is None:
                continue
            delta, new_p2 = result
            new_energy = current_energy + delta
            if new_energy < current_energy:
                _apply_kink(i, new_p2)
                current_energy = new_energy
                improved = True
                no_improve = 0
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
            elif delta > 0 and rng.random() < math.exp(-delta / max(temperature, 0.01)):
                _apply_kink(i, new_p2)
                current_energy = new_energy
                no_improve = 0
            # else: don't apply
        # Reversals
        for _ in range(15):
            if _time_remaining() < 1.0:
                break
            seg_len = rng.choice([2, 3, 4, 5, 6, 8, 10, 12, 15])
            if seg_len >= n:
                continue
            start = rng.randint(0, n - seg_len - 1)
            end = start + seg_len
            result = _reversal_delta(start, end)
            if result is None:
                continue
            delta, rev = result
            new_energy = current_energy + delta
            if new_energy < current_energy:
                _apply_reversal(start, end, rev)
                current_energy = new_energy
                improved = True
                no_improve = 0
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
            elif delta > 0 and rng.random() < math.exp(-delta / max(temperature, 0.01)):
                _apply_reversal(start, end, rev)
                current_energy = new_energy
                no_improve = 0
        # Pivot moves
        for _ in range(8):
            if _time_remaining() < 1.0:
                break
            pivot = rng.randint(1, n-2)
            result = _pivot_delta(pivot)
            if result is None:
                continue
            delta, rotated = result
            new_energy = current_energy + delta
            if new_energy < current_energy:
                _apply_pivot(pivot, rotated)
                current_energy = new_energy
                improved = True
                no_improve = 0
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
            elif delta > 0 and rng.random() < math.exp(-delta / max(temperature, 0.01)):
                _apply_pivot(pivot, rotated)
                current_energy = new_energy
                no_improve = 0
        if not improved:
            no_improve += 1
            temperature = max(0.01, temperature * 0.9)
    
    return best_chain, best_energy

def fold(sequence, dimension):
    _init_time()
    n = len(sequence)
    if n <= 2:
        return [[i, 0] for i in range(n)]
    
    rng = random.Random(12345 + len(sequence))
    initial_chains = []
    
    for _ in range(4):
        c = _grow_greedy(sequence, dimension, rng)
        if c:
            initial_chains.append(c)
    for _ in range(3):
        c = _grow_random_walk(sequence, dimension, rng)
        if c:
            initial_chains.append(c)
    for _ in range(3):
        c = _grow_compact(sequence, dimension, rng)
        if c:
            initial_chains.append(c)
    
    if not initial_chains:
        chain = [(0, 0)]
        for i in range(1, n):
            chain.append((i, 0))
        initial_chains.append(chain)
    
    best_chain = None
    best_energy = float('inf')
    
    for chain in initial_chains:
        if _time_remaining() < 3.0:
            break
        sc, se = _local_search(sequence, chain)
        if se < best_energy:
            best_energy = se
            best_chain = sc
    
    if best_chain is None:
        best_chain = initial_chains[0]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
