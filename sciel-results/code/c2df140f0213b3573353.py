import time
import random
import math

# EVOLVE-BLOCK-START

_DEADLINE = 315.0
_START_TIME = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

_N = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

def _energy(seq, chain):
    n = len(chain)
    if n == 0: return 0
    pos_map = {p: i for i, p in enumerate(chain)}
    is_h = [c == 'H' for c in seq]
    contacts = 0
    for i, p in enumerate(chain):
        if is_h[i]:
            x, y, z = p
            for dx, dy, dz in _N:
                j = pos_map.get((x+dx, y+dy, z+dz), -1)
                if j > i + 1 and is_h[j]:
                    contacts += 1
    return -contacts

def _kink_move(chain, i):
    if i <= 0 or i >= len(chain) - 1:
        return None
    occ = set(chain)
    occ.discard(chain[i])
    prev_p, next_p, curr = chain[i-1], chain[i+1], chain[i]
    cx, cy, cz = curr
    for dx, dy, dz in _N:
        site = (cx+dx, cy+dy, cz+dz)
        if site in occ: continue
        if abs(site[0]-prev_p[0])+abs(site[1]-prev_p[1])+abs(site[2]-prev_p[2]) != 1: continue
        if abs(site[0]-next_p[0])+abs(site[1]-next_p[1])+abs(site[2]-next_p[2]) != 1: continue
        new_chain = list(chain)
        new_chain[i] = site
        return new_chain
    return None

def _pivot_move(chain, i, axis, sign):
    n = len(chain)
    if i <= 1 or i >= n - 1: return None
    head = chain[:i+1]
    tail = chain[i+1:]
    pivot = chain[i]
    head_set = set(head)
    new_tail = []
    seen = set()
    px, py, pz = pivot
    for pos in tail:
        vx, vy, vz = pos[0]-px, pos[1]-py, pos[2]-pz
        if axis == 0:
            nv = (vx, -vz, vy) if sign == 1 else (vx, vz, -vy)
        elif axis == 1:
            nv = (vz, vy, -vx) if sign == 1 else (-vz, vy, vx)
        else:
            nv = (-vy, vx, vz) if sign == 1 else (vy, -vx, vz)
        np_ = (px+nv[0], py+nv[1], pz+nv[2])
        if np_ in seen or np_ in head_set: return None
        seen.add(np_)
        new_tail.append(np_)
    return head + new_tail

def _reversal_move(chain, i, j):
    n = len(chain)
    if i <= 0 or j >= n - 1 or j - i < 1:
        return None
    # Check connectivity at the reversal boundaries
    if i > 0:
        d1 = abs(chain[i-1][0]-chain[j][0]) + abs(chain[i-1][1]-chain[j][1]) + abs(chain[i-1][2]-chain[j][2])
        if d1 != 1:
            return None
    if j < n - 1:
        d2 = abs(chain[j+1][0]-chain[i][0]) + abs(chain[j+1][1]-chain[i][1]) + abs(chain[j+1][2]-chain[i][2])
        if d2 != 1:
            return None
    new_chain = chain[:i] + list(reversed(chain[i:j+1])) + chain[j+1:]
    if len(set(new_chain)) != len(new_chain):
        return None
    return new_chain

def _beam_search(seq, dim, beam_width, rng):
    """Beam search for initial chain construction."""
    n = len(seq)
    is_h = [c == 'H' for c in seq]
    
    start = (0, 0, 0)
    # Each beam entry: (contacts, chain, occupied_dict)
    beam = [(0, [start], {start: 0})]
    
    for i in range(1, n):
        if _time_left() < 5.0:
            break
        
        new_beam = []
        for contacts, chain, occ in beam:
            last = chain[-1]
            x, y, z = last
            
            for dx, dy, dz in _N:
                site = (x+dx, y+dy, z+dz)
                if site in occ:
                    continue
                
                new_contacts = contacts
                if is_h[i]:
                    sx, sy, sz = site
                    for dx2, dy2, dz2 in _N:
                        j = occ.get((sx+dx2, sy+dy2, sz+dz2), -1)
                        if j is not None and j < i - 1 and is_h[j]:
                            new_contacts += 1
                
                new_chain = chain + [site]
                new_occ = dict(occ)
                new_occ[site] = i
                new_beam.append((new_contacts, new_chain, new_occ))
        
        if not new_beam:
            break
        
        # Sort by contacts descending, keep top beam_width
        new_beam.sort(key=lambda x: -x[0])
        beam = new_beam[:beam_width]
    
    if not beam:
        return None, 0
    
    # Return the best beam entry
    best = beam[0]
    return best[1], -best[0]

def _local_search(seq, chain, rng, time_budget):
    n = len(chain)
    is_h = [c == 'H' for c in seq]
    cur_chain = list(chain)
    cur_energy = _energy(seq, cur_chain)
    best_chain = list(cur_chain)
    best_energy = cur_energy
    
    T = 3.0
    alpha = 0.997
    min_T = 0.08
    
    # Phase 1: Greedy kink descent with multiple attempts
    max_greedy = 3000 if n < 50 else 1500
    for _ in range(max_greedy):
        if _time_left() < 3.0: break
        i = rng.randint(1, n-2)
        new_chain = _kink_move(cur_chain, i)
        if new_chain:
            new_e = _energy(seq, new_chain)
            if new_e < cur_energy:
                cur_chain = new_chain
                cur_energy = new_e
                if cur_energy < best_energy:
                    best_energy = cur_energy
                    best_chain = list(cur_chain)
    
    # Phase 2: SA with kinks, pivots, and reversals
    while _time_left() > 3.0 and T > min_T:
        r = rng.random()
        new_chain = None
        delta = 0
        
        if r < 0.55:
            i = rng.randint(1, n-2)
            new_chain = _kink_move(cur_chain, i)
            if new_chain:
                new_e = _energy(seq, new_chain)
                delta = new_e - cur_energy
        elif r < 0.85 and n > 15:
            i = rng.randint(2, n-2)
            axis = rng.randint(0, 2)
            sign = rng.choice([-1, 1])
            new_chain = _pivot_move(cur_chain, i, axis, sign)
            if new_chain:
                new_e = _energy(seq, new_chain)
                delta = new_e - cur_energy
        else:
            # Reversal move
            if n > 6:
                i = rng.randint(0, n-3)
                max_len = min(8, n - i - 1)
                j = rng.randint(i+1, i + max_len)
                new_chain = _reversal_move(cur_chain, i, j)
                if new_chain:
                    new_e = _energy(seq, new_chain)
                    delta = new_e - cur_energy
        
        if new_chain:
            if delta < 0 or (delta > 0 and rng.random() < math.exp(-delta/T)):
                cur_chain = new_chain
                cur_energy += delta
                if cur_energy < best_energy:
                    best_energy = cur_energy
                    best_chain = list(cur_chain)
        T *= alpha
    
    # Phase 3: Final greedy polish
    for _ in range(500):
        if _time_left() < 2.0: break
        i = rng.randint(1, n-2)
        new_chain = _kink_move(cur_chain, i)
        if new_chain:
            new_e = _energy(seq, new_chain)
            if new_e < cur_energy:
                cur_chain = new_chain
                cur_energy = new_e
                if cur_energy < best_energy:
                    best_energy = cur_energy
                    best_chain = list(cur_chain)
    
    return best_chain, best_energy

def _multi_start_fold(seq, dim):
    n = len(seq)
    best_chain = None
    best_energy = float('inf')
    
    # Determine beam width and restarts based on length
    if n <= 30:
        beam_width = 15
        n_restarts = 6
    elif n <= 50:
        beam_width = 10
        n_restarts = 5
    elif n <= 80:
        beam_width = 8
        n_restarts = 3
    else:
        beam_width = 6
        n_restarts = 2
    
    # Reduce if time is tight
    if _time_left() < 90:
        n_restarts = min(n_restarts, 3)
        beam_width = max(4, beam_width // 2)
    if _time_left() < 30:
        n_restarts = 1
        beam_width = 4
    if _time_left() < 10:
        n_restarts = 1
        beam_width = 3
    
    for r in range(n_restarts):
        if _time_left() < 5.0: break
        
        seed = 42 + r * 7919 + n * 131
        rng = random.Random(seed)
        
        # Beam search initialization
        chain, energy = _beam_search(seq, dim, beam_width, rng)
        
        if chain is None or len(chain) < n:
            # Fallback: simple greedy growth
            chain = [(0, 0, 0)]
            occ = {(0, 0, 0): 0}
            is_h = [c == 'H' for c in seq]
            for i in range(1, n):
                if _time_left() < 3.0:
                    break
                last = chain[-1]
                x, y, z = last
                candidates = []
                for dx, dy, dz in _N:
                    site = (x+dx, y+dy, z+dz)
                    if site in occ:
                        continue
                    gain = 0
                    if is_h[i]:
                        for dx2, dy2, dz2 in _N:
                            j = occ.get((site[0]+dx2, site[1]+dy2, site[2]+dz2), -1)
                            if j is not None and j < i - 1 and is_h[j]:
                                gain += 1
                    candidates.append((-gain, rng.random(), site))
                if not candidates:
                    break
                candidates.sort(key=lambda c: (c[0], c[1]))
                site = candidates[0][2]
                chain.append(site)
                occ[site] = i
            if len(chain) < n:
                continue
        
        # Local search
        opt_chain, opt_energy = _local_search(seq, chain, rng, _time_left())
        
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_chain = opt_chain
    
    if best_chain is None:
        # Fallback: straight line
        return [[i, 0, 0] for i in range(n)]
    
    return [list(p) for p in best_chain]

def fold(sequence, dimension):
    n = len(sequence)
    return _multi_start_fold(sequence, dimension)

# EVOLVE-BLOCK-END
