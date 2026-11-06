"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import random
import time
import sys

_DEADLINE = None
_DEADLINE_MARGIN = 5.0  # seconds to keep before hard limit


def _setup_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 235.0  # 235s budget, leaving 5s margin


def _time_left():
    return _DEADLINE - time.time()


def _neighbours(point):
    """Yield 6 neighbors on cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute energy: -number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _beam_search(sequence, dimension, rng, beam_width=50, max_time=10.0):
    """Beam search with dynamic width and improved heuristic."""
    start = (0,) * dimension
    # Beam entry: (chain_list, occupied_dict, current_energy)
    beams = [([start], {start: 0}, 0)]
    n = len(sequence)
    
    t0 = time.time()
    for i in range(1, n):
        if time.time() - t0 > max_time:
            break
        
        new_beams = []
        for chain, occupied, energy in beams:
            last = chain[-1]
            options = []
            for site in _neighbours(last):
                if site in occupied:
                    continue
                gain = 0
                if sequence[i] == "H":
                    for q in _neighbours(site):
                        if q in occupied:
                            idx = occupied[q]
                            if idx < i - 1 and sequence[idx] == "H":
                                gain += 1
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                # Look-ahead: count H residues in next few positions that could contact
                lookahead = 0
                if sequence[i] == "H":
                    for j in range(i + 2, min(i + 5, n)):
                        if sequence[j] == "H":
                            lookahead += 1
                # Heuristic: prefer high gain, then free space, then lookahead potential
                score = gain * 100 + free * 10 + lookahead * 5 + rng.random()
                options.append((score, site, gain))
            
            if not options:
                continue
            
            options.sort(key=lambda x: -x[0])
            # Keep top candidates per beam
            keep = min(beam_width, max(1, beam_width // 2))
            for score, site, gain in options[:keep]:
                new_chain = chain + [site]
                new_occupied = dict(occupied)
                new_occupied[site] = i
                new_energy = energy - gain
                new_beams.append((new_chain, new_occupied, new_energy))
        
        if not new_beams:
            break
        
        # Keep top beam_width beams
        new_beams.sort(key=lambda x: x[2])
        beams = new_beams[:beam_width]
    
    if not beams:
        return None
    
    best = min(beams, key=lambda x: x[2])
    return best[0]


def _local_search(sequence, chain, max_iters=2000, seed=42):
    """Improve a conformation via 180-degree turns, 3-residue shifts, and single-residue moves."""
    n = len(chain)
    if n < 4:
        return chain
    
    rng = random.Random(seed)
    best_chain = list(chain)
    best_energy = _compute_energy(sequence, best_chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    # Precompute contact count for incremental energy updates
    def get_contact_count(position, residue_idx):
        """Count H-H contacts involving residue at residue_idx if it were at position."""
        if sequence[residue_idx] != "H":
            return 0
        count = 0
        for q in _neighbours(position):
            j = occupied.get(q)
            if j is not None and j != residue_idx - 1 and j != residue_idx + 1 and j != residue_idx:
                if sequence[j] == "H":
                    count += 1
        return count
    
    def compute_delta_energy_for_move(moved_indices, new_positions):
        """Compute energy change for moving residues at moved_indices to new_positions.
        Returns the delta in energy (negative is improvement)."""
        # Remove old contacts involving moved residues
        old_delta = 0
        for idx in moved_indices:
            old_pos = best_chain[idx]
            if sequence[idx] == "H":
                for q in _neighbours(old_pos):
                    j = occupied.get(q)
                    if j is not None and j not in moved_indices and j != idx - 1 and j != idx + 1:
                        if sequence[j] == "H":
                            old_delta += 1
        
        # Add new contacts involving moved residues
        new_delta = 0
        for idx in moved_indices:
            new_pos = new_positions[idx]
            if sequence[idx] == "H":
                for q in _neighbours(new_pos):
                    j = occupied.get(q)
                    if j is not None and j not in moved_indices and j != idx - 1 and j != idx + 1:
                        if sequence[j] == "H":
                            new_delta += 1
        
        # Also count contacts between moved residues themselves (if both H and non-adjacent in chain)
        for a_idx in moved_indices:
            for b_idx in moved_indices:
                if a_idx >= b_idx:
                    continue
                if abs(a_idx - b_idx) <= 1:
                    continue
                if sequence[a_idx] == "H" and sequence[b_idx] == "H":
                    a_pos = new_positions[a_idx]
                    b_pos = new_positions[b_idx]
                    if abs(a_pos[0]-b_pos[0]) + abs(a_pos[1]-b_pos[1]) + abs(a_pos[2]-b_pos[2]) == 1:
                        new_delta += 1
        
        # Energy = -contacts, so delta_energy = -(new_contacts - old_contacts)
        # We want to return the change in energy
        return -(new_delta - old_delta)
    
    def manhattan(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])
    
    def try_180_turn(i):
        """Try a 180-degree turn at position i (moves residues i-1, i, i+1)."""
        if i < 1 or i > n - 2:
            return None
        a, b, c = best_chain[i-1], best_chain[i], best_chain[i+1]
        new_a = (2*b[0] - c[0], 2*b[1] - c[1], 2*b[2] - c[2])
        new_c = (2*b[0] - a[0], 2*b[1] - a[1], 2*b[2] - a[2])
        
        # Check self-avoidance
        if new_a in occupied and occupied[new_a] != i-1:
            return None
        if new_c in occupied and occupied[new_c] != i+1:
            return None
        if new_a == new_c:
            return None
        if new_a == b or new_c == b:
            return None
        
        return (i-1, i+1, new_a, new_c)
    
    def try_3shift(i):
        """Try shifting residues i-1, i, i+1 as a unit in some direction."""
        if i < 1 or i > n - 2:
            return None
        a, b, c = best_chain[i-1], best_chain[i], best_chain[i+1]
        for axis in range(3):
            for step in (-1, 1):
                new_a = list(a)
                new_b = list(b)
                new_c = list(c)
                new_a[axis] += step
                new_b[axis] += step
                new_c[axis] += step
                new_a = tuple(new_a)
                new_b = tuple(new_b)
                new_c = tuple(new_c)
                
                # Check self-avoidance
                if new_a in occupied and occupied[new_a] != i-1:
                    continue
                if new_b in occupied and occupied[new_b] != i:
                    continue
                if new_c in occupied and occupied[new_c] != i+1:
                    continue
                
                # Check unit steps
                if manhattan(new_a, new_b) != 1:
                    continue
                if manhattan(new_b, new_c) != 1:
                    continue
                
                return (i-1, i, i+1, new_a, new_b, new_c)
        return None
    
    def try_single_move(i):
        """Try moving a single residue i to an adjacent free position."""
        if i < 1 or i > n - 2:
            return None
        # For a single residue move, we need to check that the neighbors (i-1 and i+1) 
        # can still be connected. Actually, single residue moves are tricky because
        # moving one residue breaks the chain. Let's use a different approach:
        # Try moving residue i to a position that maintains connectivity.
        # This is complex, so let's skip for now and focus on other moves.
        return None
    
    for it in range(max_iters):
        if _time_left() < 1.0:
            break
        improved = False
        
        for _attempt in range(30):
            move_type = rng.random()
            i = rng.randint(1, n - 2)
            
            if move_type < 0.55:
                # 180-degree turn
                result = try_180_turn(i)
                if result is None:
                    continue
                i1, i2, new_a, new_c = result
                old_a, old_c = best_chain[i1], best_chain[i2]
                
                # Apply
                best_chain[i1] = new_a
                best_chain[i2] = new_c
                occupied.pop(old_a)
                occupied.pop(old_c)
                occupied[new_a] = i1
                occupied[new_c] = i2
                
                new_energy = _compute_energy(sequence, best_chain)
                if new_energy < best_energy:
                    best_energy = new_energy
                    improved = True
                else:
                    # Revert
                    best_chain[i1] = old_a
                    best_chain[i2] = old_c
                    occupied.pop(new_a)
                    occupied.pop(new_c)
                    occupied[old_a] = i1
                    occupied[old_c] = i2
            elif move_type < 0.85:
                # 3-residue shift
                result = try_3shift(i)
                if result is None:
                    continue
                i1, i2, i3, new_a, new_b, new_c = result
                old_a, old_b, old_c = best_chain[i1], best_chain[i2], best_chain[i3]
                
                # Apply
                best_chain[i1] = new_a
                best_chain[i2] = new_b
                best_chain[i3] = new_c
                occupied.pop(old_a)
                occupied.pop(old_b)
                occupied.pop(old_c)
                occupied[new_a] = i1
                occupied[new_b] = i2
                occupied[new_c] = i3
                
                new_energy = _compute_energy(sequence, best_chain)
                if new_energy < best_energy:
                    best_energy = new_energy
                    improved = True
                else:
                    # Revert
                    best_chain[i1] = old_a
                    best_chain[i2] = old_b
                    best_chain[i3] = old_c
                    occupied.pop(new_a)
                    occupied.pop(new_b)
                    occupied.pop(new_c)
                    occupied[old_a] = i1
                    occupied[old_b] = i2
                    occupied[old_c] = i3
            else:
                # 2-residue pivot: move residues i and i+1
                if i < 1 or i > n - 3:
                    continue
                a, b = best_chain[i], best_chain[i+1]
                # Try rotating the segment a-b around a
                for axis in range(3):
                    for step in (-1, 1):
                        new_b = list(b)
                        new_b[axis] += step
                        new_b = tuple(new_b)
                        
                        # Check self-avoidance
                        if new_b in occupied and occupied[new_b] != i+1:
                            continue
                        
                        # Check unit step from a to new_b
                        if manhattan(a, new_b) != 1:
                            continue
                        
                        # Check unit step from new_b to next residue
                        if i + 2 < n:
                            next_res = best_chain[i+2]
                            if manhattan(new_b, next_res) != 1:
                                continue
                        
                        # Apply
                        old_b = best_chain[i+1]
                        best_chain[i+1] = new_b
                        occupied.pop(old_b)
                        occupied[new_b] = i+1
                        
                        new_energy = _compute_energy(sequence, best_chain)
                        if new_energy < best_energy:
                            best_energy = new_energy
                            improved = True
                        else:
                            # Revert
                            best_chain[i+1] = old_b
                            occupied.pop(new_b)
                            occupied[old_b] = i+1
        
        if not improved:
            # Try a few more iterations before giving up
            if it > 8:
                break
    
    return best_chain


def _spiral_fold(sequence, dimension, seed=0):
    """Create a spiral-like initial conformation."""
    n = len(sequence)
    rng = random.Random(seed)
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    
    directions = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1)]
    dir_idx = rng.randint(0, 5)
    
    for i in range(1, n):
        placed = False
        # Try preferred direction first, then others
        order = [dir_idx % 6] + [d for d in range(6) if d != dir_idx % 6]
        rng.shuffle(order[1:])
        for d_idx in order:
            d = directions[d_idx % 6]
            new_pos = (chain[-1][0] + d[0], chain[-1][1] + d[1], chain[-1][2] + d[2])
            if new_pos not in occupied:
                chain.append(new_pos)
                occupied[new_pos] = i
                placed = True
                # Occasionally turn
                if rng.random() < 0.3:
                    dir_idx = (d_idx + 1) % 6
                break
        
        if not placed:
            # Stuck, try all 6 directions
            for d in directions:
                new_pos = (chain[-1][0] + d[0], chain[-1][1] + d[1], chain[-1][2] + d[2])
                if new_pos not in occupied:
                    chain.append(new_pos)
                    occupied[new_pos] = i
                    break
            else:
                break
    
    return chain


def fold(sequence, dimension):
    """Fold a sequence on the cubic lattice using multiple strategies."""
    _setup_deadline()
    n = len(sequence)
    
    if n <= 1:
        return [[0] * dimension]
    
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = 0
    
    # Strategy 1: Multiple spiral initializations
    for seed in range(5):
        if _time_left() < 3.0:
            break
        chain1 = _spiral_fold(sequence, dimension, seed=seed)
        if len(chain1) == n:
            e1 = _compute_energy(sequence, chain1)
            if best_chain is None or e1 < best_energy:
                best_chain = chain1
                best_energy = e1
    
    # Strategy 2: Beam search with different widths
    if _time_left() > 5.0:
        for bw in [30, 80, 150]:
            if _time_left() < 3.0:
                break
            max_t = min(8.0, _time_left() * 0.25)
            chain2 = _beam_search(sequence, dimension, rng, beam_width=bw, max_time=max_t)
            if chain2 and len(chain2) == n:
                e2 = _compute_energy(sequence, chain2)
                if best_chain is None or e2 < best_energy:
                    best_chain = chain2
                    best_energy = e2
    
    # Strategy 3: Local search refinement on best chain
    if best_chain is not None and _time_left() > 2.0:
        # Run local search multiple times with different seeds
        for ls_seed in range(6):
            if _time_left() < 2.0:
                break
            max_iters = min(4000, int(_time_left() * 120))
            refined = _local_search(sequence, best_chain, max_iters=max_iters, seed=ls_seed * 7 + n)
            if refined and len(refined) == n:
                er = _compute_energy(sequence, refined)
                if er < best_energy:
                    best_chain = refined
                    best_energy = er
    
    # Fallback: straight line
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
