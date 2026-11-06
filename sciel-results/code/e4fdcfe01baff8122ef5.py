"""HP lattice protein folding: beam search growth + optimized local search with delta energy."""
# EVOLVE-BLOCK-START
import time
import random
import heapq
import math

_T0 = time.time()
_DEADLINE = 198.0  # 210s wall-clock limit, 12s safety margin


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _neighbours(x, y):
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _compute_energy(seq, chain):
    """Compute energy: -count of H-H contacts on non-adjacent lattice neighbours."""
    where = {p: i for i, p in enumerate(chain)}
    h_set = set(i for i, c in enumerate(seq) if c == "H")
    contacts = 0
    for i in h_set:
        x, y = chain[i]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = where.get((x + dx, y + dy))
            if j is not None and j > i + 1 and j in h_set:
                contacts += 1
    return -contacts


def _delta_energy(seq, chain, pos, new_pos, idx):
    """
    Compute the change in energy when moving residue idx from pos to new_pos.
    Returns (delta_energy, valid) where delta_energy is new_energy - old_energy.
    valid is False if the move would create a collision.
    """
    n = len(chain)
    # Check validity: new_pos must not be occupied by another residue
    # and must be adjacent to chain[idx-1] and chain[idx+1] (if they exist)
    occupied = set(chain)
    if new_pos in occupied and new_pos != pos:
        return 0, False
    
    # Check adjacency to neighbors
    if idx > 0:
        px, py = chain[idx - 1]
        if abs(new_pos[0] - px) + abs(new_pos[1] - py) != 1:
            return 0, False
    if idx < n - 1:
        nx, ny = chain[idx + 1]
        if abs(new_pos[0] - nx) + abs(new_pos[1] - ny) != 1:
            return 0, False
    
    # Compute delta energy
    # Remove old contacts involving idx, add new contacts involving idx
    h_set = set(i for i, c in enumerate(seq) if c == "H")
    if idx not in h_set:
        return 0, True  # No energy change for P residues
    
    # Build position->index map for quick lookup
    where = {p: i for i, p in enumerate(chain)}
    
    # Contacts lost when moving from pos
    lost = 0
    x, y = pos
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        j = where.get((x + dx, y + dy))
        if j is not None and j != idx and abs(j - idx) > 1 and j in h_set:
            lost += 1
    
    # Contacts gained when moving to new_pos
    gained = 0
    x, y = new_pos
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        j = where.get((x + dx, y + dy))
        if j is not None and j != idx and abs(j - idx) > 1 and j in h_set:
            gained += 1
    
    delta = -gained + lost  # energy = -contacts, so delta_energy = -(new_contacts - old_contacts) = old_contacts - new_contacts
    return delta, True


def _grow_beam_search(sequence, dimension, beam_width=50):
    """
    Beam search growth: maintain top beam_width partial chains,
    extending each by one residue at a time, scoring by energy + heuristic.
    """
    n = len(sequence)
    h_set = set(i for i, c in enumerate(sequence) if c == "H")
    
    # Each beam entry: (score, chain, occupied_set)
    # score = energy_so_far + heuristic_estimate
    beam = [((0, 0), {(0, 0): 0}, 0.0)]  # (position, occupied_dict, score)
    # Actually, let's store chains as lists for simplicity
    beam = [(0.0, [(0, 0)], {(0, 0): 0})]  # (score, chain, occupied)
    
    for i in range(1, n):
        if _remaining() < 5.0:
            break
        new_beam = []
        for score, chain, occupied in beam:
            x, y = chain[-1]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                site = (x + dx, y + dy)
                if site in occupied:
                    continue
                
                # Compute immediate energy gain
                gain = 0
                if i in h_set:
                    for adx, ady in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        q = (site[0] + adx, site[1] + ady)
                        if q in occupied:
                            idx = occupied[q]
                            if idx < i - 1 and idx in h_set:
                                gain += 1
                
                # Heuristic: estimate remaining potential contacts
                # Count H residues remaining that could potentially contact this site
                remaining_h = sum(1 for j in range(i + 1, n) if sequence[j] == "H")
                heuristic = 0.1 * remaining_h  # weak heuristic
                
                # Free space score
                free = 0
                for adx, ady in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    if (site[0] + adx, site[1] + ady) not in occupied:
                        free += 1
                
                new_score = -gain + heuristic + 0.05 * free  # lower is better
                new_chain = chain + [site]
                new_occupied = dict(occupied)
                new_occupied[site] = i
                
                new_beam.append((new_score, new_chain, new_occupied))
        
        if not new_beam:
            break
        
        # Keep top beam_width by score (lowest score = best)
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]
    
    if not beam:
        return None
    
    # Return the best chain
    best = min(beam, key=lambda x: _compute_energy(sequence, x[1]))
    return best[1]


def _pivot_move(chain, rng):
    """
    Pivot move: pick a segment of length L and rotate it 90 degrees.
    Try all 4 rotations and return the first valid one.
    """
    n = len(chain)
    if n < 4:
        return None
    
    # Pick pivot position and segment length
    L = rng.randint(2, min(6, n - 2))
    a = rng.randint(1, n - L - 1)  # segment starts at a, ends at a+L-1
    # The pivot is at chain[a-1]
    pivot = chain[a - 1]
    px, py = pivot
    
    # The segment to rotate: chain[a:a+L]
    segment = chain[a:a + L]
    
    # Try 4 rotations (0, 90, 180, 270 degrees)
    for rot in range(4):
        if rot == 0:
            continue  # Skip original position
        new_segment = []
        for (x, y) in segment:
            # Rotate (x,y) around pivot by 90*rot degrees
            dx = x - px
            dy = y - py
            if rot == 1:  # 90 degrees CCW
                nx, ny = -dy, dx
            elif rot == 2:  # 180 degrees
                nx, ny = -dx, -dy
            else:  # 270 degrees CCW
                nx, ny = dy, -dx
            new_segment.append((px + nx, py + ny))
        
        # Check if new segment is valid
        new_chain = chain[:a] + new_segment + chain[a + L:]
        
        # Check self-avoidance
        seen = set()
        valid = True
        for i, p in enumerate(new_chain):
            if p in seen:
                valid = False
                break
            seen.add(p)
            if i > 0:
                if abs(p[0] - new_chain[i - 1][0]) + abs(p[1] - new_chain[i - 1][1]) != 1:
                    valid = False
                    break
        
        if valid:
            return new_chain
    
    return None


def _reversal_move(chain, rng):
    """Random reversal move."""
    n = len(chain)
    if n < 4:
        return None
    L = rng.randint(2, min(8, n - 2))
    a = rng.randint(0, n - L - 1)
    b = a + L
    new_chain = chain[:a] + chain[a:b + 1][::-1] + chain[b + 1:]
    
    # Check validity
    seen = set()
    for i, p in enumerate(new_chain):
        if p in seen:
            return None
        seen.add(p)
        if i > 0:
            if abs(p[0] - new_chain[i - 1][0]) + abs(p[1] - new_chain[i - 1][1]) != 1:
                return None
    
    return new_chain


def _local_search(sequence, chain, rng, max_iters=500):
    """
    Local search with pivot, reversal, and single-residue moves.
    Uses delta energy for fast evaluation.
    """
    best_chain = list(chain)
    best_energy = _compute_energy(sequence, best_chain)
    n = len(best_chain)
    no_improve = 0
    max_no_improve = 50
    
    for it in range(max_iters):
        if _remaining() < 3.0:
            break
        
        if no_improve >= max_no_improve:
            # Perturb to escape local minimum
            perturbed = _reversal_move(best_chain, rng)
            if perturbed is not None:
                best_chain = perturbed
                best_energy = _compute_energy(sequence, best_chain)
            no_improve = 0
            max_no_improve = min(100, max_no_improve * 1.3)
        
        # Choose move type
        r = rng.random()
        if r < 0.4:
            new_chain = _pivot_move(best_chain, rng)
        elif r < 0.7:
            new_chain = _reversal_move(best_chain, rng)
        else:
            # Single residue move (kink)
            i = rng.randint(1, n - 2)
            x, y = best_chain[i]
            moved = False
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                site = (x + dx, y + dy)
                # Check if site is free and move is valid
                delta, valid = _delta_energy(sequence, best_chain, (x, y), site, i)
                if valid:
                    new_chain = best_chain[:]
                    new_chain[i] = site
                    # Full validation
                    seen = set()
                    ok = True
                    for j, p in enumerate(new_chain):
                        if p in seen:
                            ok = False
                            break
                        seen.add(p)
                        if j > 0:
                            if abs(p[0] - new_chain[j - 1][0]) + abs(p[1] - new_chain[j - 1][1]) != 1:
                                ok = False
                                break
                    if ok:
                        new_chain = new_chain
                        moved = True
                        break
            if not moved:
                new_chain = None
        
        if new_chain is None:
            no_improve += 1
            continue
        
        e = _compute_energy(sequence, new_chain)
        if e < best_energy:
            best_chain = new_chain
            best_energy = e
            no_improve = 0
        else:
            no_improve += 1
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold the sequence using beam search growth + local search with time guard.
    """
    rng = random.Random(12345 + len(sequence))
    best = None
    best_energy = 0
    n = len(sequence)
    
    # Phase 1: Beam search growth with multiple restarts
    num_growths = 20
    for r in range(num_growths):
        if _remaining() < 10.0:
            break
        
        # Vary beam width for diversity
        bw = 30 + rng.randint(0, 40)
        chain = _grow_beam_search(sequence, dimension, beam_width=bw)
        if chain is None:
            continue
        
        e = _compute_energy(sequence, chain)
        if e < best_energy:
            best_energy = e
            best = chain
        
        # Periodic local search
        if r % 2 == 0 and best is not None and _remaining() > 15.0:
            iters = 200 if _remaining() > 40 else 100
            improved, imp_e = _local_search(sequence, best, rng, max_iters=iters)
            if imp_e < best_energy:
                best_energy = imp_e
                best = improved
    
    # Phase 2: Intensive local search on best
    if best is not None and _remaining() > 5.0:
        iters = 500 if _remaining() > 20 else 200
        improved, imp_e = _local_search(sequence, best, rng, max_iters=iters)
        if imp_e < best_energy:
            best_energy = imp_e
            best = improved
    
    # Phase 3: Final polish
    if best is not None and _remaining() > 3.0:
        improved, imp_e = _local_search(sequence, best, rng, max_iters=200)
        if imp_e < best_energy:
            best_energy = imp_e
            best = improved
    
    if best is None:
        # Fallback: straight line
        best = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
