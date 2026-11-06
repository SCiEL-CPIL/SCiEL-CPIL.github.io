"""HP lattice protein folding: priority growth + simulated annealing with deadline guard."""
# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE = 205.0  # safety margin before 210 s wall-clock kill
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _time_left():
    return _DEADLINE - _elapsed()


def _neighbours(point):
    """Yield the 4 (or 2*dimension) lattice neighbours of a point."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute the HP contact energy (negative = better)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _grow(sequence, dimension, rng, deadline_check=None):
    """
    Priority-growth: build a self-avoiding walk that prefers sites creating
    new H-H contacts. Returns (chain, energy) or (None, 0) on failure.
    """
    n = len(sequence)
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}

    for i in range(1, n):
        if deadline_check and deadline_check():
            return None, 0

        best_site = None
        best_score = -float('inf')

        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue

            # Score: new H-H contacts (high weight) + free space (moderate) + random
            score = rng.random() * 0.1

            if sequence[i] == 'H':
                # Count new H-H contacts this site would create
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        score += 10.0

            # Free space: number of unoccupied neighbours (helps avoid dead-ends)
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            score += 1.0 * free

            if score > best_score:
                best_score = score
                best_site = site

        if best_site is None:
            return None, 0

        chain.append(best_site)
        occupied[best_site] = i

    energy = _energy(sequence, chain)
    return chain, energy


def _try_move(chain, i, new_pos, occupied, sequence):
    """
    Try moving residue i to new_pos. Returns True if valid (self-avoiding).
    Modifies chain and occupied in place if valid.
    """
    old_pos = chain[i]
    if new_pos == old_pos:
        return False

    # Check self-avoidance: new_pos must not be occupied by any other residue
    if new_pos in occupied:
        return False

    # Check that new_pos is adjacent to both neighbours (i-1 and i+1) if they exist
    if i > 0:
        prev = chain[i - 1]
        if abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) != 1:
            return False
    if i < len(chain) - 1:
        nxt = chain[i + 1]
        if abs(new_pos[0] - nxt[0]) + abs(new_pos[1] - nxt[1]) != 1:
            return False

    # Valid move: update
    del occupied[old_pos]
    chain[i] = new_pos
    occupied[new_pos] = i
    return True


def _try_reversal(chain, i, occupied, sequence):
    """
    Try reversing a 3-residue segment starting at index i (i, i+1, i+2 -> i, i+2, i+1).
    Actually, reversal means: chain[i], chain[i+1], chain[i+2] becomes chain[i], chain[i+2], chain[i+1]?
    No — a 3-residue reversal flips the direction: positions i, i+1, i+2 become i, i+2, i+1? 
    Let me think: original path goes A -> B -> C. Reversal makes it A -> C -> B? No, that's not a valid path.
    
    Correct 3-residue reversal: the segment of 3 consecutive residues (indices i, i+1, i+2) 
    is reversed in direction. So if the path was ... -> p_i -> p_{i+1} -> p_{i+2} -> ...,
    after reversal it becomes ... -> p_i -> p_{i+2} -> p_{i+1} -> ...? No, that breaks adjacency.
    
    Actually, a standard 3-residue reversal (or "flip") in protein folding takes three consecutive
    residues and reverses their order along the chain, but the lattice positions must remain
    a valid self-avoiding walk. The correct operation is:
    
    Original: ... -> A -> B -> C -> ...
    After reversal: ... -> A -> C' -> B' -> ... where C' and B' are new positions?
    
    No, that's not right either. Let me use the standard definition:
    A 3-residue reversal takes the segment (p_i, p_{i+1}, p_{i+2}) and replaces it with
    (p_i, p_{i+2}, p_{i+1})? That would require p_i adjacent to p_{i+2} and p_{i+2} adjacent to p_{i+1}.
    
    Actually, the simplest and most common move is a **single-residue move** and a **2-residue 
    reversal** (which just swaps the direction of a bond, but on a lattice that's trivial).
    
    Let me use a different approach: **kink moves**. A kink move takes two consecutive bonds and
    changes them to form a different shape. For 3 residues A-B-C, we can move B to a new position
    that is adjacent to both A and C. This is equivalent to a single-residue move on the middle residue.
    
    So let me just use single-residue moves and a **4-residue reversal** (which is more powerful).
    
    For a 4-residue reversal: segment (p_i, p_{i+1}, p_{i+2}, p_{i+3}) becomes
    (p_i, p_{i+3}, p_{i+2}, p_{i+1})? No, that breaks adjacency.
    
    The correct 4-residue reversal: the path ... -> A -> B -> C -> D -> ... becomes
    ... -> A -> D' -> C' -> B' -> ... where the new positions are computed by reflecting?
    
    This is getting complicated. Let me use a simpler and well-known move: **single-residue move**
    and **3-residue reversal** defined as follows:
    
    For a 3-residue reversal at index i: the segment (p_i, p_{i+1}, p_{i+2}) is replaced by
    (p_i, p_{i+2}, p_{i+1}) only if p_i is adjacent to p_{i+2} and p_{i+2} is adjacent to p_{i+1}.
    But p_{i+2} and p_{i+1} are already adjacent (they were consecutive), so we just need p_i 
    adjacent to p_{i+2}. This is a valid move if p_i and p_{i+2} are adjacent lattice points.
    
    Let me implement that.
    """
    n = len(chain)
    if i < 0 or i + 2 >= n:
        return False

    a = chain[i]
    b = chain[i + 1]
    c = chain[i + 2]

    # After reversal: chain[i] = a, chain[i+1] = c, chain[i+2] = b
    # Need: a adjacent to c, and c adjacent to b (already true since b and c were adjacent)
    if abs(a[0] - c[0]) + abs(a[1] - c[1]) != 1:
        return False

    # Check self-avoidance: c must not be occupied by any residue other than i+1 (which is moving away)
    # and b must not be occupied by any residue other than i+2 (which is moving away)
    # Actually, after the move, c will be at index i+1 and b at index i+2.
    # We need to ensure c is not occupied by any other residue, and b is not occupied by any other residue.
    # But b and c are currently occupied by i+1 and i+2, so we need to check that no OTHER residue is at c or b.
    # Since the chain is self-avoiding, no other residue is at b or c. So the only issue is if c or b 
    # is occupied by a residue outside the segment, which can't happen.
    
    # However, we also need to check that the new positions don't collide with neighbours outside the segment.
    # a is at index i, so its previous neighbour (if i > 0) is chain[i-1], which is unchanged.
    # b is now at index i+2, so its next neighbour (if i+2 < n-1) is chain[i+3], which must be adjacent to b.
    # b was originally at index i+1, and c was at index i+2. chain[i+3] was adjacent to c, not necessarily to b.
    
    if i + 3 < n:
        d = chain[i + 3]
        if abs(b[0] - d[0]) + abs(b[1] - d[1]) != 1:
            return False

    # Valid reversal: update
    del occupied[b]
    del occupied[c]
    chain[i + 1] = c
    chain[i + 2] = b
    occupied[c] = i + 1
    occupied[b] = i + 2
    return True


def _anneal(sequence, chain, dimension, rng, deadline_check=None):
    """
    Simulated annealing polish on a given chain. Returns the best chain found.
    """
    n = len(chain)
    if n < 3:
        return chain

    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy

    t = 10.0
    t_min = 0.01
    cooling = 0.995

    while t > t_min:
        if deadline_check and deadline_check():
            break

        # Try a single-residue move
        i = rng.randint(1, n - 2)  # avoid endpoints
        old_pos = chain[i]
        for new_pos in _neighbours(old_pos):
            if _try_move(chain, i, new_pos, occupied, sequence):
                new_energy = _energy(sequence, chain)
                delta = new_energy - current_energy
                if delta <= 0 or rng.random() < math.exp(-delta / t):
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                else:
                    # Reject: revert
                    del occupied[new_pos]
                    chain[i] = old_pos
                    occupied[old_pos] = i
                break  # only try one new position per move attempt

        # Try a 3-residue reversal
        i = rng.randint(0, n - 3)
        if _try_reversal(chain, i, occupied, sequence):
            new_energy = _energy(sequence, chain)
            delta = new_energy - current_energy
            if delta <= 0 or rng.random() < math.exp(-delta / t):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
            else:
                # Reject: revert
                a = chain[i]
                b = chain[i + 1]
                c = chain[i + 2]
                del occupied[b]
                del occupied[c]
                chain[i + 1] = c
                chain[i + 2] = b
                occupied[c] = i + 1
                occupied[b] = i + 2

        t *= cooling

    return best_chain


def fold(sequence, dimension):
    """
    Fold an HP sequence into a low-energy self-avoiding conformation.
    Uses priority growth + simulated annealing with a deadline guard.
    """
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]

    rng = random.Random(42 + len(sequence))
    best_chain = None
    best_energy = float('inf')

    # Stage 1: Multiple priority-growth attempts
    num_growths = 50
    for _ in range(num_growths):
        if _time_left() < 5.0:
            break
        chain, energy = _grow(sequence, dimension, rng, deadline_check=lambda: _time_left() < 5.0)
        if chain is not None:
            if energy < best_energy:
                best_energy = energy
                best_chain = chain

    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy(sequence, best_chain)

    # Stage 2: Simulated annealing polish
    if _time_left() > 10.0:
        polished = _anneal(sequence, best_chain, dimension, rng, deadline_check=lambda: _time_left() < 5.0)
        polished_energy = _energy(sequence, polished)
        if polished_energy < best_energy:
            best_energy = polished_energy
            best_chain = polished

    # Stage 3: If we have time, try more growths + annealing
    if _time_left() > 15.0:
        for _ in range(20):
            if _time_left() < 10.0:
                break
            chain, energy = _grow(sequence, dimension, rng, deadline_check=lambda: _time_left() < 10.0)
            if chain is not None:
                polished = _anneal(sequence, chain, dimension, rng, deadline_check=lambda: _time_left() < 10.0)
                polished_energy = _energy(sequence, polished)
                if polished_energy < best_energy:
                    best_energy = polished_energy
                    best_chain = polished

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
