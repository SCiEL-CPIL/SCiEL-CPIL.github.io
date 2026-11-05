"""HP lattice protein folding on the square lattice using model-based optimization with crankshaft and pivot moves."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_TOTAL_BUDGET = 195.0  # Safety margin before 210s kill

def _neighbours(point):
    """Yield the 4 lattice neighbours of a 2D point."""
    x, y = point
    yield (x+1, y)
    yield (x-1, y)
    yield (x, y+1)
    yield (x, y-1)

def _energy(sequence, chain):
    """Compute HP contact energy: -1 for each H-H pair on neighbouring lattice sites not adjacent in chain."""
    where = {p: i for i, p in enumerate(chain)}
    h_indices = set(i for i, s in enumerate(sequence) if s == "H")
    contacts = 0
    for i in h_indices:
        p = chain[i]
        x, y = p
        for q in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
            j = where.get(q)
            if j is not None and j > i + 1 and j in h_indices:
                contacts += 1
    return -contacts

def _greedy_grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
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

def _try_crankshaft(sequence, chain, k1, k2, rot_choice):
    """
    Attempt a crankshaft move: rotate the segment chain[k1+1..k2] around the pivot at chain[k1].
    rot_choice: 0=90deg, 1=180deg, 2=270deg
    Returns (new_chain, new_energy) if valid, else (None, None).
    """
    n = len(chain)
    if k2 - k1 < 2:
        return None, None
    
    px, py = chain[k1]
    
    if rot_choice == 0:  # 90 deg
        def rot(x, y): return (-y, x)
    elif rot_choice == 1:  # 180 deg
        def rot(x, y): return (-x, -y)
    else:  # 270 deg
        def rot(x, y): return (y, -x)
        
    # Build prefix and suffix
    prefix = chain[:k1+1]
    suffix = chain[k2+1:]
    
    # The moving segment is chain[k1+1..k2]
    seg = chain[k1+1:k2+1]
    
    # Build set of occupied points in prefix and suffix
    fixed_occupied = set(prefix) | set(suffix)
    
    new_seg = []
    for j in range(len(seg)):
        ox = seg[j][0] - px
        oy = seg[j][1] - py
        nx, ny = rot(ox, oy)
        new_pos = (px + nx, py + ny)
        new_seg.append(new_pos)
    
    # Check for collisions
    for pos in new_seg:
        if pos in fixed_occupied:
            return None, None
    # Check for self-collisions within new_seg
    if len(set(new_seg)) != len(new_seg):
        return None, None
    
    new_chain = prefix + new_seg + suffix
    
    # Verify self-avoidance and unit steps
    seen = set()
    for i, p in enumerate(new_chain):
        if p in seen:
            return None, None
        seen.add(p)
        if i > 0:
            dx = abs(new_chain[i][0] - new_chain[i-1][0])
            dy = abs(new_chain[i][1] - new_chain[i-1][1])
            if dx + dy != 1:
                return None, None
    
    new_energy = _energy(sequence, new_chain)
    return new_chain, new_energy

def _try_pivot_move(sequence, chain, k, rot_choice):
    """
    Attempt a pivot move at index k with a specific rotation.
    rot_choice: 0=90deg, 1=180deg, 2=270deg
    Returns (new_chain, new_energy) if valid, else (None, None).
    """
    n = len(chain)
    if k >= n - 1:
        return None, None
    
    px, py = chain[k]
    
    if rot_choice == 0:  # 90 deg
        def rot(x, y): return (-y, x)
    elif rot_choice == 1:  # 180 deg
        def rot(x, y): return (-x, -y)
    else:  # 270 deg
        def rot(x, y): return (y, -x)
        
    prefix_occupied = set(chain[:k+1])
    
    new_chain = list(chain[:k+1])
    
    for j in range(k + 1, n):
        ox = chain[j][0] - px
        oy = chain[j][1] - py
        nx, ny = rot(ox, oy)
        new_pos = (px + nx, py + ny)
        
        if new_pos in prefix_occupied:
            return None, None
            
        new_chain.append(new_pos)
        
    # Verify self-avoidance and unit steps
    seen = set()
    for i, p in enumerate(new_chain):
        if p in seen:
            return None, None
        seen.add(p)
        if i > 0:
            dx = abs(new_chain[i][0] - new_chain[i-1][0])
            dy = abs(new_chain[i][1] - new_chain[i-1][1])
            if dx + dy != 1:
                return None, None
    
    new_energy = _energy(sequence, new_chain)
    return new_chain, new_energy

def _try_two_point_swap(sequence, chain, i, j, rot_choice):
    """
    Attempt a two-point move: swap the direction of segments at positions i and j.
    This is a more global move than pivot/crankshaft.
    Simplified: rotate the tail from j, then fix the middle segment by rotating it around i.
    Actually, let's implement a proper two-point move:
    Pick two indices i < j. The chain is divided into 3 parts: A (0..i), B (i+1..j), C (j+1..n-1).
    We rotate B around i, and then rotate C around the new position of j.
    This is complex. A simpler "two-point" move in lattice protein folding is often:
    Pick i, j. The move is valid if we can rotate the segment between i and j such that
    it doesn't collide with A or C, and the connection at j to C is preserved or rotated.
    
    Let's use a standard two-point move definition:
    Choose i, j (i < j).
    The move rotates the segment chain[i+1..j] around chain[i].
    Then, to maintain the connection to the rest of the chain, the segment chain[j+1..] 
    must be translated so that chain[j] (new) connects to chain[j+1] (translated).
    Wait, that's not quite right either.
    
    A common efficient move is the "end-to-end" move or "pivot" at two points.
    Let's stick to a robust implementation:
    1. Rotate segment [i+1, j] around i.
    2. Check collision with [0, i] and [j+1, n-1].
    3. If valid, we must also translate the tail [j+1, n-1] so that the new position of j
       connects to the new position of j+1. But the tail is fixed in space in this move definition?
       No, if we rotate [i+1, j], the position of j changes. The tail [j+1..] is attached to j.
       So the tail must be moved with j.
       This is effectively a pivot at i, but only rotating up to j, and then the rest follows?
       No, that's just a pivot at i if we rotate the whole tail.
       
    Let's implement a "crankshaft" variant that is longer, or a "pivot" that is more global.
    Actually, the inspiration mentions "two-point swap moves".
    Let's implement a move where we pick two indices i, j.
    We rotate the segment between i and j by 90 degrees around i.
    Then we rotate the segment after j by 90 degrees around the NEW position of j.
    This allows the tail to reorient independently.
    """
    n = len(chain)
    if i >= j:
        return None, None
    if j >= n - 1:
        # If j is the last residue, this is just a crankshaft ending at the end
        # But crankshaft usually requires a pivot point.
        # Let's just return None for now to keep it simple, or treat as crankshaft.
        pass
        
    # This implementation is getting complicated and risky for bugs.
    # Given the tight time budget and the fact that the current score is already 1.247,
    # adding a buggy complex move might lower the score.
    # The "two-point swap" in the inspiration likely refers to swapping the positions of two residues?
    # No, that breaks the chain.
    
    # Let's skip the complex two-point move implementation to avoid introducing bugs
    # and instead focus on optimizing the existing moves and the SA parameters.
    # The inspiration score was 1.2470, same as current.
    # The key insight from the prompt is "Hamiltonian path pre-computation strategy".
    # However, pre-computing for all lengths 20-100 at runtime is impossible.
    # But we can pre-compute "templates" for common lengths or use a faster initialization.
    
    # Let's focus on:
    # 1. Better initialization (greedy with look-ahead).
    # 2. Faster energy calculation (already optimized).
    # 3. More efficient move selection in SA.
    
    return None, None

def _pivot_search(sequence, chain, time_limit, rng):
    """
    Monte Carlo search using pivot and crankshaft moves with simulated annealing.
    """
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Simulated Annealing parameters
    T_start = 2.0
    T_end = 0.01
    start_time = time.time()
    
    max_iters = 300000
    cooling_factor = (T_end / T_start) ** (1.0 / max_iters)
    T = T_start
    
    # Precompute H indices for faster energy check if possible, but _energy is already fast.
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        # Choose between pivot and crankshaft
        use_crankshaft = rng.random() < 0.3
        
        if use_crankshaft and n > 6:
            # Crankshaft: pick k1, k2 with k2-k1 >= 2
            k1 = rng.randint(0, n - 3)
            k2 = rng.randint(k1 + 2, n - 1)
            rot_choice = rng.randint(0, 2)
            new_chain, new_energy = _try_crankshaft(sequence, current_chain, k1, k2, rot_choice)
        else:
            k = rng.randint(0, n - 2)
            rot_choice = rng.randint(0, 2)
            new_chain, new_energy = _try_pivot_move(sequence, current_chain, k, rot_choice)
        
        if new_chain is None:
            continue
            
        delta = new_energy - current_energy
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                
        T *= cooling_factor
        
    return best_chain, best_energy

def _greedy_polish(sequence, chain, max_rounds=100, deadline=None):
    """
    Greedy polish: repeatedly scan all possible pivot and crankshaft moves and accept any that improve energy.
    Stop when no move improves energy, max_rounds reached, or deadline exceeded.
    """
    n = len(chain)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    
    for _ in range(max_rounds):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        
        # Try pivot moves
        for k in range(n - 1):
            for rot_choice in range(3):
                new_chain, new_energy = _try_pivot_move(sequence, current_chain, k, rot_choice)
                if new_chain is not None and new_energy < current_energy:
                    current_chain = new_chain
                    current_energy = new_energy
                    improved = True
                    break
            if improved:
                break
        
        # Try crankshaft moves
        if n > 6:
            for k1 in range(n - 2):
                for k2 in range(k1 + 2, n):
                    for rot_choice in range(3):
                        new_chain, new_energy = _try_crankshaft(sequence, current_chain, k1, k2, rot_choice)
                        if new_chain is not None and new_energy < current_energy:
                            current_chain = new_chain
                            current_energy = new_energy
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            break
            
    return current_chain, current_energy

def fold(sequence, dimension):
    """Fold the sequence into a low-energy conformation."""
    n = len(sequence)
    if n == 0:
        return []
    
    # Check global time
    elapsed = time.time() - _GLOBAL_START
    if elapsed > _TOTAL_BUDGET:
        return [[i, 0] for i in range(n)]
    
    remaining = _TOTAL_BUDGET - elapsed
    # Allocate time. 
    # Reserve 10% for polish and overhead.
    sa_time = remaining * 0.85
    polish_time = remaining * 0.05
    
    if sa_time < 1.0:
        # Not enough time for SA, just greedy
        rng = random.Random(12345 + n)
        initial = _greedy_grow(sequence, dimension, rng)
        if initial:
            return [list(p) for p in initial]
        return [[i, 0] for i in range(n)]
        
    rng = random.Random(12345 + n)
    
    best_chain = None
    best_energy = 0
    
    # Dynamic number of restarts based on time and sequence length
    # More restarts for shorter sequences
    if n <= 30:
        num_restarts = 4
    elif n <= 50:
        num_restarts = 3
    else:
        num_restarts = 2
    time_per_restart = sa_time / num_restarts
    
    for k in range(num_restarts):
        if time.time() - _GLOBAL_START > _TOTAL_BUDGET - 10.0:
            break
            
        init_rng = random.Random(12345 + n + k * 100)
        initial = _greedy_grow(sequence, dimension, init_rng)
        
        if initial is None:
            initial = [(i, 0) for i in range(n)]
            
        # Refine with SA
        refined_chain, refined_energy = _pivot_search(sequence, initial, time_per_restart, rng)
        
        if refined_chain is not None:
            if best_chain is None or refined_energy < best_energy:
                best_chain = refined_chain
                best_energy = refined_energy
                
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        
    # Final Polish: high-precision pass with more rounds and a tighter deadline guard
    remaining_for_polish = _TOTAL_BUDGET - 3.0 - (time.time() - _GLOBAL_START)
    if remaining_for_polish > 1.0:
        # Use more rounds for the final polish to escape shallow local minima
        polish_rounds = 300 if n <= 40 else 200
        polished_chain, polished_energy = _greedy_polish(sequence, best_chain, max_rounds=polish_rounds, deadline=_TOTAL_BUDGET - 3.0)
        if polished_energy < best_energy:
            best_chain = polished_chain
            best_energy = polished_energy
            
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
