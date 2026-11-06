"""HP lattice protein folding: contact-guided growth with pivot refinement."""
# EVOLVE-BLOCK-START
import time
import random

_T0 = time.time()
_DEADLINE = 200.0  # 210s limit, 10s safety margin


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _neighbours2(p):
    x, y = p
    return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]


def _compute_energy(seq, chain):
    """Compute HP energy: -number of H-H non-bonded contacts."""
    n = len(chain)
    if n < 3:
        return 0
    pos_to_idx = {}
    for i, p in enumerate(chain):
        pos_to_idx[p] = i
    h_indices = set(i for i, c in enumerate(seq) if c == 'H')
    contacts = 0
    for i in h_indices:
        x, y = chain[i]
        for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
            j = pos_to_idx.get((x+dx, y+dy))
            if j is not None and j > i + 1 and seq[j] == 'H':
                contacts += 1
    return -contacts


def _grow_contact_guided(seq, dimension, rng, contact_pairs):
    """Grow chain prioritizing sites that realize H-H contacts."""
    n = len(seq)
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    # Build lookup: for each position index, which contact pairs involve it
    # contact_pairs is list of (i, j) pairs where i < j and seq[i]==seq[j]=='H'
    
    for i in range(1, n):
        best_site = None
        best_score = -float('inf')
        best_free = 0
        
        last = chain[-1]
        for site in _neighbours2(last):
            if site in occupied:
                continue
            
            # Score: how many new H-H contacts would this create?
            gain = 0
            if seq[i] == 'H':
                x, y = site
                for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
                    j = occupied.get((x+dx, y+dy))
                    if j is not None and j <= i - 2 and seq[j] == 'H':
                        gain += 1
            
            # Count free neighbors (for future growth flexibility)
            free = sum(1 for nb in _neighbours2(site) if nb not in occupied)
            
            # Composite score: prioritize contacts, then flexibility
            score = gain * 10.0 + free * 0.1 + rng.random() * 0.5
            
            if score > best_score or (score == best_score and free > best_free):
                best_score = score
                best_site = site
                best_free = free
        
        if best_site is None:
            return None
        
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain


def _try_pivot(chain, occupied, i, rng):
    """Try a pivot move at position i (1 <= i <= n-2).
    Returns (new_chain, new_occupied) or (None, None) if invalid."""
    n = len(chain)
    if i < 1 or i >= n - 1:
        return None, None
    
    # The segment from i to n-1 will be rotated
    pivot = chain[i-1]  # fixed point
    seg = chain[i:]     # segment to rotate
    seg_occupied = {p: (i + k) for k, p in enumerate(seg)}
    
    # Try all 4 rotations around pivot
    for angle in range(4):
        new_seg = []
        new_occupied = {}
        valid = True
        
        # Remove original segment from occupied
        orig_occupied = dict(occupied)
        for p in seg:
            if p in orig_occupied:
                del orig_occupied[p]
        
        # Rotate each point in segment
        # Rotation around pivot by 90° * angle
        for p in seg:
            # Translate to origin, rotate, translate back
            dx = p[0] - pivot[0]
            dy = p[1] - pivot[1]
            
            if angle == 0:
                nx, ny = dx, dy
            elif angle == 1:
                nx, ny = -dy, dx
            elif angle == 2:
                nx, ny = -dx, -dy
            else:
                nx, ny = dy, -dx
            
            new_p = (pivot[0] + nx, pivot[1] + ny)
            
            if new_p in new_occupied:
                valid = False
                break
            if new_p in orig_occupied:
                valid = False
                break
            
            new_seg.append(new_p)
            new_occupied[new_p] = i + len(new_seg) - 1
        
        if not valid:
            continue
        
        # Check that the first rotated point connects to chain[i-1]
        if new_seg[0] != chain[i]:  # This should always be true by construction
            continue
        
        # Build new chain
        new_chain = chain[:i] + new_seg
        
        # Verify self-avoidance
        if len(set(new_chain)) != n:
            continue
        
        return new_chain, new_occupied
    
    return None, None


def _local_search(seq, chain, max_steps, rng):
    """Pivot-based local search to refine a conformation."""
    n = len(chain)
    current_energy = _compute_energy(seq, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    occupied = {p: i for i, p in enumerate(chain)}
    
    for step in range(max_steps):
        # Choose random pivot position
        i = rng.randint(1, n - 2)
        
        new_chain, new_occupied = _try_pivot(chain, occupied, i, rng)
        if new_chain is None:
            continue
        
        new_energy = _compute_energy(seq, new_chain)
        delta = new_energy - current_energy
        
        # Accept if better or with small probability if worse
        if delta < 0 or rng.random() < 0.1:
            chain = new_chain
            occupied = new_occupied
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_chain = list(chain)
                best_energy = current_energy
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold HP sequence using contact-guided growth with local refinement."""
    n = len(sequence)
    if n <= 2:
        return [[i, 0] for i in range(n)]
    
    # Identify H-H contact candidate pairs
    h_positions = [i for i, c in enumerate(sequence) if c == 'H']
    contact_pairs = []
    for a in range(len(h_positions)):
        for b in range(a+1, len(h_positions)):
            i, j = h_positions[a], h_positions[b]
            if j - i > 1:  # Not adjacent in chain
                contact_pairs.append((i, j))
    
    best_chain = None
    best_energy = 1  # Energy is negative, so 1 is "worse" than any valid fold
    
    # Phase 1: Multiple contact-guided growths
    rng = random.Random(42 + n)
    num_growths = 300
    
    for _ in range(num_growths):
        if _remaining() < 15.0:
            break
        
        chain = _grow_contact_guided(sequence, dimension, rng, contact_pairs)
        if chain is None:
            continue
        
        energy = _compute_energy(sequence, chain)
        if energy < best_energy:
            best_chain = chain
            best_energy = energy
    
    if best_chain is None:
        # Fallback: straight line
        return [[i, 0] for i in range(n)]
    
    # Phase 2: Local refinement on best solution
    if _remaining() > 20.0:
        refined_chain, refined_energy = _local_search(
            sequence, best_chain, 5000, rng
        )
        if refined_energy < best_energy:
            best_chain = refined_chain
            best_energy = refined_energy
    
    # Phase 3: More growths if time allows
    if _remaining() > 30.0:
        extra_growths = min(200, int(_remaining() * 5))
        for _ in range(extra_growths):
            if _remaining() < 10.0:
                break
            chain = _grow_contact_guided(sequence, dimension, rng, contact_pairs)
            if chain is None:
                continue
            energy = _compute_energy(sequence, chain)
            if energy < best_energy:
                best_chain = chain
                best_energy = energy
                # Refine immediately
                if _remaining() > 15.0:
                    refined_chain, refined_energy = _local_search(
                        sequence, best_chain, 2000, rng
                    )
                    if refined_energy < best_energy:
                        best_chain = refined_chain
                        best_energy = refined_energy
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
