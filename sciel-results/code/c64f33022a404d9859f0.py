"""HP lattice protein folding: Iterated Local Search with incremental energy updates."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
# Safety margin: Evaluator kills at 210s. We target finishing by 205s.
_GLOBAL_DEADLINE = _GLOBAL_START + 205.0

_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _remaining_time():
    return max(0.0, _GLOBAL_DEADLINE - time.time())


def _energy(sequence, chain):
    """Compute energy: -1 for each H-H contact on non-adjacent lattice neighbours."""
    n = len(chain)
    if n < 2:
        return 0
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            x, y = p
            for dx, dy in _DELTAS:
                q = (x + dx, y + dy)
                j = where.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2


def _energy_delta_reversal(sequence, chain, i, j):
    """
    Calculate the change in energy when reversing the segment chain[i:j+1].
    Internal contacts within the segment remain the same in count.
    Only contacts between the segment and the rest of the chain change.
    """
    n = len(chain)
    if i >= j:
        return 0
    
    # Identify H indices in the segment
    h_in_seg = set()
    for k in range(i, j + 1):
        if sequence[k] == 'H':
            h_in_seg.add(k)
    
    # Identify H indices outside the segment
    h_outside = set()
    for k in range(n):
        if k < i or k > j:
            if sequence[k] == 'H':
                h_outside.add(k)
    
    if not h_in_seg or not h_outside:
        return 0

    # Build position map for the current chain
    where = {}
    for k, p in enumerate(chain):
        where[p] = k

    # Count contacts between segment and outside in the original state
    old_contacts = 0
    # For each H in segment, check its neighbors in the outside
    # We only need to check neighbors that are in the outside set
    for k in h_in_seg:
        x, y = chain[k]
        for dx, dy in _DELTAS:
            q = (x + dx, y + dy)
            m = where.get(q)
            if m is not None and m in h_outside:
                # Ensure they are not adjacent in the chain (they can't be if one is in seg and other outside, except at boundaries)
                # Wait, boundary nodes i-1 and i are adjacent. i and i+1 are adjacent.
                # If k=i and m=i-1, they are adjacent. We must exclude chain-adjacent pairs.
                if abs(k - m) > 1:
                    old_contacts += 1

    # Perform the reversal on a copy of the relevant parts to get new positions
    # We don't want to modify the original chain yet, just compute new positions for the segment
    # The segment is reversed.
    # New position of chain[k] for k in [i, j] is the position of chain[i+j-k] in the original? No.
    # Let's just construct the new chain segment.
    new_segment = chain[i:j+1][::-1]
    
    # We need to check contacts between the new segment and the outside.
    # The outside chain parts remain at the same coordinates.
    # The segment moves to new coordinates.
    
    # Actually, it's easier to just calculate the new energy of the whole chain if we were to reverse,
    # but that's O(N). We want O(L) where L is segment length.
    
    # Let's map the new positions of the segment nodes.
    # The outside nodes stay put.
    # The segment nodes move.
    
    # Let's create a dict for the new positions of the segment nodes.
    new_pos_in_seg = {}
    for idx, p in enumerate(new_segment):
        new_pos_in_seg[p] = i + idx
        
    # Wait, the coordinates change.
    # Original segment: chain[i], chain[i+1], ..., chain[j]
    # New segment: chain[j], chain[j-1], ..., chain[i]
    # The coordinates of the nodes are the same, just the order in the chain is reversed.
    # So the set of occupied coordinates by the segment is the same!
    # This means contacts between two nodes both inside the segment are unchanged in count.
    # Contacts between a node inside and a node outside depend on coordinates.
    # Since the set of coordinates occupied by the segment is identical, 
    # the set of potential geometric neighbors for any node in the segment is identical.
    # However, the *identity* of the node at that coordinate changes.
    
    # Let's re-evaluate:
    # Node at coord C in old segment was index k_old.
    # Node at coord C in new segment is index k_new such that k_new + k_old = i + j.
    
    # So for each coordinate C in the segment:
    # Old occupant: k_old
    # New occupant: k_new = i + j - k_old
    
    # We need to check for each C in segment, its neighbors in the outside.
    # Let N(C) be the set of outside indices adjacent to C.
    # Old contact count for C: count of k_old in H_in_seg such that any m in N(C) is in H_outside.
    # Actually, we count pairs (k, m) where k in seg, m in outside, adjacent, both H.
    
    # Let's just iterate over the segment coordinates.
    seg_coords = set(chain[i:j+1])
    
    new_contacts = 0
    for C in seg_coords:
        # Find the old index at C
        # We can use 'where' from before, but 'where' maps coord->index.
        k_old = where.get(C)
        if k_old is None or not (i <= k_old <= j):
            continue # Should not happen
        
        # Find the new index at C
        k_new = i + j - k_old
        
        # Check neighbors of C
        x, y = C
        for dx, dy in _DELTAS:
            q = (x + dx, y + dy)
            m = where.get(q)
            if m is not None and m in h_outside:
                if abs(k_new - m) > 1:
                    # Check if both are H in the new configuration
                    # k_new is in segment, so sequence[k_new] is H if k_new in h_in_seg
                    # m is outside, so sequence[m] is H if m in h_outside
                    if sequence[k_new] == 'H' and sequence[m] == 'H':
                        new_contacts += 1

    return new_contacts - old_contacts


def _energy_delta_pivot(sequence, chain, i, angle):
    """
    Calculate energy change for pivoting the tail chain[i+1:] around chain[i] by angle (1, 2, or 3 clockwise).
    This changes the coordinates of the tail.
    Internal contacts in the tail change.
    Contacts between tail and head change.
    """
    n = len(chain)
    if i <= 0 or i >= n - 1:
        return 0
    
    px, py = chain[i]
    tail = chain[i+1:]
    
    # Rotate tail
    rotated = []
    for tx, ty in tail:
        rx, ry = tx - px, ty - py
        for _ in range(angle % 4):
            rx, ry = -ry, rx
        rotated.append((px + rx, py + ry))
    
    new_chain = chain[:i+1] + rotated
    
    # Check self-avoidance of new chain
    # If not self-avoiding, return a large penalty or invalid flag
    # For now, let's assume we check validity outside or return a large positive delta if invalid
    # Actually, it's better to return None if invalid, and let the caller handle it.
    # But _energy_delta is expected to return a number.
    # Let's return a very large positive number if invalid, so it's never accepted.
    
    if len(set(new_chain)) != n:
        return 1000000
    
    # Calculate energy of new chain
    # This is O(N). For pivots, it's hard to do incrementally faster than O(N) without complex bookkeeping.
    # Given N is small (<=100), O(N) is fine.
    # We can just compute the full energy difference.
    old_e = _energy(sequence, chain)
    new_e = _energy(sequence, new_chain)
    return new_e - old_e


def _backtrack_fold(sequence, time_limit, rng):
    """
    Fold sequence using Iterated Local Search (ILS) with fast incremental moves.
    """
    n = len(sequence)
    if n < 2:
        return [(0, 0)] * n

    is_h = [c == 'H' for c in sequence]
    
    # 1. Generate diverse initial solutions
    # Use a simple random walk with backtracking or greedy growth
    def _greedy_grow(seq, rng):
        nn = len(seq)
        if nn < 2:
            return [(0, 0)] * nn
        start = (0, 0)
        chain = [start]
        occupied = {start: 0}
        
        for i in range(1, nn):
            options = []
            x, y = chain[-1]
            for site in _DELTAS:
                cand = (x + site[0], y + site[1])
                if cand in occupied:
                    continue
                options.append(cand)
            
            if not options:
                return None
            
            # Pick randomly to ensure diversity
            best_site = rng.choice(options)
            occupied[best_site] = i
            chain.append(best_site)

        return chain

    initial_chains = []
    growth_deadline = time.time() + min(1.0, time_limit * 0.05)
    for _ in range(10):
        if time.time() > growth_deadline:
            break
        chain = _greedy_grow(sequence, rng)
        if chain is not None:
            initial_chains.append(chain)

    if not initial_chains:
        initial_chains.append([(i, 0) for i in range(n)])

    # Evaluate and sort
    scored = []
    for chain in initial_chains:
        e = _energy(sequence, chain)
        scored.append((e, chain))
    scored.sort(key=lambda x: x[0])
    
    best_chain = scored[0][1]
    best_energy = scored[0][0]
    
    # 2. Iterated Local Search
    deadline = time.time() + time_limit
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        if _remaining_time() < 1.0:
            break
        
        # Local Search
        current_chain = best_chain[:]
        current_energy = best_energy
        T = 1.0
        cooling = 0.999
        min_t = 0.05
        ls_deadline = time.time() + min(5.0, (deadline - time.time()) * 0.2) # Limit LS time
        
        while time.time() < ls_deadline:
            if _remaining_time() < 2.0:
                break
            
            move_type = rng.random()
            if move_type < 0.7:
                # Reversal
                i = rng.randint(0, n - 2)
                j = rng.randint(i + 1, n - 1)
                delta_e = _energy_delta_reversal(sequence, current_chain, i, j)
                if delta_e == 1000000: # Invalid
                    continue
                
                # Apply move
                current_chain[i:j+1] = current_chain[i:j+1][::-1]
                new_energy = current_energy + delta_e
                
                # Check if valid (reversal always maintains self-avoidance if it was valid before? 
                # Yes, because the set of coordinates is the same. 
                # Wait, if the segment touches itself in a way that creates a collision with the rest?
                # No, the coordinates are the same. The only issue is if the reversal causes a node to be adjacent to a node it wasn't before?
                # No, adjacency in the chain is what matters for the "non-adjacent" constraint in energy.
                # The geometric adjacency is the same.
                # So reversal is always valid if the original was valid.
                
                if new_energy <= current_energy or (T > min_t and rng.random() < math.exp(-new_energy / T)):
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = current_chain[:]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    # Revert
                    current_chain[i:j+1] = current_chain[i:j+1][::-1]
                    no_improve_count += 1
            else:
                # Pivot
                i = rng.randint(1, n - 2)
                angle = rng.choice([1, 2, 3])
                
                # Check validity of pivot
                px, py = current_chain[i]
                tail = current_chain[i+1:]
                rotated = []
                for tx, ty in tail:
                    rx, ry = tx - px, ty - py
                    for _ in range(angle % 4):
                        rx, ry = -ry, rx
                    rotated.append((px + rx, py + ry))
                new_chain = current_chain[:i+1] + rotated
                
                if len(set(new_chain)) != n:
                    no_improve_count += 1
                    continue
                
                new_energy = _energy(sequence, new_chain)
                delta_e = new_energy - current_energy
                
                if delta_e <= 0 or (T > min_t and rng.random() < math.exp(-delta_e / T)):
                    current_chain = new_chain
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = current_chain[:]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
            
            T *= cooling
            if T < min_t:
                break
        
        # Perturbation
        if no_improve_count > max_no_improve or _remaining_time() < 5.0:
            # Perturb best_chain
            perturbed = best_chain[:]
            # Apply 2-3 random reversals
            for _ in range(rng.randint(2, 4)):
                i = rng.randint(0, n - 2)
                j = rng.randint(i + 1, n - 1)
                perturbed[i:j+1] = perturbed[i:j+1][::-1]
            
            # Check validity
            if len(set(perturbed)) == n:
                best_chain = perturbed
                best_energy = _energy(sequence, best_chain)
            
            no_improve_count = 0
            max_no_improve = int(max_no_improve * 0.8)
            if max_no_improve < 50:
                break

    return best_chain


def fold(sequence, dimension):
    """
    Fold the sequence into a low-energy conformation.
    """
    rng = random.Random(42)
    n = len(sequence)

    if n < 2:
        return [[0, 0]] * n

    # Reserve 5s for validation and overhead
    time_limit = max(0.5, _remaining_time() - 5.0)

    best_chain = _backtrack_fold(sequence, time_limit, rng)

    # Validate: ensure self-avoiding and unit steps
    if best_chain is None or len(best_chain) != n:
        best_chain = [(i, 0) for i in range(n)]

    seen = set()
    valid = True
    for i, (x, y) in enumerate(best_chain):
        if (x, y) in seen:
            valid = False
            break
        seen.add((x, y))
        if i > 0:
            px, py = best_chain[i-1]
            if abs(px - x) + abs(py - y) != 1:
                valid = False
                break

    if not valid:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
