"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 200.0  # Safety margin for 210s kill


def _time_left():
    return _GLOBAL_LIMIT - (time.time() - _GLOBAL_START)


def _neighbours(point):
    """Yield the 4 neighbours of a 2D lattice point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _energy(sequence, chain):
    """Compute the HP contact energy of a conformation.
    Energy = - (number of H-H contacts)
    """
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            x, y = p
            for q in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


class HPFolder:
    def __init__(self, sequence):
        self.seq = sequence
        self.n = len(sequence)
        self.h_mask = [1 if c == 'H' else 0 for c in sequence]
        
    def _calc_delta_e_pivot(self, chain, pivot_idx, angle, new_seg):
        """
        Calculate delta energy for a pivot move.
        chain: current chain list
        pivot_idx: index of pivot
        angle: 90, -90, 180
        new_seg: list of new positions for the moving segment
        """
        # Moving segment indices: pivot_idx + 1 to n-1
        # Fixed segment indices: 0 to pivot_idx
        
        # We need to count H-H contacts between Moving and Fixed.
        # Old contacts:
        # For each i in moving, check neighbors in fixed.
        # New contacts:
        # For each i in moving, check neighbors of new_pos[i] in fixed.
        
        # To be efficient, we can just iterate over the moving segment.
        # We need a fast way to check if a neighbor is in the fixed segment and is H.
        # We can precompute a set of fixed H positions.
        
        # However, doing this from scratch every time is O(N).
        # With N=100, O(N) is fine. 100 * 4 checks = 400 ops.
        # If we do 10000 iterations, that's 4M ops. Fast in Python? Maybe.
        # Let's try to optimize by using a global pos_to_idx map.
        
        # But wait, the map needs to be updated.
        # In this design, we calculate delta E *before* accepting.
        # So we have the current chain and the proposed new positions.
        
        # Let's use the global map approach in the main loop instead.
        # This function is a placeholder for the logic, but we'll integrate
        # the map updates in the main search loop for maximum speed.
        
        pass

    def _grow_greedy(self, rng, bias):
        """Greedy growth with bias."""
        n = self.n
        if n == 0: return []
        chain = [(0, 0)]
        occupied = {(0, 0): 0}
        
        for i in range(1, n):
            curr = chain[-1]
            options = []
            
            # Centroid for compactness
            cx = sum(p[0] for p in chain) / len(chain)
            cy = sum(p[1] for p in chain) / len(chain)
            
            for site in _neighbours(curr):
                if site in occupied:
                    continue
                
                gain = 0
                if self.h_mask[i]:
                    sx, sy = site
                    for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                        q = (sx+dx, sy+dy)
                        if q in occupied:
                            j = occupied[q]
                            if j < i - 1 and self.h_mask[j]:
                                gain += 1
                
                dist = math.sqrt((site[0] - cx)**2 + (site[1] - cy)**2)
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                
                score = bias * gain - 0.5 * dist + 0.1 * free + rng.random()
                options.append((score, site))
            
            if not options:
                return None
            
            best_site = max(options, key=lambda x: x[0])[1]
            chain.append(best_site)
            occupied[best_site] = i
            
        return chain


def fold(sequence, dimension):
    """
    Main entry point.
    """
    seq_len = len(sequence)
    if seq_len == 0:
        return []
    
    remaining = _time_left()
    if remaining < 5:
        # Emergency: straight line
        return [[i, 0] for i in range(seq_len)]
    
    rng = random.Random(12345 + seq_len)
    folder = HPFolder(sequence)
    
    # Phase 1: Initialization
    # Generate multiple diverse initial conformations
    candidates = []
    init_time = min(1.0, remaining * 0.1)
    start_t = time.time()
    
    # Try different biases
    biases = [0.5, 1.0, 2.0, 3.0, 1.5, 0.8, 2.5, 1.2]
    for b in biases:
        if time.time() - start_t > init_time:
            break
        chain = folder._grow_greedy(rng, b)
        if chain:
            e = _energy(sequence, chain)
            candidates.append((e, chain))
            
    # Also try a simple spiral-like random walk if greedy fails or for diversity
    if not candidates:
        # Fallback: random walk
        chain = [(0,0)]
        occupied = {(0,0):0}
        for i in range(1, seq_len):
            curr = chain[-1]
            valid = [s for s in _neighbours(curr) if s not in occupied]
            if not valid:
                break
            nxt = rng.choice(valid)
            chain.append(nxt)
            occupied[nxt] = i
        if len(chain) == seq_len:
            e = _energy(sequence, chain)
            candidates.append((e, chain))
            
    if not candidates:
        return [[i, 0] for i in range(seq_len)]
        
    # Sort by energy
    candidates.sort(key=lambda x: x[0])
    
    # Keep top 3
    top_candidates = candidates[:3]
    
    # Phase 2: Local Search (Simulated Annealing)
    best_chain = top_candidates[0][1]
    best_energy = top_candidates[0][0]
    
    search_time = min(10.0, remaining * 0.8)
    search_start = time.time()
    
    for e, chain in top_candidates:
        if time.time() - search_start > search_time:
            break
        if _time_left() < 15:
            break
            
        current_chain = list(chain)
        current_energy = e
        
        # Build maps
        pos_to_idx = {p: i for i, p in enumerate(current_chain)}
        occupied = set(current_chain)
        
        n = len(current_chain)
        h_mask = folder.h_mask
        
        T = 5.0
        T_min = 0.01
        # Cooling rate
        if n > 80:
            cooling = 0.9999
        elif n > 50:
            cooling = 0.9997
        else:
            cooling = 0.9995
            
        max_iters = 20000
        
        for iter in range(max_iters):
            if time.time() - search_start > search_time:
                break
            if _time_left() < 15:
                break
                
            r = rng.random()
            new_chain = None
            delta_e = 0
            
            if r < 0.8:
                # Pivot
                pivot_idx = rng.randint(1, n - 2)
                angle = rng.choice([90, -90, 180])
                
                pivot = current_chain[pivot_idx]
                fixed = current_chain[:pivot_idx + 1]
                segment = current_chain[pivot_idx + 1:]
                
                # Calculate new positions
                new_segment = []
                valid = True
                px, py = pivot
                fixed_set = set(fixed)
                
                # We need to check collision with fixed_set
                # And calculate delta E
                
                # Pre-calculate old contacts between segment and fixed
                # This is expensive if done naively.
                # Let's try a different approach:
                # 1. Remove segment from occupied/pos_to_idx
                # 2. Calculate new positions
                # 3. Check collision
                # 4. Calculate delta E
                # 5. If accepted, update occupied/pos_to_idx
                # 6. If not, re-add segment to occupied/pos_to_idx
                
                # This "remove and re-add" is safe but might be slow if many rejections.
                # However, it ensures correctness.
                
                # Let's try to just calculate delta E without modifying global state first?
                # No, we need to know if new positions collide with fixed.
                # We can check collision against fixed_set directly.
                
                for p in segment:
                    dx, dy = p[0] - px, p[1] - py
                    if angle == 90:
                        ndx, ndy = -dy, dx
                    elif angle == -90:
                        ndx, ndy = dy, -dx
                    else:
                        ndx, ndy = -dx, -dy
                    new_p = (px + ndx, py + ndy)
                    if new_p in fixed_set:
                        valid = False
                        break
                    new_segment.append(new_p)
                    
                if not valid:
                    continue
                    
                # Check self-collision of new_segment?
                # Since it's a rigid rotation of a self-avoiding segment, it remains self-avoiding.
                # So we only need to check against fixed.
                
                # Now calculate delta E
                # Old contacts: H-H pairs (i, j) where i in segment, j in fixed, and adjacent.
                # New contacts: H-H pairs (i, j) where i in new_segment, j in fixed, and adjacent.
                
                # We need to map positions to indices for fixed part to check H status?
                # Or just check if the position is in fixed and corresponds to H.
                # We have pos_to_idx for the whole chain.
                
                old_contacts = 0
                # Iterate over segment
                for idx, p in enumerate(segment):
                    if h_mask[pivot_idx + 1 + idx]:
                        sx, sy = p
                        for q in ((sx+1, sy), (sx-1, sy), (sx, sy+1), (sx, sy-1)):
                            j = pos_to_idx.get(q)
                            if j is not None and j <= pivot_idx: # j in fixed
                                if h_mask[j]:
                                    old_contacts += 1
                                    
                new_contacts = 0
                for idx, p in enumerate(new_segment):
                    if h_mask[pivot_idx + 1 + idx]:
                        sx, sy = p
                        for q in ((sx+1, sy), (sx-1, sy), (sx, sy+1), (sx, sy-1)):
                            j = pos_to_idx.get(q)
                            if j is not None and j <= pivot_idx: # j in fixed
                                if h_mask[j]:
                                    new_contacts += 1
                                    
                delta_e = old_contacts - new_contacts # Energy is -contacts, so delta E = -(new - old) = old - new
                
                # Accept/Reject
                accept = False
                if delta_e < 0:
                    accept = True
                else:
                    if T > T_min:
                        if rng.random() < math.exp(-delta_e / T):
                            accept = True
                            
                if accept:
                    # Update chain and maps
                    # Remove old segment
                    for p in segment:
                        del pos_to_idx[p]
                        occupied.discard(p)
                        
                    # Add new segment
                    for p in new_segment:
                        pos_to_idx[p] = pivot_idx + 1 + segment.index(p) # This is wrong, index is lost
                        # We need to update indices correctly.
                        # This is getting complicated.
                        pass
                    
                    # Let's simplify: just rebuild maps if accepted?
                    # No, that's O(N).
                    # Let's store the chain as a list and update the list.
                    # current_chain = fixed + new_segment
                    # Then rebuild pos_to_idx and occupied?
                    # For N=100, rebuild is fast.
                    
                    current_chain = fixed + new_segment
                    # Rebuild maps
                    pos_to_idx = {p: i for i, p in enumerate(current_chain)}
                    occupied = set(current_chain)
                    
                    current_energy += delta_e
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                        
            else:
                # Slither
                if n < 2:
                    continue
                prev = current_chain[-2]
                last = current_chain[-1]
                last_idx = n - 1
                occupied_without_last = set(current_chain[:-1])
                
                candidates = []
                for site in _neighbours(prev):
                    if site != last and site not in occupied_without_last:
                        candidates.append(site)
                        
                if not candidates:
                    continue
                    
                new_last = rng.choice(candidates)
                
                # Delta E
                # Old contacts involving last
                old_c = 0
                if h_mask[last_idx]:
                    lx, ly = last
                    for q in ((lx+1, ly), (lx-1, ly), (lx, ly+1), (lx, ly-1)):
                        j = pos_to_idx.get(q)
                        if j is not None and j < last_idx:
                            if h_mask[j]:
                                old_c += 1
                                
                new_c = 0
                if h_mask[last_idx]:
                    lx, ly = new_last
                    for q in ((lx+1, ly), (lx-1, ly), (lx, ly+1), (lx, ly-1)):
                        j = pos_to_idx.get(q)
                        if j is not None and j < last_idx:
                            if h_mask[j]:
                                new_c += 1
                                
                delta_e = old_c - new_c
                
                accept = False
                if delta_e < 0:
                    accept = True
                else:
                    if T > T_min:
                        if rng.random() < math.exp(-delta_e / T):
                            accept = True
                            
                if accept:
                    # Update
                    del pos_to_idx[last]
                    occupied.discard(last)
                    pos_to_idx[new_last] = last_idx
                    occupied.add(new_last)
                    current_chain[-1] = new_last
                    current_energy += delta_e
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                        
            T *= cooling
            
    # Phase 3: Final Polish
    if best_chain and _time_left() > 15:
        polish_time = min(2.0, _time_left() - 15)
        current = list(best_chain)
        curr_e = best_energy
        n = len(current)
        pos_to_idx = {p: i for i, p in enumerate(current)}
        h_mask = folder.h_mask
        p_start = time.time()
        
        while time.time() - p_start < polish_time and _time_left() > 15:
            improved_step = False
            # Try all pivots
            for pivot_idx in range(1, n-1):
                for angle in (90, -90, 180):
                    pivot = current[pivot_idx]
                    fixed = current[:pivot_idx + 1]
                    segment = current[pivot_idx + 1:]
                    
                    new_segment = []
                    valid = True
                    px, py = pivot
                    fixed_set = set(fixed)
                    
                    for p in segment:
                        dx, dy = p[0] - px, p[1] - py
                        if angle == 90:
                            ndx, ndy = -dy, dx
                        elif angle == -90:
                            ndx, ndy = dy, -dx
                        else:
                            ndx, ndy = -dx, -dy
                        new_p = (px + ndx, py + ndy)
                        if new_p in fixed_set:
                            valid = False
                            break
                        new_segment.append(new_p)
                            
                    if not valid:
                        continue
                        
                    # Delta E
                    old_c = 0
                    for idx, p in enumerate(segment):
                        if h_mask[pivot_idx + 1 + idx]:
                            sx, sy = p
                            for q in ((sx+1, sy), (sx-1, sy), (sx, sy+1), (sx, sy-1)):
                                j = pos_to_idx.get(q)
                                if j is not None and j <= pivot_idx:
                                    if h_mask[j]:
                                        old_c += 1
                                        
                    new_c = 0
                    for idx, p in enumerate(new_segment):
                        if h_mask[pivot_idx + 1 + idx]:
                            sx, sy = p
                            for q in ((sx+1, sy), (sx-1, sy), (sx, sy+1), (sx, sy-1)):
                                j = pos_to_idx.get(q)
                                if j is not None and j <= pivot_idx:
                                    if h_mask[j]:
                                        new_c += 1
                                        
                    delta_e = old_c - new_c
                    if delta_e < 0:
                        # Accept
                        current = fixed + new_segment
                        curr_e += delta_e
                        pos_to_idx = {p: i for i, p in enumerate(current)}
                        improved_step = True
                        break
                if improved_step:
                    break
            if not improved_step:
                break
                
        if curr_e < best_energy:
            best_energy = curr_e
            best_chain = current

    if best_chain is None:
        best_chain = [[i, 0] for i in range(seq_len)]
        
    return best_chain
# EVOLVE-BLOCK-END
