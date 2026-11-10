"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE = time.time() + 175.0  # 35s safety margin before 210s kill


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _neighbours(point):
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _compute_energy_full(sequence, chain):
    """Compute HP contact energy: -1 per non-adjacent H-H contact."""
    n = len(chain)
    if n < 3:
        return 0
    pos_to_idx = {}
    for i, p in enumerate(chain):
        pos_to_idx[p] = i
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y = chain[i]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = pos_to_idx.get((x + dx, y + dy))
            if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                contacts += 1
    return -(contacts // 2)


class _ChainState:
    """Maintains chain, occupied map, and energy with O(1) move updates."""
    
    def __init__(self, sequence, chain):
        self.seq = sequence
        self.n = len(chain)
        self.chain = list(chain)
        self.occupied = {}
        for i, p in enumerate(chain):
            self.occupied[p] = i
        self.energy = self._calc_energy()
    
    def _calc_energy(self):
        seq = self.seq
        n = self.n
        chain = self.chain
        occ = self.occupied
        contacts = 0
        for i in range(n):
            if seq[i] != 'H':
                continue
            x, y = chain[i]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                j = occ.get((x + dx, y + dy))
                if j is not None and abs(i - j) > 1 and seq[j] == 'H':
                    contacts += 1
        return -(contacts // 2)
    
    def _contacts_for(self, i):
        """Count H-H contacts involving residue i (each contact counted once from i's side)."""
        seq = self.seq
        chain = self.chain
        occ = self.occupied
        if seq[i] != 'H':
            return 0
        x, y = chain[i]
        count = 0
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = occ.get((x + dx, y + dy))
            if j is not None and abs(i - j) > 1 and seq[j] == 'H':
                count += 1
        return count
    
    def _remove_residue_contacts(self, i):
        """Remove residue i from occupied, return contact count it had."""
        c = self._contacts_for(i)
        del self.occupied[self.chain[i]]
        return c
    
    def _add_residue_contacts(self, i, pos):
        """Add residue i at pos to occupied, return contact count it gains."""
        self.occupied[pos] = i
        return self._contacts_for(i)
    
    def try_single_move(self, i, new_pos, rng):
        """Try moving residue i to new_pos. Returns (success, delta_e)."""
        n = self.n
        chain = self.chain
        occ = self.occupied
        
        if i == 0 or i == n - 1:
            return False, 0
        if new_pos in occ:
            return False, 0
        
        old_pos = chain[i]
        prev_pos = chain[i - 1]
        next_pos = chain[i + 1]
        
        # Check adjacency
        if _manhattan(new_pos, prev_pos) != 1:
            return False, 0
        if _manhattan(new_pos, next_pos) != 1:
            return False, 0
        # Check that new_pos doesn't collide with prev or next (it shouldn't since they're adjacent)
        if new_pos == prev_pos or new_pos == next_pos:
            return False, 0
        
        # Remove i
        del occ[old_pos]
        
        # Check if any neighbor of new_pos is occupied (excluding prev and next which are fine)
        # Actually we just need to check new_pos isn't occupied - already done
        # But also need to make sure the move doesn't create self-intersection
        # Since we removed i and new_pos wasn't occupied, we're fine
        
        # Add i at new_pos
        occ[new_pos] = i
        chain[i] = new_pos
        
        # Compute delta energy
        # Old contacts involving i (before removal) + old contacts of neighbors that were with i
        # Simpler: recompute contacts for i and its 4 neighbors
        delta = 0
        # Contacts lost: neighbors of old_pos that are H and not adjacent in chain
        x, y = old_pos
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = occ.get((x + dx, y + dy))
            if j is not None and abs(i - j) > 1 and self.seq[j] == 'H':
                delta -= 1  # lost a contact (energy was -contacts, losing one means energy increases by 1)
        
        # Contacts gained: neighbors of new_pos
        x, y = new_pos
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = occ.get((x + dx, y + dy))
            if j is not None and abs(i - j) > 1 and self.seq[j] == 'H':
                delta += 1  # gained a contact (energy decreases by 1)
        
        # Wait, I need to be more careful. Let me just recompute.
        # Actually the delta approach above is wrong because when I remove i from old_pos,
        # the contacts I'm counting at old_pos include contacts with residues that are still there.
        # But I already removed i, so occ has i at new_pos now.
        # Let me just recompute energy for affected residues.
        
        # Undo and do it properly
        chain[i] = old_pos
        del occ[new_pos]
        occ[old_pos] = i
        
        # Proper approach: compute contacts before and after
        # Before: contacts involving i at old_pos
        old_contacts = self._contacts_for(i)
        
        # After removing i and placing at new_pos
        del occ[old_pos]
        occ[new_pos] = i
        chain[i] = new_pos
        new_contacts = self._contacts_for(i)
        
        # Also need to account for contacts that other residues had with i
        # When i moves from old_pos to new_pos:
        # - Residues near old_pos lose their contact with i
        # - Residues near new_pos gain their contact with i
        # But _contacts_for(i) only counts from i's perspective
        # A contact between i and j is counted in _contacts_for(i) if j is H and near i
        # And in _contacts_for(j) if i is H and near j
        # Since we count each pair once (by dividing by 2), we need to be careful
        
        # Actually, the total energy = -sum over all H-H pairs that are neighbors
        # = -(1/2) * sum over all H residues i of (number of H neighbors j with |i-j|>1)
        # So if I move i, the change in the sum is:
        # - old: i contributed old_contacts_i, and each neighbor j of i at old_pos contributed 1 to j's count
        # - new: i contributes new_contacts_i, and each neighbor j of i at new_pos contributes 1 to j's count
        
        # The total sum changes by: (new_contacts_i - old_contacts_i) + (change in neighbors' counts)
        # Neighbors' counts change because i moved:
        # - For each H residue j near old_pos (not adjacent in chain): j loses 1 from its count
        # - For each H residue j near new_pos (not adjacent in chain): j gains 1 to its count
        
        # So delta_sum = (new_contacts_i - old_contacts_i) + (new_neighbors_gain - old_neighbors_loss)
        # where new_neighbors_gain = number of H residues near new_pos not adjacent to i
        # and old_neighbors_loss = number of H residues near old_pos not adjacent to i
        
        # But wait, new_contacts_i already counts H residues near new_pos not adjacent to i
        # And old_contacts_i already counts H residues near old_pos not adjacent to i
        # So delta_sum = (new_contacts_i - old_contacts_i) + (new_contacts_i - old_contacts_i)
        #                = 2 * (new_contacts_i - old_contacts_i)
        
        # Therefore delta_energy = -delta_sum/2 = -(new_contacts_i - old_contacts_i) = old_contacts_i - new_contacts_i
        
        self.energy += old_contacts_i - new_contacts_i
        delta_e = old_contacts_i - new_contacts_i
        
        return True, delta_e
    
    def undo_single_move(self, i, old_pos, delta_e):
        """Undo a single move."""
        chain = self.chain
        occ = self.occupied
        new_pos = chain[i]
        chain[i] = old_pos
        del occ[new_pos]
        occ[old_pos] = i
        self.energy -= delta_e
    
    def try_crankshaft(self, i, rng):
        """Try a crankshaft move: move a segment of length 2 or 3."""
        n = self.n
        chain = self.chain
        occ = self.occupied
        
        if i < 1 or i + 1 >= n:
            return False, 0
        
        # Try moving residues i and i+1
        # The segment is chain[i-1], chain[i], chain[i+1], chain[i+2]
        # We move chain[i] and chain[i+1] to new positions
        # Constraints:
        # - new_pos_i adjacent to chain[i-1]
        # - new_pos_i adjacent to new_pos_i1
        # - new_pos_i1 adjacent to chain[i+2]
        # - new positions not occupied (excluding the segment itself)
        
        prev_pos = chain[i - 1]
        next_pos = chain[i + 2] if i + 2 < n else None
        
        if next_pos is None:
            return False, 0
        
        # Generate candidate new positions for i
        candidates_i = [p for p in _neighbours(prev_pos) if p != chain[i] and p not in occ]
        if not candidates_i:
            return False, 0
        
        best_delta = None
        best_new_i = None
        best_new_i1 = None
        
        for new_i in candidates_i:
            # Generate candidates for i+1
            candidates_i1 = [p for p in _neighbours(new_i) if p != chain[i+1] and p not in occ and p != prev_pos and p != new_i]
            # Also check adjacency to next_pos
            candidates_i1 = [p for p in candidates_i1 if _manhattan(p, next_pos) == 1]
            
            for new_i1 in candidates_i1:
                # Check that new_i and new_i1 don't conflict with each other or existing
                if new_i == new_i1:
                    continue
                if new_i in occ or new_i1 in occ:
                    continue
                
                # Compute delta energy
                # Remove i and i+1, compute contacts, add back at new positions
                # This is complex; let's just do a full recompute for the affected region
                # For now, skip crankshaft if too complex
                
                pass
        
        return False, 0


def _greedy_fold(sequence, rng, temperature=0.05):
    """Fast greedy chain growth with some randomness."""
    n = len(sequence)
    if n == 0:
        return []
    
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        candidates = []
        last = chain[-1]
        for site in _neighbours(last):
            if site not in occupied:
                gain = 0
                if sequence[i] == 'H':
                    for nb in _neighbours(site):
                        idx = occupied.get(nb)
                        if idx is not None and idx < i - 1 and sequence[idx] == 'H':
                            gain += 1
                free = sum(1 for nb in _neighbours(site) if nb not in occupied)
                score = gain * 10.0 + free * 0.5 + rng.random() * temperature
                candidates.append((score, site))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: -x[0])
        if len(candidates) > 1 and rng.random() < 0.15:
            k = min(3, len(candidates))
            choice_idx = rng.randint(0, k - 1)
        else:
            choice_idx = 0
        
        best_site = candidates[choice_idx][1]
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain


def _sa_optimize(sequence, chain, rng, time_budget=1.0):
    """Simulated annealing with single-residue moves."""
    if not chain or len(chain) < 3:
        return chain
    
    state = _ChainState(sequence, chain)
    n = state.n
    best_chain = state.chain[:]
    best_energy = state.energy
    
    start_time = time.time()
    t_start = 1.0
    t_end = 0.005
    # Estimate iterations based on time budget
    max_iters = 20000
    
    t = t_start
    cooling = (t_start - t_end) / max(max_iters, 1)
    
    for it in range(max_iters):
        t = max(t - cooling, t_end)
        
        if time.time() - start_time > time_budget:
            break
        
        # Pick a random interior residue
        i = rng.randint(1, n - 2)
        old_pos = state.chain[i]
        
        # Try a random neighbor
        new_pos = rng.choice(list(_neighbours(old_pos)))
        
        success, delta_e = state.try_single_move(i, new_pos, rng)
        
        if success:
            if delta_e <= 0 or rng.random() < math.exp(-delta_e / t):
                if state.energy < best_energy:
                    best_energy = state.energy
                    best_chain = state.chain[:]
            else:
                state.undo_single_move(i, old_pos, delta_e)
    
    return best_chain


def _refold_middle(sequence, chain, rng, temp=0.05):
    """Refold the middle portion of the chain to escape local minima."""
    n = len(chain)
    if n < 10:
        return chain
    
    # Pick a random middle segment to refold
    seg_len = max(4, n // 5)
    start = rng.randint(2, n - seg_len - 2)
    end = start + seg_len
    
    # Keep prefix and suffix fixed
    prefix = chain[:start]
    suffix = chain[end:]
    
    # Refold the middle segment
    # The middle segment must start adjacent to prefix[-1] and end adjacent to suffix[0]
    mid_seq = sequence[start:end]
    
    # Try to grow the middle segment
    # Start from prefix[-1], grow seg_len residues
    chain2 = prefix[:]
    occupied = {p: i for i, p in enumerate(prefix)}
    # Also mark suffix as occupied for collision checking
    for p in suffix:
        if p not in occupied:
            occupied[p] = -1  # Mark as occupied but not part of this segment
    
    success = True
    for k in range(seg_len):
        last = chain2[-1]
        candidates = []
        for site in _neighbours(last):
            if site not in occupied:
                gain = 0
                if mid_seq[k] == 'H':
                    for nb in _neighbours(site):
                        idx = occupied.get(nb)
                        if idx is not None and idx >= 0 and idx < len(chain2) - 1 and sequence[idx] == 'H':
                            gain += 1
                free = sum(1 for nb in _neighbours(site) if nb not in occupied)
                score = gain * 10.0 + free * 0.5 + rng.random() * temp
                candidates.append((score, site))
        
        if not candidates:
            success = False
            break
        
        candidates.sort(key=lambda x: -x[0])
        if len(candidates) > 1 and rng.random() < 0.2:
            k2 = min(3, len(candidates))
            choice_idx = rng.randint(0, k2 - 1)
        else:
            choice_idx = 0
        
        best_site = candidates[choice_idx][1]
        chain2.append(best_site)
        occupied[best_site] = len(chain2) - 1
    
    if not success:
        return chain
    
    # Check that the refolded segment connects to suffix
    if chain2[-1] != suffix[0] and _manhattan(chain2[-1], suffix[0]) != 1:
        return chain
    
    # Combine
    new_chain = chain2 + suffix
    
    # Verify self-avoidance
    if len(set(new_chain)) != len(new_chain):
        return chain
    
    return new_chain


def fold(sequence, dimension):
    """
    Find the best conformation for the given HP sequence.
    Returns a list of [x, y] coordinates.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    if n == 2:
        return [[0, 0], [1, 0]]
    
    # Fallback: straight line
    straight = [[i, 0] for i in range(n)]
    
    best_chain = None
    best_energy = 0
    
    rng = random.Random(42)
    
    # Adaptive number of restarts based on sequence length and time
    # Shorter sequences: more restarts; longer: fewer but longer SA
    if n <= 20:
        num_restarts = 800
        sa_budget = 0.5
    elif n <= 40:
        num_restarts = 400
        sa_budget = 1.0
    elif n <= 60:
        num_restarts = 200
        sa_budget = 1.5
    else:
        num_restarts = 100
        sa_budget = 2.0
    
    for r in range(num_restarts):
        if time.time() > _DEADLINE - 5:
            break
        
        # Vary temperature for diversity
        temp = 0.01 + rng.random() * 0.15
        chain = _greedy_fold(sequence, rng, temperature=temp)
        
        if chain is None:
            continue
        
        # SA optimization
        chain = _sa_optimize(sequence, chain, rng, time_budget=sa_budget)
        
        energy = _compute_energy_full(sequence, chain)
        
        if best_chain is None or energy < best_energy:
            best_chain = chain[:]
            best_energy = energy
        
        # Occasionally try refolding middle
        if r % 10 == 0 and best_chain is not None:
            refolded = _refold_middle(sequence, best_chain, rng, temp=temp)
            if refolded != best_chain:
                refolded = _sa_optimize(sequence, refolded, rng, time_budget=sa_budget * 0.5)
                ref_energy = _compute_energy_full(sequence, refolded)
                if ref_energy < best_energy:
                    best_chain = refolded
                    best_energy = ref_energy
    
    if best_chain is None:
        return straight
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
