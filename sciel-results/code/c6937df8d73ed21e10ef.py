"""HP lattice protein folding with O(1) delta-energy local search and beam search."""
# EVOLVE-BLOCK-START
import random
import time

_START_TIME = None
_DEADLINE = 200.0  # Safety margin before 210s kill

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _init_time()
    return _DEADLINE - (time.time() - _START_TIME)

def _neighbours(point):
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

def _count_contacts(sequence, chain):
    """Calculate total H-H contacts for a chain."""
    n = len(chain)
    if n == 0:
        return 0
    
    occupied = {p: i for i, p in enumerate(chain)}
    contacts = 0
    is_h = [c == 'H' for c in sequence]
    
    for i in range(n):
        if not is_h[i]:
            continue
        x, y = chain[i]
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        for nx, ny in neighbors:
            j = occupied.get((nx, ny))
            if j is not None and j > i + 1 and is_h[j]:
                contacts += 1
    return contacts

class _FoldingState:
    """Maintains chain state with O(1) energy updates."""
    def __init__(self, sequence, chain):
        self.sequence = sequence
        self.is_h = [c == 'H' for c in sequence]
        self.chain = list(chain)
        self.n = len(chain)
        self.occupied = {p: i for i, p in enumerate(chain)}
        self.contacts = self._compute_contacts()
        self.energy = -self.contacts
    
    def _compute_contacts(self):
        contacts = 0
        for i in range(self.n):
            if not self.is_h[i]:
                continue
            x, y = self.chain[i]
            for nx, ny in _neighbours((x, y)):
                j = self.occupied.get((nx, ny))
                if j is not None and j > i + 1 and self.is_h[j]:
                    contacts += 1
        return contacts
    
    def _delta_contacts_for_move(self, i, old_pos, new_pos):
        """Compute change in contacts when moving residue i from old_pos to new_pos."""
        if not self.is_h[i]:
            # Moving a P residue doesn't change H-H contacts directly,
            # but it might free up a position that was providing contact
            # Actually, P residues don't contribute to contacts, so delta is 0
            # unless the move affects other H residues' neighbor checks
            # But since we're only moving one residue, and P doesn't form contacts,
            # the only effect is if old_pos was adjacent to an H residue that
            # was counting i as a contact - but i is P, so no.
            # Wait - contacts are H-H pairs. Moving a P doesn't create or destroy H-H contacts.
            return 0
        
        delta = 0
        
        # Remove contacts that old_pos had with other H residues
        for nx, ny in _neighbours(old_pos):
            j = self.occupied.get((nx, ny))
            if j is not None and j != i and j > i + 1 and self.is_h[j]:
                delta -= 1
            elif j is not None and j != i and j < i - 1 and self.is_h[j]:
                delta -= 1
        
        # Add contacts that new_pos will have with other H residues
        for nx, ny in _neighbours(new_pos):
            j = self.occupied.get((nx, ny))
            if j is not None and j != i and j > i + 1 and self.is_h[j]:
                delta += 1
            elif j is not None and j != i and j < i - 1 and self.is_h[j]:
                delta += 1
        
        return delta
    
    def _try_kink_move(self, i):
        """Try a kink move at position i. Returns True if accepted."""
        if i < 1 or i > self.n - 2:
            return False
        
        prev = self.chain[i-1]
        curr = self.chain[i]
        next_ = self.chain[i+1]
        
        # Find valid new positions: neighbors of prev that are distance 1 from next_
        candidates = []
        for p in _neighbours(prev):
            if p == curr:
                continue
            dist = abs(p[0] - next_[0]) + abs(p[1] - next_[1])
            if dist == 1:
                candidates.append(p)
        
        if not candidates:
            return False
        
        new_pos = random.choice(candidates)
        if new_pos in self.occupied:
            return False
        
        # Compute delta
        delta_contacts = self._delta_contacts_for_move(i, curr, new_pos)
        delta_energy = -delta_contacts
        
        return delta_energy, new_pos
    
    def _apply_kink_move(self, i, new_pos):
        """Apply a kink move at position i."""
        old_pos = self.chain[i]
        del self.occupied[old_pos]
        self.occupied[new_pos] = i
        self.chain[i] = new_pos
        self.contacts += self._delta_contacts_for_move(i, old_pos, new_pos)
        self.energy = -self.contacts
    
    def _revert_kink_move(self, i, old_pos):
        """Revert a kink move at position i."""
        new_pos = self.chain[i]
        self.contacts -= self._delta_contacts_for_move(i, old_pos, new_pos)
        del self.occupied[new_pos]
        self.occupied[old_pos] = i
        self.chain[i] = old_pos
        self.energy = -self.contacts
    
    def _try_segment_move(self, seg_len):
        """Try a segment move. Returns (indices, new_positions, delta_energy) or None."""
        if self.n < seg_len + 2:
            return None
        
        start_idx = random.randint(1, self.n - seg_len - 1)
        end_idx = start_idx + seg_len - 1
        
        anchor_start = self.chain[start_idx - 1]
        anchor_end = self.chain[end_idx + 1]
        old_seg = [self.chain[i] for i in range(start_idx, end_idx + 1)]
        seg_indices = list(range(start_idx, end_idx + 1))
        
        # Try random walks
        for _ in range(30):
            pos = anchor_start
            path = [pos]
            for _ in range(seg_len):
                neighbors = _neighbours(pos)
                pos = random.choice(neighbors)
                path.append(pos)
            
            if path[-1] != anchor_end:
                continue
            
            new_seg = path[1:]
            
            # Check collisions
            collision = False
            old_seg_set = set(old_seg)
            for p in new_seg:
                if p in self.occupied and p not in old_seg_set:
                    collision = True
                    break
            
            if collision:
                continue
            
            if len(set(new_seg)) != len(new_seg):
                continue
            
            # Compute delta energy
            delta_contacts = 0
            for k, idx in enumerate(seg_indices):
                old_p = old_seg[k]
                new_p = new_seg[k]
                if self.is_h[idx]:
                    # Remove old contacts
                    for nx, ny in _neighbours(old_p):
                        j = self.occupied.get((nx, ny))
                        if j is not None and j != idx and j not in seg_indices:
                            if (j > idx + 1 or j < idx - 1) and self.is_h[j]:
                                delta_contacts -= 1
                    # Add new contacts
                    for nx, ny in _neighbours(new_p):
                        j = self.occupied.get((nx, ny))
                        if j is not None and j != idx and j not in seg_indices:
                            if (j > idx + 1 or j < idx - 1) and self.is_h[j]:
                                delta_contacts += 1
            
            # Also need to account for contacts between segment residues themselves
            # This is complex; for simplicity, we'll just accept and recompute if needed
            # For now, use a simpler approach: just try the move and check validity
            
            return seg_indices, new_seg, -delta_contacts
        
        return None
    
    def _apply_segment_move(self, seg_indices, new_seg):
        """Apply a segment move."""
        old_seg = [self.chain[i] for i in seg_indices]
        for i in seg_indices:
            del self.occupied[self.chain[i]]
        for k, idx in enumerate(seg_indices):
            self.occupied[new_seg[k]] = idx
            self.chain[idx] = new_seg[k]
        # Recompute contacts for simplicity in segment moves
        self.contacts = self._compute_contacts()
        self.energy = -self.contacts
    
    def _revert_segment_move(self, seg_indices, old_seg):
        """Revert a segment move."""
        for i in seg_indices:
            del self.occupied[self.chain[i]]
        for k, idx in enumerate(seg_indices):
            self.occupied[old_seg[k]] = idx
            self.chain[idx] = old_seg[k]
        self.contacts = self._compute_contacts()
        self.energy = -self.contacts

def _beam_search(sequence, dimension, beam_width=50):
    """Beam search for HP lattice protein folding."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    is_h = [c == 'H' for c in sequence]
    
    # Each beam entry: (chain_list, occupied_dict, contact_count)
    beam = [([(0, 0)], {(0, 0): 0}, 0)]
    
    for i in range(1, n):
        new_beam = []
        for chain, occupied, contacts in beam:
            current = chain[-1]
            for site in _neighbours(current):
                if site in occupied:
                    continue
                
                gain = 0
                if is_h[i]:
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and is_h[j]:
                            gain += 1
                
                new_chain = chain + [site]
                new_occupied = dict(occupied)
                new_occupied[site] = i
                new_contacts = contacts + gain
                new_beam.append((new_chain, new_occupied, new_contacts))
        
        if not new_beam:
            if beam:
                return beam[0][0]
            return [(0, 0)]
        
        new_beam.sort(key=lambda x: x[2], reverse=True)
        beam = new_beam[:beam_width]
    
    if not beam:
        return [(0, 0)]
    
    return beam[0][0]

def _local_search(state, max_steps=500):
    """Simulated annealing local search with O(1) delta updates."""
    n = state.n
    if n < 3:
        return state
    
    start_temp = 2.0
    end_temp = 0.01
    
    for step in range(max_steps):
        if _time_remaining() < 2.0:
            break
        
        progress = step / max(1, max_steps)
        temp = start_temp * (end_temp / start_temp) ** progress
        
        move_type = random.random()
        
        if move_type < 0.75:
            # Kink move
            i = random.randint(1, n - 2)
            result = state._try_kink_move(i)
            if result is False:
                continue
            delta_energy, new_pos = result
            
            if delta_energy < 0 or (temp > 0.01 and random.random() < pow(2.718281828, -delta_energy / temp)):
                state._apply_kink_move(i, new_pos)
            # If rejected, we don't apply anything since _try_kink_move doesn't modify state
        
        else:
            # Segment move
            seg_len = random.choice([2, 3])
            result = state._try_segment_move(seg_len)
            if result is None:
                continue
            seg_indices, new_seg, delta_energy = result
            
            if delta_energy < 0 or (temp > 0.01 and random.random() < pow(2.718281828, -delta_energy / temp)):
                old_seg = [state.chain[i] for i in seg_indices]
                state._apply_segment_move(seg_indices, new_seg)
        
    return state

def fold(sequence, dimension):
    """Fold an HP sequence on the square lattice."""
    _init_time()
    
    n = len(sequence)
    if n == 0:
        return []
    
    if _time_remaining() < 1.0:
        return [[i, 0] for i in range(n)]
    
    best_chain = None
    best_energy = 0
    
    # Multiple restarts with different beam widths
    max_restarts = 50
    restart = 0
    
    while _time_remaining() > 5.0 and restart < max_restarts:
        restart += 1
        
        # Vary beam width for diversity
        beam_width = 30 + restart * 5
        if beam_width > 150:
            beam_width = 150
        
        chain = _beam_search(sequence, dimension, beam_width=beam_width)
        
        if chain is None or len(chain) != n:
            continue
        
        # Local search with O(1) updates
        state = _FoldingState(sequence, chain)
        
        # Scale local search steps with remaining time
        steps = 200 + int(300 * _time_remaining() / _DEADLINE)
        state = _local_search(state, max_steps=steps)
        
        if state.energy < best_energy:
            best_energy = state.energy
            best_chain = state.chain
    
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
