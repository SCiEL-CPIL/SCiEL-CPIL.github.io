"""HP lattice protein folding on the square lattice using Hierarchical Motif Beam Search."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Explicit deadline: evaluator kills after 210s wall-clock; keep 30s safety margin.
_DEADLINE = time.time() + 175.0


def _neighbours(point):
    """Return the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _compute_energy(sequence, chain):
    """Compute the negative number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2


def _generate_all_5mers():
    """
    Generate all unique self-avoiding 5-mers on the square lattice.
    Each 5-mer is represented as a tuple of 5 relative coordinates,
    normalized so the first point is (0,0).
    Returns a list of (coords, internal_energy, contact_mask) tuples.
    """
    mers = []
    seen = set()
    
    def _grow(pos, idx, path):
        if idx == 5:
            # Normalize: translate so first point is (0,0)
            start = path[0]
            norm = tuple((p[0] - start[0], p[1] - start[1]) for p in path)
            # Canonical form: also try rotations/reflections to reduce duplicates?
            # For simplicity, we keep all unique relative shapes but we can deduplicate
            # by sorting the set of relative vectors.
            # Actually, for placement, we need to try all orientations.
            # So we generate all unique *shapes* (up to rotation/reflection) is complex.
            # Instead, we generate all 5-mers starting at (0,0) with first step fixed to (1,0)
            # to break rotational symmetry, then we will rotate them during placement.
            # Wait, the first step in the generation should be fixed to (1,0) to avoid
            # generating rotated versions of the same shape.
            pass
        
        last = path[-1]
        # Directions: Right, Left, Up, Down
        # To avoid backtracking to immediate predecessor, we skip the reverse of last step
        # But for the first step, we only allow (1,0) to fix orientation.
        
        if idx == 1:
            # Only allow (1,0) as first step to fix global orientation of the motif
            candidates = [(1, 0)]
        else:
            prev_dx = last[0] - path[-2][0]
            prev_dy = last[1] - path[-2][1]
            # All 4 directions
            candidates = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            # Exclude the direction that goes back to path[-2]
            back_dx = -prev_dx
            back_dy = -prev_dy
            candidates = [(dx, dy) for dx, dy in candidates if (dx, dy) != (back_dx, back_dy)]
        
        for dx, dy in candidates:
            new_pos = (last[0] + dx, last[1] + dy)
            # Check self-avoidance within the 5-mer
            if new_pos in path:
                continue
            new_path = path + [new_pos]
            if idx == 4:
                start = new_path[0]
                norm = tuple((p[0] - start[0], p[1] - start[1]) for p in new_path)
                if norm not in seen:
                    seen.add(norm)
                    # Calculate internal energy for this specific shape
                    # Internal energy depends on the sequence, so we can't pre-calc it here.
                    # We just store the shape.
                    mers.append(norm)
            else:
                _grow(new_pos, idx + 1, new_path)
    
    _grow((0, 0), 0, [(0, 0)])
    return mers


# Pre-compute all unique 5-mer shapes (relative coordinates)
_ALL_5MERS = _generate_all_5mers()

# Pre-compute rotations and reflections for each 5-mer
def _get_orientations(mer):
    """Return all 8 orientations (4 rotations * 2 reflections) of a 5-mer."""
    orients = []
    seen = set()
    
    def _rotate(pts, k):
        new_pts = []
        for x, y in pts:
            for _ in range(k % 4):
                x, y = y, -x
            new_pts.append((x, y))
        return tuple(new_pts)
    
    def _reflect(pts):
        return tuple((x, -y) for x, y in pts)
    
    base = list(mer)
    for rot in range(4):
        r_pts = _rotate(base, rot)
        if r_pts not in seen:
            seen.add(r_pts)
            orients.append(r_pts)
        
        rf_pts = _reflect(r_pts)
        if rf_pts not in seen:
            seen.add(rf_pts)
            orients.append(rf_pts)
            
    return orients

# Pre-compute orientations for all 5-mers
_MER_ORIENTATIONS = {}
for m in _ALL_5MERS:
    _MER_ORIENTATIONS[m] = _get_orientations(m)


def _motif_energy(sequence, start_idx, mer_coords):
    """
    Calculate the internal H-H contacts within a single motif.
    mer_coords is a tuple of 5 relative coordinates.
    """
    # Map relative coords to indices within the motif
    # We need to check pairs (i, j) within the motif that are adjacent on the lattice
    # and not adjacent in the sequence (i.e., |i-j| > 1).
    # Note: The motif is a contiguous segment of the sequence.
    
    # For a 5-mer, the indices are 0, 1, 2, 3, 4 relative to the motif start.
    # Adjacent in sequence: (0,1), (1,2), (2,3), (3,4).
    # Non-adjacent in sequence but adjacent in space:
    # We need to check all pairs (i, j) with |i-j| > 1 such that their relative coords are neighbours.
    
    contacts = 0
    for i in range(5):
        if sequence[start_idx + i] != 'H':
            continue
        for j in range(i + 2, 5):  # j > i+1 ensures non-adjacent in sequence
            if sequence[start_idx + j] != 'H':
                continue
            # Check if mer_coords[i] and mer_coords[j] are neighbours
            dx = mer_coords[i][0] - mer_coords[j][0]
            dy = mer_coords[i][1] - mer_coords[j][1]
            if abs(dx) + abs(dy) == 1:
                contacts += 1
    return -contacts // 2


def _beam_search_fold(sequence, dimension, beam_width=50, max_time=150.0):
    """
    Hierarchical Beam Search using 5-mer motifs.
    """
    n = len(sequence)
    if n < 5:
        # Fallback for very short sequences
        return [(i, 0) for i in range(n)], 0
    
    # Start time
    start_time = time.time()
    
    # Beam: list of (chain_prefix, occupied_set, energy, last_idx)
    # chain_prefix: list of absolute coordinates
    # occupied_set: set of occupied coordinates
    # energy: current energy
    # last_idx: index of the last residue in the prefix
    
    # Initialize beam with the first motif (first 5 residues)
    # We try all orientations of all 5-mers for the first 5 residues.
    # But wait, the first 5 residues must form a valid SAW.
    # We can just iterate through _ALL_5MERS and place them at (0,0).
    
    initial_beam = []
    for mer in _ALL_5MERS:
        # Place mer at origin
        coords = list(mer)
        # Calculate energy for this motif
        e = _motif_energy(sequence, 0, mer)
        # Check if this is a valid start (it is, by construction)
        initial_beam.append((coords, set(coords), e, 4))
        
    # Sort initial beam by energy
    initial_beam.sort(key=lambda x: x[2])
    beam = initial_beam[:beam_width]
    
    # Expand the beam
    # We add residues in chunks of 5? Or one by one?
    # The prompt suggests "motif segments". Let's add one residue at a time but use the motif structure to guide?
    # Actually, the strategy is "places these motifs on the lattice".
    # This implies we decide the shape of the next 5 residues (a motif) and place it.
    # However, the motif must connect to the previous chain.
    # So we need to find a motif (5-mer) that can be attached to the current chain end.
    
    # Let's refine: 
    # State: (chain, occupied, energy, last_idx)
    # Next step: Choose a 5-mer shape from _ALL_5MERS.
    # Try to attach it to the current chain end.
    # The first residue of the new motif must be a neighbour of the last residue of the current chain.
    # And it must not collide with the occupied set.
    # The rest of the motif residues must also not collide.
    
    # This is expensive. Let's try a simpler "greedy motif" approach first?
    # No, beam search is better.
    
    # Optimization: Pre-compute for each 5-mer and each possible "entry direction" (from where the chain comes in),
    # what the relative coordinates of the 5-mer would be?
    # Actually, it's easier to just try all orientations of all 5-mers.
    
    # To speed up, we can filter motifs based on the first step.
    # The first step of the new motif (from residue i to i+1) must be a valid move from the last residue.
    
    # Let's implement the expansion.
    
    while beam and time.time() - start_time < max_time:
        new_beam = []
        
        for chain, occupied, energy, last_idx in beam:
            if last_idx >= n - 1:
                # Complete chain
                new_beam.append((chain, occupied, energy, last_idx))
                continue
                
            # Determine the next 5 residues (or fewer if near end)
            start_next = last_idx + 1
            end_next = min(n, start_next + 5)
            motif_len = end_next - start_next
            
            # If motif_len < 5, we can't use the pre-computed 5-mers directly.
            # We can pad the motif? Or just grow one by one for the last few?
            # Let's handle full 5-mers first. If remaining < 5, we do a final greedy growth.
            
            if motif_len == 5:
                # Try all 5-mers
                for mer in _ALL_5MERS:
                    # The first residue of the mer is at (0,0) in relative coords.
                    # We need to place it such that mer[0] is a neighbour of chain[-1].
                    # And the sequence of moves in the mer must be consistent.
                    
                    # Actually, the 5-mer is a shape. We need to rotate/translate it.
                    # Let's iterate through orientations.
                    for orient in _MER_ORIENTATIONS[mer]:
                        # orient[0] is (0,0).
                        # We need to place orient[0] at a neighbour of chain[-1].
                        # Let last_pos = chain[-1]
                        # The first step of the motif is from orient[0] to orient[1].
                        # This step must be a valid move from last_pos.
                        
                        # Let's try placing the motif such that orient[0] is at a specific neighbour of last_pos.
                        # But orient[0] is the START of the new segment.
                        # So chain[-1] is connected to orient[0].
                        # This means orient[0] must be a neighbour of chain[-1].
                        
                        # Let's try all 4 neighbours of chain[-1] as potential positions for orient[0].
                        for new_pos0 in _neighbours(chain[-1]):
                            if new_pos0 in occupied:
                                continue
                            
                            # Check if the rest of the motif fits.
                            # Calculate absolute positions for the motif.
                            # rel_pos[i] = new_pos0 + (orient[i][0] - orient[0][0], orient[i][1] - orient[0][1])
                            # Since orient[0] is (0,0), abs_pos[i] = new_pos0 + orient[i]
                            
                            abs_pos = []
                            valid = True
                            for i in range(5):
                                px = new_pos0[0] + orient[i][0]
                                py = new_pos0[1] + orient[i][1]
                                p = (px, py)
                                if p in occupied:
                                    valid = False
                                    break
                                abs_pos.append(p)
                            
                            if not valid:
                                continue
                            
                            # Check connectivity between chain[-1] and abs_pos[0]
                            # Already ensured by new_pos0 being a neighbour.
                            
                            # Calculate energy of the new motif
                            e_motif = _motif_energy(sequence, start_next, mer)
                            
                            # Calculate inter-motif energy
                            # Contacts between the new motif and the existing chain.
                            # We need to check H-H contacts between residues in the new motif
                            # and residues in the existing chain.
                            # Note: We only count non-adjacent in sequence.
                            # The new motif residues are indices start_next to start_next+4.
                            # The existing chain residues are 0 to last_idx.
                            # Adjacent in sequence across the boundary: (last_idx, start_next) is adjacent.
                            # So we must NOT count contact between chain[last_idx] and abs_pos[0].
                            # We DO count contacts between chain[i] (i < last_idx) and abs_pos[j] (j >= 0).
                            # And contacts within the motif are already counted in e_motif.
                            
                            e_inter = 0
                            for i in range(len(abs_pos)):
                                if sequence[start_next + i] != 'H':
                                    continue
                                for q in _neighbours(abs_pos[i]):
                                    j = occupied.get(q) if hasattr(occupied, 'get') else None
                                    # occupied is a set, so we can't get index directly.
                                    # We need a mapping from position to index.
                                    # Let's change state to include pos_to_idx dict.
                                    pass
                            
                            # This approach with set is slow for energy calculation.
                            # Let's refactor state to use a dict for positions.
                            
                            # For now, let's just add to beam with a placeholder energy and refine later?
                            # No, we need accurate energy for beam pruning.
                            
                            # Let's optimize the state representation.
                            # State: (chain_list, pos_to_idx_dict, energy)
                            
                            # We will rebuild the beam logic with pos_to_idx.
                            
                            pass # Skip this inner loop for now, will rewrite below.
                            
            else:
                # Handle short suffix with greedy growth
                pass
        
        # Refactor: Use pos_to_idx dict in state
        # This is getting complex. Let's simplify.
        
        # Alternative: Use a standard beam search on single residues but with a bias?
        # The prompt specifically asks for "motif" strategy.
        
        # Let's implement a simpler version:
        # 1. Generate a set of "candidate motifs" for the next 5 residues.
        # 2. For each candidate motif, check if it can be attached.
        # 3. Calculate total energy.
        # 4. Add to new beam.
        
        # To make it efficient, we'll use a dict for occupied positions.
        
        # Restart loop with better state
        break # Break to restart with corrected logic
        
    # Corrected Beam Search Implementation
    
    beam = []
    # Initialize
    for mer in _ALL_5MERS:
        for orient in _MER_ORIENTATIONS[mer]:
            # Place at (0,0)
            # Check if valid (it is)
            coords = list(orient)
            pos_to_idx = {coords[i]: i for i in range(5)}
            e = _motif_energy(sequence, 0, mer)
            beam.append((coords, pos_to_idx, e, 4))
            
    beam.sort(key=lambda x: x[2])
    beam = beam[:beam_width]
    
    while beam and time.time() - start_time < max_time:
        new_beam = []
        for chain, pos_to_idx, energy, last_idx in beam:
            if last_idx >= n - 1:
                new_beam.append((chain, pos_to_idx, energy, last_idx))
                continue
                
            start_next = last_idx + 1
            end_next = min(n, start_next + 5)
            motif_len = end_next - start_next
            
            if motif_len == 5:
                # Try attaching a 5-mer
                for mer in _ALL_5MERS:
                    for orient in _MER_ORIENTATIONS[mer]:
                        # orient[0] is (0,0)
                        # Try placing orient[0] at neighbours of chain[-1]
                        for new_pos0 in _neighbours(chain[-1]):
                            if new_pos0 in pos_to_idx:
                                continue
                            
                            # Check if the whole motif fits
                            new_coords = []
                            valid = True
                            for i in range(5):
                                px = new_pos0[0] + orient[i][0]
                                py = new_pos0[1] + orient[i][1]
                                p = (px, py)
                                if p in pos_to_idx:
                                    valid = False
                                    break
                                new_coords.append(p)
                            
                            if not valid:
                                continue
                            
                            # Calculate inter-motif energy
                            # Contacts between new motif and old chain
                            # Exclude contact between chain[last_idx] and new_coords[0]
                            e_inter = 0
                            for i in range(5):
                                if sequence[start_next + i] != 'H':
                                    continue
                                for q in _neighbours(new_coords[i]):
                                    j = pos_to_idx.get(q)
                                    if j is not None:
                                        # Check if j is the immediate predecessor (last_idx)
                                        # If i==0 and j==last_idx, skip (adjacent in sequence)
                                        if i == 0 and j == last_idx:
                                            continue
                                        if sequence[j] == 'H':
                                            e_inter += 1
                            
                            e_inter //= 2
                            e_motif = _motif_energy(sequence, start_next, mer)
                            new_energy = energy + e_motif + e_inter
                            
                            new_chain = chain + new_coords
                            new_pos_to_idx = dict(pos_to_idx)
                            for i, p in enumerate(new_coords):
                                new_pos_to_idx[p] = start_next + i
                                
                            new_beam.append((new_chain, new_pos_to_idx, new_energy, last_idx + 5))
                            
            else:
                # Greedy growth for the last few residues
                # Simple backtracking for the remaining residues
                # To keep it fast, we do a limited search
                # For simplicity, just try all SAWs for the remaining length?
                # If motif_len < 5, it's small (1-4).
                
                # Use a recursive helper to find best completion
                # This is a small search space.
                
                def _complete_suffix(chain, pos_to_idx, energy, start_idx, seq):
                    if start_idx == len(seq):
                        return (chain, pos_to_idx, energy)
                    
                    best = None
                    last_pos = chain[-1]
                    for np in _neighbours(last_pos):
                        if np in pos_to_idx:
                            continue
                        
                        # Calculate energy gain
                        e_gain = 0
                        if seq[start_idx] == 'H':
                            for q in _neighbours(np):
                                j = pos_to_idx.get(q)
                                if j is not None and abs(j - (start_idx - 1)) > 1 and seq[j] == 'H':
                                    e_gain += 1
                        e_gain //= 2
                        
                        new_chain = chain + [np]
                        new_pos_to_idx = dict(pos_to_idx)
                        new_pos_to_idx[np] = start_idx
                        
                        res = _complete_suffix(new_chain, new_pos_to_idx, energy - e_gain, start_idx + 1, seq)
                        if res:
                            if best is None or res[2] < best[2]:
                                best = res
                    return best
                
                res = _complete_suffix(chain, pos_to_idx, energy, start_next, sequence)
                if res:
                    new_chain, new_pos_to_idx, new_energy = res
                    new_beam.append((new_chain, new_pos_to_idx, new_energy, n - 1))
        
        if not new_beam:
            break
            
        new_beam.sort(key=lambda x: x[2])
        beam = new_beam[:beam_width]
        
        # Early stop if we have a complete chain with good energy?
        # No, keep expanding until time runs out or beam is empty.
        
    if not beam:
        return None, None
        
    # Return the best chain
    best = min(beam, key=lambda x: x[2])
    return best[0], best[2]


def fold(sequence, dimension):
    """
    Fold the sequence using Hierarchical Motif Beam Search.
    """
    rng = random.Random(42)
    n = len(sequence)
    
    # Try beam search
    chain, energy = _beam_search_fold(sequence, dimension, beam_width=100, max_time=160.0)
    
    if chain is None:
        # Fallback: straight line
        chain = [(i, 0) for i in range(n)]
        energy = _compute_energy(sequence, chain)
        
    # Verify self-avoidance
    if len(set(chain)) != len(chain):
        # Fallback
        chain = [(i, 0) for i in range(n)]
        energy = _compute_energy(sequence, chain)
        
    # Verify connectivity
    for i in range(1, len(chain)):
        if abs(chain[i][0] - chain[i-1][0]) + abs(chain[i][1] - chain[i-1][1]) != 1:
            # Fallback
            chain = [(i, 0) for i in range(n)]
            energy = _compute_energy(sequence, chain)
            break
            
    return [list(p) for p in chain]
# EVOLVE-BLOCK-END
