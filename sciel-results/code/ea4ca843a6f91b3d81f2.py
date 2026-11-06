"""Rewritten program: simulated annealing on 2D grid for HP lattice protein folding."""
# EVOLVE-BLOCK-START
import random
import math


def _energy_full(sequence, positions):
    """Compute total H-H non-bonded contact energy (negative count)."""
    n = len(sequence)
    pos_to_idx = {positions[i]: i for i in range(n)}
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y = positions[i]
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            j = pos_to_idx.get((x+dx, y+dy))
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


def _contacts_of_residue(sequence, positions, i, pos_to_idx):
    """Return set of indices j > i+1 that are H-H contacts with residue i."""
    contacts = set()
    if sequence[i] != 'H':
        return contacts
    x, y = positions[i]
    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
        j = pos_to_idx.get((x+dx, y+dy))
        if j is not None and j > i + 1 and sequence[j] == 'H':
            contacts.add(j)
    return contacts


def _initial_saw(sequence, rng):
    """Generate a random self-avoiding walk on the 2D lattice."""
    n = len(sequence)
    if n == 0:
        return []
    positions = [(0, 0)]
    occupied = {(0, 0): 0}
    directions = [(1,0),(-1,0),(0,1),(0,-1)]
    for i in range(1, n):
        last = positions[-1]
        candidates = []
        for dx, dy in directions:
            new_pos = (last[0]+dx, last[1]+dy)
            if new_pos not in occupied:
                candidates.append(new_pos)
        if not candidates:
            return None
        new_pos = rng.choice(candidates)
        positions.append(new_pos)
        occupied[new_pos] = i
    return positions


def _simulated_annealing(sequence, initial_positions, rng, max_iters=5000, T0=1.0, T1=0.01):
    """Run simulated annealing starting from initial_positions."""
    n = len(sequence)
    if n == 0:
        return initial_positions, 0
    positions = list(initial_positions)
    pos_to_idx = {positions[i]: i for i in range(n)}
    
    # Compute initial energy
    best_energy = _energy_full(sequence, positions)
    current_energy = best_energy
    best_positions = list(positions)
    
    T = T0
    cooling = (T1 / T0) ** (1.0 / max(max_iters, 1))
    
    for it in range(max_iters):
        T *= cooling
        # Pick a random residue to move
        i = rng.randint(0, n - 1)
        x, y = positions[i]
        old_pos = (x, y)
        
        # Find candidate empty neighbor sites
        candidates = []
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            new_pos = (x+dx, y+dy)
            if new_pos not in pos_to_idx:
                candidates.append(new_pos)
        
        if not candidates:
            continue
        
        new_pos = rng.choice(candidates)
        
        # Compute energy delta
        # Remove old contacts involving residue i
        old_contacts = _contacts_of_residue(sequence, positions, i, pos_to_idx)
        # Also need to check if moving i breaks contacts with other residues
        # that were counted via i's perspective. Since we count each contact once (j > i+1),
        # we need to be careful.
        
        # Let's compute delta by:
        # 1. Remove all contacts that residue i participates in (as the lower-index H)
        # 2. Add all new contacts that residue i forms
        # But also, other residues might have had contacts with i. Since we only count j > i+1,
        # if i is the lower index, we count it. If i is the higher index, it was counted when
        # processing the lower index. So when i moves, we need to:
        # - Remove contacts where i is the lower index (j > i+1, both H, adjacent)
        # - For residues j < i where j was in contact with i: those were counted when j was processed.
        #   So we need to also remove/add those.
        
        # Simpler: compute energy before and after for affected residues.
        # Affected residues: i and all its neighbors (old and new)
        
        # Old neighbors of i
        old_neighbors = []
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            j = pos_to_idx.get((x+dx, y+dy))
            if j is not None and j != i:
                old_neighbors.append(j)
        
        # New neighbors of i at new_pos
        new_neighbors = []
        # Temporarily update pos_to_idx to check new neighbors
        # But we can just check which positions are occupied
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            check_pos = (new_pos[0]+dx, new_pos[1]+dy)
            j = pos_to_idx.get(check_pos)
            if j is not None and j != i:
                new_neighbors.append(j)
        
        # Compute delta: sum over affected pairs of (new_contact - old_contact)
        # A contact exists if both are H and not adjacent in chain (|i-j| > 1)
        
        def contact_value(a, b):
            """1 if a and b form an H-H non-bonded contact, else 0."""
            if sequence[a] != 'H' or sequence[b] != 'H':
                return 0
            if abs(a - b) <= 1:
                return 0
            return 1
        
        # Old state: i at old_pos
        # Pairs to consider: (i, j) for j in old_neighbors, and (j1, j2) for j1,j2 in old_neighbors
        # that become non-adjacent? No, moving i doesn't change distances between other residues.
        # Only pairs involving i change.
        
        delta = 0
        # Remove old contacts involving i
        for j in old_neighbors:
            delta -= contact_value(i, j)
        # Add new contacts involving i
        for j in new_neighbors:
            delta += contact_value(i, j)
        
        # Accept if delta < 0 or with probability exp(-delta/T)
        if delta < 0 or (T > 0 and rng.random() < math.exp(-delta / max(T, 1e-10))):
            # Accept the move
            # Update positions
            pos_to_idx.pop(old_pos)
            positions[i] = new_pos
            pos_to_idx[new_pos] = i
            current_energy += delta
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = list(positions)
    
    return best_positions, best_energy


def fold(sequence, dimension):
    """Best conformation via simulated annealing with multiple restarts."""
    n = len(sequence)
    if n == 0:
        return []
    
    rng = random.Random(42 + len(sequence))
    
    best_positions = None
    best_energy = 0  # 0 means no contacts (unfolded)
    
    # Try multiple restarts
    num_restarts = 10
    iters_per_restart = 3000
    
    for restart in range(num_restarts):
        # Generate initial SAW
        init = _initial_saw(sequence, rng)
        if init is None:
            continue
        
        pos, energy = _simulated_annealing(sequence, init, rng, max_iters=iters_per_restart)
        
        if energy < best_energy:
            best_energy = energy
            best_positions = pos
    
    if best_positions is None:
        # Fallback: straight line
        best_positions = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_positions]
# EVOLVE-BLOCK-END
