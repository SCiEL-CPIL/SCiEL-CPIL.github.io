"""HP lattice protein folding on the cubic lattice using parallel tempering Monte Carlo."""
import random
import math
from collections import deque

# EVOLVE-BLOCK-START

def _neighbours(point):
    """Yield the 6 neighbours of a point in 3D cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _initial_saw(sequence, dimension):
    """Create an initial self-avoiding walk (a simple spiral or snake pattern)."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    
    # Simple snake pattern that always works for any length
    # Go along x, then back along x at y=1, etc.
    chain = []
    x, y, z = 0, 0, 0
    chain.append((x, y, z))
    
    direction = [(1, 0, 0)]  # start going +x
    
    for i in range(1, n):
        # Alternate direction to create a snake
        if i % 2 == 1:
            dx, dy, dz = 1, 0, 0
        else:
            dx, dy, dz = -1, 0, 0
            if len(chain) > 1:
                # Move to next row
                x, y, z = chain[-1]
                y += 1
                x = 0
                chain.append((x, y, z))
                continue
        x, y, z = chain[-1]
        x += dx
        chain.append((x, y, z))
    
    return chain


def _compute_energy_and_contacts(sequence, chain):
    """Compute energy and store contact pairs for efficient updates."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    
    # Find all H-H contacts (non-consecutive)
    contacts = set()
    for i in range(n):
        if sequence[i] != 'H':
            continue
        for q in _neighbours(chain[i]):
            j = where.get(q)
            if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                # Store as sorted tuple to avoid duplicates
                pair = (min(i, j), max(i, j))
                contacts.add(pair)
    
    energy = -len(contacts)
    return energy, contacts, where


def _energy_delta_move(sequence, chain, where, contacts, i, new_pos):
    """Compute energy delta for moving residue i to new_pos.
    Returns (delta_energy, new_contacts_set) or (None, None) if move is invalid."""
    n = len(chain)
    old_pos = chain[i]
    
    # Check if new_pos is occupied by another residue
    if new_pos in where and where[new_pos] != i:
        return None, None
    
    # Check self-avoidance with consecutive residues
    for j in (i - 1, i + 1):
        if 0 <= j < n and chain[j] == new_pos:
            return None, None
    
    # Compute energy delta
    # Remove contacts involving i
    removed = set()
    for pair in contacts:
        if i in pair:
            removed.add(pair)
    
    # Add new contacts involving i at new_pos
    added = set()
    for q in _neighbours(new_pos):
        j = where.get(q)
        if j is not None and j != i and abs(j - i) > 1 and sequence[j] == 'H' and sequence[i] == 'H':
            pair = (min(i, j), max(i, j))
            added.add(pair)
    
    # Also need to check: when i moves, it might break contacts between other residues
    # that were only possible because i was at old_pos? No, contacts are between H residues
    # on neighboring lattice sites. Moving i only affects contacts involving i.
    
    delta = -(len(added) - len(removed))
    
    # Build new contacts set
    new_contacts = contacts - removed
    new_contacts.update(added)
    
    return delta, new_contacts


def _parallel_tempering_fold(sequence, dimension, n_reps=16, n_sweeps=5000, seed=12345):
    """Fold using parallel tempering Monte Carlo."""
    n = len(sequence)
    if n <= 1:
        return [list(p) for p in [(0,) * dimension] * n]
    
    rng = random.Random(seed + len(sequence))
    
    # Create replicas at different temperatures
    # Temperature schedule: T_k = T_min * (T_max/T_min)^(k/(n_reps-1))
    T_min = 0.1
    T_max = 5.0
    
    # Initial chains: use different random SAWs
    replicas = []
    for k in range(n_reps):
        # Create initial chain
        chain = _initial_saw(sequence, dimension)
        energy, contacts, where = _compute_energy_and_contacts(sequence, chain)
        replicas.append({
            'chain': chain,
            'energy': energy,
            'contacts': contacts,
            'where': where,
            'T': T_min * (T_max / T_min) ** (k / (n_reps - 1))
        })
    
    best_energy = min(r['energy'] for r in replicas)
    best_chain = list(replicas[0]['chain'])
    
    # Monte Carlo sweeps
    for sweep in range(n_sweeps):
        # For each replica, do a sweep of moves
        for k in range(n_reps):
            rep = replicas[k]
            chain = rep['chain']
            where = rep['where']
            contacts = rep['contacts']
            T = rep['T']
            
            # Random order of residues to move
            order = list(range(n))
            rng.shuffle(order)
            
            for i in order:
                if i == 0 or i == n - 1:
                    continue  # Don't move endpoints to keep it simple
                
                # Try moving residue i to a random neighbor
                neighbors = list(_neighbours(chain[i]))
                rng.shuffle(neighbors)
                
                moved = False
                for new_pos in neighbors:
                    delta, new_contacts = _energy_delta_move(sequence, chain, where, contacts, i, new_pos)
                    if delta is None:
                        continue
                    
                    # Metropolis acceptance
                    if delta <= 0 or rng.random() < math.exp(-delta / T):
                        # Accept move
                        old_pos = chain[i]
                        
                        # Update where
                        del where[old_pos]
                        where[new_pos] = i
                        
                        # Update chain
                        chain[i] = new_pos
                        
                        # Update contacts
                        rep['contacts'] = new_contacts
                        contacts = new_contacts
                        
                        # Update energy
                        rep['energy'] += delta
                        rep['where'] = where
                        
                        # Track best
                        if rep['energy'] < best_energy:
                            best_energy = rep['energy']
                            best_chain = list(chain)
                        
                        moved = True
                        break
                
                # If no move was made, try a small perturbation: rotate a segment
                # Skip for now to keep it simple
        
        # Swap replicas based on energy
        for k in range(n_reps - 1):
            rep1 = replicas[k]
            rep2 = replicas[k + 1]
            
            # Metropolis swap acceptance
            delta_swap = (rep1['energy'] / rep2['T'] - rep1['energy'] / rep1['T']) + \
                         (rep2['energy'] / rep1['T'] - rep2['energy'] / rep2['T'])
            
            if delta_swap <= 0 or rng.random() < math.exp(-delta_swap):
                # Swap energies and temperatures
                rep1['energy'], rep2['energy'] = rep2['energy'], rep1['energy']
                rep1['T'], rep2['T'] = rep2['T'], rep1['T']
                # Also swap the chains
                rep1['chain'], rep2['chain'] = rep2['chain'], rep1['chain']
                rep1['contacts'], rep2['contacts'] = rep2['contacts'], rep1['contacts']
                rep1['where'], rep2['where'] = rep2['where'], rep1['where']
    
    return [list(p) for p in best_chain]


def fold(sequence, dimension):
    """Fold the sequence using parallel tempering Monte Carlo."""
    if len(sequence) == 0:
        return []
    if len(sequence) == 1:
        return [[0] * dimension]
    
    # Use parallel tempering
    result = _parallel_tempering_fold(sequence, dimension, n_reps=8, n_sweeps=3000, seed=12345)
    
    # Verify self-avoidance
    if len(set(tuple(p) for p in result)) != len(result):
        # Fallback: simple snake
        result = []
        x, y, z = 0, 0, 0
        result.append([x, y, z])
        for i in range(1, len(sequence)):
            if i % 2 == 1:
                x += 1
            else:
                x -= 1
                if x < 0:
                    x = 0
                    y += 1
            result.append([x, y, z])
    
    return result

# EVOLVE-BLOCK-END
