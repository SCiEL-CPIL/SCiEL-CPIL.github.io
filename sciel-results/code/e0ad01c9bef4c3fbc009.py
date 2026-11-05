"""
HP Lattice Protein Folding on Cubic Lattice
Uses Monte Carlo Chain Growth (MCCG) with contact-based energy bias and local search.
"""
import math

import random
import time
import sys

# EVOLVE-BLOCK-START

def _neighbours(point):
    """Yield the 6 neighbors of a point in 3D cubic lattice."""
    x, y, z = point
    yield (x + 1, y, z)
    yield (x - 1, y, z)
    yield (x, y + 1, z)
    yield (x, y - 1, z)
    yield (x, y, z + 1)
    yield (x, y, z - 1)

def _calculate_energy_delta(sequence, index, pos, occupied):
    """
    Calculate the change in energy if residue at 'index' is placed at 'pos'.
    Energy = - (number of H-H contacts).
    A contact is formed if:
    1. Current residue is H.
    2. Neighbor residue is H.
    3. Neighbor index is not adjacent (|i - j| > 1).
    
    Returns delta_E (negative for favorable contacts).
    """
    if sequence[index] != 'H':
        return 0
    
    delta = 0
    for n_pos in _neighbours(pos):
        if n_pos in occupied:
            n_idx = occupied[n_pos]
            # Check if not adjacent in sequence
            if abs(index - n_idx) > 1:
                if sequence[n_idx] == 'H':
                    delta -= 1  # Energy decreases (becomes more negative)
    return delta

def _count_hh_contacts(sequence, chain):
    """
    Count the number of H-H contacts in the chain.
    A contact is between two H residues that are on neighboring lattice sites
    but not adjacent along the chain (|i - j| > 1).
    """
    n = len(chain)
    if n == 0:
        return 0
    
    occ = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for n_pos in _neighbours(p):
                j = occ.get(n_pos)
                if j is not None and abs(i - j) > 1:
                    if sequence[j] == 'H':
                        contacts += 1
    # Each contact is counted twice (once from each H), so divide by 2
    return contacts // 2

def _calculate_energy(sequence, chain):
    """Calculate total energy of a chain. Energy = -number of H-H contacts."""
    return -_count_hh_contacts(sequence, chain)

def _grow_contact_biased(sequence, rng, T=1.0, contact_weight=2.0):
    """
    Grow a self-avoiding walk with strong bias toward creating H-H contacts.
    Uses a contact-based scoring function that prioritizes placements
    that maximize the total number of favorable interactions.
    
    Returns None if stuck.
    """
    n = len(sequence)
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        neighbors = list(_neighbours(last))
        # Filter out occupied
        available = [p for p in neighbors if p not in occupied]
        
        if not available:
            return None
        
        # Calculate score for each available position using contact-based energy
        scores = []
        for p in available:
            # Calculate immediate contact gain
            dE = _calculate_energy_delta(sequence, i, p, occupied)
            
            # Look-ahead: estimate potential future contacts
            # Count how many H residues are within distance 2 of p
            # that could potentially form contacts later
            potential_contacts = 0
            if sequence[i] == 'H':
                for n_pos in _neighbours(p):
                    if n_pos not in occupied:
                        # This position is free, could be an H later
                        # Weight by proximity (closer = more likely to be used soon)
                        potential_contacts += 0.5
                        # Check distance-2 neighbors for even more potential
                        for n2_pos in _neighbours(n_pos):
                            if n2_pos not in occupied and n2_pos != p:
                                potential_contacts += 0.1
            
            # Combined score: immediate contacts weighted heavily + potential
            # dE is negative for good contacts, so -dE is positive
            immediate_score = max(0, -dE) * contact_weight
            total_score = immediate_score + potential_contacts * 0.3
            
            # Boltzmann factor with temperature
            weight = math.exp(total_score / T)
            scores.append((weight, p))
        
        # Normalize and pick
        total_w = sum(w for w, _ in scores)
        if total_w == 0:
            # Fallback: pick randomly
            chosen = rng.choice(available)
        else:
            r = rng.random() * total_w
            cumulative = 0
            chosen = scores[0][1]
            for w, p in scores:
                cumulative += w
                if r <= cumulative:
                    chosen = p
                    break
        
        chain.append(chosen)
        occupied[chosen] = i
        
    return chain

def _local_search_contacts(sequence, chain, rng, steps=200):
    """
    Apply local search moves to optimize the chain by maximizing H-H contacts.
    Uses pivot moves that preserve self-avoidance.
    """
    n = len(chain)
    if n < 4:
        return chain
    
    c = chain[:]
    occ = {p: i for i, p in enumerate(c)}
    
    # Calculate initial contacts
    current_contacts = _count_hh_contacts(sequence, c)
    best_chain = c[:]
    best_contacts = current_contacts
    
    T = 1.0
    cooling = 0.01
    
    for step in range(steps):
        # Try a pivot move
        # Choose pivot point i (1 to n-2)
        i = rng.randint(1, n - 2)
        
        # Get the segment to rotate
        segment = c[i:]
        
        # Try rotating the segment in different directions
        # For simplicity, try a few random rotations
        rotated = False
        for _ in range(3):
            # Create a copy of the chain
            new_c = c[:i] + segment[:]
            
            # Try to move the end of the segment to a new position
            end_pos = new_c[-1]
            neighbors = list(_neighbours(end_pos))
            
            # Filter valid positions
            valid_positions = []
            for np in neighbors:
                if np not in occ or occ.get(np) == n - 1:
                    # Check if this position is not occupied by any other part of the chain
                    # except the last position which we're moving
                    if np in occ and occ[np] != n - 1:
                        continue
                    valid_positions.append(np)
            
            if not valid_positions:
                continue
            
            new_end = rng.choice(valid_positions)
            
            # Check if the rotation is valid (no collisions)
            valid = True
            for j in range(i, n):
                pos = new_c[j]
                if pos in occ and occ[pos] != j:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Calculate new contacts
            new_occ = {p: idx for idx, p in enumerate(new_c)}
            new_contacts = _count_hh_contacts(sequence, new_c)
            
            # Accept if better or with some probability
            delta = new_contacts - current_contacts
            if delta >= 0 or rng.random() < math.exp(delta / T):
                c = new_c
                occ = new_occ
                current_contacts = new_contacts
                rotated = True
                break
        
        if not rotated:
            # Try a simple swap of two non-adjacent positions if possible
            # This is complex, so just skip
            pass
        
        # Update best
        if current_contacts > best_contacts:
            best_contacts = current_contacts
            best_chain = c[:]
        
        T = max(0.1, T - cooling)
    
    return best_chain

def _mccg_fold(sequence, dimension, time_limit, seed):
    """
    Monte Carlo Chain Growth folding algorithm with contact-based bias and local search.
    """
    rng = random.Random(seed)
    n = len(sequence)
    
    start_pos = (0, 0, 0)
    
    best_chain = None
    best_energy = 0  # 0 is the energy of no contacts
    
    start_time = time.time()
    
    # Strategy: Run many contact-biased growths, then apply local search to the best ones.
    # Time budget: ~8 seconds per sequence.
    
    best_chain = None
    best_energy = 0  # Unfolded
    
    attempts = 0
    max_attempts = 3000
    
    # Track the top few chains for local search
    top_chains = []  # List of (energy, chain)
    
    while attempts < max_attempts:
        if time.time() - start_time > time_limit:
            break
        
        # Vary parameters for diversity
        T = 0.3 + rng.random() * 0.7
        contact_weight = 1.5 + rng.random() * 1.5
        
        chain = _grow_contact_biased(sequence, rng, T=T, contact_weight=contact_weight)
        if chain is not None:
            # Calculate energy
            e = _calculate_energy(sequence, chain)
            if e < best_energy:
                best_energy = e
                best_chain = chain[:]
            
            # Keep top chains for local search
            top_chains.append((e, chain))
            top_chains.sort(key=lambda x: x[0])
            if len(top_chains) > 5:
                top_chains.pop()
        
        attempts += 1
    
    # Apply local search to the best chains if we have time
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 0.5 and best_chain is not None:
        # Apply local search to the top chains
        for e, chain in top_chains[:3]:
            if time.time() - start_time > time_limit - 0.1:
                break
            
            # Time for local search
            ls_time = min(1.0, remaining_time * 0.3)
            steps = int(ls_time * 100)
            
            refined_chain = _local_search_contacts(sequence, chain, rng, steps=steps)
            refined_e = _calculate_energy(sequence, refined_chain)
            
            if refined_e < best_energy:
                best_energy = refined_e
                best_chain = refined_chain
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Fold the sequence using MCCG with contact-based bias and local search.
    """
    n = len(sequence)
    # Base time 2s + 0.05s per residue
    time_limit = 2.0 + 0.05 * n
    if time_limit > 10.0:
        time_limit = 10.0
    if time_limit < 1.0:
        time_limit = 1.0

    # Use a unique seed for each call
    seed = hash(sequence) % (2**32)
    
    chain, energy = _mccg_fold(sequence, dimension, time_limit, seed)
    
    # Ensure format: list of lists
    return [list(p) for p in chain]

# EVOLVE-BLOCK-END
