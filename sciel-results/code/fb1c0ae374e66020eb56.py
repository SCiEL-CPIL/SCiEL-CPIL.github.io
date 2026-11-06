"""Improved HP lattice protein folding using beam search with contact-based heuristics."""
# EVOLVE-BLOCK-START
import random
from collections import defaultdict


def _neighbours(point):
    """Yield all 2*dimension neighbours of a lattice point."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute the HP energy (negative number of H-H contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _beam_search(sequence, dimension, beam_width=50, max_iterations=None):
    """
    Beam search over partial chains. Each state is (chain_tuple, occupied_set, current_energy).
    We keep the top `beam_width` states at each step, ranked by a heuristic score.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0,) * dimension]
    
    # Initialize beam with the first residue at origin
    start = (0,) * dimension
    beam = [((start,), frozenset([start]), 0)]
    
    # For efficiency, we'll track positions as tuples and use sets for occupied
    # To speed up, we precompute H positions
    h_positions = set(i for i, c in enumerate(sequence) if c == 'H')
    
    for step in range(1, n):
        new_beam = []
        seen_states = set()  # To avoid duplicate chains in the beam
        
        for chain, occupied, energy in beam:
            last = chain[-1]
            for site in _neighbours(last):
                if site in occupied:
                    continue
                
                # Check if this creates any new H-H contacts
                new_energy = energy
                if sequence[step] == 'H':
                    for q in _neighbours(site):
                        if q in occupied:
                            # Find the index of q in the chain
                            # We need to know which residue is at q
                            # Since occupied is a set of positions, we need a mapping
                            # This is inefficient. Let's restructure.
                            pass
                
                # This approach is getting complicated with the energy tracking.
                # Let's use a simpler heuristic: just track the chain and compute energy at the end,
                # but use a heuristic score to prune.
                
                # Actually, let's switch to a different strategy:
                # Use a beam of chains, and score them by:
                # 1. Current number of H-H contacts (computed incrementally)
                # 2. Potential for future contacts (heuristic)
                
                # For now, let's just do a simple beam search that keeps the best chains
                # based on a greedy contact count.
                
                new_chain = chain + (site,)
                new_occupied = occupied | {site}
                
                # Compute incremental energy change
                delta_energy = 0
                if sequence[step] == 'H':
                    for q in _neighbours(site):
                        if q in occupied:
                            # Need to check if the residue at q is H
                            # We don't have direct access to this without a position->index map
                            # Let's rebuild the map for now (inefficient but correct)
                            pass
                
                # This is getting too complex. Let's use a simpler approach:
                # Beam search where we keep top K chains by a simple heuristic:
                # number of H-H contacts formed so far (computed at the end for each candidate)
                
                # Actually, let me restart with a cleaner approach.
                
                break  # Break out to restart with cleaner code
        
        # Restart with cleaner beam search
        break
    
    # This is getting messy. Let me write a clean version.
    return _beam_search_clean(sequence, dimension, beam_width)


def _beam_search_clean(sequence, dimension, beam_width=100):
    """Clean beam search implementation."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0,) * dimension]
    
    # Each beam entry: (chain_tuple, occupied_frozenset, contact_count)
    # We'll track contact_count incrementally
    
    start = (0,) * dimension
    beam = [((start,), frozenset([start]), 0)]
    
    for step in range(1, n):
        candidates = []
        
        for chain, occupied, contacts in beam:
            last = chain[-1]
            for site in _neighbours(last):
                if site in occupied:
                    continue
                
                new_chain = chain + (site,)
                new_occupied = occupied | {site}
                
                # Compute new contacts if current residue is H
                new_contacts = contacts
                if sequence[step] == 'H':
                    for q in _neighbours(site):
                        if q in occupied:
                            # Check if residue at q is H
                            # We need to know which index q corresponds to
                            # Build a temporary map (inefficient but works for small beams)
                            pass
                
                # To properly track contacts, we need to know which residues are H
                # and their positions. Let's use a different approach:
                # Store the chain and compute contacts from scratch for scoring,
                # but that's O(n^2) per candidate.
                
                # Better approach: store position->index mapping in the beam state
                # But that makes states heavier.
                
                # For now, let's just use a simple heuristic:
                # Score = -len(chain) + (some potential)
                # Or just keep all candidates and prune by a simple metric.
                
                # Simplest correct approach: keep top beam_width by a greedy score
                # where score = number of H-H contacts so far.
                # To compute this, we need the mapping.
                
                # Let's just add the candidate and score it later.
                candidates.append((new_chain, new_occupied))
        
        # Score candidates by computing their contact count
        scored = []
        for chain, occupied in candidates:
            # Compute H-H contacts
            pos_to_idx = {p: i for i, p in enumerate(chain)}
            contacts = 0
            for i, p in enumerate(chain):
                if sequence[i] == 'H':
                    for q in _neighbours(p):
                        j = pos_to_idx.get(q)
                        if j is not None and j > i + 1 and sequence[j] == 'H':
                            contacts += 1
            scored.append((contacts, chain, occupied))
        
        # Keep top beam_width by contacts (more contacts = better)
        scored.sort(key=lambda x: -x[0])
        beam = [(c, o) for _, c, o in scored[:beam_width]]
    
    # Find the best chain in the final beam
    if not beam:
        # Fallback: straight line
        return [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    # Score all final candidates
    best_chain = None
    best_energy = 0
    for chain, occupied in beam:
        e = _energy(sequence, chain)
        if e < best_energy:
            best_energy = e
            best_chain = chain
    
    if best_chain is None:
        return [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return list(best_chain)


def fold(sequence, dimension):
    """
    Fold the sequence using beam search.
    Returns a list of lattice points (lists of ints).
    """
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension for _ in range(n)]
    
    # Use beam search with adaptive beam width
    # Larger chains need larger beams but also more time
    # Let's use a beam width that scales with chain length
    if n <= 20:
        beam_width = 200
    elif n <= 48:
        beam_width = 150
    elif n <= 80:
        beam_width = 100
    else:
        beam_width = 50
    
    try:
        chain = _beam_search_clean(sequence, dimension, beam_width)
        if chain is None or len(chain) != n:
            raise ValueError("Invalid chain from beam search")
        
        # Verify self-avoidance
        pos_set = set(chain)
        if len(pos_set) != n:
            raise ValueError("Chain is not self-avoiding")
        
        # Verify unit steps
        for i in range(1, n):
            diff = sum(abs(a - b) for a, b in zip(chain[i-1], chain[i]))
            if diff != 1:
                raise ValueError("Non-unit step detected")
        
        return [list(p) for p in chain]
    
    except Exception:
        # Fallback: simple greedy or straight line
        # Try a simple greedy approach
        try:
            return _greedy_fold(sequence, dimension)
        except Exception:
            # Last resort: straight line
            return [[i] + [0] * (dimension - 1) for i in range(n)]


def _greedy_fold(sequence, dimension):
    """Simple greedy fallback: always pick the neighbor with most H-H contacts."""
    n = len(sequence)
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        best_site = None
        best_score = -1
        
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            
            score = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == 'H':
                        score += 1
            
            # Add small random tie-breaker
            score += random.random() * 0.01
            
            if score > best_score:
                best_score = score
                best_site = site
        
        if best_site is None:
            # No valid move; return what we have (shouldn't happen for valid inputs)
            # Pad with a straight line extension
            last = chain[-1]
            for j in range(i, n):
                new_site = tuple([last[k] + (1 if k == 0 else 0) for k in range(dimension)])
                # This might collide; just return the partial chain padded
                # For safety, use a different axis
                for axis in range(dimension):
                    candidate = list(last)
                    candidate[axis] += 1
                    if tuple(candidate) not in occupied:
                        chain.append(tuple(candidate))
                        occupied[tuple(candidate)] = j
                        last = tuple(candidate)
                        break
                else:
                    # Shouldn't happen
                    chain.append(list(last))
            return [list(p) for p in chain]
        
        chain.append(best_site)
        occupied[best_site] = i
    
    return [list(p) for p in chain]
# EVOLVE-BLOCK-END
