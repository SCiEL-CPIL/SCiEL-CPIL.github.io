"""HP lattice protein folding with Simulated Annealing and Fragment Moves."""
# EVOLVE-BLOCK-START
import random
import time
import math

_START_TIME = time.time()
_DEADLINE = 205.0  # Safety margin before 210s kill

def _remaining_time():
    return _DEADLINE - (time.time() - _START_TIME)

def _neighbours(point):
    x, y = point
    return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

def _compute_energy(sequence, chain):
    """Compute energy from scratch. O(N)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _grow(sequence, dimension, rng):
    """Fast greedy growth to get a valid initial chain."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        last = chain[-1]
        candidates = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Heuristic: prefer sites that create H-H contacts
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H":
                        gain += 1
            # Add some randomness to avoid deterministic traps
            candidates.append((gain + rng.random(), site))
        
        if not candidates:
            return None # Should not happen for reasonable N on infinite lattice
            
        # Pick best candidate
        candidates.sort(reverse=True)
        best_site = candidates[0][1]
        chain.append(best_site)
        occupied[best_site] = i
        
    return chain

def _rotate_point(p, rot):
    """Rotate point (x, y) by 90 degrees 'rot' times counter-clockwise."""
    x, y = p
    for _ in range(rot):
        x, y = -y, x
    return (x, y)

def fold(sequence, dimension):
    """
    Fold the sequence using Simulated Annealing with Fragment Moves.
    """
    if dimension != 2:
        raise ValueError("Only 2D supported")
        
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
        
    rng = random.Random(12345 + n)
    
    # 1. Initialization
    chain = _grow(sequence, dimension, rng)
    if chain is None:
        # Fallback to a straight line if growth fails (rare)
        chain = [(i, 0) for i in range(n)]
        
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = current_energy
    
    # 2. Simulated Annealing
    # Temperature schedule
    T_start = 1.0
    T_end = 0.01
    # We don't know exact steps in advance, so we use a cooling rate based on time or steps
    # Let's try a fixed number of steps if time permits, or time-based.
    # Given N <= 100, 10k-50k steps is fast.
    
    steps_done = 0
    max_steps = 20000  # Cap steps to avoid infinite loop if time check fails
    
    while steps_done < max_steps:
        if _remaining_time() < 5.0:
            break
            
        # Select fragment
        # Fragment length L between 2 and n/2
        max_len = n // 2
        if max_len < 2:
            break
        L = rng.randint(2, max_len)
        
        # Select start index i such that fragment i+1 .. i+L is valid
        # Fragment is chain[i+1 : i+L+1]
        # It connects to chain[i] and chain[i+L+1] (if i+L+1 < n)
        # Wait, standard fragment move:
        # Cut at i and j. Move segment i+1 .. j.
        # Let's pick i and L. j = i + L.
        # The segment is chain[i+1 ... j].
        # It is attached to chain[i] and chain[j+1] (if exists).
        
        # To ensure we don't cut at the very ends in a way that breaks logic:
        # i can be 0 .. n-L-2 (so j+1 <= n-1)
        # Actually, if j == n-1, the fragment is at the end.
        # Let's allow i from 0 to n-L-1.
        # If i + L == n-1, the right side is empty.
        
        max_i = n - L - 1
        if max_i < 0:
            continue
        i = rng.randint(0, max_i)
        j = i + L
        
        # Get fragment coordinates relative to chain[i]
        anchor = chain[i]
        # Fragment points: chain[i+1] .. chain[j]
        # Relative vectors
        rel_frags = []
        for k in range(i+1, j+1):
            px, py = chain[k]
            rx, ry = px - anchor[0], py - anchor[1]
            rel_frags.append((rx, ry))
        
        # Try 4 rotations
        # We need to remove the fragment from occupied to check collision
        # But we must be careful: the anchor chain[i] is NOT part of the fragment.
        # The points chain[i+1]..chain[j] are removed.
        
        # Optimization: Instead of removing/adding to set every time (which is O(L)),
        # we can just check collision against `occupied` but ignore the fragment points themselves.
        # However, `occupied` contains the fragment points.
        # So we check: for new_point in new_frag:
        #    if new_point in occupied and occupied[new_point] not in range(i+1, j+1):
        #         collision
        
        # This is O(L) per rotation check.
        
        accepted_move = False
        for rot in range(4):
            # Generate new absolute positions
            new_frag = []
            valid = True
            for rx, ry in rel_frags:
                nrx, nry = _rotate_point((rx, ry), rot)
                nx, ny = anchor[0] + nrx, anchor[1] + nry
                new_point = (nx, ny)
                new_frag.append(new_point)
                
                # Check collision
                if new_point in occupied:
                    idx = occupied[new_point]
                    # If it's part of the fragment being moved, it's fine (self-overlap check later?)
                    # No, if it's part of the fragment, it's the "old" position or another fragment point.
                    # We need to ensure the NEW fragment is self-avoiding and doesn't hit the STATIC parts.
                    
                    # Static parts are indices 0..i and j+1..n-1
                    if not (i < idx <= j):
                        valid = False
                        break
            
            if not valid:
                continue
                
            # Check self-avoidance of the new fragment
            # Since we generated them from a valid fragment rotated, they are relatively self-avoiding?
            # Yes, rotation is an isometry. If the original fragment was self-avoiding, the rotated one is too.
            # Wait, the original fragment is a part of a SAW, so it is self-avoiding.
            # So new_frag is self-avoiding.
            
            # Calculate new energy
            # We need to update the chain to calculate energy?
            # Or calculate delta energy?
            # For simplicity and speed (N is small), let's just build the new chain and calc energy.
            # But building the chain is O(N).
            
            # Let's try to be smarter:
            # Old energy E_old.
            # New chain: chain[:i+1] + new_frag + chain[j+1:]
            
            # We can compute E_new by taking E_old and adjusting for contacts involving the fragment.
            # This is complex.
            # Given N=100, let's just compute full energy.
            # But we need the full chain.
            
            # Let's create a temporary chain list.
            # This is O(N) memory and time.
            # If we do this for 4 rotations, it's 4*O(N).
            # 20000 steps * 4 * 100 = 8,000,000 operations. This is fine.
            
            # However, we can avoid building the full list if we just check validity first.
            # We already checked validity (collision with static parts).
            # So we can compute energy.
            
            # To compute energy, we need the `where` map.
            # Building `where` is O(N).
            
            # Let's just do it.
            
            # Optimization: Only compute energy if the move looks promising?
            # No, SA needs energy for acceptance.
            
            # Let's construct the new chain tuple for energy calc
            # Actually, we can just update the `chain` list and `occupied` set if we accept.
            # But we need E_new to decide.
            
            # Let's compute E_new.
            # To save time, we can pre-compute the "static" part of the energy?
            # No, contacts can form between fragment and static parts.
            
            # Let's just use a helper that takes a list of points.
            # But we don't want to allocate a new list of 100 tuples 80,000 times.
            
            # Alternative:
            # Compute Delta E.
            # E = - (Contacts_Static + Contacts_Fragment_Internal + Contacts_Fragment_Static)
            # E_old has:
            #   C_SS (Static-Static)
            #   C_FF_old (Fragment-Fragment old)
            #   C_SF_old (Static-Fragment old)
            
            # E_new has:
            #   C_SS (Same)
            #   C_FF_new (Fragment-Fragment new) -- Same as C_FF_old because rotation preserves internal distances!
            #   C_SF_new (Static-Fragment new)
            
            # So Delta E = - (C_SF_new - C_SF_old)
            # We only need to count contacts between the fragment and the static parts.
            
            # This is MUCH faster. O(L * 4) for new contacts + O(L * 4) for old contacts.
            
            # Let's implement Delta E calculation.
            
            # Static indices: 0..i and j+1..n-1
            # Fragment indices: i+1..j
            
            # 1. Count Old SF contacts
            old_sf_contacts = 0
            for k in range(i+1, j+1):
                if sequence[k] == "H":
                    p = chain[k]
                    for q in _neighbours(p):
                        if q in occupied:
                            idx = occupied[q]
                            if (idx <= i or idx > j) and sequence[idx] == "H":
                                # Check if k and idx are not adjacent in chain
                                # k is in fragment, idx is in static.
                                # They are adjacent in chain if |k - idx| == 1.
                                # Since fragment is i+1..j and static is 0..i and j+1..n-1:
                                # Adjacent pairs are (i, i+1) and (j, j+1).
                                # If idx == i and k == i+1, they are chain-adjacent.
                                # If idx == j+1 and k == j, they are chain-adjacent.
                                if not (idx == i and k == i+1) and not (idx == j+1 and k == j):
                                    old_sf_contacts += 1
                                    # Wait, this counts each contact twice?
                                    # No, we iterate over fragment H's.
                                    # If p is H and q is H, we count 1.
                                    # The other side (iterating over static H's) would also count it.
                                    # But we are only iterating over fragment H's.
                                    # So we count each SF contact exactly once (from the fragment side).
                                    # UNLESS both are H and we count from fragment side.
                                    # Yes, that's correct.
            
            # 2. Count New SF contacts
            new_sf_contacts = 0
            # We need to know which static points are H.
            # We can just check sequence[idx] == "H" on the fly.
            
            for k in range(i+1, j+1):
                if sequence[k] == "H":
                    # New position of k
                    # k is the (k-i-1)-th point in the fragment?
                    # Fragment points in new_frag correspond to chain[i+1]...chain[j]
                    # So new_frag[m] corresponds to chain[i+1+m]
                    # So for chain index k, m = k - (i+1)
                    m = k - (i + 1)
                    p_new = new_frag[m]
                    for q in _neighbours(p_new):
                        if q in occupied:
                            idx = occupied[q]
                            if (idx <= i or idx > j) and sequence[idx] == "H":
                                # Check chain adjacency
                                if not (idx == i and k == i+1) and not (idx == j+1 and k == j):
                                    new_sf_contacts += 1
            
            delta_e = - (new_sf_contacts - old_sf_contacts)
            
            # Accept/Reject
            if delta_e < 0 or rng.random() < math.exp(delta_e / T_start):
                # Accept
                # Update chain and occupied
                # Remove old fragment points from occupied
                for k in range(i+1, j+1):
                    del occupied[chain[k]]
                
                # Update chain
                for m, p_new in enumerate(new_frag):
                    k = i + 1 + m
                    chain[k] = p_new
                    occupied[p_new] = k
                
                current_energy += delta_e
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                
                accepted_move = True
                break # Stop trying other rotations for this fragment selection
            
        # Cool down
        # Simple linear cooling or geometric
        # Let's use geometric with a fixed rate per step
        T_start *= 0.9999
        if T_start < T_end:
            T_start = T_end
            
        steps_done += 1
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
