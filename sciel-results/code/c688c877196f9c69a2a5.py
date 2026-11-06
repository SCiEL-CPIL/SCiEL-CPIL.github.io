"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

# EVOLVE-BLOCK-START

def _get_paleyn_orientation(n):
    """
    Constructs a Paley tournament orientation for prime n.
    Returns list of bits or None if n is not a valid prime for Paley (n % 4 == 3).
    """
    if n < 2:
        return None
    # Check prime
    is_prime = True
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            is_prime = False
            break
    if not is_prime:
        return None
    if n % 4 != 3:
        return None
        
    squares = set()
    for x in range(1, n):
        squares.add((x * x) % n)
        
    orientation = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if diff in squares:
                orientation.append(0) # i -> j
            else:
                orientation.append(1) # j -> i
    return orientation

def _count_bad_sets(n, in_bits):
    """
    Counts the number of 4-sets with no common dominator.
    in_bits[i] is a bitmask of vertices pointing to i.
    """
    bad = 0
    # Iterate all 4-subsets
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                    # Common dominators are in In[a] & In[b] & In[c] & In[d]
                    # Must be outside {a,b,c,d}
                    common = in_bits[a] & in_bits[b] & in_bits[c] & in_bits[d]
                    if (common & ~mask) == 0:
                        bad += 1
    return bad

def _local_search(n, time_limit, seed=42):
    """
    Attempts to find a valid tournament on n vertices using local search.
    Returns orientation list if successful, else None.
    """
    start_time = time.time()
    rng = random.Random(seed)
    
    # Initialize random tournament
    # in_bits[i] bitmask of j such that j -> i
    in_bits = [0] * n
    # orientation list for return
    orientation = [0] * (n * (n - 1) // 2)
    
    # Build initial random tournament
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                # i -> j
                orientation[idx] = 0
                in_bits[j] |= (1 << i)
            else:
                # j -> i
                orientation[idx] = 1
                in_bits[i] |= (1 << j)
            idx += 1
            
    # Precompute lists of 4-subsets for affected sets calculation
    # tuples: list of (a,b,c,d)
    # states: list of 1 if bad, 0 if good
    # vertex_to_tuples: list of lists of indices into tuples
    tuples = []
    states = []
    idx_t = 0
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    tuples.append((a, b, c, d))
                    mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                    common = in_bits[a] & in_bits[b] & in_bits[c] & in_bits[d]
                    is_bad = 1 if (common & ~mask) == 0 else 0
                    states.append(is_bad)
                    idx_t += 1
                    
    bad = sum(states)
    if bad == 0:
        return orientation

    vertex_to_tuples = [[] for _ in range(n)]
    for i, t in enumerate(tuples):
        for v in t:
            vertex_to_tuples[v].append(i)
            
    # Precompute edge index map for O(1) lookup
    # edge_idx[i][j] = index in orientation list for pair (i,j) where i<j
    # We need to map (u,v) to index.
    # Index formula for (i,j) with i<j:
    # sum_{k=0}^{i-1} (n-1-k) + (j-i-1)
    # Or simpler: just store a 2D array or a dictionary.
    # n is small (<=90), so a 2D list of size 90x90 is fine.
    edge_idx = [[-1] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            idx += 1

    T = 1.0
    T_min = 0.01
    cooling = 0.99
    
    while time.time() - start_time < time_limit:
        # Pick random edge
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
            
        # Get affected tuple indices
        # Union of tuples containing i or j
        # Since we need to check state change, we iterate over unique indices
        # Using a set is slow. We can just iterate both lists and use a set for deduplication
        # or just iterate and handle duplicates carefully?
        # Actually, a 4-set can contain both i and j.
        # If we iterate vertex_to_tuples[i] and vertex_to_tuples[j], we might visit the same tuple twice.
        # We can use a set of indices.
        aff_idx = set(vertex_to_tuples[i])
        aff_idx.update(vertex_to_tuples[j])
        
        # Compute delta bad
        delta = 0
        changed_states = [] # list of (idx, new_state)
        
        # Determine current direction
        # i -> j iff (in_bits[j] >> i) & 1
        if (in_bits[j] >> i) & 1:
            # Currently i -> j
            # Flip to j -> i
            # i removed from In[j], j added to In[i]
            new_in_i = in_bits[i] | (1 << j)
            new_in_j = in_bits[j] & ~(1 << i)
            flip_to_j_to_i = True
        else:
            # Currently j -> i
            # Flip to i -> j
            # j removed from In[i], i added to In[j]
            new_in_i = in_bits[i] & ~(1 << j)
            new_in_j = in_bits[j] | (1 << i)
            flip_to_j_to_i = False
            
        for idx_t in aff_idx:
            a, b, c, d = tuples[idx_t]
            mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
            
            # Get new in_bits for a,b,c,d
            ni_a = new_in_i if a == i else (new_in_j if a == j else in_bits[a])
            ni_b = new_in_i if b == i else (new_in_j if b == j else in_bits[b])
            ni_c = new_in_i if c == i else (new_in_j if c == j else in_bits[c])
            ni_d = new_in_i if d == i else (new_in_j if d == j else in_bits[d])
            
            common = ni_a & ni_b & ni_c & ni_d
            is_bad_now = 1 if (common & ~mask) == 0 else 0
            
            old_state = states[idx_t]
            if old_state != is_bad_now:
                delta += (is_bad_now - old_state)
                changed_states.append((idx_t, is_bad_now))
        
        # Acceptance
        if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
            # Apply flip
            if flip_to_j_to_i:
                # i -> j becomes j -> i
                in_bits[i] |= (1 << j)
                in_bits[j] &= ~(1 << i)
                orientation[edge_idx[i][j]] = 1 # j -> i
            else:
                in_bits[i] &= ~(1 << j)
                in_bits[j] |= (1 << i)
                orientation[edge_idx[i][j]] = 0 # i -> j
                
            # Update states
            for idx_t, state in changed_states:
                states[idx_t] = state
            bad += delta
            
            if bad == 0:
                return orientation
                
            T *= cooling
            if T < T_min:
                T = T_min
        else:
            pass # Reject
        
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Strategy:
    # 1. Try to find a valid tournament with n < 67 using local search.
    # 2. If found, return it.
    # 3. If not, return Paley(67) which is likely valid.
    
    best_n = 91
    best_orientation = None
    
    # Try n from 66 down to 31
    # We allocate more time to larger n (closer to 67) as they are more likely to be solvable quickly?
    # Actually, smaller n is harder. Larger n is easier.
    # So we should try 66 first.
    
    # We have ~140s.
    # Let's try 66, 65, 64...
    # If we find a solution for n, we keep it and try n-1.
    
    for n in range(66, 30, -1):
        if time.time() > deadline - 10:
            break
            
        # Time allocation:
        # We have 140s total.
        # Let's reserve 10s for final check/fallback.
        # Remaining 130s.
        # Number of n to try: 36 (66 down to 31).
        # Average time per n: 130 / 36 ~ 3.6s.
        # Let's use 3s per n to be safe, and if we run out of time, stop.
        
        time_alloc = 3.0
        
        # Try multiple seeds if time permits?
        # For now, single seed.
        
        orientation = _local_search(n, time_alloc, seed=42)
        
        if orientation is not None:
            best_n = n
            best_orientation = orientation
            # Continue searching for smaller n
            
    # Fallback to Paley 67
    if best_orientation is None or best_n > 67:
        # Check if we have time to verify Paley 67?
        # Just assume it's valid.
        orientation = _get_paleyn_orientation(67)
        if orientation is not None:
            best_n = 67
            best_orientation = orientation
        else:
            # Fallback to 83 if 67 fails (should not happen)
            orientation = _get_paleyn_orientation(83)
            if orientation is not None:
                best_n = 83
                best_orientation = orientation
            else:
                # Absolute fallback: random large n?
                # Just return n=90 random? It will be invalid.
                # But we must return something.
                n = 90
                orientation = [0] * (n * (n - 1) // 2)
                best_n = n
                best_orientation = orientation

    return {'n': best_n, 'orientation': best_orientation}

# EVOLVE-BLOCK-END
