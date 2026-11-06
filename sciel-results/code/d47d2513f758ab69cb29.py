"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

N = 256
MAX_D = N // 3  # 85

def _thue_morse_fast(n):
    """Generate Thue-Morse sequence of length n using bit operations."""
    c = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        c[i] = c[i >> 1] ^ (i & 1)
    return c

def _precompute_aps():
    """Precompute all 4-term APs as (a, a+d, a+2d, a+3d) with 0-indexed positions."""
    aps = []
    for d in range(1, MAX_D + 1):
        max_a = N - 3 * d
        for a in range(max_a):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    return aps

def _count_mono4_vectorized(c):
    """Count monochromatic 4-term APs using numpy vectorization."""
    count = 0
    for d in range(1, MAX_D + 1):
        max_a = N - 3 * d
        c0 = c[0:max_a]
        c1 = c[d:d + max_a]
        c2 = c[2*d:2*d + max_a]
        c3 = c[3*d:3*d + max_a]
        mask = (c0 == c1) & (c0 == c2) & (c0 == c3)
        count += int(np.sum(mask))
    return count

def _build_ap_lookup():
    """Build lookup: for each position i, list of (ap_index, position_in_ap) where i appears in AP."""
    aps = _precompute_aps()
    ap_indices = np.array(aps, dtype=np.int32)  # shape (num_aps, 4)
    
    # For each position, which APs contain it and at which slot
    pos_to_aps = [[] for _ in range(N)]
    for idx, (a, b, c_, d_) in enumerate(aps):
        pos_to_aps[a].append((idx, 0))
        pos_to_aps[b].append((idx, 1))
        pos_to_aps[c_].append((idx, 2))
        pos_to_aps[d_].append((idx, 3))
    
    return aps, ap_indices, pos_to_aps

def _delta_flip(c, pos, aps, ap_indices, pos_to_aps, mono_flags):
    """
    Compute the change in monochromatic 4-AP count when flipping position pos.
    mono_flags: boolean array of shape (num_aps,) indicating if AP is currently monochromatic.
    Returns the delta (new_count - old_count).
    """
    old_val = c[pos]
    new_val = 1 - old_val
    delta = 0
    
    for ap_idx, slot in pos_to_aps[pos]:
        # Get the other 3 positions in this AP
        other_slots = [s for s in range(4) if s != slot]
        other_vals = c[ap_indices[ap_idx, other_slots]]
        
        # Check if the AP is currently monochromatic
        is_mono_now = mono_flags[ap_idx]
        
        # After flip: the flipped position has new_val.
        # The AP becomes monochromatic iff all 4 values are equal.
        # Since we're flipping one position, the AP is mono after flip iff:
        #   - new_val equals all other 3 values (which must be equal to each other)
        
        # Check if other 3 are all equal
        o0, o1, o2 = other_vals
        others_equal = (o0 == o1) and (o1 == o2)
        
        if others_equal:
            if new_val == o0:
                # After flip, all 4 are equal -> mono
                was_mono = is_mono_now
                if not was_mono:
                    delta += 1  # gained a mono AP
                # else: was mono, still mono, no change
            else:
                # After flip, not all equal
                if is_mono_now:
                    delta -= 1  # lost a mono AP
        else:
            # Others not all equal, so AP can never be mono
            # But wait - if it was mono before, that means old_val == o0 == o1 == o2
            # which contradicts others_equal being False unless... 
            # Actually if is_mono_now is True, then old_val == o0 == o1 == o2, so others_equal must be True
            # So this case can't happen if is_mono_now is True
            pass
    
    return delta

def _local_search_incremental(c, aps, ap_indices, pos_to_aps, deadline):
    """Local search with incremental delta computation."""
    num_aps = len(aps)
    mono_flags = np.zeros(num_aps, dtype=bool)
    
    # Initialize mono_flags
    for idx in range(num_aps):
        a, b, cc, d_ = ap_indices[idx]
        mono_flags[idx] = (c[a] == c[b]) and (c[b] == c[cc]) and (c[cc] == c[d_])
    
    current_count = int(np.sum(mono_flags))
    best_count = current_count
    best_c = c.copy()
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for pos in range(N):
            if time.time() > deadline:
                break
            old_val = c[pos]
            new_val = 1 - old_val
            
            # Compute delta
            delta = 0
            for ap_idx, slot in pos_to_aps[pos]:
                other_slots = [s for s in range(4) if s != slot]
                other_vals = c[ap_indices[ap_idx, other_slots]]
                o0, o1, o2 = other_vals
                others_equal = (o0 == o1) and (o1 == o2)
                
                if others_equal:
                    if new_val == o0:
                        if not mono_flags[ap_idx]:
                            delta += 1
                    else:
                        if mono_flags[ap_idx]:
                            delta -= 1
            
            if delta < 0:
                # Accept the flip
                c[pos] = new_val
                # Update mono_flags for affected APs
                for ap_idx, slot in pos_to_aps[pos]:
                    a, b, cc, d_ = ap_indices[ap_idx]
                    mono_flags[ap_idx] = (c[a] == c[b]) and (c[b] == c[cc]) and (c[cc] == c[d_])
                current_count += delta
                if current_count < best_count:
                    best_count = current_count
                    best_c = c.copy()
                improved = True
            # else: revert (c[pos] already unchanged since we didn't modify it)
    
    return best_c, best_count

def _two_flip_search(c, aps, ap_indices, pos_to_aps, deadline):
    """Try pairs of flips to escape local optima."""
    num_aps = len(aps)
    mono_flags = np.zeros(num_aps, dtype=bool)
    for idx in range(num_aps):
        a, b, cc, d_ = ap_indices[idx]
        mono_flags[idx] = (c[a] == c[b]) and (c[b] == c[cc]) and (c[cc] == c[d_])
    
    current_count = int(np.sum(mono_flags))
    best_count = current_count
    best_c = c.copy()
    
    # Try random pairs of flips
    positions = list(range(N))
    random.shuffle(positions)
    
    attempts = 0
    while time.time() < deadline and attempts < 50000:
        attempts += 1
        # Pick two random positions
        p1, p2 = random.sample(positions, 2)
        
        if p1 == p2:
            continue
        
        old_v1 = c[p1]
        old_v2 = c[p2]
        new_v1 = 1 - old_v1
        new_v2 = 1 - old_v2
        
        # Compute combined delta
        # This is more complex because flipping both positions affects shared APs
        # Let's just compute the full count for the flipped state
        c[p1] = new_v1
        c[p2] = new_v2
        
        # Recompute mono flags for affected APs
        affected_aps = set()
        for ap_idx, slot in pos_to_aps[p1]:
            affected_aps.add(ap_idx)
        for ap_idx, slot in pos_to_aps[p2]:
            affected_aps.add(ap_idx)
        
        delta = 0
        for ap_idx in affected_aps:
            was_mono = mono_flags[ap_idx]
            a, b, cc, d_ = ap_indices[ap_idx]
            is_mono_now = (c[a] == c[b]) and (c[b] == c[cc]) and (c[cc] == c[d_])
            if is_mono_now and not was_mono:
                delta += 1
            elif was_mono and not is_mono_now:
                delta -= 1
            mono_flags[ap_idx] = is_mono_now
        
        new_count = current_count + delta
        
        if new_count < best_count:
            best_count = new_count
            best_c = c.copy()
            current_count = new_count
        else:
            # Revert
            c[p1] = old_v1
            c[p2] = old_v2
            # Revert mono flags
            for ap_idx in affected_aps:
                a, b, cc, d_ = ap_indices[ap_idx]
                mono_flags[ap_idx] = (c[a] == c[b]) and (c[b] == c[cc]) and (c[cc] == c[d_])
    
    return best_c, best_count

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    # Start with Thue-Morse
    c = _thue_morse_fast(N)
    
    # Precompute
    aps, ap_indices, pos_to_aps = _build_ap_lookup()
    
    # Initial count
    current_count = _count_mono4_vectorized(c)
    best_c = c.copy()
    best_count = current_count
    
    # Phase 1: Local search with incremental updates
    c, current_count = _local_search_incremental(c, aps, ap_indices, pos_to_aps, deadline)
    if current_count < best_count:
        best_count = current_count
        best_c = c.copy()
    
    # Phase 2: Two-flip search to escape local optima
    if time.time() < deadline:
        c, current_count = _two_flip_search(c, aps, ap_indices, pos_to_aps, deadline)
        if current_count < best_count:
            best_count = current_count
            best_c = c.copy()
    
    # Phase 3: Another round of local search on the improved solution
    if time.time() < deadline:
        c, current_count = _local_search_incremental(c, aps, ap_indices, pos_to_aps, deadline)
        if current_count < best_count:
            best_count = current_count
            best_c = c.copy()
    
    # Phase 4: Final two-flip search
    if time.time() < deadline:
        c, current_count = _two_flip_search(c, aps, ap_indices, pos_to_aps, deadline)
        if current_count < best_count:
            best_count = current_count
            best_c = c.copy()
    
    # Final local search polish
    if time.time() < deadline:
        c, current_count = _local_search_incremental(c, aps, ap_indices, pos_to_aps, deadline)
        if current_count < best_count:
            best_count = current_count
            best_c = c.copy()
    
    return [int(x) for x in best_c]
# EVOLVE-BLOCK-END
