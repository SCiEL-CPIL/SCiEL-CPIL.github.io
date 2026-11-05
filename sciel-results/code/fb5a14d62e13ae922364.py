"""Rewritten program with a structural change: simulated annealing on bitmask family."""
# EVOLVE-BLOCK-START
import itertools
import random
import time

def _generate_all_sets():
    """Generate all 4-element subsets of [1, 16] as bitmasks."""
    sets = []
    for combo in itertools.combinations(range(16), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        sets.append(mask)
    return sets

def _mask_to_list(mask):
    """Convert bitmask to sorted list of 1-based indices."""
    return sorted([i + 1 for i in range(16) if (mask >> i) & 1])

def _has_sunflower_triple(family):
    """
    Check if any three distinct sets in family form a sunflower triple.
    A sunflower triple is three distinct sets A, B, C such that
    A & B == B & C == C & A (the common core).
    """
    n = len(family)
    if n < 3:
        return False
    # For each pair, compute intersection, then check if a third set has that intersection with both
    # Efficient approach: for each pair (i, j), compute v = A_i & A_j.
    # Then for each k != i, j, check if A_i & A_k == v and A_j & A_k == v.
    # But this is O(n^3). For n up to ~250, that's 15.6M checks, each with bit ops.
    # 15.6M * few ops = maybe 1-2 seconds in Python. Might be too slow for repeated calls.
    # Let's use a smarter approach with indexing.
    
    # Alternative: group sets by their intersection with each other.
    # For a fixed set A, for each possible intersection value v, keep a list of sets B such that A & B == v.
    # Then for any two sets B, C in the same group (same v with A), check if B & C == v.
    # If yes, then A, B, C form a sunflower.
    
    # Build index for the current family
    # idx[A][v] = list of B in family such that A & B == v
    idx = {}
    for i, A in enumerate(family):
        idx[A] = {}
    
    for i, A in enumerate(family):
        for j, B in enumerate(family):
            if i >= j:
                continue
            v = A & B
            if v not in idx[A]:
                idx[A][v] = []
            idx[A][v].append(B)
    
    # Now check for sunflower triples
    for A in family:
        for v, group in idx[A].items():
            if len(group) < 2:
                continue
            # For each pair B, C in group, check if B & C == v
            for x in range(len(group)):
                for y in range(x + 1, len(group)):
                    B = group[x]
                    C = group[y]
                    if (B & C) == v:
                        return True
    return False

def _count_sunflower_triples(family):
    """
    Count the number of sunflower triples in the family.
    Returns 0 if none.
    """
    if len(family) < 3:
        return 0
    
    idx = {}
    for A in family:
        idx[A] = {}
    
    for A in family:
        for B in family:
            if A >= B:
                continue
            v = A & B
            if v not in idx[A]:
                idx[A][v] = []
            idx[A][v].append(B)
    
    count = 0
    for A in family:
        for v, group in idx[A].items():
            if len(group) < 2:
                continue
            for x in range(len(group)):
                for y in range(x + 1, len(group)):
                    B = group[x]
                    C = group[y]
                    if (B & C) == v:
                        count += 1
    return count

def _can_add(family, idx, C):
    """
    Check if C can be added to family without forming a sunflower triple.
    idx[A][v] = list of B in family such that A & B == v.
    """
    for A in family:
        v = A & C
        if v in idx[A]:
            for B in idx[A][v]:
                if (B & C) == v:
                    return False
    return True

def _add_to_family(family, idx, C):
    """Add C to family and update index."""
    family.append(C)
    idx[C] = {}
    for A in family[:-1]:
        v = A & C
        if v not in idx[A]:
            idx[A][v] = []
        idx[A][v].append(C)
        
        if v not in idx[C]:
            idx[C][v] = []
        idx[C][v].append(A)

def _remove_from_family(family, idx, C):
    """Remove C from family and update index."""
    if C not in family:
        return
    family.remove(C)
    
    for A in family:
        v = A & C
        if v in idx[A]:
            if C in idx[A][v]:
                idx[A][v].remove(C)
                if not idx[A][v]:
                    del idx[A][v]
        
        if v in idx[C]:
            if A in idx[C][v]:
                idx[C][v].remove(A)
                if not idx[C][v]:
                    del idx[C][v]
    
    if C in idx:
        del idx[C]

def _greedy_construct(all_sets, order, deadline):
    """Run a greedy pass with the given order."""
    family = []
    idx = {}
    
    for C in order:
        if time.time() > deadline:
            break
        if _can_add(family, idx, C):
            _add_to_family(family, idx, C)
            
    return family, idx

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    all_sets = _generate_all_sets()
    random.seed(42)
    
    best_family = []
    best_idx = {}
    best_size = 0
    
    # Phase 1: Multiple greedy restarts to get a good initial solution
    while time.time() < deadline - 15.0:
        order = all_sets[:]
        random.shuffle(order)
        
        family, idx = _greedy_construct(all_sets, order, deadline - 15.0)
        
        if len(family) > best_size:
            best_family = family[:]
            best_idx = {k: {k2: v2[:] for k2, v2 in v.items()} for k, v in idx.items()}
            best_size = len(family)
            
            # Try to improve with local search
            if time.time() < deadline - 10.0:
                # Local search: try to add sets that were skipped
                candidates = [s for s in all_sets if s not in set(family)]
                random.shuffle(candidates)
                
                improved = True
                while improved and time.time() < deadline - 10.0:
                    improved = False
                    for C in candidates[:]:
                        if time.time() > deadline - 10.0:
                            break
                        if C in set(family):
                            continue
                        if _can_add(family, best_idx, C):
                            _add_to_family(family, best_idx, C)
                            candidates.remove(C)
                            improved = True
                            if len(family) > best_size:
                                best_size = len(family)
                            break
                    
                    # Try swaps
                    if not improved and family and candidates:
                        for _ in range(5):
                            if time.time() > deadline - 10.0:
                                break
                            idx_old = random.randint(0, len(family) - 1)
                            A_old = family[idx_old]
                            C_new = random.choice(candidates)
                            
                            _remove_from_family(family, best_idx, A_old)
                            if _can_add(family, best_idx, C_new):
                                _add_to_family(family, best_idx, C_new)
                                candidates.remove(C_new)
                                candidates.append(A_old)
                                improved = True
                                if len(family) > best_size:
                                    best_size = len(family)
                                break
                            else:
                                _add_to_family(family, best_idx, A_old)
    
    # Phase 2: Simulated annealing to further improve
    if best_family and time.time() < deadline - 5.0:
        family = best_family[:]
        idx = {k: {k2: v2[:] for k2, v2 in v.items()} for k, v in best_idx.items()}
        in_family = set(family)
        candidates = [s for s in all_sets if s not in in_family]
        
        current_size = len(family)
        best_sa_size = current_size
        best_sa_family = family[:]
        best_sa_idx = {k: {k2: v2[:] for k2, v2 in v.items()} for k, v in idx.items()}
        
        temperature = 1.0
        cooling_rate = 0.999
        min_temp = 0.01
        
        while temperature > min_temp and time.time() < deadline - 5.0:
            # Try a random move
            if random.random() < 0.3 and candidates:
                # Try to add a random candidate
                C_new = random.choice(candidates)
                if C_new not in in_family:
                    if _can_add(family, idx, C_new):
                        _add_to_family(family, idx, C_new)
                        in_family.add(C_new)
                        candidates.remove(C_new)
                        if len(family) > best_sa_size:
                            best_sa_size = len(family)
                            best_sa_family = family[:]
                            best_sa_idx = {k: {k2: v2[:] for k2, v2 in v.items()} for k, v in idx.items()}
                    else:
                        # Reject with some probability based on temperature
                        if random.random() > temperature:
                            pass  # Reject
            elif family:
                # Try a swap
                idx_old = random.randint(0, len(family) - 1)
                A_old = family[idx_old]
                
                if candidates:
                    C_new = random.choice(candidates)
                    
                    _remove_from_family(family, idx, A_old)
                    in_family.discard(A_old)
                    
                    if _can_add(family, idx, C_new):
                        _add_to_family(family, idx, C_new)
                        in_family.add(C_new)
                        candidates.remove(C_new)
                        candidates.append(A_old)
                        if len(family) > best_sa_size:
                            best_sa_size = len(family)
                            best_sa_family = family[:]
                            best_sa_idx = {k: {k2: v2[:] for k2, v2 in v.items()} for k, v in idx.items()}
                    else:
                        # Revert
                        _add_to_family(family, idx, A_old)
                        in_family.add(A_old)
                        if C_new in candidates:
                            pass
                        else:
                            candidates.append(C_new)
                        candidates.remove(A_old)
                    
                    temperature *= cooling_rate
                else:
                    # No candidates, try removing a set (shouldn't happen if we're maximizing)
                    temperature *= cooling_rate
        
        if best_sa_size > best_size:
            best_family = best_sa_family
            best_idx = best_sa_idx
            best_size = best_sa_size
    
    # Convert to sorted lists
    result = [_mask_to_list(s) for s in best_family]
    return result

# EVOLVE-BLOCK-END
