import time
import random
import sys
from itertools import combinations

# Increase recursion limit for deep branch-and-bound
sys.setrecursionlimit(10000)

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N_TRIPLES = len(ALL_TRIPLES)  # 560

# Precompute triple indices
_triple_idx = {t: i for i, t in enumerate(ALL_TRIPLES)}

# Precompute all K(2,2,2) structures
# A K(2,2,2) is defined by 3 disjoint pairs (x1,x2), (y1,y2), (z1,z2)
# and consists of the 8 triples: {xi, yj, zk} for i,j,k in {1,2}
# We need to check if a set of triples contains all 8 of any such structure.

# Strategy:
# 1. Precompute for each pair of vertices (a,b), the set of vertices c such that (a,b,c) is a triple.
# 2. For conflict checking, we use a bitmask-based approach.

# partner_mask[i*V + j] is a bitmask of vertices k such that (i,j,k) is in the current hypergraph.
# We use this to quickly check if adding a new triple creates a K(2,2,2).

def _check_conflict_fast(a, b, c, partner_mask):
    """
    Check if adding triple (a,b,c) creates a K(2,2,2).
    
    A K(2,2,2) is formed by 3 disjoint pairs (x1,x2), (y1,y2), (z1,z2).
    If we add (a,b,c), it could be part of a K(2,2,2) where:
    - a is in pair (a, a'), b in (b, b'), c in (c, c')
    - Then the 8 triples are all combinations of {a,a'} x {b,b'} x {c,c'}
    
    For a conflict to exist, there must be a', b', c' such that:
    - (a', b, c), (a, b', c), (a, b, c') are present (to form the "edges" of the K222)
    - Actually, we need ALL 8 to be present. But we're adding (a,b,c), so we need
      the other 7 to be present.
    
    More precisely: if we add (a,b,c), a K222 is completed if there exist a', b', c'
    (distinct from a,b,c and from each other) such that all of the following are present:
    (a',b,c), (a,b',c), (a,b,c'), (a',b',c), (a',b,c'), (a,b',c'), (a',b',c')
    
    Wait, that's not quite right. The 8 triples are:
    (a,b,c), (a,b,c'), (a,b',c), (a,b',c'), (a',b,c), (a',b,c'), (a',b',c), (a',b',c')
    
    So if we add (a,b,c), we need the other 7 to be present.
    """
    # For each possible a' (partner of a), b' (partner of b), c' (partner of c):
    # Check if all other 7 triples are present.
    
    # a' must be such that (a', b, c) is present -> a' in partner_mask[b*V + c]
    # b' must be such that (a, b', c) is present -> b' in partner_mask[a*V + c]
    # c' must be such that (a, b, c') is present -> c' in partner_mask[a*V + b]
    
    mask_a_candidates = partner_mask[b * V + c]
    mask_a_candidates &= ~(1 << a)
    mask_a_candidates &= ~(1 << b)
    mask_a_candidates &= ~(1 << c)
    
    mask_b_candidates = partner_mask[a * V + c]
    mask_b_candidates &= ~(1 << a)
    mask_b_candidates &= ~(1 << b)
    mask_b_candidates &= ~(1 << c)
    
    mask_c_candidates = partner_mask[a * V + b]
    mask_c_candidates &= ~(1 << a)
    mask_c_candidates &= ~(1 << b)
    mask_c_candidates &= ~(1 << c)
    
    # Iterate over possible a'
    temp_a = mask_a_candidates
    while temp_a:
        lsb_a = temp_a & -temp_a
        a_p = lsb_a.bit_length() - 1
        temp_a ^= lsb_a
        
        # Now check if there exist b', c' such that all other triples are present
        # For this a', we need:
        # (a', b, c) present (guaranteed by a' in mask_a_candidates)
        # (a', b', c) present -> b' in partner_mask[a_p*V + c]
        # (a', b, c') present -> c' in partner_mask[a_p*V + b]
        # (a', b', c') present -> c' in partner_mask[a_p*V + b_p] for each b'
        
        # Let's iterate over b'
        mask_b_valid = mask_b_candidates & partner_mask[a_p * V + c]
        mask_b_valid &= ~(1 << a_p)
        
        if mask_b_valid == 0:
            continue
        
        temp_b = mask_b_valid
        while temp_b:
            lsb_b = temp_b & -temp_b
            b_p = lsb_b.bit_length() - 1
            temp_b ^= lsb_b
            
            # Now check if there exists c' such that:
            # (a, b, c') present -> c' in mask_c_candidates
            # (a', b, c') present -> c' in partner_mask[a_p*V + b]
            # (a, b', c') present -> c' in partner_mask[a*V + b_p]
            # (a', b', c') present -> c' in partner_mask[a_p*V + b_p]
            
            mask_c_valid = mask_c_candidates
            mask_c_valid &= partner_mask[a_p * V + b]
            mask_c_valid &= partner_mask[a * V + b_p]
            mask_c_valid &= partner_mask[a_p * V + b_p]
            mask_c_valid &= ~(1 << b_p)
            
            if mask_c_valid:
                return True
    
    return False

def _add_triple(a, b, c, partner_mask, presence, idx):
    presence[idx] = True
    partner_mask[a * V + b] |= (1 << c)
    partner_mask[b * V + a] |= (1 << c)
    partner_mask[a * V + c] |= (1 << b)
    partner_mask[c * V + a] |= (1 << b)
    partner_mask[b * V + c] |= (1 << a)
    partner_mask[c * V + b] |= (1 << a)

def _remove_triple(a, b, c, partner_mask, presence, idx):
    presence[idx] = False
    partner_mask[a * V + b] &= ~(1 << c)
    partner_mask[b * V + a] &= ~(1 << c)
    partner_mask[a * V + c] &= ~(1 << b)
    partner_mask[c * V + a] &= ~(1 << b)
    partner_mask[b * V + c] &= ~(1 << a)
    partner_mask[c * V + b] &= ~(1 << a)

def _greedy(order, deadline, partner_mask):
    presence = [False] * N_TRIPLES
    count = 0
    for idx in order:
        if time.time() >= deadline:
            break
        a, b, c = ALL_TRIPLES[idx]
        if _check_conflict_fast(a, b, c, partner_mask):
            continue
        count += 1
        _add_triple(a, b, c, partner_mask, presence, idx)
    return count, presence

def _local_search(count, presence, partner_mask, deadline):
    """
    Local search with add and swap moves.
    """
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to add any missing triple
        for idx in range(N_TRIPLES):
            if time.time() >= deadline:
                return count, presence, partner_mask
            if presence[idx]:
                continue
            a, b, c = ALL_TRIPLES[idx]
            if not _check_conflict_fast(a, b, c, partner_mask):
                count += 1
                _add_triple(a, b, c, partner_mask, presence, idx)
                improved = True
        
        if improved:
            continue
        
        # Try swaps: remove one, try to add multiple
        present_indices = [i for i in range(N_TRIPLES) if presence[i]]
        if not present_indices:
            break
        
        # Try a few random swaps
        for _ in range(50):
            if time.time() >= deadline:
                return count, presence, partner_mask
            
            idx_out = random.choice(present_indices)
            a, b, c = ALL_TRIPLES[idx_out]
            
            # Save state
            old_partner_mask = partner_mask[:]
            old_presence = presence[:]
            old_count = count
            
            _remove_triple(a, b, c, partner_mask, presence, idx_out)
            count -= 1
            
            added_any = False
            for idx_in in range(N_TRIPLES):
                if time.time() >= deadline:
                    partner_mask = old_partner_mask
                    presence = old_presence
                    count = old_count
                    return count, presence, partner_mask
                if presence[idx_in]:
                    continue
                a_in, b_in, c_in = ALL_TRIPLES[idx_in]
                if not _check_conflict_fast(a_in, b_in, c_in, partner_mask):
                    count += 1
                    _add_triple(a_in, b_in, c_in, partner_mask, presence, idx_in)
                    added_any = True
            
            if count > old_count:
                improved = True
                break
            else:
                partner_mask = old_partner_mask
                presence = old_presence
                count = old_count
    
    return count, presence, partner_mask

def construct():
    start = time.time()
    deadline = start + 115
    
    best_count = 0
    best_presence = [False] * N_TRIPLES
    best_partner_mask = [0] * (V * V)
    
    random.seed(42)
    
    # Run multiple attempts with different random orders
    while time.time() < deadline:
        order = list(range(N_TRIPLES))
        random.shuffle(order)
        
        partner_mask = [0] * (V * V)
        
        count, presence = _greedy(order, deadline, partner_mask)
        count, presence, partner_mask = _local_search(count, presence, partner_mask, deadline)
        
        if count > best_count:
            best_count = count
            best_presence = presence[:]
            best_partner_mask = partner_mask[:]
        
        # If we have a good result, try to improve with more local search
        if count >= best_count and time.time() < deadline - 10:
            count, presence, partner_mask = _local_search(count, presence, partner_mask, deadline - 5)
            if count > best_count:
                best_count = count
                best_presence = presence[:]
                best_partner_mask = partner_mask[:]
    
    result = []
    for i in range(N_TRIPLES):
        if best_presence[i]:
            result.append(list(ALL_TRIPLES[i]))
    return result
