import time
import random
import sys
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N_TRIPLES = len(ALL_TRIPLES)  # 560
_triple_idx = {t: i for i, t in enumerate(ALL_TRIPLES)}

# Precompute all K(2,2,2) structures as lists of 8 triple indices
# A K(2,2,2) is defined by 3 disjoint pairs of vertices: (x1,x2), (y1,y2), (z1,z2)
# The 8 triples are all (xi, yj, zk) for i,j,k in {1,2}

def _precompute_k222s():
    """Precompute all K(2,2,2) configurations as frozensets of triple indices."""
    k222_list = []
    # Iterate over all ways to choose 3 disjoint pairs from 16 vertices
    # We need to choose 6 distinct vertices and partition them into 3 ordered pairs
    # But the order of pairs doesn't matter for the K222 structure
    # Actually, we just need all sets of 8 triples that form a K222
    
    # For each set of 6 distinct vertices, there are 15 ways to partition into 3 pairs
    # But we need to be careful about duplicates.
    # A K222 is uniquely determined by its 3 pairs (as a set of 3 unordered pairs).
    
    vertices = range(V)
    for six in combinations(vertices, 6):
        # Generate all perfect matchings of these 6 vertices
        a, b, c, d, e, f = six
        matchings = [
            ((a,b),(c,d),(e,f)),
            ((a,b),(c,e),(d,f)),
            ((a,b),(c,f),(d,e)),
            ((a,c),(b,d),(e,f)),
            ((a,c),(b,e),(d,f)),
            ((a,c),(b,f),(d,e)),
            ((a,d),(b,c),(e,f)),
            ((a,d),(b,e),(c,f)),
            ((a,d),(b,f),(c,e)),
            ((a,e),(b,c),(d,f)),
            ((a,e),(b,d),(c,f)),
            ((a,e),(b,f),(c,d)),
            ((a,f),(b,c),(d,e)),
            ((a,f),(b,d),(c,e)),
            ((a,f),(b,e),(c,d)),
        ]
        for m in matchings:
            p1, p2, p3 = m
            # Generate the 8 triples
            triples = []
            for x in p1:
                for y in p2:
                    for z in p3:
                        triples.append((x, y, z))
            # Convert to sorted tuples and get indices
            idx_set = set()
            for t in triples:
                st = tuple(sorted(t))
                idx_set.add(_triple_idx[st])
            k222_list.append(frozenset(idx_set))
    
    # Deduplicate
    unique_k222s = list(set(k222_list))
    return unique_k222s

# Precompute for fast lookup
_K222_LIST = _precompute_k222s()
# Map from triple index to list of K222 indices that contain it
_triple_to_k222 = [[] for _ in range(N_TRIPLES)]
for i, k222 in enumerate(_K222_LIST):
    for t in k222:
        _triple_to_k222[t].append(i)

def _check_conflict_fast(a, b, c, partner_mask):
    """
    Check if adding triple (a,b,c) creates a K(2,2,2).
    Uses precomputed partner masks for speed.
    """
    # For K(2,2,2) to be formed, we need:
    # There exist a', b', c' such that all 8 combinations of {a,a'}x{b,b'}x{c,c'} are present.
    # The 8 triples are:
    # (a,b,c) - the one we're adding
    # (a,b,c'), (a,b',c), (a,b',c'), (a',b,c), (a',b,c'), (a',b',c), (a',b',c')
    
    # a' must be a vertex such that (a',b,c) is present -> a' in partner_mask[b][c]
    # b' must be a vertex such that (a,b',c) is present -> b' in partner_mask[a][c]
    # c' must be a vertex such that (a,b,c') is present -> c' in partner_mask[a][b]
    
    mask_a_candidates = partner_mask[b*V + c]
    mask_a_candidates &= ~(1 << a)
    mask_a_candidates &= ~(1 << b)
    mask_a_candidates &= ~(1 << c)
    
    mask_b_base = partner_mask[a*V + c]
    mask_b_base &= ~(1 << a)
    mask_b_base &= ~(1 << b)
    mask_b_base &= ~(1 << c)
    
    mask_c_base = partner_mask[a*V + b]
    mask_c_base &= ~(1 << a)
    mask_c_base &= ~(1 << b)
    mask_c_base &= ~(1 << c)
    
    while mask_a_candidates:
        lsb = mask_a_candidates & -mask_a_candidates
        a_p = lsb.bit_length() - 1
        mask_a_candidates ^= lsb
        
        # Check if (a',b,c) and (a',b',c) can both be satisfied
        # (a',b,c) is present by construction (a_p in partner_mask[b][c])
        # We need b' in partner_mask[a][c] AND b' in partner_mask[a_p][c]
        # So b' must be in the intersection
        mask_b_valid = mask_b_base & partner_mask[a_p*V + c]
        mask_b_valid &= ~(1 << a_p)
        
        if mask_b_valid == 0:
            continue
            
        # Similarly for c'
        mask_c_valid = mask_c_base & partner_mask[a_p*V + b]
        mask_c_valid &= ~(1 << a_p)
        
        if mask_c_valid == 0:
            continue
        
        temp_b = mask_b_valid
        while temp_b:
            lsb_b = temp_b & -temp_b
            b_p = lsb_b.bit_length() - 1
            temp_b ^= lsb_b
            
            # Now we have a', b'. We need c' such that:
            # (a,b,c'), (a',b,c'), (a,b',c'), (a',b',c') are all present
            # (a,b,c') -> c' in partner_mask[a][b] (already in mask_c_base)
            # (a',b,c') -> c' in partner_mask[a'][b] (already in mask_c_valid)
            # (a,b',c') -> c' in partner_mask[a][b']
            # (a',b',c') -> c' in partner_mask[a'][b']
            mask_c_final = mask_c_valid
            mask_c_final &= partner_mask[a*V + b_p]
            mask_c_final &= partner_mask[a_p*V + b_p]
            mask_c_final &= ~(1 << b_p)
            
            if mask_c_final:
                return True
                
    return False

def _add_triple(a, b, c, partner_mask, presence, idx):
    presence[idx] = True
    partner_mask[a*V + b] |= (1 << c)
    partner_mask[b*V + a] |= (1 << c)
    partner_mask[a*V + c] |= (1 << b)
    partner_mask[c*V + a] |= (1 << b)
    partner_mask[b*V + c] |= (1 << a)
    partner_mask[c*V + b] |= (1 << a)

def _remove_triple(a, b, c, partner_mask, presence, idx):
    presence[idx] = False
    partner_mask[a*V + b] &= ~(1 << c)
    partner_mask[b*V + a] &= ~(1 << c)
    partner_mask[a*V + c] &= ~(1 << b)
    partner_mask[c*V + a] &= ~(1 << b)
    partner_mask[b*V + c] &= ~(1 << a)
    partner_mask[c*V + b] &= ~(1 << a)

def _greedy_init(order, deadline, partner_mask):
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
    Local search: repeatedly try to add any missing triple.
    Then try single swaps (remove one, add one or more).
    """
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try to add any missing triple
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
        
        # 2. Single swap: remove one triple, try to add as many as possible
        present_indices = [i for i in range(N_TRIPLES) if presence[i]]
        if not present_indices:
            break
        
        # Try a limited number of random swaps
        max_swaps = 50
        for _ in range(max_swaps):
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
                # Restore
                partner_mask = old_partner_mask
                presence = old_presence
                count = old_count
                
    return count, presence, partner_mask

def _simulated_annealing(count, presence, partner_mask, deadline):
    """
    Simulated annealing: accept worse moves with probability exp(-delta/T).
    """
    current_count = count
    current_presence = presence
    current_partner_mask = partner_mask
    best_count = count
    best_presence = presence[:]
    best_partner_mask = partner_mask[:]
    
    T = 5.0
    cooling = 0.995
    
    while time.time() < deadline:
        if T < 0.01:
            break
            
        # Pick a random move: add or remove a random triple
        if random.random() < 0.5:
            # Try to add a random missing triple
            missing = [i for i in range(N_TRIPLES) if not current_presence[i]]
            if not missing:
                continue
            idx = random.choice(missing)
            a, b, c = ALL_TRIPLES[idx]
            
            if not _check_conflict_fast(a, b, c, current_partner_mask):
                current_count += 1
                _add_triple(a, b, c, current_partner_mask, current_presence, idx)
                if current_count > best_count:
                    best_count = current_count
                    best_presence = current_presence[:]
                    best_partner_mask = current_partner_mask[:]
            else:
                # Can't add, try removing a random present triple
                present = [i for i in range(N_TRIPLES) if current_presence[i]]
                if not present:
                    continue
                idx_out = random.choice(present)
                a_out, b_out, c_out = ALL_TRIPLES[idx_out]
                _remove_triple(a_out, b_out, c_out, current_partner_mask, current_presence, idx_out)
                current_count -= 1
        else:
            # Remove a random present triple
            present = [i for i in range(N_TRIPLES) if current_presence[i]]
            if not present:
                continue
            idx_out = random.choice(present)
            a_out, b_out, c_out = ALL_TRIPLES[idx_out]
            _remove_triple(a_out, b_out, c_out, current_partner_mask, current_presence, idx_out)
            current_count -= 1
        
        T *= cooling
        
        # After each move, try to greedily add any possible triples
        for idx in range(N_TRIPLES):
            if time.time() >= deadline:
                break
            if current_presence[idx]:
                continue
            a, b, c = ALL_TRIPLES[idx]
            if not _check_conflict_fast(a, b, c, current_partner_mask):
                current_count += 1
                _add_triple(a, b, c, current_partner_mask, current_presence, idx)
        
        if current_count > best_count:
            best_count = current_count
            best_presence = current_presence[:]
            best_partner_mask = current_partner_mask[:]
    
    return best_count, best_presence, best_partner_mask

def construct():
    start = time.time()
    deadline = start + 115  # Safety margin
    
    best_count = 0
    best_presence = [False] * N_TRIPLES
    best_partner_mask = [0] * (V * V)
    
    random.seed(42)
    
    # Run multiple attempts
    while time.time() < deadline:
        order = list(range(N_TRIPLES))
        random.shuffle(order)
        
        partner_mask = [0] * (V * V)
        
        count, presence = _greedy_init(order, deadline, partner_mask)
        count, presence, partner_mask = _local_search(count, presence, partner_mask, deadline)
        
        # If time remains, try simulated annealing
        if time.time() < deadline - 5:
            count_sa, presence_sa, partner_mask_sa = _simulated_annealing(
                count, presence, partner_mask, deadline - 5
            )
            if count_sa > count:
                count, presence, partner_mask = count_sa, presence_sa, partner_mask_sa
        
        if count > best_count:
            best_count = count
            best_presence = presence[:]
            best_partner_mask = partner_mask[:]
            
    result = []
    for i in range(N_TRIPLES):
        if best_presence[i]:
            result.append(list(ALL_TRIPLES[i]))
    return result
