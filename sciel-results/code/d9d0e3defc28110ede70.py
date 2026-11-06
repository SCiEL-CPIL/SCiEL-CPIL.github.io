import time
import random
from itertools import combinations

def _precompute_k222s(n):
    """
    Precompute all K(2,2,2) configurations and triple-to-K mapping.
    Returns:
    - all_triples: list of tuples
    - triple_to_k222s: list of lists (for each triple index, list of K indices it belongs to)
    - k222_list: list of frozensets of triple indices
    """
    all_triples = list(combinations(range(n), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    triple_to_k222s = [[] for _ in range(len(all_triples))]
    k222_list = []
    
    # Generate all sets of 6 points
    for points in combinations(range(n), 6):
        # Generate all perfect matchings of these 6 points
        # There are 15 perfect matchings for 6 points
        def gen_matchings(pts):
            if not pts:
                yield []
                return
            first = pts[0]
            rest = pts[1:]
            for i in range(len(rest)):
                second = rest[i]
                new_rest = rest[:i] + rest[i+1:]
                for m in gen_matchings(new_rest):
                    yield [(first, second)] + m
        
        for matching in gen_matchings(points):
            # matching is list of 3 tuples: (a,a'), (b,b'), (c,c')
            k_triples = []
            for x in [0, 1]:
                for y in [0, 1]:
                    for z in [0, 1]:
                        t = tuple(sorted((matching[0][x], matching[1][y], matching[2][z])))
                        k_triples.append(t)
            
            k_indices = set()
            for t in k_triples:
                k_indices.add(triple_to_idx[t])
            
            k222_idx = len(k222_list)
            k222_list.append(frozenset(k_indices))
            
            for idx in k_indices:
                triple_to_k222s[idx].append(k222_idx)
    
    return all_triples, triple_to_k222s, k222_list

def _get_conflicts(triple_idx, present_set, triple_to_k222s, k222_list):
    """
    Returns a list of K(2,2,2) indices that would be completed if triple_idx were added.
    """
    conflicts = []
    for k_idx in triple_to_k222s[triple_idx]:
        k_set = k222_list[k_idx]
        # Check if all other members are in present_set
        if all(other in present_set for other in k_set if other != triple_idx):
            conflicts.append(k_idx)
    return conflicts

def _greedy_init(n_triples, triple_to_k222s, k222_list):
    """
    Fast greedy initialization using a static random order.
    """
    order = list(range(n_triples))
    random.shuffle(order)
    present = set()
    for idx in order:
        if not _get_conflicts(idx, present, triple_to_k222s, k222_list):
            present.add(idx)
    return present

def construct():
    start_time = time.time()
    deadline = 120.0
    n = 16
    
    all_triples, triple_to_k222s, k222_list = _precompute_k222s(n)
    n_triples = len(all_triples)
    
    # Initial solution via greedy
    current_set = _greedy_init(n_triples, triple_to_k222s, k222_list)
    best_set = set(current_set)
    best_size = len(current_set)
    
    # Simulated Annealing Parameters
    # Start temperature high enough to accept bad moves, cool down
    # The "energy" is -size, so we want to maximize size.
    # Delta E = new_size - old_size. We accept if Delta E > 0 or exp(Delta E / T) > random
    T_start = 5.0
    T_end = 0.01
    cooling_rate = 0.995
    
    current_T = T_start
    
    # Precompute list of all triple indices for random selection
    all_indices = list(range(n_triples))
    
    while time.time() - start_time < deadline - 2.0:
        current_T *= cooling_rate
        
        # Generate a move:
        # 1. Pick a random triple to add (u)
        # 2. Pick a random triple to remove (v) from current_set
        # This is a swap move.
        
        if not current_set:
            # If empty, just add a random valid triple
            idx = random.choice(all_indices)
            if not _get_conflicts(idx, current_set, triple_to_k222s, k222_list):
                current_set.add(idx)
            continue
        
        u = random.choice(all_indices)
        v = random.choice(list(current_set))
        
        # Skip if u == v or u is already in set (we want a swap or pure add/remove)
        # Let's allow:
        # - Add u (if not in set)
        # - Remove v (if in set)
        # - Swap u and v
        
        # Simplest robust move: Try to add u. If conflict, try to repair by removing a conflicting triple.
        # This is the "error_repair" strategy.
        
        new_set = set(current_set)
        
        if u in new_set:
            # If u is already in set, just try removing v? Or skip?
            # Let's just try removing v to explore smaller sets (annealing allows this)
            # But removing usually decreases size.
            # Let's stick to: Add u. If u in set, do nothing (or remove v).
            # Let's try removing v.
            new_set.discard(v)
            delta_size = len(new_set) - len(current_set)
            # No conflicts introduced by removal.
            if delta_size > 0:
                current_set = new_set
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
            else:
                # Accept with probability
                if random.random() < min(1.0, current_T): # delta is -1, so exp(-1/T)
                    current_set = new_set
                # else revert (do nothing)
            continue
            
        # u not in set. Try to add u.
        conflicts = _get_conflicts(u, new_set, triple_to_k222s, k222_list)
        
        if not conflicts:
            new_set.add(u)
            delta_size = 1
            # Always accept improvement
            current_set = new_set
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
        else:
            # Conflict exists. 
            # Strategy: Try to repair by removing one of the other triples in the conflict.
            # Pick the first conflict.
            k_idx = conflicts[0]
            k_set = k222_list[k_idx]
            
            # Candidates to remove: any triple in k_set except u
            candidates = [t for t in k_set if t != u]
            
            repaired = False
            # Try random candidates
            random.shuffle(candidates)
            for rem_candidate in candidates:
                if rem_candidate in new_set:
                    trial_set = set(new_set)
                    trial_set.discard(rem_candidate)
                    trial_set.add(u)
                    
                    # Check if this repair resolves ALL conflicts for u?
                    # Or just check if trial_set is valid?
                    # Since we only added u and removed one other, we might have created new conflicts?
                    # No, removing triples never creates new conflicts.
                    # Adding u created conflicts. Removing a member of a conflict breaks it.
                    # But u might be in MULTIPLE conflicts.
                    # Let's check if u is valid in trial_set.
                    if not _get_conflicts(u, trial_set, triple_to_k222s, k222_list):
                        # Repair successful
                        new_set = trial_set
                        delta_size = len(new_set) - len(current_set)
                        
                        if delta_size > 0:
                            current_set = new_set
                            if len(current_set) > best_size:
                                best_size = len(current_set)
                                best_set = set(current_set)
                        else:
                            # Size decreased or stayed same (if we swapped 1 for 1, delta=0)
                            # Wait, we added u and removed rem_candidate.
                            # If u was not in set, and rem_candidate was in set.
                            # Size change: +1 - 1 = 0.
                            # So delta_size is 0.
                            # We accept with probability exp(0/T) = 1? 
                            # Actually, if delta=0, we should accept to explore.
                            if random.random() < min(1.0, current_T): # exp(0)=1, so always accept if T>0?
                                # Let's just accept to allow exploration
                                current_set = new_set
                        repaired = True
                        break
            
            if not repaired:
                # Could not repair. 
                # Option 1: Reject move (revert)
                # Option 2: Accept the "bad" move? No, we must maintain validity.
                # So we reject.
                pass

    # Final cleanup: ensure best_set is valid
    # (It should be, as we only update best_set from valid current_set states)
    
    return [list(all_triples[i]) for i in sorted(best_set)]

if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Found {len(res)} triples in {t1-t0:.2f}s")
