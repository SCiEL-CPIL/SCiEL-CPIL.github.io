# EVOLVE-BLOCK-START
import time
import random
from math import gcd

_DEADLINE_SECONDS = 85  # safety margin before 90s kill
_N = 128

# Precompute GCD table
_gcd_table = [[0] * (_N + 1) for _ in range(_N + 1)]
for _i in range(1, _N + 1):
    for _j in range(_i, _N + 1):
        _g = gcd(_i, _j)
        _gcd_table[_i][_j] = _g
        _gcd_table[_j][_i] = _g

# Precompute Bad masks
_bad_masks = [[0] * (_N + 1) for _ in range(_N + 1)]
for a in range(1, _N + 1):
    for b in range(a + 1, _N + 1):
        g_ab = _gcd_table[a][b]
        mask = 0
        for x in range(1, _N + 1):
            if x == a or x == b:
                continue
            g_ax = _gcd_table[a][x]
            g_bx = _gcd_table[b][x]
            if g_ab == g_ax == g_bx:
                mask |= (1 << x)
        _bad_masks[a][b] = mask
        _bad_masks[b][a] = mask

def _popcount(n):
    return bin(n).count('1')

def _smart_greedy_pass(all_nums, table, bad_masks, rng=None):
    """
    Greedy pass that at each step picks the candidate that blocks the fewest
    other remaining candidates.
    """
    selected = []
    selected_mask = 0
    conflict_mask = 0
    
    remaining = set(all_nums)
    
    while remaining:
        best_x = None
        best_cost = float('inf')
        best_new_conflicts = 0
        
        # Find candidates not in selected and not in conflict
        # Note: conflict_mask contains elements that CANNOT be added because they form a bad triple with some pair in selected.
        # So valid candidates are those in remaining and NOT in conflict_mask.
        
        for x in remaining:
            if (conflict_mask >> x) & 1:
                continue
            
            # Calculate new conflicts if we add x
            # new_conflicts is the union of bad_masks[a][x] for all a in selected
            new_conf = 0
            for a in selected:
                new_conf |= bad_masks[a][x]
            
            # The cost is the number of *other* remaining candidates that would be blocked.
            # These are the elements in new_conf that are in remaining and not x.
            # We also need to consider that some elements in new_conf might already be in conflict_mask?
            # No, if they were in conflict_mask, they were already blocked. Adding x might block *more* elements.
            # The "damage" is the number of *currently available* elements that become blocked.
            # Currently available = remaining - {x} - (conflict_mask & remaining)
            # Actually, simply: count bits in (new_conf & remaining) excluding x.
            
            # Optimization: we can precompute or just calculate.
            # N is 128, selected size is small initially.
            
            # We want to minimize the number of future options lost.
            # candidates_blocked = new_conf & remaining
            # cost = _popcount(candidates_blocked & ~(1 << x))
            
            candidates_blocked = new_conf & remaining
            # x is in remaining, so it is in candidates_blocked.
            cost = _popcount(candidates_blocked) - 1
            
            if cost < best_cost:
                best_cost = cost
                best_x = x
                best_new_conflicts = new_conf
                
        if best_x is None:
            break
            
        # Add best_x
        selected.append(best_x)
        selected_mask |= (1 << best_x)
        remaining.remove(best_x)
        
        # Update conflict mask
        # The new conflicts are best_new_conflicts.
        # But we must also keep the old conflicts.
        # Wait, the conflict_mask represents "elements that cannot be added".
        # If we add x, any element y in bad_masks[a][x] for any a in selected becomes forbidden.
        # So conflict_mask |= best_new_conflicts.
        conflict_mask |= best_new_conflicts
        
    return selected

def _greedy_pass(order, table, bad_masks):
    """
    Original simple greedy pass for diversity.
    """
    selected = []
    conflict_mask = 0
    
    for x in order:
        if (conflict_mask >> x) & 1:
            continue
        
        new_conflicts = 0
        for a in selected:
            new_conflicts |= bad_masks[a][x]
        
        conflict_mask |= new_conflicts
        selected.append(x)
        
    return selected

def _local_improve_add(selected, candidates, table, bad_masks, deadline):
    improved = list(selected)
    in_set = set(improved)
    
    conflict_mask = 0
    for i in range(len(improved)):
        a = improved[i]
        for j in range(i + 1, len(improved)):
            b = improved[j]
            conflict_mask |= bad_masks[a][b]
            
    to_try = [c for c in candidates if c not in in_set]
    random.shuffle(to_try)
    
    for x in to_try:
        if time.time() > deadline:
            break
        if not ((conflict_mask >> x) & 1):
            new_conflicts = 0
            for a in improved:
                new_conflicts |= bad_masks[a][x]
            conflict_mask |= new_conflicts
            improved.append(x)
            
    return improved

def _local_improve_swap(selected, candidates, table, bad_masks, deadline, max_swaps=100):
    improved = list(selected)
    in_set = set(improved)
    excluded = [c for c in candidates if c not in in_set]
    
    def get_conflict_mask(lst):
        cm = 0
        for i in range(len(lst)):
            a = lst[i]
            for j in range(i + 1, len(lst)):
                b = lst[j]
                cm |= bad_masks[a][b]
        return cm

    current_mask = get_conflict_mask(improved)
    
    for _ in range(max_swaps):
        if time.time() > deadline:
            break
        if not excluded:
            break
        
        idx = random.randint(0, len(improved) - 1)
        removed = improved[idx]
        
        improved.pop(idx)
        
        new_mask = get_conflict_mask(improved)
        
        added_any = False
        candidates_to_add = [c for c in candidates if c not in improved]
        random.shuffle(candidates_to_add)
        
        for x in candidates_to_add:
            if time.time() > deadline:
                break
            if not ((new_mask >> x) & 1):
                for a in improved:
                    new_mask |= bad_masks[a][x]
                improved.append(x)
                added_any = True
        
        if not added_any or len(improved) < len(selected):
            improved.insert(idx, removed)
        else:
            current_mask = new_mask
            in_set = set(improved)
            excluded = [c for c in candidates if c not in in_set]
            
    return improved

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE_SECONDS
    
    all_nums = list(range(1, _N + 1))
    
    best_set = []
    best_len = 0
    
    # Strategy 1: Smart Greedy
    if time.time() < deadline:
        s = _smart_greedy_pass(all_nums, _gcd_table, _bad_masks)
        if len(s) > best_len:
            best_set, best_len = s, len(s)
            
    # Strategy 2: Simple Greedy (Sequential)
    if time.time() < deadline:
        s = _greedy_pass(all_nums, _gcd_table, _bad_masks)
        if len(s) > best_len:
            best_set, best_len = s, len(s)
            
    # Strategy 3: Simple Greedy (Reverse)
    if time.time() < deadline:
        s = _greedy_pass(list(reversed(all_nums)), _gcd_table, _bad_masks)
        if len(s) > best_len:
            best_set, best_len = s, len(s)

    # Strategy 4: Randomized Smart Greedy + Local Search
    # We will run a loop that performs local search on the best set, 
    # and also tries to find better sets via perturbation.
    
    rng = random.Random(42)
    
    # First, improve the initial best_set
    if time.time() < deadline:
        improved = _local_improve_add(best_set, all_nums, _gcd_table, _bad_masks, deadline)
        if len(improved) > best_len:
            best_set, best_len = improved, len(improved)
        if time.time() < deadline:
            improved2 = _local_improve_swap(best_set, all_nums, _gcd_table, _bad_masks, deadline, max_swaps=50)
            if len(improved2) > best_len:
                best_set, best_len = improved2, len(improved2)
                
    # Iterated Local Search / Perturbation
    # We keep the best_set. We create perturbed versions of it and try to improve them.
    # If a perturbed version improves beyond best_len, we update best_set.
    
    while time.time() < deadline - 5: # Reserve 5s for final polish
        # Create a perturbation of best_set
        # Remove k random elements from best_set
        k = rng.randint(1, 3)
        perturbed = list(best_set)
        for _ in range(min(k, len(perturbed))):
            idx = rng.randint(0, len(perturbed) - 1)
            perturbed.pop(idx)
            
        # Try to improve this perturbed set
        # First, try to add elements
        improved = _local_improve_add(perturbed, all_nums, _gcd_table, _bad_masks, deadline)
        
        # Then try swaps
        improved = _local_improve_swap(improved, all_nums, _gcd_table, _bad_masks, deadline, max_swaps=50)
        
        if len(improved) > best_len:
            best_set, best_len = improved, len(improved)
            
    # Final polish on the very best set
    if time.time() < deadline:
        improved = _local_improve_add(best_set, all_nums, _gcd_table, _bad_masks, deadline)
        if len(improved) > best_len:
            best_set, best_len = improved, len(improved)
        if time.time() < deadline:
            improved2 = _local_improve_swap(best_set, all_nums, _gcd_table, _bad_masks, deadline, max_swaps=100)
            if len(improved2) > best_len:
                best_set, best_len = improved2, len(improved2)
                
    if not best_set:
        best_set = [1]
        
    return best_set
# EVOLVE-BLOCK-END
