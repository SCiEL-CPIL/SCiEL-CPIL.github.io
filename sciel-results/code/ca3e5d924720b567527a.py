import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin before 60s limit

    n = 16
    k = 4
    full_mask = (1 << n) - 1

    # Generate all 4-element subsets as masks
    all_masks = []
    for c in combinations(range(n), k):
        mask = 0
        for x in c:
            mask |= (1 << x)
        all_masks.append(mask)
    
    num_sets = len(all_masks)
    
    # Precompute popcount for all masks up to 2^16
    popcount_table = [0] * (1 << n)
    for i in range(1, 1 << n):
        popcount_table[i] = popcount_table[i >> 1] + (i & 1)
    
    # Precompute indices for all masks
    all_indices = [None] * (1 << n)
    for m in range(1 << n):
        inds = []
        temp = m
        idx = 0
        while temp:
            if temp & 1:
                inds.append(idx)
            temp >>= 1
            idx += 1
        all_indices[m] = inds
    
    def popcount(x):
        return popcount_table[x]
    
    # Precompute for each pair (i, j) the set of masks that form a sunflower with them
    # This is expensive: C(1820, 2) * avg_blocked_count
    # Instead, we'll compute on-the-fly but cache results
    
    # For local search: maintain a family as a set of masks
    # A move is: add one mask, remove one mask, or swap two masks
    
    def get_blocked_masks(x, y):
        """Returns frozenset of masks z that form a sunflower triple with x and y"""
        k_mask = x & y
        f_mask = (x | y) ^ k_mask
        avail_mask = full_mask ^ f_mask
        need = k - popcount(k_mask)
        
        if need == 0:
            return frozenset()
            
        avail_inds = all_indices[avail_mask]
        blocked = set()
        for combo in combinations(avail_inds, need):
            z_extra = 0
            for idx in combo:
                z_extra |= (1 << idx)
            z_mask = k_mask | z_extra
            if z_mask != x and z_mask != y:
                blocked.add(z_mask)
        return frozenset(blocked)
    
    # Cache for blocked masks
    blocked_cache = {}
    
    def get_blocked_cached(x, y):
        if x > y:
            x, y = y, x
        key = (x, y)
        if key not in blocked_cache:
            blocked_cache[key] = get_blocked_masks(x, y)
        return blocked_cache[key]
    
    def is_valid_family(family_set, changed_mask=None):
        """Check if a family contains no sunflower triple.
        If changed_mask is provided, only check pairs involving that mask."""
        if changed_mask is not None:
            # Only check pairs where one element is changed_mask
            for y in family_set:
                if y == changed_mask:
                    continue
                blocked = get_blocked_cached(changed_mask, y)
                if blocked & family_set:
                    return False
            return True
        else:
            # Full check
            family_list = list(family_set)
            for i in range(len(family_list)):
                for j in range(i + 1, len(family_list)):
                    x = family_list[i]
                    y = family_list[j]
                    blocked = get_blocked_cached(x, y)
                    if blocked & family_set:
                        return False
            return True
    
    def count_conflicts(family_set):
        """Count number of sunflower triples in the family (should be 0 for valid)"""
        family_list = list(family_set)
        count = 0
        for i in range(len(family_list)):
            for j in range(i + 1, len(family_list)):
                x = family_list[i]
                y = family_list[j]
                blocked = get_blocked_cached(x, y)
                for z in blocked:
                    if z in family_set:
                        count += 1
        return count
    
    # Simulated annealing / local search
    def local_search(initial_family, max_steps=5000):
        """Perform local search starting from initial_family"""
        current = set(initial_family)
        current_size = len(current)
        best = set(current)
        best_size = current_size
        
        temperature = 5.0
        cooling = 0.9995
        
        for step in range(max_steps):
            if time.time() > deadline - 2.0:
                break
                
            # Choose a random move
            move_type = random.random()
            changed_mask = None
            
            if move_type < 0.4:
                # Add a random mask
                candidates = [m for m in all_masks if m not in current]
                if not candidates:
                    continue
                new_mask = random.choice(candidates)
                changed_mask = new_mask
                new_family = current | {new_mask}
                
            elif move_type < 0.8:
                # Remove a random mask
                if not current:
                    continue
                remove_mask = random.choice(list(current))
                new_family = current - {remove_mask}
                # No need to check validity when removing - always valid
                current = new_family
                temperature *= cooling
                continue
                
            else:
                # Swap: remove one, add one
                if not current:
                    continue
                remove_mask = random.choice(list(current))
                candidates = [m for m in all_masks if m not in current]
                if not candidates:
                    continue
                add_mask = random.choice(candidates)
                changed_mask = add_mask
                new_family = (current - {remove_mask}) | {add_mask}
            
            # Evaluate new family
            new_size = len(new_family)
            
            # Check if valid - optimize by only checking pairs involving the changed element
            if is_valid_family(new_family, changed_mask):
                # Accept if valid
                current = new_family
                if new_size > best_size:
                    best = set(current)
                    best_size = new_size
                # Cool down
                temperature *= cooling
            else:
                # Invalid: reject (keep solution valid always)
                pass
        
        return best, best_size
    
    # Phase 1: Greedy construction to get initial family
    def greedy_construct(rng):
        order = list(all_masks)
        rng.shuffle(order)
        
        selected = set()
        forbidden = set()
        
        for x in order:
            if x in forbidden or x in selected:
                continue
            
            # Check if x conflicts with any pair in selected
            conflicts = False
            for y in selected:
                blocked = get_blocked_cached(x, y)
                if blocked & selected:
                    conflicts = True
                    break
            
            if not conflicts:
                selected.add(x)
                # Update forbidden
                for y in selected - {x}:
                    blocked = get_blocked_cached(x, y)
                    forbidden |= (blocked - selected)
        
        return selected
    
    # Phase 2: Local search from multiple starting points
    best_family = set()
    best_size = 0
    
    rng = random.Random(42)
    
    # Try several greedy constructions
    num_greedy = 20
    for i in range(num_greedy):
        if time.time() > deadline - 5.0:
            break
        
        initial = greedy_construct(rng)
        if len(initial) > best_size:
            best_size = len(initial)
            best_family = set(initial)
        
        # Local search from this starting point
        improved, improved_size = local_search(initial, max_steps=3000)
        if improved_size > best_size:
            best_size = improved_size
            best_family = improved
    
    # If time remains, do more local search from the best
    if time.time() < deadline - 3.0 and best_size < 250:
        improved, improved_size = local_search(best_family, max_steps=10000)
        if improved_size > best_size:
            best_size = improved_size
            best_family = improved
    
    # Convert back to sorted lists
    result = []
    for m in sorted(best_family):
        s = []
        temp = m
        idx = 0
        while temp:
            if temp & 1:
                s.append(idx + 1)
            temp >>= 1
            idx += 1
        result.append(s)
        
    return result

# EVOLVE-BLOCK-END
