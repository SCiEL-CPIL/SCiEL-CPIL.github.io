import time
import random
from itertools import combinations

def construct():
    t_start = time.time()
    deadline = t_start + 55.0  # Safety margin
    
    N = 192
    candidates = list(range(1, N + 1))
    
    # State variables
    selected = []
    selected_set = set()
    
    # pair_residues[a] is a set of residues r in [0, a-1] such that 
    # there exist b, c in selected \ {a} with (b+c) % a == r.
    # We use a list of sets or bitsets. Sets are fine for small a.
    pair_residues = {}
    
    # triple_sums: list of sums of all triples in selected
    triple_sums = []
    
    def add_element(x):
        """Add x to selected, updating all structures."""
        nonlocal selected, selected_set, pair_residues, triple_sums
        
        selected.append(x)
        selected_set.add(x)
        
        # 1. Update triple_sums
        # New triples are (x, b, c) where b, c in old selected
        old_selected = selected[:-1]
        for b, c in combinations(old_selected, 2):
            triple_sums.append(x + b + c)
        
        # 2. Update pair_residues
        # For each a in old_selected:
        #   We need to add pairs (x, z) and (z, x) effectively (x+z)
        #   to pair_residues[a]
        for a in old_selected:
            if a not in pair_residues:
                pair_residues[a] = set()
            
            # Pairs involving x and some z in old_selected \ {a}
            for z in old_selected:
                if z == a:
                    continue
                r = (x + z) % a
                pair_residues[a].add(r)
                
        # For x itself:
        # pair_residues[x] should contain residues of pairs from old_selected
        # Initialize pair_residues[x]
        if x not in pair_residues:
            pair_residues[x] = set()
            for b, c in combinations(old_selected, 2):
                r = (b + c) % x
                pair_residues[x].add(r)

    def remove_element(x):
        """Remove x from selected, updating all structures."""
        nonlocal selected, selected_set, pair_residues, triple_sums
        
        if x not in selected_set:
            return
            
        selected.remove(x)
        selected_set.remove(x)
        
        # 1. Update triple_sums
        # Remove all triples containing x
        # This is O(k^2) to find them, or we can rebuild if k is small.
        # Rebuilding is O(k^3). k<=50, 125000 ops. Acceptable.
        # Let's rebuild for simplicity and safety.
        new_triple_sums = []
        for b, c, d in combinations(selected, 3):
            new_triple_sums.append(b + c + d)
        triple_sums = new_triple_sums
        
        # 2. Update pair_residues
        # This is tricky to do incrementally for removals.
        # Rebuild pair_residues for all a in selected.
        for a in selected:
            new_res = set()
            for b, c in combinations(selected, 2):
                if b == a or c == a:
                    continue
                r = (b + c) % a
                new_res.add(r)
            pair_residues[a] = new_res
            
        # Clean up pair_residues for x if it exists (it shouldn't be used anymore)
        if x in pair_residues:
            del pair_residues[x]

    def can_add(x):
        """Check if x can be added to current selected set."""
        if x in selected_set:
            return False
        
        # 1. Check if x divides any existing triple sum
        for s in triple_sums:
            if s % x == 0:
                return False
        
        # 2. Check if any a in selected divides x + b + c for b,c in selected \ {a}
        for a in selected:
            # We need to know if there exists b, c in selected \ {a} 
            # such that (x + b + c) % a == 0
            # i.e., (b + c) % a == (-x) % a
            target = (-x) % a
            
            # Get current residues for a
            # Note: pair_residues[a] contains residues of pairs from selected \ {a}
            if target in pair_residues.get(a, set()):
                return False
                
        return True

    # --- Greedy Construction ---
    # Order by "safety": large numbers are safer.
    # Also, numbers with fewer divisors/multiples?
    # Let's just try descending, then ascending, then random.
    
    best_set = []
    
    def try_greedy(order):
        nonlocal selected, selected_set, pair_residues, triple_sums
        # Reset state
        selected = []
        selected_set = set()
        pair_residues = {}
        triple_sums = []
        
        for x in order:
            if time.time() > deadline:
                break
            if can_add(x):
                add_element(x)
        
        return selected[:]

    # 1. Descending
    best_set = try_greedy(range(N, 0, -1))
    
    # 2. Ascending
    if time.time() < deadline:
        alt = try_greedy(range(1, N + 1))
        if len(alt) > len(best_set):
            best_set = alt

    # 3. Randomized Greedy
    if time.time() < deadline:
        random.seed(42)
        for _ in range(20):
            if time.time() > deadline:
                break
            order = list(range(1, N + 1))
            random.shuffle(order)
            # Bias towards larger numbers
            order.sort(key=lambda x: x * 0.5 + random.random() * 100, reverse=True)
            res = try_greedy(order)
            if len(res) > len(best_set):
                best_set = res

    # --- Local Search ---
    # Start from best_set
    selected = best_set[:]
    selected_set = set(selected)
    pair_residues = {}
    triple_sums = []
    for x in selected:
        add_element(x)
        
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add any unselected element
        unselected = [x for x in range(1, N + 1) if x not in selected_set]
        random.shuffle(unselected)
        
        for x in unselected:
            if time.time() > deadline:
                break
            if can_add(x):
                add_element(x)
                improved = True
                # Restart search from new state? Or just continue?
                # Continuing is fine, but we might miss other additions.
                # Let's just break and restart the outer loop to re-evaluate all.
                break
        
        if improved:
            continue
            
        # Phase 2: Try 1-for-2 swaps
        # Remove one y, try to add two u, v
        # This increases size by 1 if successful.
        
        # To make this efficient, we can try removing each y in selected
        # and then trying to add pairs.
        
        # Optimization: Only try if we have time.
        if time.time() > deadline - 5:
            break
            
        for y in selected[:]:
            if time.time() > deadline:
                break
                
            # Remove y
            remove_element(y)
            
            # Try to add two elements
            # Candidate pool: unselected elements
            # We want to find u, v such that can_add(u) and can_add(v) after u is added.
            
            # Heuristic: Try adding u, then v.
            # To limit search, try a few random pairs or sorted pairs.
            
            # Let's try adding one element u first.
            unselected = [x for x in range(1, N + 1) if x not in selected_set]
            random.shuffle(unselected)
            
            added_u = False
            for u in unselected:
                if time.time() > deadline:
                    break
                if can_add(u):
                    # Temporarily add u
                    add_element(u)
                    added_u = True
                    
                    # Now try to add v
                    unselected2 = [x for x in range(1, N + 1) if x not in selected_set]
                    random.shuffle(unselected2)
                    
                    added_v = False
                    for v in unselected2:
                        if time.time() > deadline:
                            break
                        if can_add(v):
                            add_element(v)
                            added_v = True
                            improved = True
                            break
                            
                    if not added_v:
                        # Revert u
                        remove_element(u)
                    
                    if improved:
                        break
                        
            if not improved:
                # Revert y
                add_element(y)
                
            if improved:
                break

    # Final verification
    def is_valid_set(S):
        S_list = list(S)
        for i, a in enumerate(S_list):
            others = [S_list[j] for j in range(len(S_list)) if j != i]
            for t in combinations(others, 3):
                if sum(t) % a == 0:
                    return False
        return True

    if not is_valid_set(selected):
        # Fallback to greedy result if local search corrupted state
        selected = best_set
        if not is_valid_set(selected):
            # Extreme fallback
            selected = try_greedy(range(N, 0, -1))

    return selected
