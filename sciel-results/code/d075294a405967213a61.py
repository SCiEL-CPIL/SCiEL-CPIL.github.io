import time
from fractions import Fraction
import random

def _get_primes(limit=1000):
    """Generate primes up to limit using simple sieve."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    primes = _get_primes(1000)
    L_target = 100000  # Cap for evaluation
    
    # We will use a greedy approach that builds the set of classes.
    # State:
    # - classes: list of [p, r]
    # - covered: bytearray of size L_target + 1, 1 if covered
    # - current_L: the largest L such that 1..L is fully covered
    # - budget_used: Fraction sum of 1/p
    
    # Pre-compute all possible classes? No, too many (168 primes * avg 500 residues ~ 84k classes).
    # But we can generate them on the fly or pre-compute the indices they cover.
    
    # To speed up evaluation of "gain in L", we need to quickly see if a new class
    # covers the first uncovered number and extends the prefix.
    
    # Let's maintain:
    # 1. covered: bytearray
    # 2. current_L: int
    # 3. first_gap: the first index > current_L that is NOT covered? 
    #    Actually, current_L is defined as the max L where 1..L are covered.
    #    So current_L + 1 is the first gap (or L_target+1 if full).
    
    # Strategy:
    # 1. Initialize empty.
    # 2. Loop:
    #    a. Identify the "bottleneck": the first number `n` in `current_L + 1` that is not covered.
    #       Wait, `current_L` is contiguous. So `current_L + 1` is the first uncovered number.
    #    b. Find all classes (p, r) such that (current_L + 1) % p == r.
    #       These are the only classes that can *immediately* extend the prefix.
    #    c. Among these candidates, pick the one that maximizes the new prefix length after addition.
    #       To do this efficiently, we can simulate the addition and check how far the prefix extends.
    #    d. If no class covers current_L + 1, we are stuck? 
    #       No, we might need to cover a gap *before* current_L + 1? 
    #       No, by definition of current_L, 1..current_L are covered. 
    #       So the only way to increase L is to cover current_L + 1.
    #       Therefore, we MUST pick a class that covers current_L + 1.
    #       If no available prime p has a residue r such that (current_L + 1) % p == r and (p, r) is not used,
    #       then we cannot extend the prefix with a single new class?
    #       Actually, we can use ANY unused prime p. The residue is determined: r = (current_L + 1) % p.
    #       So for every unused prime p, there is exactly one residue r that covers current_L + 1.
    #       So candidates are all unused primes p.
    #       We evaluate the gain for each candidate p (with r = (current_L+1)%p).
    #       Pick the best one.
    
    # This is a very specific greedy: "Cover the next number".
    # Is this optimal? 
    # Adding a class that covers current_L+1 might also cover many future numbers, extending L significantly.
    # Or it might only cover current_L+1 and then hit a gap at current_L+2.
    # So we must evaluate the *new* L after adding the class.
    
    # Efficiency:
    # Evaluating new L for a candidate (p, r) requires checking from current_L+1 onwards.
    # We can do this by iterating n = current_L+1, current_L+2, ... until we find a gap.
    # A number n is covered if covered[n] == 1 OR (n % p == r).
    # Since we are simulating, we don't update covered yet.
    
    # Let's implement this.
    
    covered = bytearray(L_target + 1)
    current_L = 0
    used_primes = set()
    classes = []
    budget_used = Fraction(0)
    
    best_L = 0
    best_classes = []
    
    # Pre-compute prime list for quick access
    # We need to iterate through unused primes.
    
    # To make "find best candidate" fast, we can limit the search space.
    # Small primes are more likely to cover many numbers (higher density).
    # But small primes cost more budget.
    
    # Let's try a randomized greedy with restarts if stuck or to find better paths.
    # But the "cover next number" logic is deterministic if we pick the best gain.
    # Tie-breaking: prefer smaller p (higher density) or larger p (cheaper)?
    # Let's evaluate the gain L_new - current_L.
    
    def evaluate_candidate(p, r):
        """
        Simulate adding class (p, r) to current state.
        Returns new_L.
        """
        # We know 1..current_L are covered.
        # We know p, r covers current_L + 1 (by construction of candidates).
        # Check current_L + 2, current_L + 3, ...
        new_L = current_L
        n = current_L + 1
        while n <= L_target:
            # Check if n is covered by existing classes
            if covered[n]:
                new_L = n
                n += 1
                continue
            # Check if n is covered by candidate (p, r)
            if n % p == r:
                new_L = n
                n += 1
                continue
            # Gap found
            break
        return new_L

    # Main loop
    # We can't just pick the best greedily forever because we might run out of budget
    # or get stuck where no prime can extend L (which shouldn't happen as long as we have primes,
    # because for any n, and any unused prime p, we can pick r = n % p to cover n).
    # Wait, the constraint is distinct primes.
    # If we use all primes, we stop.
    # Also budget constraint: sum(1/p) <= 1.
    
    # Since we have ~168 primes, and sum(1/p) for all primes <= 1000 is ~ 2.2 (roughly),
    # we can only use a subset.
    
    # Greedy choice:
    # At each step, consider all unused primes p such that budget_used + 1/p <= 1.
    # For each such p, let r = (current_L + 1) % p.
    # Calculate new_L = evaluate_candidate(p, r).
    # Pick p that maximizes new_L.
    # If max new_L == current_L, we are stuck? 
    # No, if new_L == current_L, it means current_L+1 is covered by (p,r) but current_L+2 is not covered by anything.
    # This is still an improvement in "potential", but L doesn't increase immediately.
    # Actually, if L doesn't increase, we shouldn't add it unless we expect it to help later?
    # But if L doesn't increase, the "next gap" is still current_L+2? No, if L doesn't increase, 
    # it means current_L+1 was NOT the first gap? 
    # Wait. current_L is the max prefix. So current_L+1 is the first gap.
    # If we add a class covering current_L+1, the new prefix MUST be at least current_L+1.
    # So new_L >= current_L + 1.
    # So L always increases if we add a valid candidate.
    
    # So the algorithm is:
    # While budget allows and unused primes exist:
    #   Find best p among unused primes fitting budget.
    #   Add it.
    #   Update state.
    
    # This is deterministic. To improve, we might need to backtrack or randomize tie-breaking.
    # Or, we might want to add a prime that *doesn't* cover current_L+1?
    # No, that wouldn't increase L immediately. It might cover a future gap, but we can't see past the current gap.
    # So "Cover the next number" is the only logical move to increase L.
    
    # However, different primes cover different *patterns* of future numbers.
    # A small prime p covers 1/p of numbers. A large prime covers 1/p.
    # Small primes have more "conflicts" (more residues used? no, distinct primes).
    # Actually, small primes are "denser".
    
    # Let's implement the deterministic greedy first, then add randomization if it gets stuck or to explore variations.
    
    # Optimization: evaluate_candidate can be slow if L increases by a lot.
    # But L increases by at least 1. Max L is 100k.
    # Total work is sum of (L_new - L_old) over all steps? No, we scan from current_L+1 each time.
    # In worst case, we scan 100k numbers for each of 168 primes. 16.8M operations. Fast enough in Python?
    # 16.8M modulo operations might be ~1-2 seconds.
    
    # Let's refine evaluate_candidate to be faster.
    # We can pre-compute nothing, but the loop is simple.
    
    # One issue: if we pick the "best" greedy, we might paint ourselves into a corner where
    # the remaining budget/primes can't cover the next gap?
    # No, because for ANY gap n, and ANY unused prime p, we can cover n.
    # The only limit is budget.
    # If we run out of budget, we stop.
    # If we run out of primes, we stop.
    
    # So the only variation is WHICH prime we pick when multiple extend L.
    # We pick the one that extends L the most.
    # If tied, we might pick randomly or by some heuristic.
    
    # Let's try multiple random seeds for tie-breaking or initial perturbations?
    # Actually, the greedy is quite rigid.
    # Let's try to improve by looking ahead 1 or 2 steps?
    # Or just run the greedy, and if we have time, do local search on the result.
    
    # Plan:
    # 1. Run Greedy "Cover Next" to get a baseline solution.
    # 2. Use remaining time for Local Search (swap primes, change residues) to improve L.
    
    # Step 1: Greedy Construction
    current_L = 0
    used_primes = set()
    classes = []
    budget_used = Fraction(0)
    covered = bytearray(L_target + 1)
    
    # Sort primes? Doesn't matter for iteration, but we check all.
    
    while time.time() < deadline - 10: # Reserve 10s for local search
        # Find candidates
        best_p = None
        best_r = None
        best_new_L = current_L
        
        # Iterate through all primes
        for p in primes:
            if p in used_primes:
                continue
            cost = Fraction(1, p)
            if budget_used + cost > 1:
                continue
            
            r = (current_L + 1) % p
            
            # Evaluate
            # Optimization: if p is very large, it covers sparse numbers.
            # If p is small, it covers dense.
            
            new_L = evaluate_candidate(p, r)
            
            if new_L > best_new_L:
                best_new_L = new_L
                best_p = p
                best_r = r
            elif new_L == best_new_L and best_p is not None:
                # Tie-breaking: prefer smaller p (denser) or larger p (cheaper)?
                # Smaller p usually leads to longer runs because it hits more often.
                # But larger p is cheaper.
                # Let's prefer smaller p for now, as density helps bridge gaps.
                if p < best_p:
                    best_p = p
                    best_r = r
                    best_new_L = new_L
        
        if best_p is None:
            break
        
        # Apply choice
        used_primes.add(best_p)
        classes.append([best_p, best_r])
        budget_used += Fraction(1, best_p)
        
        # Update covered array
        p = best_p
        r = best_r
        start = r if r >= 1 else r + p
        for n in range(start, L_target + 1, p):
            covered[n] = 1
            
        current_L = best_new_L
        
        if current_L >= L_target:
            break
    
    # Save best so far
    best_L = current_L
    best_classes = classes[:]
    
    # Step 2: Local Search
    # We have a valid set of classes.
    # We can try to swap a prime with another, or change a residue.
    # Changing residue: 
    #   If we change residue of p, we might lose coverage of some numbers, but gain others.
    #   This is hard to evaluate incrementally without full recalc.
    #   But we can just recalc L from scratch? 
    #   Recalc L from scratch is O(L_target * num_classes) if naive, or O(L_target) with bitset.
    #   With bytearray, recalc is:
    #   1. Zero out covered.
    #   2. Repaint all classes.
    #   3. Find prefix.
    #   This is ~ 100k * (sum 1/p) operations. Sum 1/p is ~ 2. So 200k ops. Very fast.
    #   So we can afford to recalc L for every move in local search.
    
    def recalc_L(classes_list):
        covered_local = bytearray(L_target + 1)
        for p, r in classes_list:
            start = r if r >= 1 else r + p
            for n in range(start, L_target + 1, p):
                covered_local[n] = 1
        L = 0
        for i in range(1, L_target + 1):
            if covered_local[i]:
                L = i
            else:
                break
        return L

    # Local Search Loop
    no_improve_count = 0
    while time.time() < deadline:
        # Random move
        move_type = random.choice(['swap_prime', 'change_residue', 'add_prime', 'remove_prime'])
        
        if move_type == 'change_residue' and classes:
            idx = random.randint(0, len(classes) - 1)
            p, old_r = classes[idx]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
            classes[idx] = [p, new_r]
            new_L = recalc_L(classes)
            if new_L > best_L:
                best_L = new_L
                best_classes = classes[:]
                no_improve_count = 0
            else:
                classes[idx] = [p, old_r]
                no_improve_count += 1
                
        elif move_type == 'swap_prime' and classes:
            idx = random.randint(0, len(classes) - 1)
            p_old, r_old = classes[idx]
            
            # Find a prime to swap in
            available = [p for p in primes if p not in used_primes]
            if not available:
                continue
            p_new = random.choice(available)
            
            # Check budget
            current_sum = sum(Fraction(1, p) for p, _ in classes)
            new_sum = current_sum - Fraction(1, p_old) + Fraction(1, p_new)
            if new_sum > 1:
                continue
            
            # Apply swap
            r_new = r_old % p_new
            classes[idx] = [p_new, r_new]
            # Update used_primes set
            used_primes.remove(p_old)
            used_primes.add(p_new)
            
            new_L = recalc_L(classes)
            if new_L > best_L:
                best_L = new_L
                best_classes = classes[:]
                no_improve_count = 0
            else:
                # Revert
                classes[idx] = [p_old, r_old]
                used_primes.remove(p_new)
                used_primes.add(p_old)
                no_improve_count += 1
                
        elif move_type == 'add_prime':
            current_sum = sum(Fraction(1, p) for p, _ in classes)
            if current_sum >= 1:
                continue
            available = [p for p in primes if p not in used_primes]
            if not available:
                continue
            p_add = random.choice(available)
            if current_sum + Fraction(1, p_add) > 1:
                continue
            
            # Pick residue to cover first gap?
            # Find first gap in best_classes
            # Actually, just pick random residue first?
            # Better: pick residue to cover current_L + 1 of the *current* state?
            # Let's just pick random for now, local search will fix it.
            r_add = random.randint(0, p_add - 1)
            
            classes.append([p_add, r_add])
            used_primes.add(p_add)
            
            new_L = recalc_L(classes)
            if new_L > best_L:
                best_L = new_L
                best_classes = classes[:]
                no_improve_count = 0
            else:
                classes.pop()
                used_primes.remove(p_add)
                no_improve_count += 1
                
        elif move_type == 'remove_prime' and len(classes) > 1:
            idx = random.randint(0, len(classes) - 1)
            p_old, r_old = classes[idx]
            
            # Check if removal is beneficial? Unlikely for L, but might free budget for swap.
            # Only remove if we can immediately add something better? 
            # For simplicity, just try removing and see if L drops. If it does, revert.
            # If L stays same or increases (unlikely), keep.
            
            classes.pop(idx)
            used_primes.remove(p_old)
            
            new_L = recalc_L(classes)
            if new_L >= best_L: # Allow equal to explore? No, strict improvement for L.
                # If L is equal, we saved budget. This is a "free" move.
                # Let's accept if L is equal, as it might enable future swaps.
                if new_L == best_L:
                    # Accept, but mark as no improvement in L score
                    best_classes = classes[:] 
                    # Don't update best_L
                    no_improve_count += 1 # Count as non-improving for L
                else:
                    best_L = new_L
                    best_classes = classes[:]
                    no_improve_count = 0
            else:
                # Revert
                classes.insert(idx, [p_old, r_old])
                used_primes.add(p_old)
                no_improve_count += 1
        
        # Restart if stuck too long?
        if no_improve_count > 500:
            # Randomize current state slightly?
            # Or just reset to best and try different random seed?
            # Let's just continue, random moves will eventually find something.
            pass

    return best_classes

# To ensure we don't return an empty list if something goes wrong, 
# we can wrap in a try-except in the actual construct if needed, 
# but the logic above is robust.
