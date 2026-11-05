import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit (or 150s if that's the real limit, but 120 is stated in contract)
    # Actually contract says "Each evaluation must finish within 120 s".
    # Directive says "evaluator KILLS the program after 150s wall-clock".
    # Let's use 110s to be safe.
    
    n = 16
    points = list(range(n))
    
    # Precompute all triples
    all_triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                all_triples.append((i, j, k))
                
    num_triples = len(all_triples)
    
    # Precompute masks for triples
    triple_masks = [0] * num_triples
    for i, t in enumerate(all_triples):
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks[i] = mask
        
    # Precompute conflicts
    # conflict[i] is a set of indices j such that triples i and j share >= 2 points
    # Two triples (a,b,c) and (x,y,z) conflict if they share at least 2 points.
    # Since they are distinct triples, they can share exactly 2 points.
    # We can compute this by checking pairs of points.
    
    # For each pair of points (u, v), which triples contain them?
    # If two triples contain the same pair (u, v), they conflict.
    
    # Let's build a map: pair -> list of triple indices
    pair_to_triples = {}
    for i, t in enumerate(all_triples):
        pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
        for p in pairs:
            if p not in pair_to_triples:
                pair_to_triples[p] = []
            pair_to_triples[p].append(i)
            
    conflicts = [set() for _ in range(num_triples)]
    for p, idxs in pair_to_triples.items():
        # All triples in idxs conflict with each other
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                conflicts[idxs[i]].add(idxs[j])
                conflicts[idxs[j]].add(idxs[i])
                
    # Convert to lists for faster iteration if needed, but sets are good for membership
    # Actually, for "can I add triple X to system S?", we need to check if X conflicts with ANY triple in S.
    # If we maintain a set of triples in S, we can check intersection.
    # But S is small.
    
    # Helper: Check if a set of triples (indices) is linear
    # This is guaranteed by construction if we only add non-conflicting triples.
    
    # Helper: Calculate independence number (alpha)
    # Alpha is the size of the largest subset of points containing no triple.
    # We want to minimize alpha.
    
    # To check if alpha <= K, we need to verify that EVERY subset of points of size K+1 contains at least one triple.
    # This is equivalent to saying that the set of "bad" subsets (those containing a triple) covers all subsets of size K+1.
    # Or, we can just try to find an independent set of size K+1. If we find one, alpha >= K+1.
    # If we don't, alpha <= K.
    
    # Efficient check for alpha:
    # We can use a backtracking search to find the maximum independent set.
    # Given the small size (16 points), we can use bitmasks.
    
    def calc_alpha(system_indices):
        if not system_indices:
            return n
            
        sys_masks = [triple_masks[i] for i in system_indices]
        best = 0
        
        # Sort points by degree in the system
        degrees = [0] * n
        for idx in system_indices:
            t = all_triples[idx]
            degrees[t[0]] += 1
            degrees[t[1]] += 1
            degrees[t[2]] += 1
            
        ordered_points = sorted(range(n), key=lambda p: degrees[p])
        
        def is_independent(mask):
            for m in sys_masks:
                if (mask & m) == m:
                    return False
            return True
            
        def backtrack(pos, current_mask, current_count):
            nonlocal best
            if pos == n:
                if current_count > best:
                    best = current_count
                return
                
            if current_count + (n - pos) <= best:
                return
                
            p = ordered_points[pos]
            
            # Option 1: Include p
            new_mask = current_mask | (1 << p)
            if is_independent(new_mask):
                backtrack(pos + 1, new_mask, current_count + 1)
                
            # Option 2: Exclude p
            backtrack(pos + 1, current_mask, current_count)
            
        backtrack(0, 0, 0)
        return best

    # Optimization:
    # The calc_alpha function might be slow if called too often.
    # We can cache results? The system changes slightly.
    # Maybe we can do a local search that only evaluates alpha when we have a candidate improvement.
    
    # Strategy:
    # 1. Generate a random maximal linear system (greedy).
    # 2. Calculate its alpha.
    # 3. Try to improve by swapping triples.
    #    - Remove a triple.
    #    - Add a different triple that doesn't conflict.
    #    - Check if alpha decreases.
    
    # To make this faster, we can use a heuristic for alpha.
    # Or we can just run the exact calc_alpha, but limit the number of evaluations.
    
    # Let's optimize calc_alpha a bit.
    # Precompute for each mask of size 2, if it's part of a triple in the system?
    # No, the system changes.
    
    # Let's try to improve the search.
    # We can start with a dense system.
    # A "dense" system covers many pairs.
    # The maximum number of triples in a linear triple system on 16 points is bounded by Steiner Triple System?
    # STS(16) doesn't exist (16 is not 1 or 3 mod 6).
    # Max packing number D(2, 3, 16).
    # 16*15/2 = 120 pairs. Each triple covers 3 pairs.
    # 120/3 = 40. So max 40 triples.
    # If we have 40 triples, we cover all pairs.
    # Then any 4 points contain a pair, which is in a triple?
    # No, if a pair is in a triple, it doesn't mean the 4 points contain the triple.
    # But if we have a Steiner Triple System (if it existed), alpha would be small.
    
    # Let's just run the local search.
    
    def random_maximal_system():
        available = set(range(num_triples))
        system = []
        while available:
            # Pick random triple
            t_idx = random.choice(list(available))
            system.append(t_idx)
            # Remove conflicts
            to_remove = {t_idx} | conflicts[t_idx]
            available.difference_update(to_remove)
        return system
        
    best_system = None
    best_alpha = n + 1
    
    # Run multiple restarts
    # Each restart:
    # 1. Generate random maximal system.
    # 2. Local search.
    
    # To fit in time, we need to be efficient.
    # calc_alpha is the bottleneck.
    # Let's see how fast calc_alpha is.
    # Worst case 2^16 = 65536 states.
    # Python might do ~1M simple ops/sec.
    # So ~0.06s per alpha calc?
    # If we do 1000 evals, that's 60s.
    # We have 110s.
    # So we can do ~1500-2000 alpha calcs.
    
    # Let's optimize calc_alpha with bit-level tricks or C-like speed?
    # No, standard library only.
    # We can use `functools.lru_cache` on the backtrack function?
    # The state is (point, current_mask).
    # current_mask is up to 2^16.
    # This might be too large for cache.
    
    # Alternative: Use a different alpha estimator.
    # We only care if alpha <= 7.
    # We can check if there is an independent set of size 8.
    # If yes, alpha >= 8.
    # If no, alpha <= 7.
    # This is a boolean check, which might be faster if we prune early.
    
    def has_independent_set_of_size(system_indices, size):
        # Returns True if there is an independent set of size 'size'
        if not system_indices:
            return True
            
        sys_masks = [triple_masks[i] for i in system_indices]
        
        # Backtracking
        def backtrack(point, current_mask, current_count):
            if current_count == size:
                return True
            if point == n:
                return False
            if (n - point) < (size - current_count):
                return False
                
            # Try include
            new_mask = current_mask | (1 << point)
            # Check independence
            ok = True
            for m in sys_masks:
                if (new_mask & m) == m:
                    ok = False
                    break
            if ok:
                if backtrack(point + 1, new_mask, current_count + 1):
                    return True
                    
            # Try exclude
            if backtrack(point + 1, current_mask, current_count):
                return True
                
            return False
            
        return backtrack(0, 0, 0)
        
    # We can use this to guide the search.
    # We want to find a system where has_independent_set_of_size(..., 8) is False.
    # i.e., alpha <= 7.
    
    # Let's run a loop.
    
    # To make the search more effective, we can bias the random system generation
    # or the local search moves.
    
    # Local search move:
    # Pick a triple to remove.
    # Pick a triple to add (from those not conflicting with system \ {removed}).
    # Check if new system has alpha <= 7.
    
    # To speed up, we can maintain the "available" set?
    # Or just check conflicts on the fly.
    
    # Let's implement the main loop.
    
    # We will do a series of restarts.
    # In each restart, we do a local search for a fixed number of steps or time.
    
    # Let's try to improve the initial system by "adding" triples if possible?
    # No, it's maximal.
    # So we must swap.
    
    # One idea: Start with a known good structure?
    # e.g. Affine geometry AG(4,2)?
    # Points are vectors in F2^4. 16 points.
    # Lines are triples? No, lines in AG(4,2) have 2 points.
    # Flats?
    # Maybe we can construct a specific system and then optimize.
    
    # Let's stick to random restarts + local search.
    
    # Optimization:
    # If we find a system with alpha <= 7, we can try to improve to 6.
    # But first, let's just aim for 7.
    
    # To make the local search faster, we can use a "greedy" improvement:
    # Try to swap a triple with another one that "covers" more "dangerous" pairs?
    
    # Let's just run it.
    
    # To ensure we return something valid even if we run out of time,
    # we keep the best system found.
    
    # We need to be careful with the time.
    # calc_alpha (or has_independent_set_of_size) is the most expensive part.
    
    # Let's refine has_independent_set_of_size to be as fast as possible.
    # We can precompute the list of sys_masks outside the backtrack function?
    # Yes, we did that.
    
    # We can also sort sys_masks by "density" or something?
    # Maybe check masks that are "likely" to be subsets first?
    # No, they are all size 3.
    
    # Let's run the search.
    
    # We will run until deadline - 5s.
    
    # To improve the chance of finding a good system, we can vary the "density" of the initial system.
    # Greedy maximal system is usually fairly dense.
    
    # Let's add a "repair" step:
    # If the system is not maximal, add random non-conflicting triples.
    # (Though random_maximal_system should produce maximal systems).
    
    # Let's run.
    
    best_alpha_found = n + 1
    best_system_indices = []
    
    # We can also try to fix the system to be "more uniform".
    
    # Let's just do the loop.
    
    # To avoid infinite loops or getting stuck, we restart if no improvement for a while.
    
    # We'll use a simple hill climbing with random restarts.
    
    # Since we have ~100s, and each alpha check takes ~0.01-0.05s, we can do ~2000-5000 checks.
    # Let's aim for that.
    
    # We'll structure it as:
    # while time < deadline:
    #   if stuck or first time:
    #     system = random_maximal_system()
    #   else:
    #     system = local_search_step(system)
    #   alpha = check_alpha(system)
    #   if alpha < best_alpha:
    #     best_alpha = alpha
    #     best_system = system
    #     if best_alpha <= 7:
    #       # We can break or continue to try for 6
    #       # Let's continue to see if we can get 6
    #       pass
    
    # To make local_search_step faster, we can pre-compute the available triples for the current system?
    # That would be O(N^2) per step.
    # N=560. 560^2 = 313,600.
    # That's fast in C, but in Python?
    # Maybe 0.1s?
    # That's too slow if we do it every step.
    
    # So we should pick a random triple to remove, and then pick a random triple to add
    # and check if it conflicts.
    # This is O(1) expected if the system is sparse?
    # No, checking conflict is O(len(system)) ~ 40.
    # So O(40) per candidate.
    # We might need to try many candidates to find a non-conflicting one.
    # But the system is maximal, so almost all triples conflict?
    # No, maximal means no MORE can be added.
    # So all triples NOT in the system conflict with SOME triple in the system.
    # So if we remove one, many become available.
    
    # Let's implement the step.
    
    def local_search_step(system_indices):
        # Pick a random triple to remove
        if not system_indices:
            return system_indices
            
        remove_idx = random.randint(0, len(system_indices) - 1)
        t_remove = system_indices[remove_idx]
        
        # Find a triple to add
        # We want a triple that does not conflict with (system_indices \ {t_remove})
        
        # We can pick random triples and test.
        # To make it smarter, we can pick triples that share points with t_remove?
        # Or just random.
        
        for _ in range(20): # Try 20 times
            t_add = random.randint(0, num_triples - 1)
            if t_add == t_remove:
                continue
            if t_add in system_indices:
                continue
                
            # Check conflict
            conflict = False
            for t_other in system_indices:
                if t_other == t_remove:
                    continue
                if t_add in conflicts[t_other]:
                    conflict = True
                    break
            if not conflict:
                new_system = system_indices[:]
                new_system[remove_idx] = t_add
                return new_system
                
        return system_indices # No change
        
    # Main loop
    # We need to track the current system and the best system.
    
    current_system = random_maximal_system()
    current_alpha = calc_alpha(current_system) # Use exact calc for initial?
    # Or use has_independent_set_of_size?
    # Let's use calc_alpha to get the true value, but it's slower.
    # Maybe use has_independent_set_of_size(8) first.
    # If False, alpha <= 7. Then calc_alpha to get exact.
    # If True, alpha >= 8.
    
    # Let's just use calc_alpha for now, and optimize if needed.
    # Wait, calc_alpha is exponential.
    # Let's use has_independent_set_of_size for the loop, and only calc_alpha when we find a candidate with alpha <= 7?
    # No, we need to know if we improved.
    # If we go from 8 to 7, that's an improvement.
    # If we go from 9 to 8, that's an improvement.
    
    # Let's define a function that returns a "score" which is the alpha.
    # We can use a fast approximation?
    # Or just use the boolean check for 8.
    # Score = 8 if has_independent_set_of_size(8) else 7 (approx).
    # This is good enough to guide the search towards alpha <= 7.
    
    def fast_alpha_check(system_indices):
        # Returns 8 if alpha >= 8, 7 if alpha <= 7 (assumed)
        # Actually, if alpha is 6, it returns 7.
        # This is fine for optimization.
        if has_independent_set_of_size(system_indices, 8):
            return 8
        else:
            return 7
            
    # Let's use this in the loop.
    
    # We'll keep the best system based on this fast check.
    # At the end, we can do a more thorough check on the best system?
    # Or just return the best system found.
    
    # Let's run the loop.
    
    best_fast_alpha = n + 1
    best_system = []
    
    # We can also try to diversify the search.
    
    # Let's run.
    
    # To prevent getting stuck in a local minimum with alpha=8,
    # we can occasionally restart.
    
    # We'll use a counter for no-improvement steps.
    
    no_improve_steps = 0
    
    while time.time() < deadline - 2:
        # Generate or modify system
        if no_improve_steps > 50 or not best_system:
            # Restart
            current_system = random_maximal_system()
            no_improve_steps = 0
        else:
            current_system = local_search_step(best_system)
            
        # Evaluate
        fast_alpha = fast_alpha_check(current_system)
        
        if fast_alpha < best_fast_alpha:
            best_fast_alpha = fast_alpha
            best_system = current_system[:]
            no_improve_steps = 0
            
            # If we found alpha <= 7, we can try to find alpha <= 6
            # But for now, let's just keep going.
            # We can switch to a more precise check if we want.
        else:
            no_improve_steps += 1
            
    # Final verification
    # We have best_system.
    # We should probably verify its exact alpha?
    # If best_fast_alpha is 7, it might be 6 or 7.
    # If it's 8, it's at least 8.
    
    # The score is the actual alpha.
    # So we should return the system with the lowest actual alpha.
    
    # Since we only tracked fast_alpha, we might have a system with fast_alpha=7 (actual 6)
    # and another with fast_alpha=7 (actual 7).
    # We prefer the one with actual 6.
    
    # So let's calculate the exact alpha for the best_system found.
    # And maybe for a few other candidates?
    
    # We have ~2s left.
    # calc_alpha takes ~0.05s.
    # We can do ~40 calcs.
    
    # Let's keep a history of "good" systems (fast_alpha <= 7).
    # And evaluate them.
    
    # Actually, the current best_system is the one with the lowest fast_alpha.
    # If fast_alpha is 7, it's a candidate for alpha 6 or 7.
    # If fast_alpha is 8, it's alpha >= 8.
    
    # Let's just calculate the exact alpha of best_system.
    # If time permits, we can do a final local search on this best_system
    # using exact alpha evaluation.
    
    # But we are close to deadline.
    
    # Let's just return the best_system.
    # The evaluator will calculate the exact alpha.
    
    # Wait, if we have time, we can do a final polish.
    # Let's see how much time is left.
    
    time_left = deadline - time.time()
    
    if time_left > 5 and best_fast_alpha <= 7:
        # We have a system with alpha <= 7.
        # Let's try to improve it to alpha <= 6 using exact alpha checks.
        
        # We'll do a local search with exact alpha.
        # This will be slower, so fewer steps.
        
        current_alpha = calc_alpha(best_system)
        
        # If current_alpha is 7, we try to get 6.
        # If it's 6, we try to get 5.
        
        # Let's run a small loop.
        while time.time() < deadline - 1:
            new_system = local_search_step(best_system)
            if new_system == best_system:
                break # No change possible? Or just unlucky.
                
            new_alpha = calc_alpha(new_system)
            
            if new_alpha < current_alpha:
                best_system = new_system
                current_alpha = new_alpha
            else:
                # Accept with some probability?
                # Or just reject.
                pass
                
    # Convert best_system indices to triples
    result = [list(all_triples[i]) for i in best_system]
    
    return result

if __name__ == "__main__":
    t = construct()
    print(f"Constructed {len(t)} triples")
