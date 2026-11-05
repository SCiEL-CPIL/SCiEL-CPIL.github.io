"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        """Verify S is sum-free."""
        S_set = set(S)
        for a in S:
            for b in S:
                if (a + b) in S_set:
                    return False
        return True

    def count_violations(S_set):
        """Count the number of triples (a,b,c) in S such that a+b=c.
        We count each distinct triple once to avoid double counting symmetric pairs if needed, 
        but for optimization, counting pairs (a,b) where a+b in S is sufficient and faster.
        """
        count = 0
        # To avoid O(n^2) overhead in the inner loop of WalkSAT, we might just check specific moves.
        # But for the objective function, we need a score.
        # A faster proxy: For each x in S, count how many pairs (a,b) in S satisfy a+b=x.
        # This is still O(n^2). 
        # For n<=240, O(n^2) is ~50k ops, which is fast enough for ~1000s of iterations in Python if optimized.
        
        # Optimization: Precompute sums? No, S changes.
        # We will use a "delta" evaluation for WalkSAT steps to avoid full recount.
        pass

    def get_violation_pairs(S_set):
        """Return a set of pairs (a, b) such that a+b is in S_set. 
        This helps identify which elements are 'bad'.
        """
        pairs = []
        S_list = list(S_set)
        for i, a in enumerate(S_list):
            for j in range(i, len(S_list)):
                b = S_list[j]
                if a + b in S_set:
                    pairs.append((a, b))
        return pairs

    def walk_sat_search(elements, seed_offset=0, time_budget=30):
        """
        WalkSAT-style search for maximum sum-free subset.
        State: A set S of elements.
        Objective: Maximize |S| subject to SumFree(S).
        Heuristic: 
          1. Start with a random valid subset (e.g., odds).
          2. Try to add elements that don't create violations.
          3. If stuck, "walk": randomly remove an element involved in a violation, 
             then greedily add back as many elements as possible.
          4. Alternatively, use a cost function: Cost = -|S| + lambda * Violations.
             Minimize Cost.
        """
        rng = random.Random(seed_offset)
        t_start = time.time()
        
        # Initial solution: Odds are always sum-free.
        # If there are few odds, use positives or negatives.
        odds = [x for x in elements if x % 2 != 0]
        if len(odds) < len(elements) // 3:
            # Try other structural starts
            pos = [x for x in elements if x > 0]
            neg = [x for x in elements if x < 0]
            # Pick the larger one or a mix
            start_set = set(odds) if len(odds) > max(len(pos), len(neg)) else (set(pos) if len(pos) > len(neg) else set(neg))
        else:
            start_set = set(odds)
        
        # Ensure start_set is sum-free (odds are, but pos/neg might not be if 0 is involved? 0 is excluded. 
        # Positives: a,b>0 => a+b>0. If a+b in Pos, violation. So Pos is NOT necessarily sum-free.
        # We need to repair the start set.
        
        def repair(S_set):
            """Remove elements to make S_set sum-free."""
            changed = True
            while changed:
                changed = False
                S_list = list(S_set)
                # Find a violation
                viol_elem = None
                for i, a in enumerate(S_list):
                    for b in S_list[i:]:
                        if a + b in S_set:
                            # Remove the one with highest "degree" or just random?
                            # Remove 'c' = a+b often helps as it's the result.
                            # Or remove a or b.
                            # Let's remove the element that is part of the most violations.
                            # For speed, just remove one and restart loop.
                            # Strategy: Remove the largest magnitude element from the triple to preserve small structure?
                            # Or just random.
                            candidates = [a, b, a+b]
                            # Prefer removing a+b if it's in S
                            if a+b in S_set:
                                viol_elem = a+b
                            else:
                                viol_elem = rng.choice([a, b])
                            S_set.discard(viol_elem)
                            changed = True
                            break
                    if changed:
                        break
            return S_set

        # Initial repair
        best_set = repair(set(start_set))
        best_size = len(best_set)
        
        # Greedy expansion
        def greedy_expand(S_set, candidates):
            # Try to add elements from candidates in random order
            cand_list = list(candidates)
            rng.shuffle(cand_list)
            for x in cand_list:
                if x in S_set:
                    continue
                # Check if x can be added
                valid = True
                for a in S_set:
                    if a + x in S_set or x - a in S_set:
                        valid = False
                        break
                    if a == x and 2*x in S_set:
                        valid = False
                        break
                if valid:
                    # Also check 2x
                    if 2*x in S_set:
                        valid = False
                if valid:
                    S_set.add(x)
            return S_set

        best_set = greedy_expand(best_set, set(elements) - best_set)
        best_size = len(best_set)
        
        # WalkSAT Loop
        no_improve_count = 0
        restart_count = 0
        max_restarts = 50
        
        while time.time() - t_start < time_budget:
            if no_improve_count > 100:
                # Restart with a new random subset
                restart_count += 1
                if restart_count > max_restarts:
                    break
                # New random start
                k = int(len(elements) * rng.uniform(0.3, 0.7))
                new_start = rng.sample(elements, k)
                new_set = repair(set(new_start))
                new_set = greedy_expand(new_set, set(elements) - new_set)
                if len(new_set) > best_size:
                    best_set = new_set
                    best_size = len(new_set)
                no_improve_count = 0
                continue
            
            # Current set
            S_set = set(best_set)
            V_set = set(elements)
            
            # Try to find an element to add
            added = False
            candidates = [x for x in V_set if x not in S_set]
            rng.shuffle(candidates)
            
            # Limit candidates to check for speed
            for x in candidates[:50]: 
                # Check if x can be added to S_set
                # Violation if: 
                # 1. a + x in S for some a in S
                # 2. x - a in S for some a in S  (i.e., x = a + b)
                # 3. 2x in S
                can_add = True
                for a in S_set:
                    if (a + x) in S_set:
                        can_add = False
                        break
                    if (x - a) in S_set:
                        can_add = False
                        break
                if can_add and (2 * x) in S_set:
                    can_add = False
                
                if can_add:
                    S_set.add(x)
                    best_size = len(S_set)
                    best_set = S_set
                    added = True
                    no_improve_count = 0
                    break
            
            if added:
                continue
                
            # If no simple add works, try a "swap" or "walk" move
            # Walk: Remove an element that is in a violation, then try to add more.
            # Identify violations in current S_set
            S_list = list(S_set)
            violations = []
            for i, a in enumerate(S_list):
                for j in range(i, len(S_list)):
                    b = S_list[j]
                    c = a + b
                    if c in S_set:
                        violations.append((a, b, c))
            
            if not violations:
                # No violations, but couldn't add? Then we are locally optimal for this set.
                no_improve_count += 1
                continue
            
            # Pick a random violation and remove one of its elements
            a, b, c = rng.choice(violations)
            # Remove the element with highest "cost" or random
            # Let's remove c (the sum) as it's often "heavy"
            # Or remove random from (a,b,c)
            to_remove = rng.choice([a, b, c])
            S_set.discard(to_remove)
            
            # Now try to greedily add as many as possible from remaining candidates
            # This is the "Walk" step
            cand_list = [x for x in V_set if x not in S_set]
            rng.shuffle(cand_list)
            
            # We want to see if this removal allows us to add MORE than we removed (which is 1)
            # So if we can add >= 1, we might improve or maintain size.
            # We actually want to improve size.
            # Let's try to add back to_max_size
            new_S = set(S_set)
            for x in cand_list:
                if len(new_S) >= best_size: # Don't go below best size too much? 
                    # Actually, we can go below if we expect to add more later.
                    # But for simplicity, just greedy add.
                    pass
                
                # Check if x can be added to new_S
                can_add = True
                for a in new_S:
                    if (a + x) in new_S:
                        can_add = False
                        break
                    if (x - a) in new_S:
                        can_add = False
                        break
                if can_add and (2 * x) in new_S:
                    can_add = False
                
                if can_add:
                    new_S.add(x)
                    # If we've already exceeded best size, we can stop checking others? 
                    # No, we want max size. Keep adding.
            
            if len(new_S) > best_size:
                best_set = new_S
                best_size = len(new_S)
                no_improve_count = 0
            else:
                # If no improvement, we might still keep the new state if it has same size but different structure?
                # WalkSAT usually accepts neutral moves.
                if len(new_S) == best_size:
                    # Accept with some probability to escape local minima
                    if rng.random() < 0.5:
                        best_set = new_S
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1

        return list(best_set)

    # Generate instances
    instances = []
    for size, seed in [(60, 801), (120, 802), (240, 803)]:
        instances.append(instance(size, seed))

    # Allocate time budget
    # n=60: ~5s, n=120: ~15s, n=240: ~35s
    # Total ~55s, leaving plenty of margin.
    time_budgets = [10, 20, 40]

    results = []
    for i, elems in enumerate(instances):
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed - 5  # 5s safety
        if remaining < 3:
            # Fallback: odd numbers
            results.append([x for x in elems if x % 2 != 0])
            continue
        
        budget = min(time_budgets[i], max(3, remaining))
        result = walk_sat_search(elems, seed_offset=801 + i, time_budget=budget)
        
        # Verify and fallback if invalid
        if not is_sum_free(result):
             # Repair
             S_set = set(result)
             changed = True
             while changed:
                 changed = False
                 S_list = list(S_set)
                 for i, a in enumerate(S_list):
                     for j in range(i, len(S_list)):
                         b = S_list[j]
                         if a + b in S_set:
                             S_set.discard(a+b)
                             changed = True
                             break
                     if changed: break
             result = list(S_set)
             
        results.append(result)

    return results
# EVOLVE-BLOCK-END
