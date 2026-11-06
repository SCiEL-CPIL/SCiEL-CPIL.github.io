import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def optimize_set(V, initial_set, time_limit, rng):
        """
        Optimizes a sum-free set by trying to add elements.
        V: The full set of available integers.
        initial_set: A list of integers forming a sum-free subset.
        time_limit: Max seconds to run.
        rng: Random number generator.
        """
        S = set(initial_set)
        if not S:
            return []
        
        # Precompute sums and diffs for current S
        # S_sums = {a+b for a in S for b in S}
        # S_diffs = {a-b for a in S for b in S}
        
        def get_sums_diffs(S):
            sums = set()
            diffs = set()
            S_list = list(S)
            for a in S_list:
                for b in S_list:
                    sums.add(a + b)
                    diffs.add(a - b)
            return sums, diffs

        S_sums, S_diffs = get_sums_diffs(S)
        
        remaining = list(set(V) - S)
        
        # Shuffle remaining to break symmetries
        rng.shuffle(remaining)
        
        improved = True
        while improved and time.time() < start_time + time_limit:
            improved = False
            new_remaining = []
            
            # Try to add each remaining element
            for x in remaining:
                if time.time() > start_time + time_limit:
                    new_remaining.append(x)
                    continue
                
                # Check if x is safe to add
                # 1. x not in S_sums (a+b=x)
                # 2. x not in S_diffs (a-b=x => a=x+b => x+b in S? No, a-b=x means x is a difference. If x is a difference, then there exist a,b in S such that a-b=x. This implies a = b+x. So b+x in S. This is the condition for (b, x, a) or (x, b, a) conflict? 
                # Let's re-verify: 
                # Conflict: u+v=w.
                # If we add x, new conflicts involve x.
                # 1. x + a = c. (a,c in S). => c-a = x. So x is in S-S (differences).
                # 2. a + x = c. (a,c in S). => c-a = x. Same.
                # 3. x + x = c. (c in S). => 2x in S.
                # 4. a + b = x. (a,b in S). => x in S+S (sums).
                
                if x in S_sums:
                    new_remaining.append(x)
                    continue
                if x in S_diffs:
                    new_remaining.append(x)
                    continue
                if 2 * x in S:
                    new_remaining.append(x)
                    continue
                
                # Safe to add
                S.add(x)
                
                # Update S_sums and S_diffs
                # New sums: x+a for a in S (including x itself for 2x)
                # Note: S was modified, so we should iterate over the old S or handle carefully.
                # Let's store old S elements
                old_S = list(S)
                # Actually, we just added x. So old S is S without x.
                # Let's just recompute? No, N is small (240). Recomputing is O(N^2) = 240^2 = 57600 ops.
                # With 240 adds, that's 13M ops. Might be too slow in Python if done repeatedly.
                # Better to update incrementally.
                
                # Let's revert to incremental update.
                # We need to add sums/diffs involving x and the *old* S.
                # But S already contains x.
                # Let's get the list of elements in S BEFORE adding x.
                # This is inefficient if we don't store it.
                
                # Alternative: Just recompute S_sums and S_diffs every time an element is added.
                # N=240. N^2 = 57600.
                # If we add 100 elements, 100 * 57600 = 5.7M operations.
                # Python can do ~10M simple ops/sec. So ~0.5s per full recompute.
                # If we do this in a loop of hill climbing, it might be slow.
                
                # Let's try incremental update properly.
                # We need the set of elements in S before adding x.
                # Let's restructure:
                pass # Placeholder

            if not improved:
                # Try 1-2 swap? Or just break?
                # For now, break if no single add works.
                break
            remaining = new_remaining
            
            # Recompute sums/diffs if we added anything? 
            # The logic above was flawed because I didn't update S_sums/S_diffs.
            # Let's rewrite the inner loop to properly handle updates or just recompute.
            # Given N=240, recomputing is likely fast enough if we limit restarts.
            
        return list(S)

    # The above optimize_set was a draft. Let's write a cleaner version.
    
    def optimize_hill_climb(V, initial_set, time_limit, rng):
        S = set(initial_set)
        if not S:
            return []
        
        def check_safe(x, S, S_sums, S_diffs):
            if x in S_sums: return False
            if x in S_diffs: return False
            if 2*x in S: return False
            return True

        def update_sums_diffs(S, S_sums, S_diffs, x):
            # x is already in S
            # We need to add pairs involving x and other elements in S
            # But S_sums/S_diffs currently contain sums/diffs of S\{x}?
            # No, this is hard to manage state.
            # Given N<=240, O(N^2) recompute is ~60k ops.
            # 100 additions -> 6M ops. ~0.5s.
            # Let's just recompute. It's safer and simpler.
            pass

        remaining = list(set(V) - S)
        rng.shuffle(remaining)
        
        improved = True
        while improved and time.time() < start_time + time_limit:
            improved = False
            new_remaining = []
            
            # Compute sums and diffs for current S
            S_list = list(S)
            S_sums = set()
            S_diffs = set()
            for a in S_list:
                for b in S_list:
                    S_sums.add(a + b)
                    S_diffs.add(a - b)
            
            for x in remaining:
                if time.time() > start_time + time_limit:
                    new_remaining.append(x)
                    continue
                
                if check_safe(x, S, S_sums, S_diffs):
                    S.add(x)
                    improved = True
                else:
                    new_remaining.append(x)
            
            if improved:
                remaining = new_remaining
            else:
                break
        
        return list(S)

    def generate_initial_strategies(V, rng):
        strategies = []
        
        # 1. All Odds
        odds = [x for x in V if x % 2 != 0]
        strategies.append(odds)
        
        # 2. Large Magnitude Greedy
        # Sort by abs value descending
        V_sorted = sorted(V, key=lambda x: abs(x), reverse=True)
        S_mag = []
        S_mag_set = set()
        # We need to check safety as we add.
        # For speed in init, we can just pick top K and then clean up?
        # Or do the greedy add.
        # Greedy add is O(N^3) if naive, but N=240 is small.
        # Let's do a quick greedy add.
        for x in V_sorted:
            # Check if x is safe w.r.t S_mag
            safe = True
            # 1. x in sums of S_mag?
            # 2. x in diffs of S_mag?
            # 3. 2x in S_mag?
            # To speed up, we can just check against S_mag directly for now in init, 
            # or use the sum/diff sets.
            # Let's just use a simple check for init speed, or reuse the logic.
            # Simple check:
            for a in S_mag:
                if x + a in S_mag_set:
                    safe = False
                    break
                if a - x in S_mag_set: # x = a - b => a-x = b. If b in S, then a-x in S.
                    safe = False
                    break
            if safe and 2*x in S_mag_set:
                safe = False
            
            if safe:
                S_mag.append(x)
                S_mag_set.add(x)
        
        strategies.append(S_mag)
        
        # 3. Random Greedy
        V_rand = list(V)
        rng.shuffle(V_rand)
        S_rand = []
        S_rand_set = set()
        for x in V_rand:
            safe = True
            for a in S_rand:
                if x + a in S_rand_set:
                    safe = False
                    break
                if a - x in S_rand_set:
                    safe = False
                    break
            if safe and 2*x in S_rand_set:
                safe = False
            
            if safe:
                S_rand.append(x)
                S_rand_set.add(x)
        
        strategies.append(S_rand)
        
        return strategies

    results = []
    
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        V = instance(size, seed)
        
        best_set = []
        best_size = 0
        
        # Time allocation
        if size <= 60:
            budget = 10.0
        elif size <= 120:
            budget = 15.0
        else:
            budget = 25.0
        
        # Generate initial sets
        rng = random.Random(seed)
        initial_sets = generate_initial_strategies(V, rng)
        
        # Optimize each
        time_per_opt = budget / len(initial_sets)
        
        for init_set in initial_sets:
            if time.time() > start_time + DEADLINE - 5:
                break
            
            t_left = min(time_per_opt, DEADLINE - (time.time() - start_time))
            if t_left < 0.5:
                break
            
            # Use a specific rng for this optimization to ensure reproducibility
            opt_rng = random.Random(seed + len(init_set))
            res = optimize_hill_climb(V, init_set, t_left, opt_rng)
            
            if len(res) > best_size:
                best_size = len(res)
                best_set = res
        
        # If time remains, try more random restarts of the best performing init type?
        # Or just run more random greedy inits.
        elapsed = time.time() - start_time
        remaining_total = DEADLINE - elapsed - 3
        
        if remaining_total > 2.0:
            # Do a few more random restarts
            num_restarts = int(remaining_total / 2.0)
            for i in range(num_restarts):
                if time.time() > start_time + DEADLINE - 2:
                    break
                t_left = min(2.0, (DEADLINE - 2) - (time.time() - start_time))
                if t_left < 0.5:
                    break
                
                rng_restart = random.Random(seed + 1000 + i)
                # Generate a random greedy init
                V_rand = list(V)
                rng_restart.shuffle(V_rand)
                S_rand = []
                S_rand_set = set()
                for x in V_rand:
                    safe = True
                    for a in S_rand:
                        if x + a in S_rand_set:
                            safe = False
                            break
                        if a - x in S_rand_set:
                            safe = False
                            break
                    if safe and 2*x in S_rand_set:
                        safe = False
                    if safe:
                        S_rand.append(x)
                        S_rand_set.add(x)
                
                res = optimize_hill_climb(V, S_rand, t_left, rng_restart)
                if len(res) > best_size:
                    best_size = len(res)
                    best_set = res
        
        results.append(best_set)

    return results

construct()
