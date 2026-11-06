"""Strategy: CSP-based construction with randomized backtracking.
Instead of local search on S, we target a set of differences D and try to find S 
that realizes them. This allows us to aim for high counts directly."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def _count_progression_diffs(S):
    """Count distinct d > 0 such that there exists a with a, a+d, a+2d in S."""
    S_set = set(S)
    diffs = set()
    for a in S:
        for b in S:
            d = b - a
            if d > 0:
                c = a + 2 * d
                if c in S_set:
                    diffs.add(d)
    return len(diffs)

def _try_construct_for_diffs(target_diffs, max_attempts=200):
    """
    Try to construct a set S of size 32 in 1..512 that realizes all diffs in target_diffs.
    Uses randomized backtracking.
    """
    n = 32
    max_val = 512
    
    # Sort diffs to prioritize larger ones? Or smaller? 
    # Smaller diffs are easier to satisfy (more room). Larger ones constrain positions more.
    # Let's try random order or descending. Descending might fail faster if impossible.
    # Actually, let's just shuffle to break symmetry.
    diffs_to_check = list(target_diffs)
    
    for attempt in range(max_attempts):
        random.shuffle(diffs_to_check)
        
        # We will build S incrementally.
        # Start with some random seeds? Or start empty?
        # Starting empty is hard. Let's start with a random valid AP for the first diff?
        # Or just pick random elements and check constraints.
        
        # Better: Randomized Local Search on S, but with a penalty for missing target diffs.
        # Or: Backtracking on positions.
        
        # Let's use a simpler heuristic: 
        # 1. Pick a random "core" set that is dense.
        # 2. Check if it covers the diffs.
        # 3. If not, mutate.
        
        # Actually, let's use a specific CSP solver approach:
        # Variables: The 32 elements of S.
        # Domain: 1..512.
        # Constraints: For each d in target_diffs, exists a such that {a, a+d, a+2d} subset S.
        
        # This is hard to backtrack directly on 32 vars.
        
        # Alternative: 
        # Start with a random set of size 32.
        # Calculate current diffs.
        # Identify missing diffs from target_diffs.
        # Pick a missing diff d.
        # Try to adjust S to include an AP with difference d.
        #   - Pick a random a in 1..512-2d.
        #   - If a, a+d, a+2d are not all in S:
        #     - Replace some elements in S with these to maintain size 32.
        #     - Specifically, remove elements that are "least useful" (part of fewest APs) 
        #       and add the new AP elements.
        
        S = set(random.sample(range(1, max_val + 1), n))
        current_diffs = _count_progression_diffs(S)
        current_diff_set = set()
        S_set = set(S)
        for a in S:
            for b in S:
                d = b - a
                if d > 0:
                    c = a + 2 * d
                    if c in S_set:
                        current_diff_set.add(d)
                        
        missing = [d for d in target_diffs if d not in current_diff_set]
        
        # If we already have all, return S
        if not missing:
            return list(S)
        
        # Try to fix missing diffs
        # We have limited time per attempt.
        # Simple hill climbing on "number of target diffs covered"
        
        for _ in range(100): # Inner loop steps
            if not missing:
                return list(S)
            
            # Pick a random missing diff
            d = random.choice(missing)
            
            # Try to insert an AP with diff d
            # Random start point a
            max_a = max_val - 2 * d
            if max_a < 1:
                continue
            a = random.randint(1, max_a)
            ap = [a, a + d, a + 2 * d]
            
            # Check how many of these are already in S
            in_s = [x for x in ap if x in S]
            not_in_s = [x for x in ap if x not in S]
            
            if not not_in_s:
                # Already covered? Should have been in current_diff_set
                # Maybe my current_diff_set calculation was slow/stale?
                # Let's just re-evaluate if needed, but for now assume logic holds.
                # If all in S, then d should be in current_diff_set. 
                # So this case implies d was NOT missing. Contradiction?
                # Ah, current_diff_set is calculated once. If we modified S, it's stale.
                # We need to update current_diff_set after every move.
                pass
            
            # Strategy: Remove k elements from S and add len(not_in_s) elements from ap.
            # To maintain size 32, if we add m new elements, we must remove m elements.
            m = len(not_in_s)
            if m == 0:
                continue
                
            # Which elements to remove?
            # Prefer elements that are NOT part of any other target AP?
            # Heuristic: Remove random elements, or elements with low "degree" in AP graph.
            # For speed, just remove random elements from S, avoiding those in 'in_s' if possible?
            # Actually, we want to KEEP in_s. So remove from S \ in_s.
            
            candidates_to_remove = [x for x in S if x not in in_s]
            if len(candidates_to_remove) < m:
                # Not enough to remove while keeping in_s
                # We can remove from in_s too, but then we lose progress on this AP?
                # If we remove an element from in_s, we break the AP we're trying to form.
                # So we must remove m elements from S \ in_s.
                continue
                
            to_remove = random.sample(candidates_to_remove, m)
            
            # Execute move
            S.difference_update(to_remove)
            S.update(not_in_s)
            
            # Recalculate diffs (expensive but necessary for correctness)
            new_diff_set = set()
            S_set = set(S)
            for a in S:
                for b in S:
                    d = b - a
                    if d > 0:
                        c = a + 2 * d
                        if c in S_set:
                            new_diff_set.add(d)
            
            current_diff_set = new_diff_set
            missing = [d for d in target_diffs if d not in current_diff_set]
            
            # Note: This inner loop is O(N^2) per step. 100 steps * 1024 ops * 200 attempts * many targets = too slow?
            # 200 attempts * 100 steps * 1024 * 32 (target size) ... 
            # 1024 checks per score. 
            # 100 steps * 1024 = 100k ops per attempt.
            # 200 attempts = 20M ops. 
            # Python does ~10M simple ops/sec. So 2 seconds per target set?
            # If we try 50 target sets, that's 100s. Too slow.
            
            # Optimization: Only recalculate if the move likely helped.
            # Or reduce inner steps.
            
            # Let's abort this inner loop if it's taking too long or not improving.
            # Actually, let's simplify: 
            # Just do a few random restarts of the "insert AP" strategy.
            
            # Break out of inner loop if no progress in last 20 steps?
            # For now, keep it simple and rely on outer timeout.
            
            # To speed up, let's limit inner steps to 20.
            if _ > 19:
                break
                
        # End inner loop
        if not missing:
            return list(S)

    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    best_set = None
    best_score = -1
    
    # We will try to construct sets targeting increasing numbers of diffs.
    # What is a good target?
    # Random set of 32 in 512 has ~0 diffs.
    # Structured sets (like intervals) have many small diffs.
    # Let's target random sets of diffs.
    
    # Strategy: 
    # 1. Try to find a set with K diffs for K = 5, 10, 15, 20, 25...
    # 2. Start with small K to get a baseline, then increase.
    
    # However, _try_construct_for_diffs is probabilistic.
    # Let's interleave:
    # - Run a few SA steps (from parent) to get a good baseline quickly.
    # - Then try CSP targets based on that baseline + extra diffs.
    
    # Actually, the parent SA scored 4.4. That's very low. 
    # Wait, 4.4 is the score? 
    # "score 4.4000". 
    # If the max possible is high, 4.4 is bad. 
    # An interval of 32 integers {1..32} has diffs 1..15. Score 15.
    # So 4.4 is indeed very low. The parent SA is stuck in a bad local optimum or the random start is bad.
    # 
    # My CSP approach should easily beat 15 by targeting diffs 1..20.
    # Can we get 30? 40?
    # A set like {1, 2, ..., 32} U {some others} might have more.
    
    # Let's target specific diff sets.
    # Start with diffs 1..15 (guaranteed by interval 1..32).
    # Then try to add 16, 17, ...
    
    # 1. Get a strong baseline: Interval 1..32.
    base_set = list(range(1, 33))
    base_score = _count_progression_diffs(base_set)
    best_set = base_set
    best_score = base_score
    
    # 2. Try to improve by targeting more diffs.
    # We will try to construct a set that has diffs 1..K for increasing K.
    
    current_target_size = 15
    while time.time() < deadline - 5.0:
        # Try to add one more diff
        current_target_size += 1
        if current_target_size > 50: break
        
        target_diffs = list(range(1, current_target_size + 1))
        
        # Try to construct
        # To save time, limit attempts per target size
        attempts = 50 if current_target_size < 20 else 20
        
        candidate = _try_construct_for_diffs(target_diffs, max_attempts=attempts)
        
        if candidate:
            score = _count_progression_diffs(candidate)
            if score > best_score:
                best_score = score
                best_set = candidate
                # If we succeeded, we might try to go higher.
                # If we failed, maybe this K is hard, try next K? 
                # Or stick with current best and try random perturbations?
        else:
            # If we failed to construct for K, maybe K is too high?
            # Or our heuristic is weak.
            # Let's try a different strategy: 
            # Take the best_set found so far and do a few SA steps on it.
            # This combines the benefits.
            
            # Simple SA step on best_set
            S = set(best_set)
            T = 1.0
            for _ in range(500): # Few steps
                if time.time() > deadline - 5.0: break
                
                to_remove = random.choice(list(S))
                new_set = set(S)
                new_set.discard(to_remove)
                
                to_add = random.randint(1, 512)
                while to_add in new_set:
                    to_add = random.randint(1, 512)
                new_set.add(to_add)
                
                new_score = _count_progression_diffs(new_set)
                delta = new_score - best_score
                
                if delta > 0 or random.random() < pow(2.718, delta / T):
                    S = new_set
                    if new_score > best_score:
                        best_score = new_score
                        best_set = list(S)
                
                T *= 0.999
                if T < 0.1:
                    T = 0.1 # Keep some exploration
            
            # If we improved via SA, update best_set
            # Note: best_set is already updated in the loop above if improved
    
    # Final check
    if best_set is None:
        return list(range(1, 33))
        
    return best_set
# EVOLVE-BLOCK-END
