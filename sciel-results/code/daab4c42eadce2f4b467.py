import math
import time
import random
import numpy as np

def _score(u):
    """Compute the score for set u."""
    if not u:
        return 0.0
    max_u = max(u)
    if max_u == 0:
        return 1.0
    diffs = set()
    sums = set()
    for a in u:
        for b in u:
            diffs.add(a - b)
            sums.add(a + b)
    num_diffs = len(diffs)
    num_sums = len(sums)
    if num_sums == 0:
        return 0.0
    ratio = num_diffs / num_sums
    denom = 2 * max_u + 1
    if denom <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(denom)

def _spectral_guided_greedy(max_val, target_size, time_limit):
    """
    Greedy construction guided by spectral properties.
    We want to maximize |U-U| / |U+U|.
    This is equivalent to minimizing the collision rate in sums while maximizing spread in differences.
    
    We represent U as a binary vector v of length max_val + 1.
    The sumset corresponds to the support of v * v (convolution).
    The difference set corresponds to the support of v * reverse(v) (autocorrelation).
    
    Instead of full convolution every step (expensive), we use a heuristic:
    Prefer elements that are 'far' from existing ones in a spectral sense (low frequency interference).
    However, for exact scoring we must compute the sets.
    
    Strategy:
    1. Start with {0, 1, max_val}.
    2. Iteratively add the element x in [1, max_val] \ U that maximizes the score improvement.
    3. To speed up, we pre-filter candidates based on a simple spectral proxy:
       We want elements that don't create too many new sums.
       A new element x adds sums {x+u for u in U}.
       Collisions happen if x+u1 = x+u2 (impossible for distinct u) or x+u = v+w for existing sums.
       Actually, the number of new sums is |{x+u} \ (U+U)|.
       The number of new diffs is |{x-u} \ (U-U)|.
       
       We approximate the gain by counting how many new sums/diffs are created.
    """
    start_time = time.time()
    u = [0, 1, max_val]
    u_set = set(u)
    current_score = _score(u)
    
    # Pre-compute current sums and diffs to speed up marginal gain estimation
    # But for exactness, we recompute score only when a candidate is promising.
    # Let's use a fast marginal gain estimator.
    
    sums_set = set(a + b for a in u for b in u)
    diffs_set = set(a - b for a in u for b in u)
    
    best_candidate = None
    best_gain = -1.0
    
    # We will try to add elements until time runs out or size is reached.
    # To avoid O(N^2 * M) where M is max_val, we sample candidates or use a smarter search.
    # Given max_val can be large, we'll sample a subset of candidates each iteration.
    
    candidates_pool = list(range(2, max_val))
    random.shuffle(candidates_pool)
    
    while time.time() - start_time < time_limit:
        if len(u) >= target_size:
            break
            
        # Try a batch of candidates
        batch_size = 20
        found_improvement = False
        
        for _ in range(batch_size):
            if not candidates_pool:
                break
            x = candidates_pool.pop()
            if x in u_set:
                continue
            
            # Estimate marginal gain
            # New sums: {x + u for u in u}
            new_sums = [x + a for a in u]
            new_diffs = [x - a for a in u]
            
            # Count how many are truly new
            new_sums_count = sum(1 for s in new_sums if s not in sums_set)
            new_diffs_count = sum(1 for d in new_diffs if d not in diffs_set)
            
            # Heuristic score: We want high new_diffs and low new_sums.
            # The ratio change is roughly (|D| + new_diffs) / (|S| + new_sums)
            # Let's compute the actual score for this candidate if the heuristic looks good.
            # To save time, only compute full score if heuristic gain is positive.
            
            old_ratio = len(diffs_set) / len(sums_set)
            new_ratio = (len(diffs_set) + new_diffs_count) / (len(sums_set) + new_sums_count)
            
            if new_ratio > old_ratio:
                # Compute full score to be sure (includes log terms which depend on max_u, which might change if x > max_val? No, max_val is fixed bound, but max(u) might change)
                # Wait, max(u) determines the denominator. If x > max(u), max(u) increases, denominator increases, score might drop.
                # So we should only add x if it doesn't increase max(u) too much or if the ratio gain outweighs it.
                # Actually, the problem says "maximize ... on C6". The score formula uses max U.
                # If we add a large element, max U increases, log(2*max U + 1) increases, score decreases.
                # Therefore, it is generally better to stay within a fixed range or grow slowly.
                # Let's compute the exact score for the candidate set.
                
                candidate_u = u + [x]
                candidate_score = _score(candidate_u)
                
                if candidate_score > current_score:
                    # Accept
                    u = candidate_u
                    u_set.add(x)
                    current_score = candidate_score
                    found_improvement = True
                    
                    # Update sums and diffs sets
                    for a in u:
                        sums_set.add(x + a)
                        diffs_set.add(x - a)
                        # Note: x+x and x-x are covered since a can be x? No, a is from old u.
                        # We need to add sums/diffs involving x and all elements in the NEW u?
                        # Actually, sums_set should contain all sums of pairs in u.
                        # Previously it had sums of pairs in old_u.
                        # New sums are {x+a | a in old_u} U {x+x}.
                        # My loop adds x+a for a in u (which includes x now? No, u was updated before loop? No, u is updated, then loop runs over new u? No, I iterate over 'u' which is now the new list.
                        # This is inefficient and slightly wrong if I don't handle x+x.
                        # Let's just recompute sums_set and diffs_set from scratch after accepting? 
                        # That's O(N^2). If N is small (<100), it's fine.
                        pass
                    
                    # Recompute sets for correctness, as N is likely small during greedy phase
                    sums_set = set(a + b for a in u for b in u)
                    diffs_set = set(a - b for a in u for b in u)
                    
                    if found_improvement:
                        break
        
        if not found_improvement:
            # If no improvement found in this batch, try a different batch or stop?
            # Let's just continue to next batch.
            pass
            
    return u, current_score

def construct():
    """
    Construct a finite set U of nonnegative integers containing 0 to maximize
    the sumset/difference-set exponent lower bound.
    
    Strategy: Spectral-Guided Greedy Construction.
    1. Determine a good maximum value M.
    2. Use a greedy algorithm to build U by adding elements that maximize the score,
       using a fast marginal gain estimator to filter candidates.
    3. Refine with local search.
    """
    start_time = time.time()
    total_time = 115  # Safety margin
    
    # We need to choose max_val.
    # The score is 1 + log(|D|/|S|) / log(2*max_u + 1).
    # To maximize, we want |D|/|S| to be large (close to 2, max possible is 2*max_u+1 / (2*max_u+1) = 1? No.
    # Max |D| is 2*max_u + 1. Max |S| is 2*max_u + 1.
    # For a set with 0, |D| >= |S|.
    # If U is a Sidon-like set, |S| ~ N^2, |D| ~ N^2. Ratio ~ 1.
    # If U is arithmetic, |S| ~ 2N, |D| ~ 2N. Ratio ~ 1.
    # We want |D| to be large and |S| to be small.
    # This happens when the set is "spread out" but "sum-compressed".
    # Actually, for any set, |D| >= |S| is not always true?
    # |U-U| is always >= |U+U| - 1? No.
    # Cauchy-Davenport: |A+B| >= |A|+|B|-1.
    # |U-U| is at least |U|.
    
    # Let's try a few different max_val values.
    # Based on previous insights, moderate sizes work well.
    # The paper mentions size 2003 and 54265.
    # Let's try max_val around 1000-5000.
    
    best_set = [0]
    best_score = 1.0
    
    # Phase 1: Greedy construction with varying max_val
    max_vals = [50, 100, 200, 500, 1000, 2000, 5000]
    
    for mv in max_vals:
        if time.time() - start_time > total_time - 20:
            break
            
        # Allocate time based on remaining time
        time_left = total_time - (time.time() - start_time)
        alloc = time_left * 0.6 / len(max_vals)
        
        # Target size: we don't know optimal size, let greedy decide.
        # But greedy stops if no improvement.
        # Let's cap target size to avoid huge sets if not beneficial.
        target_size = 50
        
        u, score = _spectral_guided_greedy(mv, target_size, alloc)
        
        if score > best_score:
            best_score = score
            best_set = u
            
    # Phase 2: Local search refinement
    if time.time() - start_time < total_time - 10:
        time_left = total_time - (time.time() - start_time)
        # Simple local search: try swapping elements
        current = list(best_set)
        current_score = best_score
        max_val = max(current)
        
        while time.time() - start_time < total_time - 5:
            improved = False
            # Try removing an element
            for i in range(1, len(current)):
                if time.time() - start_time > total_time - 5:
                    break
                new_set = current[:i] + current[i+1:]
                s = _score(new_set)
                if s > current_score:
                    current = new_set
                    current_score = s
                    improved = True
                    break
            
            # Try adding an element
            if not improved:
                # Sample some candidates
                candidates = random.sample(range(1, max_val + 1), min(20, max_val))
                for x in candidates:
                    if x in current:
                        continue
                    new_set = sorted(current + [x])
                    s = _score(new_set)
                    if s > current_score:
                        current = new_set
                        current_score = s
                        max_val = max(current)
                        improved = True
                        break
                        
            if not improved:
                # Try a few random swaps
                if len(current) > 2:
                    i = random.randint(1, len(current)-1)
                    x = random.randint(1, max_val)
                    if x not in current:
                        new_set = current[:i] + [x] + current[i+1:]
                        new_set = sorted(set(new_set))
                        if 0 in new_set:
                            s = _score(new_set)
                            if s > current_score:
                                current = new_set
                                current_score = s
                                improved = True
                if not improved:
                    break
                    
        if current_score > best_score:
            best_score = current_score
            best_set = current
            
    return best_set
