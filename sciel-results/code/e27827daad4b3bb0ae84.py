import time
import random

def construct():
    target_diffs = set(range(1, 31))
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s kill
    best_solution = None
    best_diameter = float('inf')
    
    # Lower bound for diameter is roughly 50-60 for small sets covering 1..30
    # We will try to find solutions with diameter < 100 first, then expand if needed.
    # But since we want to minimize the max element, we can just run random greedy
    # and keep the best one found.
    
    # To guide the search, we can try to force specific small differences early.
    # A known small Sidon set covering 1..30 might look like:
    # e.g., {0, 1, 4, 10, 12, 17, 21, 23, 28, ...}
    
    # Strategy: Randomized Greedy
    # 1. Start with {0}
    # 2. At each step, pick a random candidate x that is valid (doesn't create duplicate diffs)
    #    and prioritizes covering missing differences.
    # 3. Repeat until all 1..30 are covered or max size reached.
    # 4. Keep the solution with the smallest max element.
    
    num_restarts = 10000
    attempt = 0
    
    while time.time() < deadline and attempt < num_restarts:
        attempt += 1
        sol = greedy_random(target_diffs)
        if sol is not None:
            max_el = max(sol)
            if max_el < best_diameter:
                best_diameter = max_el
                best_solution = sol
                # If we find a very good solution, we can stop early or tighten search
                # Lower bound is ~50. If we get < 60, it's likely near optimal.
                if best_diameter <= 55:
                    break

    if best_solution is not None:
        return best_solution
    
    # Fallback: Deterministic Greedy if random failed (unlikely)
    return greedy_deterministic(target_diffs)

def greedy_random(target_diffs):
    """
    Attempts to construct a Sidon set covering target_diffs using a randomized greedy approach.
    Returns the set if successful, else None.
    """
    current = [0]
    used_diffs = set()
    missing = set(target_diffs)
    
    # Optional: Force some small elements to help cover small differences quickly
    # But randomization handles this.
    
    max_size = 32
    # We don't want the set to grow too large or elements too big.
    # If we can't cover missing diffs, we might get stuck.
    
    while missing and len(current) < max_size:
        candidates = []
        
        # Search for candidates. 
        # We should look in a range that is likely to be small to minimize diameter.
        # However, if we're stuck, we might need larger numbers.
        # Let's search up to a dynamic limit.
        # If we have k elements, the next element x creates k new diffs.
        # To cover a specific missing diff d, we need x such that x-y=d or y-x=d.
        # So x = y+d or x = y-d.
        
        # Strategy: Generate candidates based on missing diffs
        # For each missing diff d, and each existing y, consider x = y+d and x = y-d (if x>0).
        # This focuses the search on elements that *will* cover a missing diff.
        
        candidate_set = set()
        for d in missing:
            for y in current:
                x1 = y + d
                x2 = y - d
                if x1 > 0:
                    candidate_set.add(x1)
                if x2 > 0:
                    candidate_set.add(x2)
        
        # Also add some random small numbers to explore
        # But the diff-based candidates are more promising for covering 'missing'
        
        if not candidate_set:
            return None
        
        # Filter candidates
        valid_candidates = []
        for x in candidate_set:
            if x in current:
                continue
            # Check Sidon condition
            valid = True
            new_diffs = set()
            for y in current:
                diff = abs(x - y)
                if diff in used_diffs:
                    valid = False
                    break
                new_diffs.add(diff)
            
            if valid:
                # Score: how many missing diffs does this cover?
                score = len(new_diffs & missing)
                if score > 0:
                    valid_candidates.append((score, x, new_diffs))
        
        if not valid_candidates:
            return None
        
        # Pick a candidate.
        # Heuristic: Pick one with highest score. If tie, random.
        # To add randomness, we can pick from top 3 scores.
        valid_candidates.sort(key=lambda t: -t[0])
        
        # Randomly select from the top few to diversify search
        top_n = min(5, len(valid_candidates))
        choice = random.choice(valid_candidates[:top_n])
        
        score, x, new_diffs = choice
        
        current.append(x)
        used_diffs.update(new_diffs)
        missing.difference_update(new_diffs)
        
        # Sort current to keep it tidy (not strictly necessary but helps debugging)
        current.sort()
        
    if not missing:
        return current
    return None

def greedy_deterministic(target_diffs):
    """
    Fallback deterministic greedy.
    """
    current = [0]
    used_diffs = set()
    missing = set(target_diffs)
    
    while missing and len(current) < 32:
        best_x = None
        best_score = -1
        
        # Search range: try to keep numbers small
        # Max element so far is M. Try x in 1..M+50
        M = max(current)
        search_limit = M + 50
        
        for x in range(1, search_limit + 1):
            if x in current:
                continue
            valid = True
            new_diffs = set()
            for y in current:
                diff = abs(x - y)
                if diff in used_diffs:
                    valid = False
                    break
                new_diffs.add(diff)
            
            if valid:
                score = len(new_diffs & missing)
                if score > best_score:
                    best_score = score
                    best_x = x
                    if best_score == len(missing):
                        break
        
        if best_x is None:
            return None
        
        current.append(best_x)
        current.sort()
        
        # Update state
        for y in current[:-1]:
            diff = abs(best_x - y)
            used_diffs.add(diff)
            if diff in missing:
                missing.remove(diff)
        
    if not missing:
        return current
    return None
