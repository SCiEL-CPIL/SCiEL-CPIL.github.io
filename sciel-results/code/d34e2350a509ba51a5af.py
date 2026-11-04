import math
import random

def calc_score(U):
    if not U:
        return 0
    U = list(set(U))
    if 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 0
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    num = len(diffs) / len(sums)
    den = math.log(2 * max_u + 1)
    if den == 0:
        return 0
    return 1 + math.log(num) / den

def construct():
    best = [0, 1, 3]
    best_score = calc_score(best)
    
    # Search parameters
    bases = [2, 3, 4, 5]
    ns = [4, 5, 6, 7]
    
    # Set random seed for reproducibility in local search if needed, 
    # but random exploration is key.
    
    for B in bases:
        for n in ns:
            # Generate all possible values from base-B binary digits
            candidates = []
            for mask in range(1 << n):
                val = 0
                for j in range(n):
                    if mask & (1 << j):
                        val += B ** j
                candidates.append(val)
            
            # candidates always includes 0 (mask=0)
            # We want to find a subset U of candidates that maximizes score.
            # U must contain 0.
            
            # Local Search / Hill Climbing
            # Start with a random subset of a certain size
            # Size heuristic: maybe 1/4 to 1/2 of candidates?
            
            num_candidates = len(candidates)
            
            # Try a few different starting points and sizes
            for trial in range(5):
                # Random initial size between 10 and num_candidates//2
                if num_candidates < 20:
                    target_size = num_candidates // 2
                else:
                    target_size = random.randint(10, num_candidates // 2)
                
                # Ensure 0 is included. 0 is candidates[0].
                # Pick target_size - 1 other random candidates
                if target_size <= 1:
                    current_U = [0]
                else:
                    others = random.sample(candidates[1:], target_size - 1)
                    current_U = [0] + others
                
                current_score = calc_score(current_U)
                if current_score > best_score:
                    best_score = current_score
                    best = current_U[:]
                
                # Hill Climbing
                # Max steps to avoid timeout
                max_steps = 2000
                for step in range(max_steps):
                    # Pick a random action: add or remove
                    # If current size is small, bias towards add. If large, bias towards remove.
                    if len(current_U) < 20:
                        do_add = random.random() < 0.8
                    elif len(current_U) > num_candidates // 2:
                        do_add = random.random() < 0.2
                    else:
                        do_add = random.random() < 0.5
                    
                    if do_add:
                        # Pick a candidate not in current_U
                        # Efficient way: keep a set of current_U for O(1) lookup
                        # But we need to pick randomly from complement.
                        # For small num_candidates, just filter.
                        available = [c for c in candidates if c not in current_U]
                        if not available:
                            continue
                        new_elem = random.choice(available)
                        new_U = current_U + [new_elem]
                    else:
                        # Pick a candidate in current_U to remove
                        # Must keep 0
                        if len(current_U) <= 1:
                            continue
                        # Pick from current_U excluding 0
                        removable = [c for c in current_U if c != 0]
                        if not removable:
                            continue
                        new_elem = random.choice(removable)
                        new_U = [c for c in current_U if c != new_elem]
                    
                    new_score = calc_score(new_U)
                    if new_score > current_score:
                        current_U = new_U
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best = current_U[:]
                    else:
                        # If we get stuck, maybe restart? 
                        # For simplicity, just continue. 
                        # A full restart would be better but complex to manage state.
                        pass

    return best
