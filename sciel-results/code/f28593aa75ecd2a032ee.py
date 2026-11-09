"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    t_start = time.time()
    DEADLINE = t_start + 75.0  # Safety margin
    N = 4096
    TARGET_SIZE = 40  # Theoretical max based on sum range 3..12288
    
    # Precompute conflict structure
    # Two numbers x, y conflict if adding y to a set containing x creates a duplicate triple sum
    # We'll use a bitset-based approach for fast conflict checking
    
    # For each pair (x, y), compute the triple sums they create with each other
    # Conflict: x and y conflict if there exists z such that some triple sum involving x,y,z
    # equals another triple sum
    
    # Key insight: if we have selected set S, and want to add x, we need to check:
    # 1. 3x not in triple_sums
    # 2. 2x+y not in triple_sums for all y in S
    # 3. x+2y not in triple_sums for all y in S
    # 4. x+y+z not in triple_sums for all y<=z in S
    
    # We can precompute for each x, the set of "forbidden" values that would create conflicts
    
    # Strategy: Use greedy with lookahead
    # At each step, pick the candidate that minimizes future conflicts
    
    def compute_conflict_matrix(selected):
        """Compute which candidates conflict with the current selection."""
        if not selected:
            return np.zeros(N + 1, dtype=bool)
        
        forbidden = np.zeros(N + 1, dtype=bool)
        k = len(selected)
        sel = np.array(selected)
        
        # For each candidate x in 1..N, check if it conflicts
        # Conflict if any of: 3x, 2x+y, x+2y, x+y+z (y<=z in selected) is already in triple_sums
        
        # First, compute all current triple sums
        triple_sums_set = set()
        for i in range(k):
            for j in range(i, k):
                for l in range(j, k):
                    s = sel[i] + sel[j] + sel[l]
                    triple_sums_set.add(s)
        
        # For each candidate x, check conflicts
        for x in range(1, N + 1):
            if x in selected:
                forbidden[x] = True
                continue
            
            # Check 3x
            if 3 * x in triple_sums_set:
                forbidden[x] = True
                continue
            
            # Check 2x + y for each y in selected
            conflict = False
            for y in selected:
                if 2 * x + y in triple_sums_set:
                    conflict = True
                    break
            if conflict:
                forbidden[x] = True
                continue
            
            # Check x + 2y for each y in selected
            for y in selected:
                if x + 2 * y in triple_sums_set:
                    conflict = True
                    break
            if conflict:
                forbidden[x] = True
                continue
            
            # Check x + y + z for y <= z in selected
            for i in range(k):
                for j in range(i, k):
                    if x + sel[i] + sel[j] in triple_sums_set:
                        conflict = True
                        break
                if conflict:
                    break
            if conflict:
                forbidden[x] = True
        
        return forbidden
    
    def greedy_with_lookahead(target_size):
        """Greedy algorithm with conflict-based candidate selection."""
        selected = []
        triple_sums = set()
        
        for step in range(target_size):
            if time.time() > DEADLINE:
                break
            
            if not selected:
                # First element: pick smallest
                selected.append(1)
                triple_sums.add(3)
                continue
            
            # Compute which candidates are forbidden
            forbidden = compute_conflict_matrix(selected)
            
            # Find valid candidates
            valid_candidates = [x for x in range(1, N + 1) if not forbidden[x]]
            
            if not valid_candidates:
                break
            
            # Score each candidate by how many future candidates it forbids
            # Pick the one that forbids the fewest (most flexible)
            best_x = None
            best_score = float('inf')
            
            # Limit scoring to top 200 candidates for speed
            scored_candidates = valid_candidates[:200]
            
            for x in scored_candidates:
                # Estimate how many candidates x would forbid
                # This is expensive, so use a heuristic: prefer larger x (more room)
                score = -x  # Prefer larger values
                if score < best_score:
                    best_score = score
                    best_x = x
            
            if best_x is None:
                break
            
            # Add best_x and update triple_sums
            selected.append(best_x)
            k = len(selected)
            sel = selected
            
            # Add new triple sums
            for i in range(k - 1, -1, -1):
                for j in range(i, -1, -1):
                    for l in range(j, -1, -1):
                        s = sel[i] + sel[j] + sel[l]
                        triple_sums.add(s)
        
        return selected
    
    def random_greedy_improved(target_size, seed=None):
        """Randomized greedy with better candidate selection."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        
        selected = []
        triple_sums = set()
        
        for step in range(target_size):
            if time.time() > DEADLINE:
                break
            
            if not selected:
                selected.append(rng.randint(1, N))
                triple_sums.add(3 * selected[0])
                continue
            
            # Try random candidates until we find one that works
            attempts = 0
            max_attempts = 500
            while attempts < max_attempts:
                if time.time() > DEADLINE:
                    return selected
                
                x = rng.randint(1, N)
                if x in selected:
                    attempts += 1
                    continue
                
                # Check if x can be added
                new_sums = set()
                valid = True
                
                # 3x
                s = 3 * x
                if s in triple_sums or s in new_sums:
                    valid = False
                else:
                    new_sums.add(s)
                
                if valid:
                    # 2x + y
                    for y in selected:
                        s = 2 * x + y
                        if s in triple_sums or s in new_sums:
                            valid = False
                            break
                        new_sums.add(s)
                
                if valid:
                    # x + 2y
                    for y in selected:
                        s = x + 2 * y
                        if s in triple_sums or s in new_sums:
                            valid = False
                            break
                        new_sums.add(s)
                
                if valid:
                    # x + y + z
                    k = len(selected)
                    for i in range(k):
                        for j in range(i, k):
                            s = x + selected[i] + selected[j]
                            if s in triple_sums or s in new_sums:
                                valid = False
                                break
                            new_sums.add(s)
                        if not valid:
                            break
                
                if valid:
                    selected.append(x)
                    triple_sums.update(new_sums)
                    break
                
                attempts += 1
        
        return selected
    
    best_set = []
    
    # Strategy 1: Deterministic greedy with lookahead
    current = greedy_with_lookahead(TARGET_SIZE)
    if len(current) > len(best_set):
        best_set = current
    
    # Strategy 2: Multiple randomized restarts
    attempts = 0
    while time.time() < DEADLINE - 5.0 and attempts < 30:
        seed = attempts * 12345 + 67890
        current = random_greedy_improved(TARGET_SIZE, seed=seed)
        if len(current) > len(best_set):
            best_set = current
            if len(best_set) >= TARGET_SIZE:
                break
        attempts += 1
    
    return best_set

# EVOLVE-BLOCK-END
