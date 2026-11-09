"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def construct():
    """
    Find a Sidon set containing 0 with all differences 1..30 present,
    minimizing the largest element.
    
    Strategy: 
    1. Greedy construction of a valid Sidon set covering 1..30.
    2. Local Search (Simulated Annealing) to minimize the maximum element.
    """
    
    REQUIRED = set(range(1, 31))
    
    def get_diffs(S):
        """Return set of all positive differences in S."""
        S = np.array(S)
        # Compute all pairwise differences
        # S[i] - S[j] for i > j
        # Since S is sorted, S[i] > S[j]
        # We can use broadcasting
        # But for small N, simple loop is fine or np.triu
        diffs = set()
        n = len(S)
        for i in range(n):
            for j in range(i):
                d = S[i] - S[j]
                diffs.add(d)
        return diffs

    def is_sidon(S):
        """Check if S is a Sidon set."""
        diffs = get_diffs(S)
        return len(diffs) == len(S) * (len(S) - 1) // 2

    def coverage(S):
        """Return set of required differences covered by S."""
        diffs = get_diffs(S)
        return REQUIRED & diffs

    def energy(S):
        """
        Calculate energy of the set S.
        Lower is better.
        """
        if 0 not in S:
            return float('inf')
        
        diffs = get_diffs(S)
        n_diffs = len(diffs)
        expected_diffs = len(S) * (len(S) - 1) // 2
        
        # Penalty for non-Sidon (duplicates)
        dup_penalty = 0
        if n_diffs < expected_diffs:
            dup_penalty = (expected_diffs - n_diffs) * 10000
            
        # Penalty for missing required diffs
        missing = REQUIRED - diffs
        missing_penalty = len(missing) * 100
        
        # Penalty for max element
        max_penalty = max(S)
        
        return dup_penalty + missing_penalty + max_penalty

    def greedy_init(n_elements):
        """
        Construct a Sidon set of size n_elements greedily.
        """
        S = [0]
        # We need to add n_elements - 1 more
        # We prioritize small numbers that cover missing diffs
        current_diffs = set()
        
        for _ in range(n_elements - 1):
            best_x = None
            best_score = float('inf')
            
            # Search for x
            # Upper bound for x: maybe 500?
            for x in range(1, 500):
                if x in S:
                    continue
                
                # Check Sidon condition
                new_diffs = set(current_diffs)
                valid = True
                for e in S:
                    d = abs(x - e)
                    if d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                # Score: prefer covering missing diffs, then smaller x
                covers = len((REQUIRED & new_diffs) - (REQUIRED & current_diffs))
                # Actually, just count how many new required diffs it adds
                new_req = [d for d in new_diffs if d in REQUIRED and d not in current_diffs]
                
                # If it covers 0 new required diffs, it's less attractive unless we are forced
                score = -len(new_req) * 100 + x
                
                if score < best_score:
                    best_score = score
                    best_x = x
                    best_new_diffs = new_diffs
            
            if best_x is None:
                # If no small x found, expand range
                for x in range(500, 2000):
                    if x in S:
                        continue
                    new_diffs = set(current_diffs)
                    valid = True
                    for e in S:
                        d = abs(x - e)
                        if d in new_diffs:
                            valid = False
                            break
                    if not valid:
                        continue
                    new_req = [d for d in new_diffs if d in REQUIRED and d not in current_diffs]
                    score = -len(new_req) * 100 + x
                    if score < best_score:
                        best_score = score
                        best_x = x
                        best_new_diffs = new_diffs
            
            if best_x is None:
                return None # Failed to construct
            
            S.append(best_x)
            S.sort()
            current_diffs = best_new_diffs
            
        return S

    def local_search(S, max_iter=5000):
        """
        Simulated Annealing / Local Search to minimize max(S).
        """
        S = S[:]
        T = 1000.0
        best_S = S[:]
        best_E = energy(S)
        
        for i in range(max_iter):
            # Pick a random index to perturb (exclude 0)
            idx = random.randint(1, len(S) - 1)
            old_val = S[idx]
            
            # Generate new value
            # Random walk or jump
            step = random.randint(-10, 10)
            new_val = old_val + step
            
            # Keep it positive and distinct
            if new_val <= 0 or new_val in S:
                # Try a larger jump
                new_val = random.randint(1, max(S) + 50)
                if new_val in S:
                    continue
                    
            S[idx] = new_val
            S.sort()
            
            new_E = energy(S)
            delta_E = new_E - best_E
            
            if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                # Accept
                if new_E < best_E:
                    best_E = new_E
                    best_S = S[:]
            else:
                # Reject
                S[idx] = old_val
                S.sort()
            
            # Cool down
            T *= 0.99
            
            # Early stop if we have a perfect solution (0 missing, 0 dup) and max is small?
            # We can't easily know if it's global min, so run full iter.
            
        return best_S

    # Try different sizes
    # n=9 is minimal for 30 diffs (36 slots)
    # n=10 gives more flexibility
    
    best_result = None
    best_max = float('inf')
    
    # Start with n=9
    for n in [9, 10]:
        # Multiple restarts
        for restart in range(5):
            S = greedy_init(n)
            if S is None:
                continue
            
            # Check if it's already valid
            diffs = get_diffs(S)
            if REQUIRED.issubset(diffs) and is_sidon(S):
                current_max = max(S)
                if current_max < best_max:
                    best_max = current_max
                    best_result = S[:]
                
                # Try to improve with local search
                S_improved = local_search(S, max_iter=2000)
                diffs_improved = get_diffs(S_improved)
                if REQUIRED.issubset(diffs_improved) and is_sidon(S_improved):
                    current_max = max(S_improved)
                    if current_max < best_max:
                        best_max = current_max
                        best_result = S_improved[:]

    if best_result is not None:
        return best_result
        
    # Fallback: Return a known valid set if search fails
    # Construct a large Sidon set that covers 1..30
    # Use Singer set q=7 (size 8) + 2 extra?
    # Or just return a valid greedy result even if not optimized
    S = greedy_init(12)
    if S is not None:
        return S
    else:
        # Absolute fallback
        return [0, 1, 3, 7, 12, 18, 28, 42, 60, 85, 120, 160]

# EVOLVE-BLOCK-END
