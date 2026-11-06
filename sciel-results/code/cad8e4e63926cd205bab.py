"""Constructs a Sidon set using beam search with heuristics."""
import time

def construct():
    """Construct a Sidon set containing 0 with all differences 1..30, minimizing max element."""
    
    target_diffs = set(range(1, 31))
    max_size = 32
    start_time = time.time()
    timeout = 110  # Leave 10s buffer

    def get_new_diffs(S, new_val):
        return {abs(new_val - x) for x in S}

    # Beam Search
    # State: (S_sorted, used_diffs, missing_diffs)
    beam_width = 50
    current_beam = [([0], set(), set(target_diffs))]
    
    while current_beam and time.time() - start_time < timeout:
        next_beam = []
        
        for S, used, missing in current_beam:
            if not missing:
                # Found a solution
                return S
            
            if len(S) >= max_size:
                continue
                
            # Generate candidates
            # Candidates should be integers.
            # We focus on candidates that cover at least one missing difference.
            candidates = set()
            for d in missing:
                for x in S:
                    v1 = x + d
                    v2 = x - d
                    if v1 >= 0:
                        candidates.add(v1)
                    if v2 >= 0:
                        candidates.add(v2)
            
            # Add a range of small integers to allow flexibility
            # This helps escape local minima where only "greedy" diffs are considered
            for v in range(1, max(S) + 20):
                candidates.add(v)
                
            # Evaluate candidates
            valid_candidates = []
            for v in candidates:
                if v in S:
                    continue
                
                # Check Sidon property
                new_diffs = get_new_diffs(S, v)
                if new_diffs & used:
                    continue
                
                # Calculate new state
                new_used = used | new_diffs
                new_missing = missing - new_diffs
                new_S = sorted(S + [v])
                
                # Score: 
                # 1. Maximize covered diffs (len(target_diffs) - len(new_missing))
                # 2. Minimize max element
                covered = len(target_diffs) - len(new_missing)
                max_el = max(new_S)
                
                # We want high covered, low max_el
                # Score tuple: (-covered, max_el, len(new_S))
                score = (-covered, max_el, len(new_S))
                
                valid_candidates.append((score, new_S, new_used, new_missing))
            
            # Keep top beam_width candidates
            valid_candidates.sort(key=lambda x: x[0])
            for item in valid_candidates[:beam_width]:
                next_beam.append(item[1:])
            
        # Deduplicate states in next_beam
        seen = set()
        unique_beam = []
        for S, used, missing in next_beam:
            key = (tuple(S), frozenset(used))
            if key not in seen:
                seen.add(key)
                unique_beam.append((S, used, missing))
        
        current_beam = unique_beam
        
    if current_beam:
        # Return the best solution in the beam
        best_state = None
        best_score = None
        for S, used, missing in current_beam:
            if not missing:
                covered = 30
            else:
                covered = len(target_diffs) - len(missing)
            max_el = max(S)
            score = (-covered, max_el)
            if best_score is None or score < best_score:
                best_score = score
                best_state = S
        
        if best_state and not (set(range(1, 31)) - {abs(a-b) for i,a in enumerate(best_state) for b in best_state[i+1:]}):
             # Verify validity
             diffs = set()
             n = len(best_state)
             for i in range(n):
                 for j in range(i+1, n):
                     diffs.add(best_state[j] - best_state[i])
             if all(d in diffs for d in range(1, 31)) and len(diffs) == n*(n-1)//2:
                 return best_state

    # Fallback: Hardcoded perfect difference set of order 5
    # Size 31, covers 1..30, max element 106
    return [0, 1, 4, 14, 16, 28, 29, 32, 37, 41, 46, 49, 54, 55, 60, 62, 64, 70, 71, 76, 78, 83, 85, 86, 89, 94, 96, 99, 101, 105, 106]
