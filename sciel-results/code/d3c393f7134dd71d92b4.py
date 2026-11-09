"""Sidon set construction minimizing diameter for differences 1..30."""
import time
import sys

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    
    def get_diffs(s):
        """Get all positive differences of sorted set s."""
        diffs = set()
        for i in range(len(s)):
            for j in range(i+1, len(s)):
                diffs.add(s[j] - s[i])
        return diffs
    
    def is_sidon(s):
        """Check if s is a Sidon set (all positive differences distinct)."""
        diffs = []
        for i in range(len(s)):
            for j in range(i+1, len(s)):
                diffs.append(s[j] - s[i])
        return len(diffs) == len(set(diffs))
    
    def covered_mask(s):
        """Return bitmask of differences 1..30 that are covered."""
        diffs = get_diffs(s)
        mask = 0
        for d in range(1, 31):
            if d in diffs:
                mask |= (1 << d)
        return mask
    
    FULL_MASK = (1 << 31) - 2  # bits 1 through 30 set
    
    # Beam search
    # State: (elements_tuple, mask, max_elem)
    # Start with {0}
    initial = ((0,), 0, 0)
    
    best_solution = None
    best_max = float('inf')
    
    # Try increasing sizes, starting small
    for target_size in range(9, 33):
        if time.time() > deadline - 10:
            break
        
        # Beam search for this size
        beam = [initial]
        
        for depth in range(target_size - 1):
            if time.time() > deadline - 5:
                break
            
            candidates = []
            
            for state in beam:
                elems, mask, max_el = state
                
                # If already complete, record and continue (might find smaller)
                if mask == FULL_MASK:
                    if max_el < best_max:
                        best_max = max_el
                        best_solution = list(elems)
                    # Don't expand further if we have a complete solution
                    # with this or smaller max
                    if max_el >= best_max:
                        continue
                
                # Generate candidate new elements
                # Try small values first for better diameter
                tried = set(elems)
                # Upper bound: don't go too far
                upper = best_max if best_max < float('inf') else 200
                # Also try some strategic values
                candidate_values = set()
                for v in range(1, min(upper + 1, max_el + 50)):
                    if v not in tried:
                        candidate_values.add(v)
                # Add some strategic candidates
                for v in [max_el + 1, max_el + 2, max_el + 3, max_el + 5]:
                    if v not in tried and v <= upper:
                        candidate_values.add(v)
                
                for x in sorted(candidate_values):
                    new_elems = tuple(sorted(elems + (x,)))
                    new_max = new_elems[-1]
                    
                    if new_max >= best_max:
                        continue
                    
                    # Check Sidon property
                    if not is_sidon(new_elems):
                        continue
                    
                    new_mask = covered_mask(new_elems)
                    
                    # Only keep if we gain new coverage or same coverage with smaller max
                    if new_mask == mask:
                        continue
                    
                    candidates.append((new_elems, new_mask, new_max))
            
            if not candidates:
                break
            
            # Score and select top beam_width states
            # Score: (max_elem, -popcount(mask)) - prefer smaller max and more coverage
            def popcount(x):
                return bin(x).count('1')
            
            candidates.sort(key=lambda s: (s[2], -popcount(s[1]), len(s[0])))
            
            # Keep diverse top candidates
            beam_width = 500
            beam = candidates[:beam_width]
            
            # Check if any complete solution found
            for state in beam:
                if state[1] == FULL_MASK:
                    if state[2] < best_max:
                        best_max = state[2]
                        best_solution = list(state[0])
        
        # After finishing this size, if we have a solution, move on
        if best_solution is not None:
            # We could try to improve but let's move to next size for now
            pass
    
    # If we found a solution, return it
    if best_solution is not None:
        return best_solution
    
    # Fallback: use the greedy approach from seed
    a = [0]
    def diffs_list(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs_list(sorted(a)):
            continue
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs_list(trial))) == len(diffs_list(trial)):
                a = trial
                break
        else:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs_list(trial))) == len(diffs_list(trial)):
                    a = trial
                    break
                y += 1
    return a

# EVOLVE-BLOCK-END
