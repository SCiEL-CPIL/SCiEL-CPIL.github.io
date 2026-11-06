"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 115
    best = [None]
    best_max = [float('inf')]
    
    def solve_csp():
        """
        CSP approach: build Sidon set containing 0, covering diffs 1..30.
        Use MRV: at each step, find the uncovered difference with fewest
        valid placements, then try placing it.
        
        State:
        - elements: sorted list of chosen elements (starts with [0])
        - used_diffs: set of all positive differences realized
        - need: set of differences still needed (subset of 1..30)
        """
        elements = [0]
        used_diffs = set()
        need = set(range(1, 31))
        
        def get_valid_placements(d):
            """For uncovered diff d, find all valid (a, b) pairs where b-a=d,
            a in elements, b not in elements, b > a, and no new diff conflicts."""
            placements = []
            for a in elements:
                b = a + d
                # b must be > all current elements (to maintain sorted order when we add)
                # Actually b just needs to not be in elements and be positive
                if b in elements:
                    continue
                # Check: adding b creates diffs b - e for all e in elements
                # These must not be in used_diffs
                valid = True
                new_diffs = []
                for e in elements:
                    diff = b - e
                    if diff <= 0:
                        continue
                    if diff in used_diffs:
                        valid = False
                        break
                    new_diffs.append(diff)
                if valid:
                    placements.append((a, b, new_diffs))
            return placements
        
        def backtrack():
            if time.time() > deadline:
                return False
            if not need:
                mx = elements[-1]
                if mx < best_max[0]:
                    best_max[0] = mx
                    best[0] = list(elements)
                return True
            
            # MRV: pick the uncovered diff with fewest valid placements
            min_count = float('inf')
            chosen_d = None
            chosen_placements = None
            for d in sorted(need):
                if time.time() > deadline:
                    return False
                placements = get_valid_placements(d)
                count = len(placements)
                if count == 0:
                    return False  # Dead end
                if count < min_count:
                    min_count = count
                    chosen_d = d
                    chosen_placements = placements
                    if count == 1:
                        break  # Can't do better than 1
            
            if chosen_d is None:
                return False
            
            # Try each placement for chosen_d
            for (a, b, new_diffs) in chosen_placements:
                if time.time() > deadline:
                    return False
                
                # Add b to elements (keep sorted)
                elements.append(b)
                elements.sort()
                
                # Add new diffs
                for diff in new_diffs:
                    used_diffs.add(diff)
                    if diff in need:
                        need.discard(diff)
                
                if backtrack():
                    # Undo
                    elements.remove(b)
                    for diff in new_diffs:
                        used_diffs.discard(diff)
                    return True
                
                # Undo
                elements.remove(b)
                for diff in new_diffs:
                    used_diffs.discard(diff)
            
            return False
        
        backtrack()
    
    solve_csp()
    
    if best[0] is not None:
        return best[0]
    
    # Fallback: greedy
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                break
        else:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    return a
# EVOLVE-BLOCK-END
