"""CSP-based construction of a Sidon set containing 0 with all differences 1..30."""
import sys
sys.setrecursionlimit(10000)

def construct():
    # We need a Sidon set S containing 0 such that every d in 1..30 is a difference.
    # Sidon condition: all positive differences y-x (x<y in S) are distinct.
    # We want to minimize max(S).
    
    # Key insight: If S is a Sidon set with |S| = n, then the number of positive differences is n*(n-1)/2.
    # We need at least 30 differences (for d=1..30), so n*(n-1)/2 >= 30 => n >= 9.
    # For n=9, we have 36 total differences, of which 30 must be exactly 1..30.
    
    # Strategy: Use backtracking with forward checking.
    # State: current elements (sorted), set of used differences, coverage bitmask for 1..30.
    # We build the set incrementally by adding elements in increasing order.
    
    def solve():
        # Backtracking search
        # We'll try to build the set by adding one element at a time.
        # Key: we add elements in increasing order to maintain sorted state.
        
        best = [None]
        
        def all_diffs(s):
            """Return list of all positive differences in sorted set s."""
            return [s[j] - s[i] for i in range(len(s)) for j in range(i+1, len(s))]
        
        def backtrack(elements, used_diffs_set, coverage_mask, max_val):
            # Check if we've covered all required differences
            if coverage_mask == (1 << 31) - 2:  # bits 1..30 set
                # Found a valid set
                if best[0] is None or max(elements) < max(best[0]):
                    best[0] = list(elements)
                return True
            
            # Pruning: if max(elements) >= max_val, stop
            if elements and max(elements) >= max_val:
                return False
            
            # Heuristic: pick the smallest missing difference
            missing = [d for d in range(1, 31) if not (coverage_mask & (1 << d))]
            if not missing:
                return False
            d = missing[0]
            
            # Try to add a new element x such that x - y = d or y - x = d for some y in elements
            # i.e., x = y + d or x = y - d for some y in elements
            # x must be >= 0, not in elements, and adding x must preserve Sidon property
            
            candidates = set()
            for y in elements:
                for cand in (y + d, y - d):
                    if cand >= 0 and cand not in elements:
                        candidates.add(cand)
            
            # Sort candidates by value (try smaller first to keep max small)
            for x in sorted(candidates):
                new_elements = sorted(elements + [x])
                
                # Check Sidon property: new diffs must not conflict with used_diffs_set
                valid = True
                new_diffs = []
                for y in elements:
                    diff = abs(x - y)
                    if diff in used_diffs_set:
                        valid = False
                        break
                    new_diffs.append(diff)
                
                if not valid:
                    continue
                
                # Update coverage mask
                new_mask = coverage_mask
                for diff in new_diffs:
                    if 1 <= diff <= 30:
                        new_mask |= (1 << diff)
                
                # Prune if new max element exceeds bound
                if max(new_elements) >= max_val:
                    continue
                
                new_used_set = used_diffs_set | set(new_diffs)
                
                if backtrack(new_elements, new_used_set, new_mask, max_val):
                    return True
            
            return False
        
        # Iterative deepening on max_val
        # Lower bound: need at least 30 distinct differences, all <= max(S), so max(S) >= 30.
        # But with Sidon constraint, it's larger.
        # For n=9: 36 distinct positive differences, max diff = max(S), so max(S) >= 36.
        
        for max_val in range(36, 100):
            backtrack([0], set(), 0, max_val)
            if best[0] is not None:
                break
        
        if best[0] is not None:
            return best[0]
        
        # Fallback: greedy approach (shouldn't reach here)
        a = [0]
        for d in range(1, 31):
            found = False
            for x in range(1, 1000000):
                trial = sorted(a + [x])
                dd = all_diffs(trial)
                if d in dd and len(set(dd)) == len(dd):
                    a = trial
                    found = True
                    break
            if not found:
                # Force add
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    dd = all_diffs(trial)
                    if len(set(dd)) == len(dd):
                        a = trial
                        break
                    y += 1
        return a
    
    return solve()
