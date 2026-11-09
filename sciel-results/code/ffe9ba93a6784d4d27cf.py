"""Construct a Sidon set containing 0 with all differences 1..30, minimizing the largest element."""

import time
import sys

def construct():
    # Deadline guard: return best found before 150s wall-clock
    deadline = time.time() + 145.0
    
    # We want a Sidon set S with 0 in S, |S| <= 32, all positive differences distinct,
    # and every d in 1..30 appears as a difference. Minimize max(S).
    
    # Key insight: We need to find a set S = {0, s1, s2, ..., sk} such that:
    # 1. All pairwise differences |si - sj| for i != j are distinct (Sidon condition)
    # 2. Every d in {1, ..., 30} is among those differences
    # 3. k+1 <= 32
    # 4. max(S) is minimized
    
    # Strategy: Use DFS/backtracking with bitmask pruning on covered differences.
    # Build the set incrementally, tracking which differences are covered.
    # At each step, try adding the smallest possible new element that:
    #   - doesn't create duplicate differences
    #   - covers at least one new difference (preferably)
    #   - keeps max(S) as small as possible
    
    # For efficiency, we'll use a different approach:
    # Since we need all 30 differences, and a Sidon set of size n has n*(n-1)/2 differences,
    # we need n*(n-1)/2 >= 30, so n >= 8 (8*7/2 = 28 < 30, so n >= 9 since 9*8/2=36).
    # Actually n=8 gives 28 diffs, not enough. n=9 gives 36, which is >= 30.
    # But we also need 0 in the set, and the set must have exactly the right structure.
    
    # Let's use a backtracking search with aggressive pruning.
    # State: current set (sorted), covered differences bitmask (30 bits)
    # We try to build the set by adding elements in increasing order.
    
    # For speed, we'll use a greedy+backtrack hybrid:
    # 1. Start with {0}
    # 2. At each step, generate candidate next elements
    # 3. Sort candidates by resulting max value
    # 4. Try each, with pruning
    
    # Better approach: Since the problem is small (need ~9-12 elements),
    # we can use a more structured search.
    
    # Let me think about this differently. A perfect difference set of order n 
    # (size n) in Z has all differences 1..n^2-n+1 distinct. We want a "truncated" 
    # version where we only need differences 1..30.
    
    # Known small Sidon sets with many small differences:
    # Let's try to find one by systematic search.
    
    # Approach: DFS with the following:
    # - Current set S (sorted list)
    # - Covered differences as a set
    # - Used differences as a set (to check Sidon condition)
    # - Try adding new elements in order of increasing value
    
    # To make this fast, I'll use a different strategy:
    # Enumerate possible sets by trying all subsets of {0, 1, 2, ..., M} of size k
    # for increasing M, checking Sidon condition and coverage.
    # But this is too slow for large M.
    
    # Better: Use the fact that 0 is in the set. So S = {0} ∪ T where T ⊂ {1, ..., M}.
    # The differences are: elements of T, and |t1 - t2| for t1, t2 in T.
    # All these must be distinct and must cover 1..30.
    
    # Let's use a backtracking search on T, building it element by element.
    # For efficiency, track:
    # - current elements of T (sorted)
    # - all differences generated so far (as a set for O(1) lookup)
    # - covered differences in 1..30 (as a bitmask)
    
    # Pruning:
    # - If we already have |T| = 31 (total 32 elements), check if all 30 diffs are covered
    # - If adding a new element would create a duplicate difference, skip
    # - Lower bound on remaining elements needed
    
    # For the search to be fast enough, I'll use a smarter ordering and more aggressive pruning.
    
    # Actually, let me try a completely different approach that's known to work well:
    # Use a simulated annealing or local search to find a good Sidon set.
    
    # But the directive says to use branch-and-bound with memoization on covered difference bitmask.
    # Let me implement that carefully.
    
    # State for memoization: (frozenset_of_current_elements, covered_mask)
    # But frozenset of elements is expensive. Instead, since elements are bounded,
    # we can use a tuple of sorted elements.
    
    # Actually, for this problem size, let me try a more direct approach:
    # 
    # We know we need at least 9 elements (including 0). Let's search for sets of size 9, 10, 11, ...
    # For each size k, we search for the minimum M such that there exists a Sidon set 
    # {0, t1, ..., t_{k-1}} with t_{k-1} <= M covering all diffs 1..30.
    
    # For the search, I'll use DFS with the following key optimizations:
    # 1. Build elements in increasing order
    # 2. At each level, try candidate values starting from the smallest valid one
    # 3. Use bitmask for covered differences
    # 4. Prune if remaining capacity can't cover all remaining differences
    
    # Let me implement this:
    
    def search():
        # We'll build the set incrementally
        # current_set: list of elements (always includes 0, sorted)
        # diffs_set: set of all positive differences generated
        # covered: bitmask of which differences in 1..30 are covered
        
        FULL_MASK = (1 << 30) - 1  # bits 0..29 represent diffs 1..30
        
        best_max = [10**7]  # mutable container for best found
        best_set = [None]
        
        # For memoization, we'll use a dict mapping (tuple of elements, covered_mask) -> True
        # to avoid re-exploring the same state. But this might use too much memory.
        # Instead, we'll just be careful about the search order.
        
        # Actually, let me try a different, more efficient approach.
        # 
        # Key observation: In a Sidon set containing 0, the differences are:
        # - Each element t_i (difference from 0)
        # - |t_i - t_j| for i < j
        # 
        # So if S = {0, a1, a2, ..., ak} with 0 < a1 < a2 < ... < ak,
        # the differences are: a1, a2, ..., ak, and |ai - aj| for i < j.
        # All of these must be distinct.
        
        # Let's do a DFS where we pick a1, a2, ..., ak in order.
        # At step i, we choose a_i > a_{i-1}.
        # The new differences added are: a_i and |a_i - a_j| for j < i.
        # None of these can be in the existing diffs_set.
        
        # Pruning:
        # - If len(diffs_set) + (k - i) * (k - i + 1) // 2 < 30, we can't cover all 30 diffs
        #   Wait, that's not quite right. Let me think...
        # - Actually, the maximum number of additional differences we can add with m more elements
        #   is m * (current_size + 1) + m * (m-1) // 2... no.
        # - If we have c elements so far and add m more, the new differences are:
        #   m (from 0) + m*c (from existing non-zero elements) + m*(m-1)//2 (among themselves)
        #   = m + m*c + m*(m-1)//2
        # - We need the total to be at least 30.
        
        # Let me just implement a clean DFS with good pruning:
        
        import sys
        sys.setrecursionlimit(10000)
        
        def dfs(current, diffs, covered, depth, max_allowed):
            """
            current: tuple of elements (sorted, includes 0)
            diffs: set of all positive differences
            covered: bitmask of covered differences in 1..30
            depth: number of elements chosen so far (including 0)
            max_allowed: upper bound on the maximum element
            """
            if time.time() > deadline:
                return False
            
            # Check if we've covered all 30 differences
            if covered == FULL_MASK:
                if max(current) < best_max[0]:
                    best_max[0] = max(current)
                    best_set[0] = list(current)
                    return True  # Found a solution, but keep searching for better
                return False
            
            # If we've used all 32 elements, we can't add more
            if depth >= 32:
                return False
            
            # Pruning: can we still cover all remaining differences?
            # Remaining diffs to cover: 30 - popcount(covered)
            # Max additional diffs with (32 - depth) more elements:
            remaining_slots = 32 - depth
            # Each new element x adds: 1 (diff from 0) + (depth-1) (diffs from other non-zero) + (index among new) 
            # Actually, if we add m new elements to a set of size c, new diffs = m + m*(c-1) + m*(m-1)//2
            # where c = depth (current size including 0), and the m new elements contribute
            # m diffs from 0, m*(depth-1) diffs from the depth-1 non-zero elements,
            # and m*(m-1)//2 diffs among themselves.
            c = depth
            m = remaining_slots
            max_new_diffs = m + m * (c - 1) + m * (m - 1) // 2
            remaining_needed = 30 - bin(covered).count('1')
            if max_new_diffs < remaining_needed:
                return False
            
            # Try adding new elements
            last = current[-1]
            # Next element must be > last and <= max_allowed
            # Also, the new element x must satisfy:
            #   x not in diffs (diff from 0)
            #   x - y not in diffs for all y in current[1:] (diffs from non-zero elements)
            #   and all these new diffs are not already in diffs
            
            # For efficiency, start from last+1 and go up
            # But to find the minimum max, we should try smaller values first
            
            start = last + 1
            for x in range(start, max_allowed + 1):
                if time.time() > deadline:
                    return False
                
                # Check if x can be added
                new_diffs = []
                valid = True
                if x in diffs:
                    continue
                new_diffs.append(x)
                
                for y in current[1:]:  # skip 0 since we already checked x
                    d = x - y
                    if d in diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Check for duplicates among new_diffs themselves
                if len(set(new_diffs)) != len(new_diffs):
                    continue
                
                # Update covered mask
                new_covered = covered
                for d in new_diffs:
                    if 1 <= d <= 30:
                        new_covered |= (1 << (d - 1))
                
                # Recurse
                new_current = current + (x,)
                new_diffs_set = diffs | set(new_diffs)
                
                # Pruning: if we already exceeded best_max, no point
                if x > best_max[0] and new_covered != FULL_MASK:
                    continue
                
                dfs(new_current, new_diffs_set, new_covered, depth + 1, max_allowed)
            
            return False
        
        # Try with increasing max_allowed
        # Start with a reasonable lower bound
        # Minimum possible max: we need 30 distinct positive differences.
        # With k elements (including 0), max diffs = k*(k-1)/2.
        # For k=9: 36 diffs, so max >= 30 (at least, since we need diff 30).
        # Actually max must be at least 30.
        
        # Let's try max_allowed from 30 upward
        for max_allowed in range(30, 200):
            if time.time() > deadline:
                break
            # Reset best for this max_allowed
            # Actually, we want the global minimum, so let's just search with this bound
            dfs((0,), set(), 0, 1, max_allowed)
            if best_set[0] is not None:
                break
        
        return best_set[0]
    
    result = search()
    
    # Fallback: if the search didn't find anything, use a known construction
    if result is None:
        # Use the greedy approach as fallback
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
        result = a
    
    # Verify the result
    def verify(s):
        s = sorted(s)
        if 0 not in s:
            return False
        if len(s) > 32:
            return False
        if max(s) > 10**6:
            return False
        d_list = [y - x for i, x in enumerate(s) for y in s[i+1:]]
        if len(set(d_list)) != len(d_list):
            return False
        for d in range(1, 31):
            if d not in d_list:
                return False
        return True
    
    if not verify(result):
        # Fallback to greedy
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
        result = a
    
    return result
