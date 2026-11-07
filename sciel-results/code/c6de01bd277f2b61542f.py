"""Maximum Independent Set via Branch-and-Bound with bitset acceleration."""
import time
import math
from functools import reduce

def _subset_sums(S):
    """Return set of all nonempty subset sums of S."""
    sums = set()
    for x in S:
        new_sums = set()
        for s in sums:
            new_sums.add(s + x)
        sums.update(new_sums)
        sums.add(x)
    return sums

def _check_pair_conflict(a, b):
    """Check if adding both a and b to an empty set creates a divisibility conflict."""
    # Subset sums of {a, b}: a, b, a+b
    sums = {a, b, a + b}
    total = a + b
    for s in sorted(sums):
        if s == 0:
            continue
        multiple = 2 * s
        while multiple <= total:
            if multiple in sums:
                return True
            multiple += s
    return False

def _check_conflict_with_set(x, S):
    """Check if adding x to set S creates any divisibility conflict among subset sums."""
    old_sums = _subset_sums(S)
    new_sums = set(old_sums)
    for s in old_sums:
        new_sums.add(s + x)
    new_sums.add(x)
    new_total = sum(S) + x
    # Check all pairs in new_sums for divisibility
    sums_list = sorted(new_sums)
    sums_set = new_sums
    for s in sums_list:
        if s == 0:
            continue
        multiple = 2 * s
        while multiple <= new_total:
            if multiple in sums_set:
                return True
            multiple += s
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    n = 256  # Numbers 1..256
    nodes = list(range(1, n + 1))
    
    # Build conflict graph: edge (i,j) if {i,j} is not a valid set
    # Two nodes conflict if their subset sums create a divisibility relation
    # For pair {a,b}: sums are a, b, a+b. Check if any divides another.
    # a divides b? a divides a+b? b divides a? b divides a+b? a+b divides a or b? (impossible since a+b > a,b)
    # So check: a|b, b|a, a|(a+b) which is always true, b|(a+b) which is always true
    # Wait: a always divides a+b. So ANY two numbers a,b where a|b or b|a conflict.
    # But also need to check: does a divide b? If a|b, then a is a subset sum and b is a subset sum, and a|b. Conflict!
    # Does b divide a? Same.
    # What about a dividing a+b? Always true. But a and a+b are both subset sums, so a|(a+b) is ALWAYS a conflict?
    # Wait no. The condition is: no two DISTINCT resulting values may divide one another.
    # a and a+b are distinct (since b>0), and a divides a+b. So EVERY pair of positive integers conflicts?!
    
    # Let me re-read: "Form every NONEMPTY subset sum; no two distinct resulting values may divide one another."
    # For set {a, b}: subset sums are a, b, a+b (assuming a != b and a,b > 0).
    # a divides a+b? Yes, always. So a and a+b are two distinct subset sums where one divides the other.
    # This means NO set of size >= 2 can be valid?!
    
    # Wait, that can't be right because the current program finds sets of size ~7.
    # Let me re-read more carefully...
    
    # "no two distinct resulting values may divide one another"
    # Hmm, maybe I'm misunderstanding. Let me check: for {3, 5}, sums are 3, 5, 8.
    # 3 divides 5? No. 3 divides 8? No. 5 divides 3? No. 5 divides 8? No. 8 divides 3? No. 8 divides 5? No.
    # So {3,5} is valid! I made an error: a does NOT always divide a+b in the sense that we need a to be a factor.
    # a divides a+b means (a+b) % a == 0, which means b % a == 0. So a|(a+b) iff a|b.
    # Similarly b|(a+b) iff b|a.
    # So the conflict for pair {a,b} is: a|b or b|a.
    
    # OK so for pairs, conflict iff one divides the other.
    
    # For larger sets, it's more complex because subset sums interact.
    # But for the MIS formulation, we need to check if a SET is valid, not just pairs.
    # The conflict graph approach only works if validity is pairwise (i.e., a set is valid iff all pairs are valid).
    # Is that true? No! A set can have all pairs valid but still have a 3-element subset sum conflict.
    # Example: need to find such a case.
    
    # Actually, let's think: if all pairs are valid (no element divides another), can there still be a conflict?
    # Consider {2, 3, 7}. Pairs: 2∤3, 2∤7, 3∤7. All good.
    # Subset sums: 2,3,7,5,9,10,12.
    # 2|4? 4 not in sums. 2|6? 6 not in sums. 2|8? 8 not in sums. 2|10? 10 IS in sums! 2|10. Conflict!
    # So {2,3,7} is invalid even though all pairs are valid.
    
    # This means the problem is NOT a standard MIS problem. The conflict is not pairwise.
    # We need a different approach.
    
    # Alternative: Use the incremental approach but with a smarter search strategy.
    # Let's use a different structural approach: 
    # 1. Precompute which numbers are "compatible" with large sets
    # 2. Use a backtracking search with strong pruning based on subset sum properties
    
    # Actually, let me try a completely different angle:
    # Key insight: if a is in the set, then no subset sum can be a multiple of a (except a itself).
    # This heavily constrains what other elements can be added.
    
    # New strategy: Use a constraint-based backtracking with memoization on the "forbidden sums" set.
    # When we add element x, all existing subset sums s become s+x as new sums.
    # The constraint is: for any two subset sums p, q in the final set, p ∤ q and q ∤ p.
    
    # Let me try a different structural approach: 
    # Greedy with strong lookahead and multiple diverse restarts using a different ordering strategy.
    # Use the concept of "sum-free" sets and number theory to guide selection.
    
    # Actually, the most promising structural change: use a SAT-like approach or 
    # integer programming relaxation. But without external solvers, let's try:
    
    # APPROACH: Backtracking with aggressive pruning using the fact that
    # if we have current subset sums S, and we want to add x, then:
    # - x must not divide any element of S, and no element of S must divide x
    # - For each s in S, s+x must not be a multiple of any element in S ∪ {x}, 
    #   and no element in S ∪ {x} must divide s+x
    # - Also s+x must not divide any other element in S ∪ {x} ∪ {s+x for all s}
    
    # Let me implement a clean backtracking with these prunings.
    
    def _can_add(x, current_sums, current_total):
        """Check if x can be added to a set with given subset sums."""
        # x itself must not conflict with existing sums
        for s in current_sums:
            if s % x == 0 or x % s == 0:
                return False
        # New sums: x and s+x for all s in current_sums
        new_sums = set(current_sums)
        new_sums.add(x)
        for s in current_sums:
            new_sums.add(s + x)
        new_total = current_total + x
        
        # Check all pairs in new_sums
        sums_list = sorted(new_sums)
        for i in range(len(sums_list)):
            s = sums_list[i]
            if s == 0:
                continue
            multiple = 2 * s
            while multiple <= new_total:
                if multiple in new_sums:
                    return False
                multiple += s
        return True
    
    best_set = []
    best_size = 0
    
    def _backtrack(remaining, current, current_sums, current_total, depth, deadline):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
        
        if not remaining or len(current) >= 30:
            return
        
        # Pruning: upper bound on possible additions
        # Each new element must be > current_total (to avoid small divisibility)
        # This is a heuristic bound
        if len(current) + len(remaining) <= best_size:
            return
        
        # Try adding elements from remaining
        for idx, x in enumerate(remaining):
            if time.time() > deadline:
                return
            
            if _can_add(x, current_sums, current_total):
                # Add x
                new_sums = set(current_sums)
                new_sums.add(x)
                for s in current_sums:
                    new_sums.add(s + x)
                new_total = current_total + x
                
                current.append(x)
                _backtrack(remaining[idx+1:], current, new_sums, new_total, depth + 1, deadline)
                current.pop()
    
    # Multiple diverse starting strategies
    strategies = []
    
    # Strategy 1: Start with large primes (less likely to divide others)
    primes = [x for x in range(2, 257) if all(x % i != 0 for i in range(2, int(math.isqrt(x)) + 1))]
    strategies.append(sorted(primes, reverse=True))
    
    # Strategy 2: Start with numbers with few divisors
    def divisor_count(x):
        if x < 2:
            return 1
        count = 2
        for i in range(2, int(math.isqrt(x)) + 1):
            if x % i == 0:
                count += 2
        return count
    
    by_divisors = sorted(range(1, 257), key=lambda x: divisor_count(x), reverse=True)
    strategies.append(by_divisors)
    
    # Strategy 3: Large numbers first
    strategies.append(list(range(256, 0, -1)))
    
    # Strategy 4: Random
    import random
    for seed in [42, 123, 456, 789]:
        rand_list = list(range(1, 257))
        random.Random(seed).shuffle(rand_list)
        strategies.append(rand_list)
    
    for strategy in strategies:
        if time.time() > deadline:
            break
        
        # Use backtracking with this ordering
        _backtrack(strategy, [], set(), 0, 0, deadline)
    
    # Final validation
    if best_set:
        final_sums = _subset_sums(best_set)
        total = sum(best_set)
        valid = True
        sums_list = sorted(final_sums)
        for s in sums_list:
            if s == 0:
                continue
            multiple = 2 * s
            while multiple <= total:
                if multiple in final_sums:
                    valid = False
                    break
                multiple += s
            if not valid:
                break
        
        if not valid:
            # Fallback: greedy with large first
            current = []
            current_sums = set()
            current_total = 0
            for x in range(256, 0, -1):
                if _can_add(x, current_sums, current_total):
                    new_sums = set(current_sums)
                    new_sums.add(x)
                    for s in current_sums:
                        new_sums.add(s + x)
                    current.append(x)
                    current_sums = new_sums
                    current_total += x
            best_set = current
    
    return [int(x) for x in best_set]
