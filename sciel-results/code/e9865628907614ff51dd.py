"""Rewritten program: maximize subset of [1,180] with no distinct triple a,b,c s.t. 1/a=1/b+1/c."""
# EVOLVE-BLOCK-START
def construct():
    N = 180
    N_RED = 90
    
    # 1. Identify all odd numbers in [1, N]
    odds = set(range(1, N + 1, 2))
    
    # 2. Work on reduced space [1, 90] for even numbers (mapped to x = e/2)
    # We want to find a subset S_red of [1, 90] such that:
    # A. No triple (x1, x2, x3) in S_red satisfies 1/x1 = 1/x2 + 1/x3
    # B. No pair (x1, x2) in S_red satisfies 2*x1*x2/(x2-x1) is an odd integer in [1, 180]
    
    # Precompute forbidden triples in [1, 90]
    forbidden_triples = set()
    for a in range(1, N_RED + 1):
        for b in range(a + 1, N_RED + 1):
            num = a * b
            den = b - a
            if num % den == 0:
                c = num // den
                if 1 <= c <= N_RED and c != a and c != b:
                    forbidden_triples.add(frozenset((a, b, c)))
                    
    # Precompute forbidden pairs in [1, 90] due to odd interactions
    forbidden_pairs = set()
    for x1 in range(1, N_RED + 1):
        for x2 in range(x1 + 1, N_RED + 1):
            num = 2 * x1 * x2
            den = x2 - x1
            if num % den == 0:
                val = num // den
                if val % 2 == 1 and 1 <= val <= N:
                    forbidden_pairs.add(frozenset((x1, x2)))
                    
    # Greedy construction of S_red
    # Start with all elements
    included = set(range(1, N_RED + 1))
    
    # Count violations for each element
    # We will iteratively remove elements that are in the most active violations
    
    def get_violations(s):
        """Yield (type, elements) for all violations in s"""
        # Check triples
        for tri in forbidden_triples:
            if tri.issubset(s):
                yield ('tri', tri)
        # Check pairs
        for pair in forbidden_pairs:
            if pair.issubset(s):
                yield ('pair', pair)
                
    # Iterative removal
    changed = True
    while changed:
        changed = False
        # Find the most "crowded" violation
        # We can just pick the first violation and remove the element with highest degree in remaining violations
        # Better: Count how many violations each element is in
        counts = {x: 0 for x in included}
        
        # To speed up, we can just iterate and count
        # But N_RED=90 is small.
        
        # Find one bad triple/pair to trigger removal
        bad_item = None
        for item in get_violations(included):
            bad_item = item
            break
            
        if bad_item is None:
            break
            
        # Count violations for each element in the bad item
        # Actually, let's count total violations for ALL elements in included to find the global best removal
        # This is O(V * E) which might be slow if V is large, but N=90 is small.
        
        # Let's just count for elements in the current bad_item first? 
        # No, removing an element not in the current bad_item might be better if it's in many other violations.
        # But usually, fixing the immediate violation is good.
        
        # Let's count violations for all elements in included
        # Optimization: Only count violations that are currently present.
        
        # We'll just pick the element in the current bad_item that is in the most total violations.
        
        # Re-count violations for elements in bad_item[1]
        candidates = list(bad_item[1])
        
        # Count how many violations each candidate is involved in
        cand_counts = {c: 0 for c in candidates}
        
        # We need to iterate through all violations to count properly for these candidates
        for v_type, v_elems in get_violations(included):
            for c in candidates:
                if c in v_elems:
                    cand_counts[c] += 1
                    
        # Remove the candidate with max count
        if cand_counts:
            best_to_remove = max(cand_counts, key=cand_counts.get)
            included.discard(best_to_remove)
            changed = True
            
    # Now try to add back elements
    excluded = set(range(1, N_RED + 1)) - included
    
    def is_valid_red(s):
        for tri in forbidden_triples:
            if tri.issubset(s):
                return False
        for pair in forbidden_pairs:
            if pair.issubset(s):
                return False
        return True
        
    improved = True
    while improved:
        improved = False
        for x in sorted(excluded, reverse=True):
            if x in included:
                continue
            if is_valid_red(included | {x}):
                included.add(x)
                excluded.discard(x)
                improved = True
                
    # Construct final set
    final_set = set(odds)
    for x in included:
        final_set.add(2 * x)
        
    return sorted(final_set)
# EVOLVE-BLOCK-END
