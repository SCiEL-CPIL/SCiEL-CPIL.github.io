import time
import sys

def construct():
    start = time.time()
    deadline = start + 85.0

    N = 128
    # 1. Structural Decomposition
    # We know that {65, ..., 128} is always valid and large.
    # Let H = {65, ..., 128}. Size 64.
    # Any a, b in H have a+b > 128, so no sums are in S.
    # Thus, we can always include H.
    # The problem reduces to finding a valid subset L of {1, ..., 64}
    # such that L U H is valid.
    
    # Constraint for L U H:
    # 1. L must be internally valid (no a,b,c in L with sums in L).
    # 2. No a,b in L such that a+b in H. (Since a,b < 65, a+b <= 126. If a+b >= 65, it's in H).
    # 3. No a in L, b,c in H such that a+b, a+c, b+c in S.
    #    b,c in H => b+c > 128 => b+c NOT in S. So condition 3 is automatically satisfied!
    #    Wait, let's re-read: "a+b, a+c and b+c are also selected".
    #    If b,c in H, b+c > 128, so b+c is not in S. Thus the triple (a,b,c) is NOT forbidden.
    #    So, the ONLY interaction between L and H is:
    #    We cannot have a,b in L such that a+b is in H.
    #    Because if a,b in L and a+b in H (which is in S), then we have a pair a,b whose sum is in S.
    #    Does this create a forbidden triple?
    #    The forbidden triple requires THREE distinct members a,b,c such that a+b, a+c, b+c are in S.
    #    If we just have a,b in L and a+b in H, do we have a forbidden triple?
    #    Not necessarily. We need a third element c.
    #    If c is in L: we need a+c in S and b+c in S.
    #    If c is in H: b+c > 128 (not in S). So c cannot be in H.
    #    So, a forbidden triple involving H elements must have c in L?
    #    Let's check: a in L, b in L, c in L. (Internal to L).
    #    a in L, b in L, c in H. b+c > 128. Not forbidden.
    #    a in L, b in H, c in H. b+c > 128. Not forbidden.
    #    a in H, b in H, c in H. Not forbidden.
    #    
    #    So, the constraint is:
    #    1. L is a "sum-free-like" set within 1..64? No, just "no triple in L with sums in L".
    #    2. AND, for any a,b in L, if a+b is in H, we must ensure there is no c in L such that a+c in S and b+c in S.
    #       Wait, if a+b in H, then a+b is in S.
    #       If there exists c in L such that a+c in S and b+c in S, then {a,b,c} is a forbidden triple.
    #       So, if a+b in H, we must NOT have any c in L such that a+c in S and b+c in S.
    #       Note that a+c and b+c could be in L or H.
    #       If a+c in H, that's fine (it's in S).
    #       If a+c in L, that's fine (it's in S).
    #       So the condition is: If a+b in H, then there is NO c in L such that (a+c in S) AND (b+c in S).
    
    # This is complex. Let's simplify by observing the range.
    # H = {65..128}.
    # If a,b in L (1..64), a+b is in 2..128.
    # If a+b <= 64, a+b is NOT in H. It might be in L or not in S.
    # If a+b >= 65, a+b IS in H (and thus in S).
    
    # So, if we pick L, we must ensure:
    # A. No triple a,b,c in L such that a+b, a+c, b+c are in L.
    # B. For any pair a,b in L with a+b >= 65:
    #    There is no c in L such that a+c in S and b+c in S.
    #    Since S = L U H, "x in S" means "x in L or x in H".
    #    x in H iff x >= 65.
    #    So "a+c in S" means "a+c in L or a+c >= 65".
    #    "b+c in S" means "b+c in L or b+c >= 65".
    
    # This looks like a hypergraph independent set problem on L={1..64}.
    # Vertices: 1..64.
    # Hyperedges (forbidden sets):
    # 1. {a,b,c} where a,b,c in L and a+b in L, a+c in L, b+c in L.
    # 2. {a,b,c} where a,b,c in L, a+b >= 65, and (a+c in L or a+c>=65) and (b+c in L or b+c>=65).
    #    Note: a,b,c distinct.
    
    # Let's precompute the forbidden hyperedges for L.
    # There are 64 vertices. Number of triples is C(64,3) ~ 41664.
    # We can iterate through all triples and mark them as forbidden if the condition holds.
    
    # However, condition 2 depends on which elements are in L (since "a+c in L" depends on L).
    # This is a "forbidden configuration" that depends on the set itself.
    # Specifically, {a,b,c} is forbidden IF:
    # (a+b in L and a+c in L and b+c in L)  OR
    # (a+b >= 65 and (a+c in L or a+c >= 65) and (b+c in L or b+c >= 65))
    
    # This is not a static hypergraph! The validity of a triple depends on the other elements in the set.
    # Example: If a+c is in L, it contributes to the condition. If a+c is NOT in L (and < 65), it doesn't.
    
    # This makes it a "dynamic" constraint.
    # However, we can still use Branch and Bound.
    # State: Current set L_cur, Remaining candidates R.
    # When we add x to L_cur, we must check if it creates a forbidden triple with any pair in L_cur.
    # Also, adding x might make some previously "safe" triples forbidden?
    # No, the constraint is "No three distinct members...".
    # If we have a set S, it is valid if NO triple in S is forbidden.
    # A triple {a,b,c} is forbidden in S if the condition holds given S.
    # The condition for {a,b,c} to be forbidden depends on whether a+b, a+c, b+c are in S.
    # Since S = L_cur U H (fixed part) U {new elements}, and H is fixed...
    
    # Let's re-evaluate:
    # S = L U H.
    # Triple {a,b,c} subset of L is forbidden if:
    # (a+b in L) AND (a+c in L) AND (b+c in L)
    # OR
    # (a+b in H) AND (a+c in L U H) AND (b+c in L U H)
    
    # Note that a+b in H is a static property of a,b (since a,b < 65, a+b is either <65 or >=65).
    # a+c in L U H is dynamic: it's true if a+c in L or a+c >= 65.
    
    # So, if we build L incrementally:
    # When we add x to L, we form new triples {x, a, b} where a,b are already in L.
    # We must check if {x, a, b} is forbidden GIVEN the current L (which now includes x).
    # Wait, the condition "a+c in L" depends on the FINAL L.
    # If we add x, and later we don't add y, then "y in L" is false.
    # But "x in L" is true.
    # So, the validity of a triple {a,b,c} can change as we add more elements to L?
    # Yes!
    # Example: a=1, b=2, c=3.
    # If L={1,2,3}, a+b=3 (in L), a+c=4 (not in L, <65), b+c=5 (not in L, <65).
    # Condition 1: 3 in L (T), 4 in L (F), 5 in L (F). -> Not forbidden by 1.
    # Condition 2: a+b=3 < 65. -> Not forbidden by 2.
    # So {1,2,3} is valid.
    # Now add 4 to L. L={1,2,3,4}.
    # Check {1,2,3} again.
    # a+b=3 (in L), a+c=4 (in L now!), b+c=5 (not in L).
    # Still not forbidden.
    # Now add 5. L={1,2,3,4,5}.
    # {1,2,3}: 3 in L, 4 in L, 5 in L. -> FORBIDDEN!
    # So, adding 5 made {1,2,3} invalid.
    
    # This means we CANNOT just check new triples when adding an element.
    # We must ensure that the FINAL set is valid.
    # This is a global constraint.
    
    # However, Branch and Bound can still work.
    # We can use a "propagation" step.
    # If we have a partial assignment, we can check if it's "potentially valid".
    # But the "forbidden" status of a triple depends on the final set.
    
    # Alternative approach:
    # Since the domain is small (64), we can use a SAT-like approach or just optimized search.
    # But 2^64 is too big.
    
    # Let's look at the structure again.
    # H = 65..128.
    # Is it possible to include MORE than 64?
    # The current best is 81.
    # 81 - 64 = 17 elements from 1..64.
    # So L has size 17.
    
    # Let's try to find the maximum size of L.
    # We can use a simple backtracking with strong pruning.
    # Pruning 1: If we have a set L_cur, and we want to add x.
    # We must check if adding x creates a forbidden triple with any pair in L_cur,
    # assuming that NO other elements will be added to L (worst case for "in L" checks? No, best case?).
    # Actually, if a condition requires "y in L", and y is not in L_cur and not added later, then "y in L" is False.
    # If we assume we will NOT add any more elements, we can check if the current set + x is valid under that assumption.
    # If it's valid under the assumption "no more elements", it might become invalid if we add more.
    # So this is a "safe" check for pruning? No, if it's invalid now, it might become valid?
    # No, adding elements only adds more "in L" truths.
    # If a triple is forbidden because "a+b in L", and a+b is in L_cur, it stays forbidden.
    # If a triple is NOT forbidden because "a+c NOT in L", and we later add a+c, it MIGHT become forbidden.
    # So, if a set is invalid under "current L", it is invalid in any superset.
    # Wait.
    # Condition: (a+b in L) AND (a+c in L) AND (b+c in L).
    # If this is True, it's Forbidden.
    # If we add more elements, "in L" stays True. So it remains Forbidden.
    # So, if a triple is Forbidden in L_cur, it is Forbidden in any superset.
    # Therefore, we can prune if L_cur contains a forbidden triple (under the assumption that L_cur is the final set? No, under the assumption that L_cur is a subset of the final set).
    # Actually, if {a,b,c} is in L_cur, and a+b in L_cur, a+c in L_cur, b+c in L_cur, then it is forbidden in ANY superset.
    # So we can prune if L_cur contains a "closed" forbidden triple.
    
    # What about Condition 2?
    # (a+b >= 65) AND (a+c in L or a+c >= 65) AND (b+c in L or b+c >= 65).
    # If a+b >= 65 (static), and (a+c >= 65 or a+c in L_cur) and (b+c >= 65 or b+c in L_cur).
    # If this holds for L_cur, it holds for any superset.
    # So, if L_cur contains a triple {a,b,c} satisfying this, it is forbidden in any superset.
    
    # So, we can define a "local forbidden" check:
    # A triple {a,b,c} is "locally forbidden" in L_cur if:
    # 1. a+b in L_cur and a+c in L_cur and b+c in L_cur.
    # OR
    # 2. a+b >= 65 and (a+c in L_cur or a+c >= 65) and (b+c in L_cur or b+c >= 65).
    
    # If L_cur contains a locally forbidden triple, we prune.
    # This is a necessary condition for validity. It is not sufficient (as shown by the 1,2,3,4,5 example).
    # But it is a strong pruning rule.
    
    # Let's implement Branch and Bound with this pruning.
    # We will also use a heuristic ordering: add elements that are "least likely to cause conflicts".
    # Elements with large sums are safer (push sums into H or out of range).
    # Elements in 33..64 are safer than 1..32.
    
    # Let's try to search for L.
    # We want to maximize |L|.
    
    # Precomputation:
    # For each pair (a,b) in 1..64, precompute:
    # sum_ab = a+b
    # in_H = sum_ab >= 65
    # sum_ac, sum_bc for each c? No, too many.
    
    # We will iterate through candidates in descending order (64 down to 1).
    # This tends to pick "safe" elements first.
    
    best_L = []
    
    # Bitmask for L_cur? 64 bits fits in Python int.
    # But we need to check triples.
    
    # Let's use a recursive search.
    # State: current_mask (int), current_size, next_candidate_index
    # To speed up, we can use a "conflict mask" for each element?
    # No, the conflict depends on the pair.
    
    # Optimization:
    # If we pick a, b, then for any c, we must check if {a,b,c} is locally forbidden.
    # We can precompute for each pair (a,b) the set of c's that would form a locally forbidden triple GIVEN the current mask?
    # No, "in L_cur" changes.
    
    # However, we can precompute the STATIC parts.
    # For pair (a,b):
    # Type 1: Requires a+b, a+c, b+c in L.
    # Type 2: Requires a+b>=65, and (a+c in L or a+c>=65), (b+c in L or b+c>=65).
    
    # Let's just do the check on the fly. 64 is small.
    
    def is_locally_forbidden(a, b, c, mask):
        # a,b,c are 1-based values in 1..64
        # mask has bits set for elements in L_cur
        # Check Type 1
        if (mask >> (a+b-1)) & 1:
            if (mask >> (a+c-1)) & 1:
                if (mask >> (b+c-1)) & 1:
                    return True
        # Check Type 2
        if a + b >= 65:
            # a+c in L or a+c >= 65
            cond_ac = ((mask >> (a+c-1)) & 1) or (a + c >= 65)
            cond_bc = ((mask >> (b+c-1)) & 1) or (b + c >= 65)
            if cond_ac and cond_bc:
                return True
        return False

    # We need to check all triples in the current mask?
    # No, only new triples involving the newly added element.
    # When adding x to mask, we check all pairs (a,b) in mask.
    # If is_locally_forbidden(a, b, x, mask | (1<<(x-1))) is True, prune.
    
    # Wait, the mask passed to is_locally_forbidden should include x?
    # Yes, because we are checking if the triple {a,b,x} is forbidden in the set that includes x.
    # And the "in L" checks should use the mask that includes x.
    
    def search(mask, size, next_val, best_size_ref):
        # next_val is the next candidate to consider (descending)
        nonlocal best_L
        if time.time() > deadline - 5:
            return
        
        if size > best_size_ref[0]:
            best_size_ref[0] = size
            # Reconstruct L from mask
            # For efficiency, we can just store the mask
            best_L[0] = mask
        
        if next_val < 1:
            return
        
        # Pruning: if size + (next_val) <= best_size, return
        # Max we can add is next_val (if we add all remaining)
        if size + next_val <= best_size_ref[0]:
            return
        
        # Option 1: Exclude next_val
        search(mask, size, next_val - 1, best_size_ref)
        
        # Option 2: Include next_val
        x = next_val
        new_mask = mask | (1 << (x - 1))
        
        # Check if adding x creates a locally forbidden triple
        valid = True
        # Iterate over all pairs in mask
        # We can iterate over set bits in mask
        m = mask
        while m:
            # Get lowest set bit
            lb = m & (-m)
            a = lb.bit_length() # 1-based index? No, bit_length gives index+1 if we use 1<<i for i=0..63
            # Wait, 1 << (v-1). So bit i corresponds to v=i+1.
            # lb.bit_length() gives the position of the highest set bit + 1.
            # So if lb = 1 << 5, bit_length is 6. Value is 6.
            # So a = lb.bit_length()
            
            # Remove lb from m
            m ^= lb
            
            # Now find b in m
            m2 = m
            while m2:
                lb2 = m2 & (-m2)
                b = lb2.bit_length()
                m2 ^= lb2
                
                if a == b or b == x or a == x:
                    continue
                if a > b:
                    continue # Ensure a < b to avoid duplicates? Not strictly needed, just efficiency.
                
                # Check triple {a, b, x}
                if is_locally_forbidden(a, b, x, new_mask):
                    valid = False
                    break
            
            if not valid:
                break
        
        if valid:
            search(new_mask, size + 1, next_val - 1, best_size_ref)

    best_size_ref = [0]
    best_L = [0]
    
    # Start search
    # Candidates: 64 down to 1
    search(0, 0, 64, best_size_ref)
    
    # best_L[0] is the mask for L
    L_mask = best_L[0]
    
    # Construct final set
    # H = 65..128
    result = list(range(65, 129))
    
    # Add L
    for i in range(64):
        if (L_mask >> i) & 1:
            result.append(i + 1)
    
    # Final validation
    # Check if result is valid
    S = set(result)
    valid = True
    # Check all triples
    # This is O(N^3) = 128^3 ~ 2M. Might be slow in Python.
    # But we only do it once.
    # We can optimize by iterating pairs and checking sums.
    
    # Let's do a fast check using the same logic as the seed, but on the final set.
    # Actually, the seed's _is_valid_mask is good.
    # Let's use it.
    
    def _is_valid_mask(mask):
        for i in range(128):
            if not (mask >> i) & 1:
                continue
            for j in range(i + 1, 128):
                if not (mask >> j) & 1:
                    continue
                if i + j + 1 >= 128:
                    continue
                if not (mask >> (i + j + 1)) & 1:
                    continue
                candidates = mask & (mask >> (i + 1)) & (mask >> (j + 1))
                if candidates & ~((1 << (j + 1)) - 1):
                    return False
        return True

    # Convert result to mask
    final_mask = 0
    for v in result:
        final_mask |= (1 << (v - 1))
        
    if not _is_valid_mask(final_mask):
        # Fallback to H only
        result = list(range(65, 129))
    
    return result

# Run construct to test? No, just define it.
# But we need to make sure it's fast.
# The search might be slow if not pruned well.
# Let's add a timeout guard inside search.

def construct():
    start = time.time()
    deadline = start + 85.0

    def is_locally_forbidden(a, b, c, mask):
        if (mask >> (a+b-1)) & 1:
            if (mask >> (a+c-1)) & 1:
                if (mask >> (b+c-1)) & 1:
                    return True
        if a + b >= 65:
            cond_ac = ((mask >> (a+c-1)) & 1) or (a + c >= 65)
            cond_bc = ((mask >> (b+c-1)) & 1) or (b + c >= 65)
            if cond_ac and cond_bc:
                return True
        return False

    best_size_ref = [0]
    best_L_mask = [0]
    
    def search(mask, size, next_val):
        if time.time() > deadline - 5:
            return
        
        if size > best_size_ref[0]:
            best_size_ref[0] = size
            best_L_mask[0] = mask
        
        if next_val < 1:
            return
        
        if size + next_val <= best_size_ref[0]:
            return
        
        # Exclude
        search(mask, size, next_val - 1)
        
        # Include
        x = next_val
        new_mask = mask | (1 << (x - 1))
        
        valid = True
        m = mask
        while m:
            lb = m & (-m)
            a = lb.bit_length()
            m ^= lb
            
            m2 = m
            while m2:
                lb2 = m2 & (-m2)
                b = lb2.bit_length()
                m2 ^= lb2
                
                if a >= b:
                    continue
                if a == x or b == x:
                    continue
                
                if is_locally_forbidden(a, b, x, new_mask):
                    valid = False
                    break
            
            if not valid:
                break
        
        if valid:
            search(new_mask, size + 1, next_val - 1)

    search(0, 0, 64)
    
    L_mask = best_L_mask[0]
    
    result = list(range(65, 129))
    for i in range(64):
        if (L_mask >> i) & 1:
            result.append(i + 1)
    
    def _is_valid_mask(mask):
        for i in range(128):
            if not (mask >> i) & 1:
                continue
            for j in range(i + 1, 128):
                if not (mask >> j) & 1:
                    continue
                if i + j + 1 >= 128:
                    continue
                if not (mask >> (i + j + 1)) & 1:
                    continue
                candidates = mask & (mask >> (i + 1)) & (mask >> (j + 1))
                if candidates & ~((1 << (j + 1)) - 1):
                    return False
        return True

    final_mask = 0
    for v in result:
        final_mask |= (1 << (v - 1))
        
    if not _is_valid_mask(final_mask):
        result = list(range(65, 129))
    
    return result

