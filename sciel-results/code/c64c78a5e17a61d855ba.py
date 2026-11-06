import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin
    
    N = 180
    
    # Precompute conflicts
    # triples contains (a, b, c) such that 1/a = 1/b + 1/c with a < b < c
    # We need to know for each element x, which pairs (u, v) form a triple with x.
    # Let's build a list of all triples first.
    triples = []
    
    for a in range(1, N + 1):
        a2 = a * a
        # Find factor pairs d, e of a^2 such that d < e
        # b = a + d, c = a + e
        # We need b, c <= N
        # Also b != c, which is guaranteed by d != e
        for d in range(1, int(a2**0.5) + 1):
            if a2 % d != 0:
                continue
            e = a2 // d
            if d >= e:
                continue
            b = a + d
            c = a + e
            if b <= N and c <= N:
                # Ensure distinctness: a, b, c are distinct because d>0 => b>a, e>d => c>b?
                # c - b = e - d > 0. So a < b < c.
                triples.append((a, b, c))
                
    # Build conflict map for efficient checking
    # conflicts[x] = list of (u, v) such that {x, u, v} is a triple
    conflicts = {i: [] for i in range(1, N + 1)}
    for a, b, c in triples:
        # Triple is {a, b, c}
        conflicts[a].append((b, c))
        conflicts[b].append((a, c))
        conflicts[c].append((a, b))
        
    # Helper to check if x can be added to S
    def can_add(x, S):
        for u, v in conflicts[x]:
            if u in S and v in S:
                return False
        return True

    # Helper to check if a set of elements can be added to S
    def can_add_multiple(elements, S):
        # Check if all elements in 'elements' can be added to S simultaneously
        # This is equivalent to checking if S U elements is valid.
        # Since we are adding multiple, we must check conflicts between S and elements,
        # and between elements themselves.
        # However, in our swap heuristic, we usually remove one element 'rem' and add two 'add1', 'add2'.
        # The new set is (S - {rem}) U {add1, add2}.
        # We need to ensure:
        # 1. add1 and add2 are not in S (they aren't, usually).
        # 2. No triple in (S - {rem}) U {add1, add2}.
        
        S_new = set(S)
        S_new.discard(rem)
        S_new.add(add1)
        S_new.add(add2)
        
        # Check all triples
        # Optimization: Only check triples involving add1 or add2 or affected by rem?
        # With N=180 and ~few hundred triples, full check is fast enough for local search steps.
        for a, b, c in triples:
            if a in S_new and b in S_new and c in S_new:
                return False
        return True

    def local_search(S, time_limit):
        S = set(S)
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            # 1. Try to add any element not in S
            for x in range(1, N + 1):
                if x not in S:
                    if can_add(x, S):
                        S.add(x)
                        changed = True
                        # Continue checking others in this pass
            if changed:
                continue # Restart loop to ensure all possible adds are done
            
            # 2. Try swap moves: Remove one element 'rem', Add two elements 'add1', 'add2'
            # We iterate over all possible removals
            for rem in list(S):
                # Find elements not in S that could potentially be added if rem is removed
                # An element y can be added if all its conflicts with S are satisfied,
                # OR if its only conflict with S is via rem.
                # Actually, simpler: just try all pairs of candidates outside S.
                # To make it efficient, we can precompute for each rem, which elements are "blocked" by rem.
                # But N is small. Let's just iterate.
                
                # Candidates for addition: elements not in S
                # We want to find add1, add2 such that (S - {rem}) U {add1, add2} is valid.
                # This implies add1 and add2 must not form a triple with any element in S - {rem}.
                
                # Optimization: For a given rem, identify elements x not in S such that
                # x is only blocked by rem.
                # If x is blocked by some other y in S, removing rem won't help add x.
                
                potential_adds = []
                for x in range(1, N + 1):
                    if x in S:
                        continue
                    # Check if x is blocked by S.
                    # If it is blocked, is it blocked ONLY by rem?
                    # If x is not blocked by S, it would have been added in step 1.
                    # So x is blocked by S.
                    # We need to see if removing rem unblocks x.
                    
                    blocked_by = []
                    for u, v in conflicts[x]:
                        if u in S and v in S:
                            # x is blocked by pair (u,v)
                            # If rem is in (u,v), then removing rem might help?
                            # No, if u,v are both in S, and rem is one of them, say u=rem,
                            # then v is still in S. The triple is (x, rem, v).
                            # If we remove rem, the triple (x, rem, v) is broken.
                            # BUT, we need to check if x is blocked by OTHER pairs too.
                            
                            if rem in (u, v):
                                # This specific conflict is resolved by removing rem.
                                pass
                            else:
                                # This conflict remains even if rem is removed.
                                # So x is still blocked.
                                blocked_by.append((u, v))
                                break
                    if not blocked_by:
                        # x is only blocked by conflicts involving rem (or not blocked at all, which is handled in step 1)
                        # Wait, if x was not blocked at all, it would be in S.
                        # So x is blocked. If removing rem resolves all conflicts, x is a candidate.
                        potential_adds.append(x)
                
                # Now try to pick two distinct elements from potential_adds
                # such that they don't conflict with each other or with S - {rem}.
                # Note: potential_adds elements are already checked against S - {rem} implicitly?
                # Not quite. We checked that they aren't blocked by S \ {rem}.
                # But we must also ensure add1 and add2 don't form a triple with each other and S.
                
                if len(potential_adds) >= 2:
                    # Try all pairs in potential_adds
                    for i in range(len(potential_adds)):
                        for j in range(i+1, len(potential_adds)):
                            add1 = potential_adds[i]
                            add2 = potential_adds[j]
                            
                            # Check if (S - {rem}) U {add1, add2} is valid
                            # We know add1 and add2 are not blocked by S \ {rem}.
                            # We need to check:
                            # 1. Do add1 and add2 conflict with each other? (i.e., form a triple with some z in S \ {rem} or each other? No, triple needs 3 distinct elements).
                            # A triple is (x, y, z).
                            # If we add add1, add2, we need to ensure no triple exists in the new set.
                            # Triples involving add1: (add1, u, v). We know u,v in S\{rem} is safe.
                            # What if u=add2? Then we have (add1, add2, v). If v is in S\{rem}, it's a conflict.
                            # Our previous check for add1 only looked at pairs in S. It did not consider add2.
                            # So we must do a full check for the new set.
                            
                            S_new = S - {rem}
                            S_new.add(add1)
                            S_new.add(add2)
                            
                            valid = True
                            for a, b, c in triples:
                                if a in S_new and b in S_new and c in S_new:
                                    valid = False
                                    break
                            
                            if valid:
                                S = S_new
                                changed = True
                                break
                        if changed:
                            break
                if changed:
                    break # Restart outer while loop
            
        return S

    best_set = set()
    
    # Attempt to solve using MILP if available
    milp_success = False
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        # Objective: maximize sum(x_i) -> minimize -sum(x_i)
        c_obj = -np.ones(N)
        
        # Constraints: for each triple, x_a + x_b + x_c <= 2
        # A_ub is the matrix for A_ub @ x <= b_ub
        A_ub = np.zeros((len(triples), N))
        b_ub = np.full(len(triples), 2)
        
        for idx, (a, b, c) in enumerate(triples):
            A_ub[idx, a-1] = 1
            A_ub[idx, b-1] = 1
            A_ub[idx, c-1] = 1
            
        # Bounds: 0 <= x_i <= 1
        bounds = Bounds(lb=np.zeros(N), ub=np.ones(N))
        
        # Integrality: all variables are integer
        integrality = np.ones(N)
        
        # Solve
        # Set time limit for solver to leave time for local search
        # scipy milp doesn't have a direct time limit param in all versions, 
        # but we can wrap it. However, it might hang.
        # Let's try it, but if it takes too long, we can't interrupt easily in standard scipy.
        # Given the risk, let's prioritize the heuristic which is more controllable.
        # But if MILP is fast, it's optimal.
        # Let's try with a very short timeout if possible? No, scipy doesn't support it directly.
        # We will rely on the heuristic which is robust.
        pass
        
    except Exception:
        pass
        
    # If MILP failed or wasn't used, use Heuristic Search
    # Strategy: Randomized Greedy Restarts + Local Search with Swaps
    
    remaining_time = deadline - time.time()
    if remaining_time < 1:
        # Not enough time for heuristic, return empty or best found (which is empty)
        # But we must return a nonempty subset.
        # Fallback to a single fast greedy if no time.
        S = set()
        for a in range(N, 0, -1):
            if can_add(a, S):
                S.add(a)
        if S:
            return sorted(S)
        return [1] # Guaranteed non-empty, valid? 1 has no conflicts with any pair from [2..N] because 1/1 = 1, 1/b+1/c < 1.
        # Actually, is {1} valid? Yes.

    # Run multiple randomized greedy restarts
    # We have 'remaining_time'. Let's divide it into chunks.
    # Each chunk: Generate random order, greedy construct, local search.
    
    # To be safe, let's do a deterministic greedy first as a baseline.
    # Greedy by ascending order
    S_base = set()
    for a in range(1, N + 1): 
        if can_add(a, S_base):
            S_base.add(a)
    
    # Greedy by descending order
    S_desc = set()
    for a in range(N, 0, -1):
        if can_add(a, S_desc):
            S_desc.add(a)
            
    # Greedy by "least conflicts" (degree in conflict graph)
    # Degree of x is len(conflicts[x])
    # Elements with fewer conflicts are "safer" to add?
    # Actually, if we add an element, it blocks others.
    # Elements with many conflicts are "central" and might block many things.
    # So we should add them LATE or NOT at all?
    # Or add them FIRST to secure them?
    # Independent set: usually greedy by degree (add lowest degree first) is a good heuristic.
    
    degrees = {i: len(conflicts[i]) for i in range(1, N+1)}
    order_by_degree = sorted(range(1, N+1), key=lambda x: degrees[x])
    S_deg = set()
    for a in order_by_degree:
        if can_add(a, S_deg):
            S_deg.add(a)
            
    candidates = [S_base, S_desc, S_deg]
    
    # Add a few random initializations
    for _ in range(3):
        order = list(range(1, N + 1))
        random.shuffle(order)
        S_rand = set()
        for a in order:
            if can_add(a, S_rand):
                S_rand.add(a)
        candidates.append(S_rand)
        
    best_set = max(candidates, key=len)
    best_size = len(best_set)
    
    # Polish the best candidate with the improved local search
    best_set = local_search(best_set, deadline - 2)
    best_size = len(best_set)
    
    # If time permits, try to polish other candidates or do more restarts
    # The local search with swaps is the most valuable part.
    # We can try to run local search on the other candidates too if time allows.
    
    for S_cand in candidates:
        if len(S_cand) + 5 < best_size: # Skip if too far behind
            continue
        if time.time() > deadline - 2:
            break
        S_polished = local_search(set(S_cand), deadline - 2)
        if len(S_polished) > best_size:
            best_size = len(S_polished)
            best_set = S_polished
            
    return sorted(best_set)
