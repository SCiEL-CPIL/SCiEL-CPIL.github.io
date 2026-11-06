import time
import random

N = 256
MAX_SUM = 2 * N

def construct():
        deadline = time.time() + 25.0
        best_size = N + 1
        best_set = None
        
        # Helper to check if x can be added to set S with sum_mask M
        # Returns True if addable
        def can_add(x, S, sum_mask):
            # 2x must not be in sum_mask
            if (sum_mask >> (2 * x)) & 1:
                return False
            # a + x must not be in sum_mask for any a in S
            # We can iterate S. S is small (~10-20).
            for a in S:
                if (sum_mask >> (a + x)) & 1:
                    return False
            return True

        def is_maximal(S, sum_mask):
            S_set = set(S)
            # For every x not in S, check if it is addable.
            # If any is addable, not maximal.
            for x in range(1, N + 1):
                if x in S_set:
                    continue
                if can_add(x, S, sum_mask):
                    return False
            return True

        def get_sum_mask(S):
            mask = 0
            for i in range(len(S)):
                a = S[i]
                for j in range(i, len(S)):
                    mask |= (1 << (a + S[j]))
            return mask

        # Phase 1: Randomized Greedy with Blocking Heuristic
        # We run multiple instances.
        
        for seed in range(200):
            if time.time() > deadline - 5.0:
                break
                
            rng = random.Random(seed)
            S = []
            sum_mask = 0
            remaining = set(range(1, N + 1))
            
            while True:
                addable = []
                for x in remaining:
                    if can_add(x, S, sum_mask):
                        addable.append(x)
                
                if not addable:
                    break
                
                # Heuristic: Pick x that blocks the most
                # Blocking(x) = count of y in remaining \ {x} such that y is NOT addable after adding x
                # To speed up, we can estimate blocking power.
                # Adding x adds sums: {2x} U {a+x for a in S}
                # A remaining y is blocked if 2y is in new_sums OR (a+y) in new_sums for some a.
                
                best_x = None
                best_blocks = -1
                
                # If many candidates, sample
                if len(addable) > 10:
                    candidates = rng.sample(addable, 10)
                else:
                    candidates = addable
                
                for x in candidates:
                    # Calculate new sums added
                    new_sums = [2 * x] + [a + x for a in S]
                    
                    # Check how many y in remaining (excluding x) become non-addable
                    # y is non-addable if:
                    # 1. 2y is in (old_sums U new_sums)
                    # 2. (a+y) is in (old_sums U new_sums) for some a in (S U {x})
                    
                    # This is expensive to check exactly for all y.
                    # Approximation: Count how many new_sums "hit" potential sums of y.
                    # Or simpler: Just pick randomly from top few?
                    # Let's use a simpler heuristic: Pick x that is "central" or random.
                    # Actually, the previous program's heuristic was "blocked count".
                    # Let's try to compute it fast.
                    
                    blocks = 0
                    # Pre-compute a set of "newly added sum values" for fast lookup?
                    # new_sums is small.
                    new_sum_set = set(new_sums)
                    
                    for y in remaining:
                        if y == x:
                            continue
                        # Check if y is addable with S + [x]
                        # If NOT addable, it's blocked.
                        # We can check if y was addable before?
                        # We know y is addable before (since it's in 'addable' list? No, 'addable' was computed for S, not S+[x]).
                        # Wait, 'addable' list contains elements addable to S.
                        # If y is in 'addable', it means y could be added to S.
                        # If y cannot be added to S U {x}, then x blocks y.
                        
                        # Check if y is addable to S U {x}
                        # Conditions for y to be addable to S U {x}:
                        # 1. 2y not in sums(S U {x})
                        # 2. (a+y) not in sums(S U {x}) for all a in S U {x}
                        
                        # sums(S U {x}) = sums(S) U {2x} U {a+x for a in S}
                        
                        # Check 1:
                        if (sum_mask >> (2 * y)) & 1:
                            # Already blocked by S
                            # Does it count as "newly blocked"? 
                            # The goal is to reach maximality.
                            # If y was already blocked, x doesn't help with y.
                            # We want to maximize the number of *currently addable* y's that become blocked.
                            # Since y is in 'addable', it was NOT blocked by S.
                            # So if it is blocked by S U {x}, it is newly blocked.
                            pass # This branch is irrelevant for 'newly blocked' count if we only count y in 'addable'.
                        
                        # So we only care about y in 'addable'.
                        # For y in 'addable', 2y is NOT in sum_mask, and a+y is NOT in sum_mask for all a in S.
                        # So y is addable to S.
                        
                        # Is y addable to S U {x}?
                        # It fails if:
                        # - 2y is in new_sums (i.e., 2y == 2x or 2y == a+x)
                        # - (a+y) is in new_sums (i.e., a+y == 2x or a+y == b+x)
                        #   Note: a is in S.
                        #   Also we must check a=x case: x+y is in new_sums? No, x+y is a new sum itself.
                        #   Wait, the condition is: for all a in S U {x}, a+y not in sums(S U {x}).
                        #   We already know for a in S, a+y not in sums(S).
                        #   We need to check:
                        #   1. x+y not in sums(S U {x})?
                        #      sums(S U {x}) contains sums(S) and new sums.
                        #      x+y is not in sums(S) (since y addable to S).
                        #      Is x+y in new_sums?
                        #      new_sums = {2x} U {a+x for a in S}.
                        #      So check if x+y == 2x (y=x, impossible) or x+y == a+x (y=a, impossible since y not in S).
                        #      So x+y is never in sums(S U {x})? 
                        #      Wait. If y is addable to S, then x+y is not in sums(S).
                        #      Is x+y in {2x} U {a+x}?
                        #      x+y = 2x => y=x (No).
                        #      x+y = a+x => y=a (No).
                        #      So the pair (x, y) itself does not create a conflict sum *within* the set of sums?
                        #      Wait, the Sidon condition is that all sums a+b are distinct.
                        #      If we add x, we add sums 2x and a+x.
                        #      We must ensure these new sums are distinct from each other and from old sums.
                        #      We already checked can_add(x), so 2x and a+x are not in old sums, and distinct from each other.
                        
                        #      Now, does adding x make y non-addable?
                        #      y is non-addable if some sum involving y conflicts with sums in S U {x}.
                        #      Sums involving y in S U {x} U {y} would be:
                        #      2y, a+y (a in S), x+y.
                        #      We need to check if any of these are in sums(S U {x}).
                        #      1. 2y in sums(S U {x})?
                        #         2y in sums(S)? No (y addable).
                        #         2y in new_sums? Check if 2y == 2x (No) or 2y == a+x.
                        #      2. a+y in sums(S U {x})?
                        #         a+y in sums(S)? No.
                        #         a+y in new_sums? Check if a+y == 2x or a+y == b+x.
                        #      3. x+y in sums(S U {x})?
                        #         x+y in sums(S)? No.
                        #         x+y in new_sums? 
                        #         x+y == 2x => y=x (No).
                        #         x+y == b+x => y=b (No, y not in S).
                        #         So x+y is NEVER in sums(S U {x}).
                        
                        #      So y is blocked if:
                        #      (2y == 2x) OR (2y == a+x for some a in S) OR
                        #      (a+y == 2x for some a in S) OR (a+y == b+x for some a,b in S)
                        
                        blocked = False
                        if 2 * y == 2 * x: pass # y=x
                        for a in S:
                            if 2 * y == a + x:
                                blocked = True
                                break
                            if a + y == 2 * x:
                                blocked = True
                                break
                            # a + y == b + x  => a - b = x - y
                            # This requires checking all pairs (a,b) in S.
                            # Since S is small, we can precompute differences?
                            # Or just check if a+y is in new_sums?
                            # new_sums is a list.
                            # Let's just use the set new_sum_set.
                            
                        if not blocked:
                            # Check a+y in new_sum_set
                            for a in S:
                                if (a + y) in new_sum_set:
                                    blocked = True
                                    break
                            
                        if blocked:
                            blocks += 1

                    if blocks > best_blocks:
                        best_blocks = blocks
                        best_x = x

                if best_x is None:
                    break

                S.append(best_x)
                # Update sum_mask
                for a in S[:-1]:
                    sum_mask |= (1 << (a + best_x))
                sum_mask |= (1 << (2 * best_x))
                remaining.remove(best_x)

            # Check if maximal
            if is_maximal(S, sum_mask):
                if len(S) < best_size:
                    best_size = len(S)
                    best_set = S[:]
                    # Try to reduce
                    # Local search: try removing elements
                    changed = True
                    while changed and time.time() < deadline - 2.0:
                        changed = False
                        for i in range(len(S)):
                            if time.time() > deadline - 2.0: break
                            x = S[i]
                            new_S = S[:i] + S[i+1:]
                            if not new_S: continue
                            new_mask = get_sum_mask(new_S)
                            if is_maximal(new_S, new_mask):
                                S = new_S
                                sum_mask = new_mask
                                best_size = len(S)
                                best_set = S[:]
                                changed = True
                                break # Restart reduction loop

        return best_set
