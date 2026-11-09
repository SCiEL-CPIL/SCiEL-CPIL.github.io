import time
import random

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 25.0  # 5s margin
    
    best_set = None
    best_size = float('inf')
    
    # Helper to check if x is addable to S given current sums U
    def is_addable(x, S, U):
        # Check 2x
        if 2 * x in U:
            return False
        # Check x + s
        for s in S:
            if x + s in U:
                return False
        return True

    def get_sums(x, S):
        return {2 * x} | {x + s for s in S}

    def domination_score(c, S, U, addable_candidates):
        # c is assumed addable
        sums_c = get_sums(c, S)
        score = 0
        for z in addable_candidates:
            if z == c:
                continue
            # Check if z is blocked by c
            # z's sums in S: sums_z
            # z's new sum with c: z + c
            # Conflict if (sums_z U {z+c}) intersects sums_c
            # sums_z = {2z} U {z+s for s in S}
            # We know sums_z is disjoint from U.
            # We just need to check intersection with sums_c.
            
            # Check 2z in sums_c
            if 2 * z in sums_c:
                score += 1
                continue # Already blocked, no need to check others? 
                         # Actually, we just need to know if it's blocked.
                         # If any conflict, it's blocked.
            
            # Check z+s in sums_c for s in S
            blocked = False
            for s in S:
                if z + s in sums_c:
                    blocked = True
                    break
            if blocked:
                score += 1
                continue
                
            # Check z+c in sums_c
            if z + c in sums_c:
                score += 1
                continue
            # Also check if z+c conflicts with sums_z?
            # z+c = 2z => c=z (False)
            # z+c = z+s => c=s (False)
            # So no internal conflict for z.
            
            # If no conflict, z is NOT blocked
        return score

    # Run multiple restarts
    num_restarts = 100
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        S = []
        U = set()
        
        # Greedy Max Domination
        while True:
            if time.time() > deadline:
                break
                
            # Find all addable candidates
            addable = []
            for x in range(1, N + 1):
                if x not in S and is_addable(x, S, U):
                    addable.append(x)
            
            if not addable:
                break
                
            # Calculate domination scores
            best_c = None
            best_score = -1
            # To avoid bias, we can randomize tie-breaking
            # Or just pick first max
            # Let's add a bit of randomness to tie-breaking
            candidates_with_scores = []
            for c in addable:
                score = domination_score(c, S, U, addable)
                candidates_with_scores.append((score, c))
            
            # Find max score
            max_score = max(s for s, c in candidates_with_scores)
            # Get all with max score
            top_candidates = [c for s, c in candidates_with_scores if s == max_score]
            
            # Random pick from top
            c = random.choice(top_candidates)
            
            # Add c
            S.append(c)
            U.update(get_sums(c, S[:-1])) # S[:-1] is S before adding c
            
        # Check if S is better
        if len(S) < best_size:
            best_size = len(S)
            best_set = S[:]
            
            # Optional: Local search to try to reduce size further?
            # If we have time, try to remove elements and re-maximize?
            # This is complex. The greedy is likely good enough.
            
    if best_set is None:
        # Fallback to simple greedy if something went wrong
        S = []
        U = set()
        for x in range(1, N + 1):
            if is_addable(x, S, U):
                S.append(x)
                U.update(get_sums(x, S[:-1]))
        best_set = S
        
    return best_set
