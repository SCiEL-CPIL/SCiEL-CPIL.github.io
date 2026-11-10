import time
import random

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    def is_sidon(S):
        """Check if S is a Sidon set."""
        if not S:
            return True
        used_sums = set()
        for i in range(len(S)):
            a = S[i]
            # Diagonal
            s = 2 * a
            if s in used_sums:
                return False
            used_sums.add(s)
            # Off-diagonal
            for j in range(i + 1, len(S)):
                b = S[j]
                s = a + b
                if s in used_sums:
                    return False
                used_sums.add(s)
        return True

    def is_maximal(S, N):
        """Check if S is inclusion-maximal in [1, N]."""
        S_set = set(S)
        used_sums = set()
        for i in range(len(S)):
            a = S[i]
            used_sums.add(2 * a)
            for j in range(i + 1, len(S)):
                b = S[j]
                used_sums.add(a + b)
        
        for z in range(1, N + 1):
            if z in S_set:
                continue
            # Check if z can be added
            conflict = False
            # Check 2z
            if 2 * z in used_sums:
                conflict = True
            else:
                # Check z + s for all s in S
                for s in S:
                    if z + s in used_sums:
                        conflict = True
                        break
            if not conflict:
                return False
        return True

    def sidon_and_maximal(S):
        return is_sidon(S) and is_maximal(S, N)

    def greedy_set_cover():
        """Greedy construction prioritizing elements that block the most other candidates."""
        selected = []
        used_sums = set()
        candidates = list(range(1, N + 1))
        
        while candidates:
            if time.time() > deadline:
                break
            
            valid_candidates = []
            for x in candidates:
                new_sums = [2 * x]
                for s in selected:
                    new_sums.append(x + s)
                if not any(s in used_sums for s in new_sums):
                    valid_candidates.append(x)
            
            if not valid_candidates:
                break
            
            # Find the candidate that blocks the most other valid candidates
            best_candidate = None
            max_blocks = -1
            
            for x in valid_candidates:
                added_sums = set([2 * x] + [x + s for s in selected])
                blocks = 0
                for y in valid_candidates:
                    if y == x:
                        continue
                    # Check if y would be blocked by adding x
                    # y is blocked if any of its new sums collide with added_sums
                    # y's sums: 2y, y+s (s in selected)
                    # Collisions with added_sums:
                    # 1. 2y in added_sums
                    # 2. y+s in added_sums for some s in selected
                    if 2 * y in added_sums:
                        blocks += 1
                        continue
                    blocked = False
                    for s in selected:
                        if y + s in added_sums:
                            blocked = True
                            break
                    if not blocked:
                        # Check if y+x creates a collision with existing used_sums
                        # If y+x is in used_sums, then y and x cannot coexist
                        if y + x in used_sums:
                            blocked = True
                    if blocked:
                        blocks += 1
                
                if blocks > max_blocks:
                    max_blocks = blocks
                    best_candidate = x
            
            if best_candidate is None:
                break
                
            selected.append(best_candidate)
            # Update used_sums
            used_sums.add(2 * best_candidate)
            for s in selected[:-1]:
                used_sums.add(best_candidate + s)
            
            # Remove blocked candidates
            new_candidates = []
            for y in candidates:
                if y == best_candidate:
                    continue
                new_sums_y = [2 * y]
                for s in selected:
                    new_sums_y.append(y + s)
                if not any(s in used_sums for s in new_sums_y):
                    new_candidates.append(y)
            candidates = new_candidates
            
        return selected

    def local_search(S, max_time=10.0):
        """Local search to minimize cardinality."""
        S = S[:]
        S.sort()
        start = time.time()
        
        improved = True
        while improved and time.time() < start + max_time:
            improved = False
            n = len(S)
            
            # Try removing one element
            for i in range(n):
                if time.time() > start + max_time:
                    break
                S_rem = S[:i] + S[i+1:]
                if is_sidon(S_rem):
                    # Check if S_rem is still maximal
                    if is_maximal(S_rem, N):
                        S = S_rem
                        improved = True
                        break
            
            if improved:
                continue
                
            # Try swapping one element for another
            # This is more complex, so we'll do a simpler version:
            # Try replacing one element with a smaller one
            for i in range(n):
                if time.time() > start + max_time:
                    break
                old_val = S[i]
                # Try smaller values
                for new_val in range(1, old_val):
                    if new_val in S:
                        continue
                    S_new = S[:i] + [new_val] + S[i+1:]
                    S_new.sort()
                    if is_sidon(S_new) and is_maximal(S_new, N):
                        S = S_new
                        improved = True
                        break
                if improved:
                    break
                    
        return S

    # Main construction
    best_S = None
    best_size = N + 1
    
    # Try greedy construction with different randomizations
    for _ in range(5):
        if time.time() > deadline:
            break
            
        S = greedy_set_cover()
        if not S:
            S = [1]
            
        if sidon_and_maximal(S):
            if len(S) < best_size:
                best_size = len(S)
                best_S = S[:]
                
        # Local search
        S_opt = local_search(S, max_time=5.0)
        if sidon_and_maximal(S_opt):
            if len(S_opt) < best_size:
                best_size = len(S_opt)
                best_S = S_opt[:]
                
    if best_S is None:
        # Fallback
        best_S = [1]
        
    return best_S
