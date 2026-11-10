import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 85.0
    n = 128

    # Known good construction: U={65..128} (sum-free upper block), L={17..32} (sum-free, pair sums < 65)
    U = list(range(65, 129))
    L = list(range(17, 33))
    best = U + L
    best_size = len(best)  # 80

    def _is_valid(S):
        S_set = set(S)
        for a, b, c in combinations(S, 3):
            if (a + b) in S_set and (a + c) in S_set and (b + c) in S_set:
                return False
        return True

    def _can_add(S_list, S_set, x):
        """Check if adding x to S creates a forbidden triple."""
        # Case 1: x is one of the three members a,b,c
        # We need to check if there exist a,b in S such that a+b in S, a+x in S, b+x in S
        # Note: a+x and b+x must be in S (which they are, as S is current set)
        # Wait, a+x is a value. It must be in S_set.
        for i in range(len(S_list)):
            a = S_list[i]
            ax = a + x
            if ax not in S_set:
                continue
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                bx = b + x
                if bx not in S_set:
                    continue
                ab = a + b
                if ab in S_set:
                    return False
        
        # Case 2: x is one of the sums, e.g., x = a+b
        # We need to check if there exist a,b in S such that a+b=x
        # and there exists c in S such that a+c in S and b+c in S
        # and a,b,c are distinct.
        for a in S_list:
            b = x - a
            if b in S_set and b != a:
                # Found a,b such that a+b=x
                # Now check if there is a c
                for c in S_list:
                    if c == a or c == b:
                        continue
                    if (a + c) in S_set and (b + c) in S_set:
                        return False
        return True

    # Phase 1: Greedy addition from 1..64
    S_list = list(best)
    S_set = set(S_list)
    for x in range(1, 65):
        if time.time() > deadline:
            break
        if x in S_set:
            continue
        if _can_add(S_list, S_set, x):
            S_list.append(x)
            S_set.add(x)

    if len(S_list) > best_size:
        best = S_list
        best_size = len(best)

    # Phase 2: Randomized Local Search (Swap and Add)
    random.seed(42)
    for _ in range(200):
        if time.time() > deadline:
            break
        
        # Start from current best
        curr = list(best)
        curr_set = set(curr)
        
        # Try to improve by swapping elements
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
            
            # Try removing a random element and adding a random one
            # Or just try adding a random element from complement
            # Let's try adding first, then swapping
            
            # 1. Try to add any element from 1..128 not in curr
            candidates = [x for x in range(1, 129) if x not in curr_set]
            random.shuffle(candidates)
            for x in candidates:
                if time.time() > deadline:
                    break
                if _can_add(curr, curr_set, x):
                    curr.append(x)
                    curr_set.add(x)
                    improved = True
                    break # Restart scan after modification
            
            if not improved and len(curr) > 0:
                # 2. Try swapping: remove u, add v
                # Pick a random u to remove
                u_idx = random.randint(0, len(curr) - 1)
                u = curr[u_idx]
                curr_temp = curr[:u_idx] + curr[u_idx+1:]
                curr_temp_set = curr_set - {u}
                
                # Try to add a random v
                candidates = [x for x in range(1, 129) if x not in curr_temp_set]
                random.shuffle(candidates)
                for v in candidates:
                    if time.time() > deadline:
                        break
                    if _can_add(curr_temp, curr_temp_set, v):
                        # Accept swap
                        curr = curr_temp
                        curr_set = curr_temp_set
                        curr.append(v)
                        curr_set.add(v)
                        improved = True
                        break
                
                if not improved and len(curr) == len(curr_temp):
                     # If we didn't add v, revert? 
                     # Actually, if we didn't find a v, we just didn't change curr.
                     # But we spent time. 
                     # To avoid getting stuck, we can force a random move even if size doesn't increase?
                     # No, we want to maximize size. 
                     # If we can't add or swap to increase size, we are stuck in a local max for this heuristic.
                     # Break inner loop.
                     break

        if len(curr) > best_size:
            best = curr
            best_size = len(best)

    # Final verification
    if not _is_valid(best):
        return list(range(65, 129)) + list(range(17, 33))

    return [int(x) for x in best]
