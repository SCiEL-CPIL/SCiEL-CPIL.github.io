"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    n = 256
    # Explicit deadline guard: evaluator kills after 90s wall-clock.
    # We reserve a safety margin and track elapsed time throughout.
    deadline = time.time() + 82.0
    
    # The proven safe tail: [171, 256] is valid because 3*171 = 513 > 511 = 2*256-1.
    # For any a<b<=c in [171,256], b+c <= 511 < 3*a (since a>=171), 
    # so a cannot divide b+c (the only multiple of a in range [2,512] that could
    # equal b+c would be 2a or 3a, but b+c > 2a is possible... wait.
    # Actually for a in [171,256], b,c > a >= 171, so b+c > 342 = 2*171.
    # The next multiple of a after 2a is 3a. We need 3a > b+c for all b,c in tail.
    # Max b+c = 255+256 = 511. Min a = 171. 3*171 = 513 > 511. 
    # So no multiple of a (other than possibly 2a, but b+c > 2a always since b,c>a)
    # can equal b+c. Wait, b+c could equal 2a? No, because b>a and c>a, so b+c>2a.
    # The next candidate is 3a. Since 3a >= 3*171 = 513 > 511 >= b+c, 
    # a never divides b+c. So [171,256] is indeed valid.
    
    tail_start = 171
    base_set = set(range(tail_start, n + 1))
    
    # Precompute residue classes for fast validity checking
    # residue_classes[a][r] = sorted list of values in [1,n] congruent to r mod a
    residue_classes = {}
    for a in range(1, n + 1):
        rc = [[] for _ in range(a)]
        for v in range(1, n + 1):
            rc[v % a].append(v)
        residue_classes[a] = rc
    
    def is_valid_addition(S, x):
        """Check if x can be added to set S without violating the property.
        Returns True if valid, False otherwise."""
        # Role 1: x is 'a' (smallest). Need x ∤ (b+c) for all b,c in S with x < b <= c.
        # For each b in S with b > x, check if there exists c in S with c >= b and c ≡ -b (mod x).
        for b in S:
            if x < b:
                target_res = (-b) % x
                # Check if any value in residue_classes[x][target_res] that is >= b is in S
                for c in residue_classes[x][target_res]:
                    if c >= b and c in S:
                        return False
                    if c > n:
                        break
        
        # Role 2: x is 'b' (middle). Need a ∤ (x+c) for all a,c in S with a < x <= c.
        # For each a in S with a < x, check if there exists c in S with c >= x and c ≡ -x (mod a).
        for a in S:
            if a < x:
                target_res = (-x) % a
                for c in residue_classes[a][target_res]:
                    if c >= x and c in S:
                        return False
                    if c > n:
                        break
        
        # Role 3: x is 'c' (largest). Need a ∤ (b+x) for all a,b in S with a < b <= x.
        # For each a in S, check if there exists b in S with a < b <= x and b ≡ -x (mod a).
        for a in S:
            target_res = (-x) % a
            for b in residue_classes[a][target_res]:
                if a < b <= x and b in S:
                    return False
                if b > x:
                    break
        
        return True
    
    def verify_set(S):
        """Verify that set S satisfies the property. Returns True if valid."""
        sorted_S = sorted(S)
        for i in range(len(sorted_S)):
            a = sorted_S[i]
            for j in range(i + 1, len(sorted_S)):
                b = sorted_S[j]
                for k in range(j, len(sorted_S)):
                    c = sorted_S[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    current_set = set(base_set)
    best_set = set(current_set)
    best_size = len(current_set)
    
    candidates = list(range(1, tail_start))
    
    # Greedy pass: try adding in ascending order
    for x in sorted(candidates):
        if time.time() > deadline:
            break
        if is_valid_addition(current_set, x):
            current_set.add(x)
            if len(current_set) > best_size:
                best_set = set(current_set)
                best_size = len(current_set)
    
    # Verify the greedy result
    if not verify_set(best_set):
        # Fallback to safe tail if verification fails
        best_set = set(base_set)
        best_size = len(best_set)
    
    # Random restarts with different orderings
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        current_set = set(base_set)
        cands = list(range(1, tail_start))
        random.shuffle(cands)
        
        for x in cands:
            if time.time() > deadline - 15:
                break
            if is_valid_addition(current_set, x):
                current_set.add(x)
        
        if len(current_set) > best_size:
            if verify_set(current_set):
                best_set = set(current_set)
                best_size = len(current_set)
    
    # Local search: 1-swap (remove one, add one)
    for sweep in range(30):
        if time.time() > deadline - 10:
            break
        
        improved = False
        current = set(best_set)
        
        for x in list(current):
            if time.time() > deadline - 10:
                break
            
            current.discard(x)
            found_better = False
            for y in range(1, n + 1):
                if y in current:
                    continue
                if is_valid_addition(current, y):
                    current.add(y)
                    if len(current) > len(best_set):
                        if verify_set(current):
                            best_set = set(current)
                            best_size = len(current)
                            improved = True
                            found_better = True
                    current.discard(y)
                    if found_better:
                        break
            if not found_better:
                current.add(x)
        
        if not improved:
            break
    
    # Local search: 2-swap (remove two, add two)
    for sweep in range(15):
        if time.time() > deadline - 5:
            break
        
        improved = False
        current = set(best_set)
        elems = list(current)
        
        for i in range(len(elems)):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, len(elems)):
                if time.time() > deadline - 5:
                    break
                
                x1, x2 = elems[i], elems[j]
                current.discard(x1)
                current.discard(x2)
                
                found_better = False
                for y1 in range(1, n + 1):
                    if time.time() > deadline - 5:
                        break
                    if y1 in current or y1 == x1 or y1 == x2:
                        continue
                    if is_valid_addition(current, y1):
                        current.add(y1)
                        for y2 in range(1, n + 1):
                            if y2 in current or y2 == x1 or y2 == x2 or y2 == y1:
                                continue
                            if is_valid_addition(current, y2):
                                current.add(y2)
                                if len(current) > len(best_set):
                                    if verify_set(current):
                                        best_set = set(current)
                                        best_size = len(current)
                                        improved = True
                                current.discard(y2)
                                if improved:
                                    break
                        current.discard(y1)
                        if improved:
                            break
                
                if not found_better:
                    current.add(x1)
                    current.add(x2)
                
                if improved:
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    # Final verification before returning
    if not verify_set(best_set):
        # Fall back to the safe tail
        best_set = set(base_set)
    
    return list(best_set)
# EVOLVE-BLOCK-END
