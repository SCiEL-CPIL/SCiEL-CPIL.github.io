"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 256

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 25.0
    best_size = N + 1
    best_set = None
    
    def is_sidon(S):
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def get_addable(S):
        if not S:
            return list(range(1, N + 1))
        S_set = set(S)
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.add(S[i] + S[j])
        
        addable = []
        for x in range(1, N + 1):
            if x in S_set:
                continue
            if 2 * x in sums:
                continue
            can_add = True
            for a in S:
                if (x + a) in sums:
                    can_add = False
                    break
            if can_add:
                addable.append(x)
        return addable
    
    def backtrack_search(S, sums, remaining_mask, depth, best_size_ref):
        """Backtracking search to find minimal maximal Sidon set."""
        nonlocal best_size, best_set
        
        if time.time() > deadline - 1.0:
            return False
        
        # Check if we've found a better solution
        if len(S) >= best_size:
            return False
        
        # Check if all remaining elements are blocked (maximal)
        # An element x is blocked if 2x in sums or x+a in sums for some a in S
        # We need to check if any element in remaining_mask can be added
        any_addable = False
        for x in range(1, N + 1):
            if not (remaining_mask >> x) & 1:
                continue
            if x in S:
                continue
            if 2 * x in sums:
                continue
            can_add = True
            for a in S:
                if (x + a) in sums:
                    can_add = False
                    break
            if can_add:
                any_addable = True
                break
        
        if not any_addable:
            # We have a maximal Sidon set
            if len(S) < best_size:
                best_size = len(S)
                best_set = list(S)
                return True
        
        # If no more elements can be added, return
        if not remaining_mask:
            return False
        
        # Try adding each remaining element, prioritizing those that block the most
        candidates = []
        for x in range(1, N + 1):
            if not (remaining_mask >> x) & 1:
                continue
            if x in S:
                continue
            if 2 * x in sums:
                continue
            can_add = True
            for a in S:
                if (x + a) in sums:
                    can_add = False
                    break
            if can_add:
                # Count how many elements this would block
                new_sums = set(sums)
                new_sums.add(2 * x)
                for a in S:
                    new_sums.add(x + a)
                
                blocked = 0
                for y in range(1, N + 1):
                    if y == x or y in S:
                        continue
                    if not (remaining_mask >> y) & 1:
                        continue
                    if 2 * y in new_sums:
                        blocked += 1
                        continue
                    y_blocked = False
                    for a in S:
                        if (y + a) in new_sums:
                            y_blocked = True
                            break
                    if y_blocked:
                        blocked += 1
                candidates.append((blocked, x))
        
        if not candidates:
            return False
        
        # Sort by blocking power (descending)
        candidates.sort(reverse=True)
        
        # Limit branching factor for performance
        max_branch = min(len(candidates), 5)
        
        for blocked, x in candidates[:max_branch]:
            # Add x to S
            S.append(x)
            old_sums = set(sums)
            sums.add(2 * x)
            for a in S[:-1]:  # Don't include x itself in the loop
                sums.add(x + a)
            
            # Update remaining_mask
            new_remaining = remaining_mask & ~(1 << x)
            
            # Recurse
            if backtrack_search(S, sums, new_remaining, depth + 1, best_size):
                # Restore state
                S.pop()
                sums.clear()
                sums.update(old_sums)
                return True
            
            # Restore state
            S.pop()
            sums.clear()
            sums.update(old_sums)
        
        return False
    
    # Initial greedy to get a good starting point
    def greedy_initial():
        S = []
        S_set = set()
        sums = set()
        remaining = list(range(1, N + 1))
        
        while remaining:
            best_x = None
            best_score = -1
            for x in remaining:
                if 2 * x in sums:
                    continue
                can_add = True
                for a in S:
                    if (x + a) in sums:
                        can_add = False
                        break
                if not can_add:
                    continue
                
                new_sums = set(sums)
                new_sums.add(2 * x)
                for a in S:
                    new_sums.add(x + a)
                
                blocked = 0
                for y in remaining:
                    if y == x:
                        continue
                    if 2 * y in new_sums:
                        blocked += 1
                        continue
                    y_blocked = False
                    for a in S:
                        if (y + a) in new_sums:
                            y_blocked = True
                            break
                    if y_blocked:
                        blocked += 1
                
                if blocked > best_score:
                    best_score = blocked
                    best_x = x
            
            if best_x is None:
                break
            
            S.append(best_x)
            S_set.add(best_x)
            sums.add(2 * best_x)
            for a in S[:-1]:
                sums.add(best_x + a)
            remaining.remove(best_x)
        
        return S
    
    initial_set = greedy_initial()
    if initial_set:
        best_size = len(initial_set)
        best_set = list(initial_set)
    
    # Run backtracking search with different random seeds
    for seed in range(10):
        if time.time() > deadline - 3.0:
            break
        
        random.seed(seed)
        
        # Create a shuffled order for backtracking
        S = []
        sums = set()
        remaining_mask = (1 << (N + 1)) - 2  # Bits 1..N set
        
        # Add a few random elements first to break symmetry
        num_initial = random.randint(2, 5)
        for _ in range(num_initial):
            candidates = [x for x in range(1, N + 1) if (remaining_mask >> x) & 1]
            if not candidates:
                break
            x = random.choice(candidates)
            
            # Check if x can be added
            if 2 * x in sums:
                continue
            can_add = True
            for a in S:
                if (x + a) in sums:
                    can_add = False
                    break
            if not can_add:
                continue
            
            S.append(x)
            sums.add(2 * x)
            for a in S[:-1]:
                sums.add(x + a)
            remaining_mask &= ~(1 << x)
        
        # Run backtracking
        backtrack_search(S, sums, remaining_mask, 0, best_size)
    
    if best_set is not None:
        return best_set
    else:
        return initial_set
# EVOLVE-BLOCK-END
