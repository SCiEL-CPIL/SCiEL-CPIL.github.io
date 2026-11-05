import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin before 120s limit
    
    initial_set = [1, 2, 5, 11]
    target_size = 20
    
    def compute_sums(s):
        """Compute all pairwise sums a+b for a<=b in sorted set s."""
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                sums.add(s[i] + s[j])
        return sums
    
    def can_add(x, curr_set, used_sums):
        """Check if x can be added to curr_set maintaining Sidon property."""
        for y in curr_set:
            if x + y in used_sums:
                return False
        if 2 * x in used_sums:
            return False
        return True
    
    def greedy_search(start_set, start_sums, target_size, deadline, max_restarts=2000):
        """
        Greedy search with randomization to find a large Sidon set.
        Returns the best set found.
        """
        best_set = start_set[:]
        best_size = len(start_set)
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            curr_set = start_set[:]
            curr_sums = start_sums.copy()
            
            # Available candidates
            candidates = list(range(12, 285))
            
            while len(curr_set) < target_size:
                if time.time() > deadline:
                    break
                
                # Filter valid candidates
                valid_cands = []
                for x in candidates:
                    if can_add(x, curr_set, curr_sums):
                        valid_cands.append(x)
                
                if not valid_cands:
                    break
                
                # Heuristic: prefer candidates that leave more room
                # Use a scoring function to pick the best candidate
                # Prefer smaller values (leave room for larger ones)
                # But also consider future constraints
                
                # Simple heuristic: pick randomly from top-k smallest valid candidates
                # Or use a more sophisticated approach
                
                if len(valid_cands) > 20:
                    # Pick from a random subset to add diversity
                    k = min(20, len(valid_cands))
                    chosen_subset = random.sample(valid_cands, k)
                    # Prefer smaller values
                    chosen_subset.sort()
                    # Pick from the first half with some randomness
                    pick_idx = random.randint(0, len(chosen_subset) // 2)
                    x = chosen_subset[pick_idx]
                else:
                    # Pick randomly for diversity
                    x = random.choice(valid_cands)
                
                # Add x
                curr_set.append(x)
                for y in curr_set[:-1]:
                    curr_sums.add(x + y)
                curr_sums.add(2 * x)
                
                if len(curr_set) > best_size:
                    best_set = curr_set[:]
                    best_size = len(curr_set)
                
                if len(curr_set) >= target_size:
                    break
                
                # Update candidates: remove x and any that are now invalid
                # Actually, we re-filter each iteration, so just remove x
                candidates = [c for c in candidates if c != x]
        
        return best_set
    
    # Compute initial sums
    initial_sums = compute_sums(initial_set)
    
    # Try greedy search with multiple restarts
    best_set = greedy_search(initial_set, initial_sums, target_size, deadline, max_restarts=5000)
    
    # If we didn't find size 20, try a backtracking approach for the remaining slots
    if len(best_set) < target_size:
        if time.time() < deadline - 5:
            # Try to extend best_set with backtracking
            def extend_with_backtrack(curr_set, used_sums, candidates, remaining_needed, deadline):
                nonlocal best_set
                
                if time.time() > deadline:
                    return False
                
                if len(curr_set) >= target_size:
                    if len(curr_set) > len(best_set):
                        best_set = curr_set[:]
                    return True
                
                if not candidates or remaining_needed > len(candidates):
                    return False
                
                for i, x in enumerate(candidates):
                    if time.time() > deadline:
                        return False
                    
                    if not can_add(x, curr_set, used_sums):
                        continue
                    
                    # Try adding x
                    new_set = curr_set + [x]
                    new_sums = used_sums.copy()
                    for y in curr_set:
                        new_sums.add(x + y)
                    new_sums.add(2 * x)
                    
                    next_cands = [c for c in candidates[i+1:] if can_add(c, new_set, new_sums)]
                    
                    if extend_with_backtrack(new_set, new_sums, next_cands, remaining_needed - 1, deadline):
                        return True
                    
                    # Backtrack
                    del new_set[-1]
                
                return False
            
            # Filter candidates for backtracking
            initial_cands = [x for x in range(12, 285) if x not in best_set]
            initial_cands = [x for x in initial_cands if can_add(x, best_set, initial_sums)]
            
            extend_with_backtrack(best_set, initial_sums, initial_cands, target_size - len(best_set), deadline)
    
    return best_set
