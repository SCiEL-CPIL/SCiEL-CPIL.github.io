import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit to be safe
    
    # Target sum
    target = 200
    # Range
    n = 100
    
    # Helper to check validity using bitset DP
    # Returns True if no subset sums to target
    def is_valid(S):
        # S is a set of integers
        # Use a boolean list or bitset
        # Max sum is 200
        reachable = [False] * (target + 1)
        reachable[0] = True
        for x in S:
            if x > target:
                continue
            # Update reachable in reverse order to avoid using x multiple times
            # Actually, we are checking subset sum, so each element used at most once.
            # Standard 0/1 knapsack DP
            for s in range(target, x - 1, -1):
                if reachable[s - x]:
                    reachable[s] = True
                    if s == target:
                        return False
        return True

    # Initial candidate: {66, ..., 100}
    # Size 35
    S = set(range(66, 101))
    best_S = set(S)
    
    # Verify initial
    if not is_valid(best_S):
        # Fallback to smaller
        while not is_valid(best_S) and best_S:
            best_S.pop()
    
    # Try to improve by adding small numbers and removing conflicts
    # Elements 1..65
    candidates = list(range(1, 66))
    random.shuffle(candidates)
    
    current_S = set(best_S)
    current_size = len(current_S)
    
    # Greedy Add/Remove
    # We try to add each candidate. If invalid, we try to remove one element from the "conflict".
    # To find conflict, we can just check validity.
    
    # A more efficient local search:
    # 1. Try to add a random small number.
    # 2. If invalid, find a subset summing to 200.
    # 3. Remove one element from that subset (preferably from the large set to keep small ones? Or vice versa?).
    
    # Let's do a simple hill climbing
    no_improvement_count = 0
    
    while time.time() < deadline and no_improvement_count < 100:
        # Pick a random candidate to add
        if candidates:
            x = random.choice(candidates)
            if x not in current_S:
                new_S = set(current_S)
                new_S.add(x)
                if is_valid(new_S):
                    current_S = new_S
                    if len(current_S) > current_size:
                        current_size = len(current_S)
                        if len(current_S) > len(best_S):
                            best_S = set(current_S)
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                else:
                    # Try to fix by removing an element
                    # Find a bad subset
                    # Simple approach: remove a random element from current_S and check
                    # This is slow. 
                    # Better: just skip this x for now.
                    pass
        else:
            break
            
        # Occasionally, try removing a large element to see if we can add more small ones
        if random.random() < 0.1:
            if len(current_S) > 10:
                # Remove a random large element
                large_elements = [y for y in current_S if y > 60]
                if large_elements:
                    y = random.choice(large_elements)
                    new_S = set(current_S)
                    new_S.discard(y)
                    # Try to add some small ones
                    # Just accept if it allows future additions? 
                    # For now, just keep it if it doesn't reduce size? No, it reduces size.
                    # Only do this if we are stuck.
                    if no_improvement_count > 20:
                         current_S = new_S
                         # Reset candidates to include removed y? No, y is large.
                         # We want to add small ones.
                         pass

    # Final check
    if len(best_S) < len(current_S) and is_valid(current_S):
        best_S = current_S
        
    # Ensure non-empty
    if not best_S:
        best_S = {100}
        
    return list(best_S)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit
    
    target = 200
    n = 100
    
    def is_valid(S):
        reachable = [False] * (target + 1)
        reachable[0] = True
        for x in S:
            if x > target:
                continue
            for s in range(target, x - 1, -1):
                if reachable[s - x]:
                    reachable[s] = True
                    if s == target:
                        return False
        return True

    # Start with {66..100}
    S = set(range(66, 101))
    best_S = set(S)
    
    # Local Search
    candidates = list(range(1, 66))
    random.shuffle(candidates)
    
    current_S = set(S)
    
    # Try to add elements
    # We iterate through candidates multiple times
    for _ in range(10):
        if time.time() > deadline - 5:
            break
        random.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 5:
                break
            if x in current_S:
                continue
            new_S = set(current_S)
            new_S.add(x)
            if is_valid(new_S):
                current_S = new_S
                if len(current_S) > len(best_S):
                    best_S = set(current_S)
            else:
                # If adding x fails, maybe removing a large element helps?
                # Try removing the largest element that is part of a conflict?
                # Heuristic: Just try removing a random large element and see if x can be added.
                # This is expensive. 
                # Instead, we can try to remove elements from S that are "critical".
                # For now, skip.
                pass

    # Try to remove large elements and add multiple small ones
    # This is complex. The simple addition above found 35.
    # If we can't beat 35, we return 35.
    
    if not best_S:
        best_S = {100}
        
    return list(best_S)
# EVOLVE-BLOCK-END
