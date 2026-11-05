import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # 1. Precomputation
    N = 24
    # Calculate LCM of 1..24
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for i in range(1, N + 1):
        L = lcm(L, i)
        
    V = [0] * (N + 1)
    for i in range(1, N + 1):
        V[i] = L // i

    # 2. Validity Check
    # mask is an integer where bit i is set if i is in the set (1-indexed, so bit 1 is element 1)
    # Actually, let's use 0-indexed bits for simplicity, bit i corresponds to element i+1
    # But V is 1-indexed. Let's stick to 1-indexed elements in the set, 
    # but use a set of integers for the state.
    
    def is_valid(S):
        """
        S: set of integers from 1 to 24
        Returns True if no reciprocal of an element equals sum of reciprocals of a subset of others.
        """
        if len(S) < 2:
            return True
            
        S_list = list(S)
        for x in S_list:
            target = V[x]
            others = [V[y] for y in S_list if y != x]
            
            # Check if target is a subset sum of others
            # Since others are positive, we can use a set of reachable sums
            # Optimization: if min(others) > target, return False (no sum possible)
            # if max(others) < target, might still be possible with multiple terms
            
            # Use a set to track possible sums
            possible_sums = {0}
            for val in others:
                # Pruning: only keep sums <= target
                new_sums = set()
                for s in possible_sums:
                    if s + val <= target:
                        new_sums.add(s + val)
                possible_sums.update(new_sums)
                
                if target in possible_sums:
                    return False
                    
            # If loop finishes, target was not found
            # Note: If target is 0, it's always in possible_sums, but V[x] > 0.
            
        return True

    # 3. Search
    best_set = set(range(9, 25)) # Start with size 16
    best_size = len(best_set)
    
    # Verify initial
    if not is_valid(best_set):
        # Fallback to smaller
        best_set = set(range(13, 25))
        best_size = len(best_set)

    # Greedy improvement
    # Try to add elements 1..8
    for i in range(1, 9):
        if i not in best_set:
            new_set = best_set | {i}
            if is_valid(new_set):
                best_set = new_set
                best_size += 1

    # Local Search / Simulated Annealing
    # We will try to swap elements to increase size or maintain size while allowing additions
    
    current_set = set(best_set)
    current_size = len(current_set)
    
    # Random restarts or continuous SA
    # Let's do a simple hill climbing with random restarts if stuck
    # Or just iterate through possible swaps
    
    # To maximize speed, we focus on adding smaller numbers
    # The bottleneck is usually the small numbers (1, 2, ...)
    
    # Try all possible swaps (remove x, add y) if y < x
    # This might allow adding a smaller y which has larger reciprocal, 
    # but wait, we want to MAXIMIZE cardinality.
    # Swapping doesn't change size.
    # We only care about size.
    # So we should only accept moves that increase size.
    # If we are stuck, we might need to remove 2 to add 3 and 4?
    
    # Let's try a more exhaustive search for size > best_size
    # Since N=24, we can try to build sets of size 17, 18...
    
    # Heuristic: The set of largest numbers is most likely valid.
    # So we should try to include as many large numbers as possible,
    # and see if we can fit in smaller ones.
    
    # Strategy:
    # 1. Start with {24, 23, ..., k}
    # 2. Try to add k-1, k-2, ...
    # 3. If conflict, remove one of the conflicting elements and try again.
    
    # Let's run a randomized search
    random.seed(42)
    
    for restart in range(1000):
        if time.time() > deadline:
            break
            
        # Start with a random valid set or the best found
        if restart == 0:
            curr = set(best_set)
        else:
            # Perturb best set
            curr = set(best_set)
            # Remove 1-2 random elements
            for _ in range(random.randint(1, 2)):
                if curr:
                    item = random.choice(list(curr))
                    curr.remove(item)
            # Add 1-2 random elements
            for _ in range(random.randint(1, 2)):
                candidates = [i for i in range(1, 25) if i not in curr]
                if candidates:
                    curr.add(random.choice(candidates))
        
        if not is_valid(curr):
            continue
            
        curr_size = len(curr)
        
        # Try to improve
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding any missing element
            candidates = [i for i in range(1, 25) if i not in curr]
            random.shuffle(candidates)
            for c in candidates:
                new_curr = curr | {c}
                if is_valid(new_curr):
                    curr = new_curr
                    curr_size += 1
                    improved = True
                    break
            
            # If no single addition works, try a swap?
            # Swap doesn't increase size, so it's only useful if it enables a future addition.
            # This is a deeper search.
            # For now, let's stick to additions.
            
        if curr_size > best_size:
            best_size = curr_size
            best_set = set(curr)
            
        # Early exit if we reach 24 (impossible) or 20+
        if best_size >= 20:
            break

    return list(best_set)

# Test
if __name__ == "__main__":
    res = construct()
    print(len(res), res)
