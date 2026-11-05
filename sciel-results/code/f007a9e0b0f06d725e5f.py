import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin
    
    def get_bitset(S):
        reachable = 1
        for x in S:
            reachable |= (reachable << x)
        return reachable

    def is_valid(S):
        reachable = get_bitset(S)
        return ((reachable >> 200) & 1) == 0

    def try_add(S, x):
        if x in S:
            return S, False
        # Check if adding x is valid
        # We need to check if 200 is reachable in S U {x}
        # This is equivalent to checking if 200 is reachable in S OR (200-x is reachable in S)
        # But we don't have the bitset for S handy in the inner loop without recomputing or storing it.
        # For speed, we should store the bitset.
        pass

    # Initial set: {66, ..., 100}
    S = set(range(66, 101))
    best_set = set(S)
    best_size = len(S)
    
    # We found {1, 2, 66, ..., 99} is valid size 36.
    # Let's use that as a strong initial seed.
    S = set([1, 2]) | set(range(66, 100))
    if is_valid(S):
        best_set = set(S)
        best_size = len(S)
    
    # Local search
    def improve(S):
        S = set(S)
        reachable = get_bitset(S)
        
        # Try to add elements
        added = True
        while added:
            added = False
            # Try adding any missing element
            # Sort by value or random? Random might find different local optima.
            # But we want max size. Adding any valid element increases size.
            for x in range(1, 101):
                if x not in S:
                    # Check validity of adding x
                    # new_reach = reachable | (reachable << x)
                    # if not (new_reach >> 200) & 1:
                    if not ((reachable >> (200 - x)) & 1):
                        # Adding x creates sum 200 iff 200-x is reachable in current S
                        # Wait, this logic is:
                        # Sums in S U {x} are Sums(S) U {s+x for s in Sums(S)}
                        # 200 is in Sums(S U {x}) iff 200 in Sums(S) OR (200-x in Sums(S))
                        # Since S is valid, 200 not in Sums(S).
                        # So we just check if 200-x is in Sums(S).
                        S.add(x)
                        reachable |= (reachable << x)
                        added = True
                        break # Restart loop to re-evaluate with new S
        return S

    def local_search(S):
        S = set(S)
        # First, greedily add
        S = improve(S)
        
        # Then try swaps to potentially allow more additions
        # We accept a swap if it keeps the set valid and size same,
        # and then we try to add again.
        # If adding fails, we might revert or keep the swap if it looks promising?
        # Simple strategy: Accept swap if valid. Then try to add.
        # If size increases, keep. If not, we are at a local optimum for size.
        
        max_iterations = 5000
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
                
            # Try to add
            S = improve(S)
            
            # Check if we can add anything
            reachable = get_bitset(S)
            can_add = False
            for x in range(1, 101):
                if x not in S:
                    if not ((reachable >> (200 - x)) & 1):
                        can_add = True
                        break
            
            if can_add:
                continue # improve() should have handled this, but just in case
            
            # No single addition possible. Try a swap.
            # Pick a random element to remove
            items = list(S)
            if not items:
                break
            y = random.choice(items)
            
            # Pick a random element to add
            missing = [x for x in range(1, 101) if x not in S]
            if not missing:
                break
            x = random.choice(missing)
            
            # Try swap S - {y} + {x}
            new_S = S - {y} | {x}
            if is_valid(new_S):
                # Accept swap
                S = new_S
                # Try to add again immediately
                S = improve(S)
                
        return S

    # Run multiple trials
    # Start with the known good set
    current_best = set(best_set)
    
    # Also try random greedy starts
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Random greedy seed
        elements = list(range(1, 101))
        random.shuffle(elements)
        S = set()
        reachable = 1
        for x in elements:
            if not ((reachable >> (200 - x)) & 1):
                S.add(x)
                reachable |= (reachable << x)
        
        S_improved = local_search(S)
        if len(S_improved) > len(current_best):
            current_best = S_improved
            
        # Also try improving the known best set
        if trial % 5 == 0:
            S_improved = local_search(best_set)
            if len(S_improved) > len(current_best):
                current_best = S_improved

    return list(current_best)
