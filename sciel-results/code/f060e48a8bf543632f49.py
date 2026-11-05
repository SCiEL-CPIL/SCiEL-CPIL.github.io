import random

def construct():
    U = set(range(1, 101))
    target = 200
    
    # Initial valid set: Tail {66..100}
    # Min 3-sum is 66+67+68 = 201 > 200.
    S = set(range(66, 101))
    
    def has_subset_sum(s, t):
        bits = 1
        for x in s:
            bits |= (bits << x)
        return (bits >> t) & 1
        
    # Ensure initial is valid
    if has_subset_sum(S, target):
        S = set(range(67, 101))
        
    best_size = len(S)
    best_set = set(S)
    
    # Greedy addition
    # Try to add elements from 1 to 65
    # Order: Larger first? Smaller first?
    # Larger numbers in the "gap" (e.g., 65) are closer to the tail, 
    # so their conflicts involve smaller sums in the tail.
    # Smaller numbers (e.g., 1) involve large sums in the tail.
    # Both are risky.
    
    # Let's try adding in descending order of value in the gap.
    for x in range(65, 0, -1):
        if x in S:
            continue
        new_S = S.union([x])
        if not has_subset_sum(new_S, target):
            S = new_S
            if len(S) > best_size:
                best_size = len(S)
                best_set = set(S)
                
    # Local Search with Swaps
    # We want to see if we can increase size further by swapping.
    # Note: Swapping keeps size constant. 
    # So we only improve if we can add an element without removing one?
    # The greedy phase already tried to add without removing.
    # So swaps won't increase size, only potentially allow *future* additions.
    # But we already did the greedy pass.
    # However, the order of greedy addition matters.
    # Adding 65 might block adding 64.
    # If we swap 65 with something else, maybe 64 becomes addable?
    # But we lost 65.
    # To increase size, we need to find a sequence of swaps that ends up with a state 
    # where a new element can be added.
    
    # Since swaps don't change size, and we want to MAXIMIZE size,
    # we should focus on finding a configuration where we can add MORE elements.
    
    # Let's do a random restart / multiple greedy passes with different orders?
    # Or just a simple hill climbing on "validity + size"?
    
    # Actually, if the greedy pass found size K, can we get K+1?
    # Only if there is a set of size K+1.
    # The local search with swaps explores the space of sets of size K.
    # If we find a "better" set of size K (one that has more "room"), we might add more.
    
    # Let's run a few iterations of:
    # 1. Pick random x not in S.
    # 2. Try to add x. If fails, try to swap y in S with x.
    # 3. If swap successful, S changes.
    # 4. Try to add another element z.
    
    # To keep it simple and fast:
    for _ in range(200):
        # Try to add a random element
        not_in_S = list(U - S)
        if not not_in_S:
            break
            
        # Prioritize elements that are "easy" to add?
        # Just pick random.
        x = random.choice(not_in_S)
        
        # Try direct add
        if not has_subset_sum(S.union([x]), target):
            S = S.union([x])
            if len(S) > best_size:
                best_size = len(S)
                best_set = set(S)
            continue
            
        # Try swap
        # To be efficient, we don't check all y.
        # We pick a random y.
        y = random.choice(list(S))
        new_S = S - {y}
        new_S.add(x)
        if not has_subset_sum(new_S, target):
            S = new_S
            # Size didn't change, but structure did.
            # Check if we can add more now?
            # We'll loop and try to add again.
            
    return list(best_set)
