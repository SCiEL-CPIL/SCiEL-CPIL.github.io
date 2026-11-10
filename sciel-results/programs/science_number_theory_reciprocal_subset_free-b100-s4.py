import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # We want a subset S of [1, 24] such that for all x in S,
    # 1/x != sum(1/y for y in T) for any T subset of S \ {x}.
    # Maximize |S|.
    
    # Strategy: Greedy construction from large to small.
    # If we process x from 24 down to 1:
    # - When considering x, all elements in current_set are > x.
    # - Their reciprocals are < 1/x.
    # - We check if 1/x can be formed by a subset of current_set.
    # - If not, we add x.
    # - Adding a smaller x to a set of larger elements is safe for the existing larger elements
    #   because 1/y (for y > x) is smaller than 1/x, so 1/x cannot be part of a sum that equals 1/y
    #   (since all terms are positive, adding 1/x makes the sum even larger than 1/y if 1/x > 1/y? 
    #   Wait. If y > x, then 1/y < 1/x. 
    #   Can 1/y = sum(T) + 1/x where T subset of S \ {y, x}?
    #   Since 1/x > 1/y, 1/x alone is already larger than 1/y. 
    #   So 1/x + anything positive > 1/y. 
    #   Thus, 1/y cannot be formed using x. 
    #   So the only constraint to check is whether 1/x is formed by the existing set.
    
    # We maintain a set of all achievable subset sums of reciprocals of current_set.
    # Since the set size is at most 24, and we add elements one by one,
    # the number of achievable sums can grow exponentially but is bounded by 2^n.
    # For n=24, 2^24 is ~16 million, which might be too large for a Python set of Fractions.
    # However, the greedy approach might not add all 24.
    # Let's estimate the size of the set.
    # The reciprocals are distinct.
    # If the set is large, the number of subset sums could be large.
    # But we can prune sums that are > 1 (since we only care about sums equal to 1/x where x>=1, so 1/x <= 1).
    # Actually, we only care about sums that are <= 1.
    # More specifically, we only care about sums that are equal to 1/k for some k in [1, 24] not yet processed.
    # But we don't know which ones will be processed.
    # A simpler optimization: only keep sums <= 1.
    # Even better: only keep sums that are "relevant".
    # But let's try the simple set of Fractions first with pruning > 1.
    
    achievable_sums = {Fraction(0, 1)}
    current_set = []
    
    for x in range(24, 0, -1):
        if time.time() > deadline:
            break
            
        target = Fraction(1, x)
        
        # Check if target is in achievable_sums
        if target in achievable_sums:
            # Cannot add x
            continue
        else:
            # Add x
            current_set.append(x)
            # Update achievable_sums
            # New sums are old_sums + target
            # Prune sums > 1 to keep the set small
            new_sums = set()
            for s in achievable_sums:
                ns = s + target
                if ns <= 1:
                    new_sums.add(ns)
            achievable_sums.update(new_sums)
            
            # Optional: If the set of achievable sums becomes too large, we might need a different approach.
            # But let's see how far we get.
            
    # If the greedy approach didn't find a good set, or if we want to try to improve,
    # we could try other orders or local search.
    # But the greedy from large to small is a very strong heuristic for this type of problem.
    # Let's verify the result size.
    # If we start with 24, 23, ..., and skip some, we might get a set of size ~12-15.
    # The seed program returns 12 elements (13..24).
    # The current program score is 1.5833, which is 12/7.5? No, score is value/seed.
    # Seed value is 12. Score 1.5833 means value is 12 * 1.5833 = 19? No.
    # "search score is value/seed". Seed program returns list(range(13, 25)) which is 12 elements.
    # So seed value is 12.
    # If score is 1.5833, then value = 12 * 1.5833 = 19.
    # So the current program found a set of size 19.
    # This is much better than the seed.
    
    # We should ensure we return a valid set.
    # The greedy construction guarantees validity by construction.
    
    if not current_set:
        return list(range(13, 25))
        
    return current_set
