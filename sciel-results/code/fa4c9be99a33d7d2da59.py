import time

def construct():
    n = 12
    max_limit = 1048576
    deadline = time.time() + 80  # Safety margin before 90s kill
    
    # Greedy minimal extension:
    # We want to pick 12 distinct positive integers a_1 < a_2 < ... < a_12
    # such that all 2^12 subset sums are distinct.
    # To minimize the largest element, we pick the smallest possible next integer
    # that doesn't create a collision with any existing subset sum.
    
    # Start with empty set of subset sums
    subset_sums = {0}
    chosen = []
    
    # We need to pick 12 numbers.
    # The first number can be 1 (smallest positive integer).
    # For each subsequent number, we try candidates starting from (last_chosen + 1)
    # and check if adding it to any existing subset sum creates a new collision.
    
    for i in range(n):
        # Start candidate from 1 if first, else last_chosen + 1
        # Actually, to ensure distinctness and minimality, we can start from 1 always,
        # but we must ensure the number is not already chosen and is larger than previous?
        # The problem says "distinct positive Python integers". It doesn't require sorted order in the output,
        # but for the greedy algorithm to work correctly for minimization, we should pick them in increasing order.
        # If we pick a smaller number later, we might have missed a better combination?
        # No, the property of distinct subset sums is symmetric with respect to the set of numbers.
        # However, the "greedy minimal extension" strategy works by building the set incrementally.
        # To minimize the maximum element, we should try to keep all elements as small as possible.
        # Picking the smallest valid next element is a standard greedy approach for this type of constraint satisfaction.
        
        # Let's maintain the set of chosen numbers to ensure distinctness.
        # We iterate candidate values starting from 1.
        # But we must be careful: if we pick 1, then 2, then 3...
        # Let's just try all candidates starting from 1, skipping those already in `chosen`.
        # However, to be efficient and ensure we don't get stuck in a bad local minimum for the *max* value,
        # sorting the chosen numbers is good. The greedy algorithm typically picks in increasing order.
        
        # Optimization: We know the minimum possible sum of k distinct positive integers is 1+2+...+k = k(k+1)/2.
        # The maximum sum of subset sums for a set S is sum(S).
        # If we pick numbers too small, we might run out of "space" in the sum range?
        # Actually, distinct subset sums require that the number of distinct sums (2^k) is <= max_sum + 1.
        # So sum(S) >= 2^k - 1.
        
        # Let's try candidates starting from 1.
        # To speed up, we can start from max(chosen) + 1 if we enforce increasing order.
        # Does enforcing increasing order lose optimality?
        # No, any set can be sorted. The subset sums are the same.
        # So we can assume a_1 < a_2 < ... < a_n.
        
        start_candidate = 1 if not chosen else chosen[-1] + 1
        
        while True:
            candidate = start_candidate
            # Check if candidate is already used? No, since we start from last+1, it's distinct.
            
            # Check if adding `candidate` creates any collision.
            # New subset sums will be {s + candidate for s in subset_sums}.
            # We need to ensure none of these are in the existing subset_sums.
            # And also, we need to ensure that the new sums don't collide with each other?
            # Since subset_sums are distinct, s1 + candidate == s2 + candidate implies s1 == s2, so new sums are distinct among themselves.
            # We only need to check if any (s + candidate) is in subset_sums.
            
            valid = True
            for s in subset_sums:
                if (s + candidate) in subset_sums:
                    valid = False
                    break
            
            if valid:
                # Accept this candidate
                chosen.append(candidate)
                # Update subset sums
                new_sums = set()
                for s in subset_sums:
                    new_sums.add(s + candidate)
                subset_sums.update(new_sums)
                break
            else:
                start_candidate += 1
                
                # Time check
                if time.time() > deadline:
                    # Fallback to powers of 2 if we run out of time
                    # This should never happen for n=12, but safe just in case.
                    return [1 << i for i in range(12)]
                
                # If candidate exceeds max_limit, we are in trouble?
                # The problem says integers <= 1048576.
                # Powers of 2 for n=12 max out at 2048, so we are well within bounds.
                if start_candidate > max_limit:
                    # This shouldn't happen for n=12
                    return [1 << i for i in range(12)]

    return chosen
