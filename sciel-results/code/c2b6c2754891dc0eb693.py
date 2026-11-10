import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit
    
    # LCM of 1..22
    # 2^4 * 3^2 * 5 * 7 * 11 * 13 * 17 * 19
    lcm_val = 232792560
    
    # Precompute weights: w_i = LCM / i
    weights = [0] * 23
    for i in range(1, 23):
        weights[i] = lcm_val // i
        
    # Helper to compute subset sums of a set of indices
    def get_subset_sums(indices):
        sums = {0}
        for i in indices:
            w = weights[i]
            # Optimization: if len(sums) is large, this is expensive.
            # But max size is ~18, so 2^18 = 262k.
            # We use a set of integers.
            new_sums = set()
            for s in sums:
                new_sums.add(s + w)
            sums.update(new_sums)
        return sums

    def is_valid(indices):
        """Returns True if all subset sums are distinct."""
        if not indices:
            return True
        sums = get_subset_sums(indices)
        return len(sums) == (1 << len(indices))

    def greedy_construct(order):
        """
        Builds a valid set by greedily adding elements in the given order.
        Returns (set, sums).
        """
        S = []
        sums = {0}
        for x in order:
            w = weights[x]
            # Check if adding x causes collision
            # Collision if (sums + w) intersects sums
            # We can check this by iterating sums, but it's O(2^k)
            # Optimization: 
            # If w is very large, it's less likely to collide.
            # Let's just do the check.
            collision = False
            # To speed up, we can check if any s + w is in sums
            # This is O(N) where N is number of sums.
            # We do this for each candidate.
            for s in sums:
                if (s + w) in sums:
                    collision = True
                    break
            
            if not collision:
                S.append(x)
                # Update sums
                # new_sums = sums U (sums + w)
                # Since we verified no collision, len(new_sums) = 2 * len(sums)
                # We can update in place or create new.
                # Creating new is safer.
                sums = sums.union({s + w for s in sums})
                
        return S, sums

    def local_search(S, sums, time_limit):
        """
        Tries to improve the set S by adding elements or swapping.
        Returns (S, sums, improved_flag)
        """
        S = list(S)
        sums = set(sums)
        remaining = set(range(1, 23)) - set(S)
        size = len(S)
        no_improve_count = 0
        max_no_improve = 200 # Reset counter to allow more exploration
        
        while time.time() < time_limit:
            # 1. Try to add any remaining element
            added = False
            # Shuffle remaining to avoid bias in local search
            rem_list = list(remaining)
            random.shuffle(rem_list)
            
            for x in rem_list:
                w = weights[x]
                collision = False
                for s in sums:
                    if (s + w) in sums:
                        collision = True
                        break
                
                if not collision:
                    S.append(x)
                    sums = sums.union({s + w for s in sums})
                    remaining.remove(x)
                    size += 1
                    added = True
                    no_improve_count = 0
                    break # Restart add-loop to check for more
            
            if added:
                continue
                
            # 2. If no addition possible, try swaps
            # Swap: remove i from S, add j from remaining
            # This requires rebuilding sums for the candidate set.
            # To be efficient, we only do this if we are stuck.
            
            if not S or not remaining:
                break
                
            # Randomly pick a swap candidate
            # To speed up, we can try a few random swaps before giving up
            found_swap = False
            for _ in range(10): # Try up to 10 random swaps per "step"
                i_idx = random.randint(0, len(S)-1)
                j = random.choice(list(remaining))
                
                # Candidate set: S with S[i_idx] replaced by j
                # We need to check if this new set is valid.
                # Rebuilding sums is O(2^k).
                # k is ~16-18. 2^18 = 262k.
                # This might be too slow if done too often.
                # Let's limit this.
                
                candidate_S = list(S)
                candidate_S[i_idx] = j
                
                # Quick check: is j the same as the one removed? (Already distinct by sets)
                # Check validity
                # Optimization: We can check if the new set is valid by computing sums.
                # But we can also check if the swap "resolves" a potential block?
                # No, we just need validity.
                
                # To save time, we only do this swap check if we've been stuck for a while
                # or if we haven't tried many swaps.
                
                if no_improve_count > 10:
                    # Perform expensive validity check
                    candidate_sums = get_subset_sums(candidate_S)
                    if len(candidate_sums) == (1 << len(candidate_S)):
                        # Valid swap
                        S = candidate_S
                        sums = candidate_sums
                        remaining = set(range(1, 23)) - set(S)
                        found_swap = True
                        no_improve_count = 0
                        break
            
            if found_swap:
                continue
            
            # 3. If no swap worked, perturb (restart local search phase slightly)
            # Or just increment no_improve
            no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                # Break out of local search to allow a new greedy restart
                break

        return S, sums

    best_set = []
    best_sums = {0}
    best_size = 0
    
    # Run multiple iterations with different random orders for greedy
    # The greedy order matters. Increasing index is good, but random might find different basins.
    
    search_deadline = start_time + 50
    
    while time.time() < search_deadline:
        # Generate a random order for greedy
        # Bias towards smaller numbers (larger weights) as they are safer
        # But allow randomness
        candidates = list(range(1, 23))
        # Sort by weight descending, then add some random noise?
        # Or just shuffle.
        # Actually, sorting by weight descending (index ascending) is the strongest heuristic.
        # Let's try:
        # 1. Standard greedy (1..22)
        # 2. Randomized greedy (shuffle 1..22)
        
        if random.random() < 0.5:
            order = list(range(1, 23))
        else:
            order = list(range(1, 23))
            random.shuffle(order)
        
        S, sums = greedy_construct(order)
        
        if len(S) > best_size:
            best_set = S
            best_sums = sums
            best_size = len(S)
        
        # Local search to improve
        if time.time() < search_deadline:
            S_ls, sums_ls = local_search(S, sums, search_deadline)
            
            if len(S_ls) > best_size:
                best_set = S_ls
                best_sums = sums_ls
                best_size = len(S_ls)
                
        # If we found a very large set, we might stop early?
        # Max possible is 22, but likely 17-18.
        # If we hit 18, we can stop.
        if best_size >= 18:
            break

    # Final validation
    if best_set:
        if not is_valid(best_set):
            # Should not happen, but trim if it does
            while best_set and not is_valid(best_set):
                best_set.pop()
        
    return best_set
