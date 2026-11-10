import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s margin
    
    # Seed solution for baseline
    # Greedy approach to get a decent start
    def greedy_construct():
        values, sums = [], set()
        for x in range(1, 257):
            tails, total = [x], x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values

    best_seq = greedy_construct()
    best_len = len(best_seq)
    
    # If we already have a good length, try to improve
    # Target lengths to try
    target_lengths = [best_len + 1, best_len + 2, best_len + 3]
    # Also try a bit higher if fast
    target_lengths = [11, 12, 13]
    
    # We will search for each target length
    # If we find a sequence of length L, we update best and try L+1
    
    current_target = best_len + 1
    if current_target < 11:
        current_target = 11
        
    # We'll iterate targets from high to low? No, low to high to ensure we don't miss a shorter one if the longer one is hard?
    # Actually, if we find 13, we are done. If we fail 13, we might find 12.
    # So try 13 first, then 12, then 11.
    # But we don't know if 13 exists.
    # Let's just run a DFS that maximizes depth, but with a pruning condition based on a target.
    
    # Better strategy:
    # 1. Try to find length 13.
    # 2. If fail, try 12.
    # 3. If fail, try 11.
    # 4. If fail, return best found (which is at least 10).
    
    def check_valid(seq, x):
        """Check if adding x to seq creates duplicate interval sums."""
        # Calculate new sums ending at x
        # seq is strictly increasing
        new_sums = []
        total = x
        new_sums.append(total)
        for a in reversed(seq):
            total += a
            new_sums.append(total)
        
        # We need to check if any of these new sums are already in the global sum set
        # But we don't pass the global set here to keep it pure? 
        # It's better to do this inside the DFS where we have the set.
        pass

    def dfs(seq, sum_set, next_min, target_len):
        """
        Recursively extend seq to reach target_len.
        Returns the sequence if found, None otherwise.
        """
        if time.time() > deadline:
            return None
            
        if len(seq) == target_len:
            return list(seq)
            
        # Pruning: Can we fit enough numbers?
        # We need target_len - len(seq) more numbers.
        # The smallest possible next numbers are next_min, next_min+1, ...
        # This doesn't help with sum collisions, but ensures we don't run out of range.
        remaining_needed = target_len - len(seq)
        if next_min + remaining_needed - 1 > 256:
            return None

        # Try each possible next value
        for x in range(next_min, 257):
            # Calculate new sums
            new_sums = []
            total = x
            new_sums.append(total)
            for a in reversed(seq):
                total += a
                new_sums.append(total)
                
            # Check for collisions with existing sums
            # If any new sum is in sum_set, invalid
            # Also, new_sums are distinct from each other because seq is strictly increasing positive
            has_collision = False
            for s in new_sums:
                if s in sum_set:
                    has_collision = True
                    break
                    
            if has_collision:
                continue
                
            # Valid move
            seq.append(x)
            sum_set.update(new_sums)
            
            result = dfs(seq, sum_set, x + 1, target_len)
            
            if result is not None:
                return result
            
            # Backtrack
            seq.pop()
            for s in new_sums:
                sum_set.remove(s)
                
        return None

    # Try targets in descending order to find the longest possible quickly
    # If 13 is found, we stop.
    # If 13 is not found, we try 12.
    # We must be careful with time.
    
    targets_to_try = [13, 12, 11]
    
    for target in targets_to_try:
        if time.time() > deadline:
            break
            
        if target <= best_len:
            continue
            
        # Reset for this target search
        # We can start from empty or from a good prefix?
        # Starting from empty is safer for correctness of the specific target search.
        # But we can use the greedy result as a hint?
        # Let's just run DFS from scratch for each target.
        # Optimization: Use the greedy sequence as a starting point if its length is close?
        # No, DFS from scratch with good pruning is standard.
        
        # To speed up, we can try to initialize with the first few elements of the greedy solution?
        # The greedy solution is valid. If we start with its prefix, we save some depth.
        # However, the greedy solution might be a dead end for longer lengths.
        # Let's just do a full DFS.
        
        # One optimization: Order of x.
        # Trying smaller x first might lead to "tighter" sequences which might be easier to extend?
        # Or larger x?
        # Let's stick to ascending x.
        
        found = dfs([], set(), 1, target)
        
        if found:
            best_seq = found
            best_len = len(found)
            # If we found a longer sequence, we can break and return it, 
            # or try even longer? 
            # The loop goes 13, 12, 11. 
            # If we find 13, we break.
            # If we fail 13, we try 12.
            # So we should break if we find the highest target we are trying?
            # No, if we find 13, we break the loop.
            # If we find 12, we continue to 11? No, 12 > 11, so we should keep 12 and not bother with 11.
            # So if we find a sequence of length `target`, and `target` is the current highest we are looking for, 
            # we should update best and break the loop?
            # Wait, the loop is 13, 12, 11.
            # If we find 13, we update best to 13 and break.
            # If we fail 13, we try 12. If we find 12, we update best to 12 and break (since 11 is worse).
            # So we should break as soon as we find ANY valid sequence for the current target, 
            # because the targets are decreasing.
            break
            
    return best_seq

# EVOLVE-BLOCK-END
