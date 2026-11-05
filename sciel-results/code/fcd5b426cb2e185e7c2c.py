import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    # Initial greedy solution (similar to seed but maybe optimized)
    def greedy():
        values = []
        sums = set()
        for x in range(1, 257):
            tails = [x]
            total = x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values
    
    best_seq = greedy()
    best_len = len(best_seq)
    
    # DFS to try to find longer sequences
    # State: current_marks (list of prefix sums, starts with 0), used_diffs (set)
    # We want to maximize the number of marks (which is len(current_marks) - 1)
    
    # To make DFS efficient, we can limit the depth we explore if we don't beat best_len
    # We will try to find a sequence of length best_len + 1, then +2, etc.
    
    # Helper to check if adding gap g to current_marks is valid
    def is_valid_gap(current_marks, used_diffs, g):
        new_mark = current_marks[-1] + g
        # Check new differences: new_mark - m for all m in current_marks
        for m in current_marks:
            d = new_mark - m
            if d in used_diffs:
                return False, None
        return True, new_mark

    # DFS function
    # max_depth: the target length we are trying to achieve
    # current_marks: list of marks
    # used_diffs: set of differences
    # found: list to store the found sequence
    
    def dfs(current_marks, used_diffs, target_len):
        nonlocal best_seq, best_len, deadline
        
        if time.time() > deadline:
            return False
            
        current_len = len(current_marks) - 1
        
        if current_len >= target_len:
            # We found a sequence of at least target_len
            # Extract gaps
            gaps = [current_marks[i+1] - current_marks[i] for i in range(current_len)]
            # Verify strictly increasing? No, the task says "strictly increasing sequence of integers".
            # Wait. The task says: "Return a strictly increasing sequence of integers in [1,256]."
            # This means the GAPS themselves must be strictly increasing?
            # "Return a strictly increasing sequence of integers in [1,256]."
            # Yes, $a_1 < a_2 < \dots < a_L$.
            # My Golomb Ruler formulation allows any gaps. I must enforce $a_i < a_{i+1}$.
            
            # Check if gaps are strictly increasing
            for i in range(current_len - 1):
                if gaps[i] >= gaps[i+1]:
                    return False
            
            # Check if all gaps in [1, 256]
            if gaps[0] < 1 or gaps[-1] > 256:
                return False
                
            best_seq = gaps
            best_len = current_len
            return True
            
        # Pruning: If we can't reach target_len with strictly increasing gaps, stop
        # Max possible gap is 256.
        # We need target_len - current_len more gaps.
        # The smallest possible next gap is current_marks[-1] - current_marks[-2] + 1 (if we require strictly increasing gaps)
        # Wait, the sequence of integers $a_i$ must be strictly increasing.
        # So $a_{current_len+1} > a_{current_len}$.
        
        last_gap = current_marks[-1] - current_marks[-2] if current_len > 0 else 0
        min_next_gap = last_gap + 1
        
        if min_next_gap > 256:
            return False
            
        # Also, we need to fit enough gaps.
        # The minimum sum of the remaining gaps (assuming strictly increasing)
        # is min_next_gap + (min_next_gap+1) + ...
        # This doesn't constrain us directly because total sum is not bounded, only gap size.
        
        # Try next gaps from min_next_gap to 256
        # To find a solution faster, we might want to try larger gaps first?
        # Or smaller?
        # Larger gaps spread the ruler, which might help with distinct differences.
        # But larger gaps also increase the total sum, which doesn't hurt.
        # However, if we pick very large gaps, we might "jump over" potential conflicts?
        # Let's try descending order (larger first) to spread out quickly.
        
        for g in range(256, min_next_gap - 1, -1):
            if time.time() > deadline:
                return False
                
            valid, new_mark = is_valid_gap(current_marks, used_diffs, g)
            if valid:
                new_diffs = set()
                for m in current_marks:
                    new_diffs.add(new_mark - m)
                
                # Recurse
                if dfs(current_marks + [new_mark], used_diffs.union(new_diffs), target_len):
                    return True
                    
        return False

    # Try to beat the greedy result
    # Greedy likely gives ~8-10.
    # Let's try to find length best_len + 1
    # If found, try best_len + 2, etc.
    
    # To avoid getting stuck in a deep unfruitful branch, we can limit the search depth or use a timeout.
    # We will attempt to find sequences of increasing length.
    
    # Initialize state
    # We start with 0.
    # First gap must be >= 1.
    
    # We'll run DFS for target_len = best_len + 1
    # If found, update best and try for +2
    
    # Optimization: Randomize the order of gaps if we want to explore different branches?
    # For now, deterministic descending order.
    
    # Note: The "strictly increasing" constraint on the sequence $a_i$ is crucial.
    # The seed program did NOT enforce strictly increasing gaps!
    # "Return a strictly increasing sequence of integers".
    # Let's re-read the seed code.
    # `for x in range(1, 257): ... if valid: values.append(x)`
    # Since it iterates $x$ in increasing order, the resulting `values` list IS strictly increasing.
    # So the seed already satisfies the strictly increasing constraint.
    
    # My DFS must also enforce strictly increasing gaps.
    
    # Let's start the search
    # We know best_len from greedy.
    # Try to find best_len + 1
    
    # To make it faster, we can use a heuristic:
    # If we are close to the deadline, stop.
    
    # We'll try a few target lengths
    # Start with target = best_len + 1
    
    # We need to be careful with recursion depth.
    # sys.setrecursionlimit(1000)
    
    # Let's try to find a sequence of length 10, 11, 12...
    # Up to some limit, say 15, or until time runs out.
    
    # We'll use a loop to increment target length
    # But DFS is expensive.
    # Maybe we should just run one DFS that tries to maximize length?
    # Standard DFS for max length:
    # Explore all branches, keep track of max length found.
    # But the space is huge.
    
    # Better approach:
    # Try to find a valid sequence of length L.
    # Start L = best_len + 1.
    # If found, L++.
    # If not found, break (assuming if we can't find L, we can't find L+1? Not necessarily, but likely).
    
    # Let's try L = best_len + 1, then +2, etc.
    # To avoid hanging, we limit the time for each L.
    
    time_per_target = 10.0
    
    for target_len in range(best_len + 1, best_len + 10):
        if time.time() > deadline - 5.0:
            break
            
        # Reset DFS state
        # We can pass a timeout to DFS
        target_deadline = time.time() + time_per_target
        
        # Modify DFS to check this specific deadline
        def dfs_limited(current_marks, used_diffs, target_len, deadline):
            if time.time() > deadline:
                return False
                
            current_len = len(current_marks) - 1
            
            if current_len >= target_len:
                gaps = [current_marks[i+1] - current_marks[i] for i in range(current_len)]
                # Check strictly increasing
                for i in range(current_len - 1):
                    if gaps[i] >= gaps[i+1]:
                        return False
                if gaps[0] < 1 or gaps[-1] > 256:
                    return False
                nonlocal best_seq, best_len
                best_seq = gaps
                best_len = current_len
                return True
                
            last_gap = current_marks[-1] - current_marks[-2] if current_len > 0 else 0
            min_next_gap = last_gap + 1
            
            if min_next_gap > 256:
                return False
                
            # Try gaps
            # Heuristic: Try larger gaps first to spread out
            for g in range(256, min_next_gap - 1, -1):
                if time.time() > deadline:
                    return False
                valid, new_mark = is_valid_gap(current_marks, used_diffs, g)
                if valid:
                    new_diffs = set()
                    for m in current_marks:
                        new_diffs.add(new_mark - m)
                    if dfs_limited(current_marks + [new_mark], used_diffs.union(new_diffs), target_len, deadline):
                        return True
            return False

        found = dfs_limited([0], set(), target_len, target_deadline)
        if not found:
            # If we failed to find a sequence of this length, we might stop
            # But it's possible a different branch works.
            # Given the time constraint, we'll break.
            break

    return best_seq
