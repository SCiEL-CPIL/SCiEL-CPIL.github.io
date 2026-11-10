import time
import sys

def construct():
    """
    Constructs a set of 12 distinct positive integers such that all 4096 subset
    sums are distinct, and the largest integer is minimized.
    
    Uses a greedy approach with backtracking to find the smallest possible
    next element that does not create duplicate subset sums.
    """
    n = 12
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    # Fallback solution: Superincreasing sequence (Powers of 2)
    # This guarantees a valid solution with max 2^11 = 2048
    fallback = [1 << i for i in range(n)]
    best_solution = fallback
    best_max_val = max(fallback)
    
    # State for backtracking search
    # sequence: list of chosen integers so far
    # sums: set of all subset sums of the current sequence
    # We will try to build the sequence element by element.
    
    def try_build(sequence, sums, index):
        """
        Recursively try to build the sequence.
        Returns True if a complete valid sequence is found.
        Modifies sequence and sums in-place if successful.
        """
        nonlocal best_solution, best_max_val
        
        if time.time() > deadline:
            return False
            
        if index == n:
            # Found a complete valid sequence
            current_max = max(sequence)
            if current_max < best_max_val:
                best_max_val = current_max
                best_solution = sequence[:]
            return True
            
        # Determine the range of candidates for the next element
        # It must be distinct from existing elements
        # It must be positive
        # Heuristic: Start from the smallest possible integer that is > max(sequence) 
        # or simply > 0 and not in sequence.
        # However, to minimize the max, we should try small numbers first.
        # But we must ensure distinctness from previous elements.
        
        # Lower bound for candidate:
        # If we want to minimize the final max, we should try small values early.
        # However, if we pick too small a value, we might block future options.
        # A safe lower bound is 1. But we must be distinct.
        # Let's iterate from 1 upwards.
        # Optimization: We know that if we have chosen k elements, the number of sums is 2^k.
        # The new element x must not create collisions.
        
        # To speed up, we can start from the minimum possible value.
        # Since elements must be distinct positive integers, and we want to minimize max,
        # we should try values starting from 1.
        
        # However, a strong lower bound for the next element in a *sorted* sequence 
        # that minimizes max is not necessarily 1.
        # But since we are building a set, order doesn't matter for validity, 
        # but for the search tree, we can assume w.l.o.g. that we build in increasing order?
        # No, the greedy choice for "smallest valid integer" doesn't require the sequence to be sorted 
        # in the input, but the set of sums is the same.
        # If we assume the final set is sorted a_1 < a_2 < ... < a_n, 
        # then a_i must be distinct.
        
        # Let's assume we are building a sorted sequence.
        # If index == 0, candidate starts at 1.
        # If index > 0, candidate must be > sequence[index-1] to maintain sorted order and distinctness.
        
        start_val = 1 if index == 0 else sequence[-1] + 1
        
        # Upper bound for candidate:
        # We only care if it's better than the current best_max_val.
        # If we are at index i, and we pick x, the final max will be at least x.
        # If x >= best_max_val, it's not better (unless we are just trying to find *any* solution 
        # but we already have the fallback). 
        # Actually, we want to find a solution with max < best_max_val.
        # So we can prune candidates x >= best_max_val.
        
        end_val = best_max_val  # We only want to try values strictly less than current best max
        
        for x in range(start_val, end_val):
            if time.time() > deadline:
                return False
                
            # Check if adding x creates any duplicate subset sums
            # New sums = {s + x for s in sums}
            # Collision if any new sum is already in sums
            
            # Optimization: 
            # If min(sums) + x > max(sums), then no collision? 
            # No, sums are not contiguous.
            # We must check intersection.
            
            # Efficient check:
            # We need to verify that (sums + x) is disjoint from sums.
            # i.e., for all s in sums, (s + x) not in sums.
            
            # Since sums can be large, iterating over all sums is O(2^index).
            # For index up to 11, 2^11 = 2048. This is fast.
            
            has_collision = False
            # We can break early if a collision is found
            for s in sums:
                if (s + x) in sums:
                    has_collision = True
                    break
            
            if not has_collision:
                # No collision, proceed with this x
                sequence.append(x)
                # Update sums
                new_sums = set(sums)
                for s in sums:
                    new_sums.add(s + x)
                sums = new_sums
                
                # Recurse
                if try_build(sequence, sums, index + 1):
                    return True
                
                # Backtrack
                sequence.pop()
                # Restore sums (we passed sums by value/reference? 
                # In Python, set is mutable. We created new_sums, so old sums is intact 
                # if we don't modify it in place. 
                # Wait, I did `sums = new_sums` inside the block? 
                # No, `sums` is a local variable in the function scope if passed by reference?
                # In Python, if I do `sums = new_sums`, I am reassigning the local variable.
                # The caller's `sums` remains unchanged.
                # So backtracking is just `sequence.pop()`. The `sums` variable in the 
                # caller's frame is still the old set.
                
        return False

    # Initialize search
    initial_sequence = []
    initial_sums = {0}
    
    # Run the search
    try_build(initial_sequence, initial_sums, 0)
    
    # If the search didn't find anything better or timed out, return the best found
    # best_solution is initialized to fallback, so it's always valid.
    
    return best_solution

# EVOLVE-BLOCK-END
