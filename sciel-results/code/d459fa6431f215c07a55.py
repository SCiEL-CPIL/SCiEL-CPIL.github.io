"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
from collections import defaultdict

def construct():
    """
    Find the largest Sidon set containing {1, 2, 5, 11} in [1, 284].
    
    Approach:
    1. Start with the mandatory elements {1, 2, 5, 11}.
    2. Compute all pairwise sums (a+b with a<=b) for the current set.
    3. For each candidate x in [12, 284], check if it can be added without conflict.
       A candidate x conflicts if:
       - x is already in the set
       - 2*x is already a sum
       - x + y is already a sum for any y in the set
    4. Use a greedy approach with backtracking to maximize the set size.
       Since the problem is NP-hard, we use a heuristic:
       - First, try to add candidates in order of "fewest conflicts" (most compatible).
       - If we reach a dead end, backtrack and try different choices.
       
    Given the time constraint (120s), we use a sophisticated greedy with limited backtracking.
    """
    
    # Mandatory elements
    mandatory = [1, 2, 5, 11]
    current_set = set(mandatory)
    
    # Precompute sums for mandatory elements
    sums = set()
    for i in range(len(mandatory)):
        for j in range(i, len(mandatory)):
            sums.add(mandatory[i] + mandatory[j])
    
    # Candidate elements
    candidates = list(range(12, 285))
    
    # For each candidate, compute which sums it would introduce
    # and check compatibility with current sums
    
    def is_compatible(x, current_sums, current_set):
        """Check if x can be added to the Sidon set."""
        if x in current_set:
            return False
        # 2*x must not be in current_sums
        if 2 * x in current_sums:
            return False
        # x + y must not be in current_sums for any y in current_set
        for y in current_set:
            if x + y in current_sums:
                return False
        return True
    
    def get_new_sums(x, current_set):
        """Get the new sums introduced by adding x."""
        new_sums = {2 * x}
        for y in current_set:
            new_sums.add(x + y)
        return new_sums
    
    # Greedy with backtracking
    # We'll use a recursive approach with a time limit
    
    best_set = set(mandatory)
    best_size = len(mandatory)
    start_time = time.time()
    time_limit = 110  # seconds, leave 10s margin
    
    # Sort candidates by some heuristic: prefer smaller numbers? Or fewer conflicts?
    # Let's compute for each candidate how many existing elements it conflicts with
    # if we were to add it to the mandatory set
    
    # Actually, let's use a simpler but effective approach:
    # 1. Build a conflict graph between candidates
    # 2. Use greedy coloring or maximum independent set approximation
    
    # Conflict between two candidates x and y (x < y) exists if:
    # - x + y is a sum that would be created by either, causing collision
    # More precisely, when we have a set S, adding x and y:
    #   2x, 2y, x+y, and x+s, y+s for all s in S must be distinct
    # Two candidates x, y conflict if:
    #   2x == 2y (impossible since x != y)
    #   2x == x + y => x == y (impossible)
    #   2x == y + s for some s in S
    #   x + y == x + s => y == s (y in S)
    #   x + y == y + s => x == s (x in S)
    #   x + y == 2x => y == x (impossible)
    #   x + y == 2y => x == y (impossible)
    # So the main conflicts are with existing elements in S.
    
    # Let's use a different approach: iterative deepening with greedy
    # Try to build a set of size k, starting from k=15 and going up
    
    # Actually, let's just do a smart greedy:
    # At each step, pick the candidate that has the fewest conflicts with future candidates
    # This is a common heuristic for maximum independent set
    
    # Precompute pairwise conflicts between all candidates
    # Two candidates x < y conflict if they cannot both be in a Sidon set with the mandatory elements
    # But this depends on the other elements chosen...
    
    # Simpler: just do a greedy that picks the candidate with the most "room"
    # i.e., the one that blocks the fewest other candidates
    
    def count_conflicts(x, current_set, current_sums, all_candidates):
        """Count how many other candidates x would block."""
        count = 0
        new_sums = get_new_sums(x, current_set)
        for y in all_candidates:
            if y == x or y in current_set:
                continue
            # Check if y would conflict with x being added
            if 2 * y in new_sums or 2 * y in current_sums:
                count += 1
                continue
            conflict = False
            for s in current_set:
                if y + s in new_sums or y + s in current_sums:
                    conflict = True
                    break
            if y + x in new_sums or y + x in current_sums:
                conflict = True
            if conflict:
                count += 1
        return count
    
    # Greedy approach with lookahead
    current_set = set(mandatory)
    current_sums = set()
    for i in range(len(mandatory)):
        for j in range(i, len(mandatory)):
            current_sums.add(mandatory[i] + mandatory[j])
    
    remaining = list(range(12, 285))
    
    while remaining:
        if time.time() - start_time > time_limit:
            break
        
        # For each remaining candidate, check if it's compatible
        compatible = []
        for x in remaining:
            if is_compatible(x, current_sums, current_set):
                compatible.append(x)
        
        if not compatible:
            break
        
        # Pick the one with the fewest conflicts (most compatible)
        # Compute conflict count for each compatible candidate
        min_conflicts = float('inf')
        best_candidate = None
        for x in compatible:
            conflicts = count_conflicts(x, current_set, current_sums, remaining)
            if conflicts < min_conflicts:
                min_conflicts = conflicts
                best_candidate = x
        
        if best_candidate is None:
            break
        
        # Add the best candidate
        current_set.add(best_candidate)
        new_sums = get_new_sums(best_candidate, current_set - {best_candidate})
        current_sums |= new_sums
        remaining.remove(best_candidate)
    
    # Update best if we found something better
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = set(current_set)
    
    # Now try a backtracking approach to see if we can do better
    # Use the greedy result as a starting point and try to improve
    
    # Actually, let's try multiple random orderings and take the best
    import random
    random.seed(42)
    
    best_result = set(best_set)
    best_result_size = len(best_result)
    
    for trial in range(50):
        if time.time() - start_time > time_limit:
            break
        
        current_set = set(mandatory)
        current_sums = set()
        for i in range(len(mandatory)):
            for j in range(i, len(mandatory)):
                current_sums.add(mandatory[i] + mandatory[j])
        
        remaining = list(range(12, 285))
        random.shuffle(remaining)
        
        while remaining:
            if time.time() - start_time > time_limit:
                break
            
            # Find the first compatible candidate in the shuffled order
            added = False
            for i, x in enumerate(remaining):
                if is_compatible(x, current_sums, current_set):
                    current_set.add(x)
                    new_sums = get_new_sums(x, current_set - {x})
                    current_sums |= new_sums
                    remaining.pop(i)
                    added = True
                    break
            
            if not added:
                break
        
        if len(current_set) > best_result_size:
            best_result_size = len(current_set)
            best_result = set(current_set)
    
    return sorted(best_result)

# EVOLVE-BLOCK-END
