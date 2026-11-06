import math
import time

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    limit_time = start_time + 290.0

    primes = _primes(N)
    
    # Strategy 1: Minimal removal approach
    # For each prime, only remove enough points to ensure one residue class is empty
    # This is more precise than removing entire classes
    
    # Start with all points
    active = [True] * (N + 1)
    active_count = N + 1
    
    # For each prime, determine which residue class to empty
    # We want to choose classes that overlap as much as possible to minimize total removals
    
    # Precompute for each x and each prime p, the residue x % p
    # Process primes in increasing order
    for p in primes:
        if time.time() > limit_time:
            break
        if p > active_count:
            break
        
        # Count active points in each residue class mod p
        residue_counts = [0] * p
        for x in range(N + 1):
            if active[x]:
                residue_counts[x % p] += 1
        
        if min(residue_counts) > 0:
            # All classes are covered, need to remove some
            # Choose the class with fewest active points
            min_count = min(residue_counts)
            min_classes = [r for r in range(p) if residue_counts[r] == min_count]
            
            # Among min classes, choose the one that overlaps most with already-emptied classes
            # For now, just pick the first one (could be improved)
            chosen_class = min_classes[0]
            
            # Remove active points in this class, but only as many as needed
            # We need to remove ALL active points in this class to make it empty
            for x in range(chosen_class, N + 1, p):
                if active[x]:
                    active[x] = False
                    active_count -= 1
    
    # Build current set
    current_set = [x for x in range(N + 1) if active[x]]
    
    # Precompute residue coverage counts for each prime
    coverage = []
    for p in primes:
        counts = [0] * p
        for x in current_set:
            counts[x % p] += 1
        coverage.append(counts)
    
    # Number of missed classes per prime
    missed = [sum(1 for c in cov if c == 0) for cov in coverage]
    
    # Local improvement: try to add back removed points one at a time
    removed = [x for x in range(N + 1) if not active[x]]
    
    # Build a mapping: for each prime, which residue is the unique missed one (if any)
    unique_missed_res = [None] * len(primes)
    for i, cov in enumerate(coverage):
        zeros = [r for r, c in enumerate(cov) if c == 0]
        if len(zeros) == 1:
            unique_missed_res[i] = zeros[0]
    
    # For each removed point, count how many primes would become fully covered
    point_cost = {}
    for x in removed:
        cost = 0
        for i, p in enumerate(primes):
            if unique_missed_res[i] is not None and x % p == unique_missed_res[i]:
                cost += 1
        point_cost[x] = cost
    
    # Sort by cost (lower cost = safer to add)
    removed_sorted = sorted(removed, key=lambda x: point_cost[x])
    
    # Try adding points back in sorted order
    for x in removed_sorted:
        if time.time() > limit_time:
            break
        
        new_size = active_count + 1
        
        # Check if adding x back keeps the set admissible
        # For each prime p <= new_size, check if some residue class mod p is still missed
        valid = True
        for i, p in enumerate(primes):
            if p > new_size:
                break
            
            r = x % p
            # After adding x, coverage[i][r] increases by 1
            # If coverage[i][r] was 0, it becomes 1, so missed count decreases by 1
            # If coverage[i][r] was > 0, missed count stays the same
            
            old_missed = missed[i]
            if coverage[i][r] == 0:
                new_missed = old_missed - 1
            else:
                new_missed = old_missed
            
            if new_missed == 0:
                # All classes would be covered -> not admissible
                valid = False
                break
        
        if valid:
            active[x] = True
            active_count += 1
            current_set.append(x)
            
            # Update coverage and missed counts
            for i, p in enumerate(primes):
                r = x % p
                coverage[i][r] += 1
                if coverage[i][r] == 1:  # was 0, now 1
                    missed[i] -= 1
                # Update unique_missed_res if needed
                if missed[i] <= 1:
                    zeros = [r2 for r2, c in enumerate(coverage[i]) if c == 0]
                    if len(zeros) == 1:
                        unique_missed_res[i] = zeros[0]
                    else:
                        unique_missed_res[i] = None
    
    return sorted(current_set)
# EVOLVE-BLOCK-END
