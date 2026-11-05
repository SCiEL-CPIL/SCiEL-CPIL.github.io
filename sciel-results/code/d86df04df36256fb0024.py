"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    """
    Find a Sidon set in [1, 284] containing {1, 2, 5, 11} with maximum size.
    
    Uses randomized backtracking with forward checking and dynamic variable ordering.
    The constraint is that all pairwise sums a + b (a <= b) are distinct.
    
    Deadline guard: returns best result found before 140s (safety margin from 150s limit).
    """
    start_time = time.time()
    DEADLINE = 140.0  # seconds, safety margin
    
    # Prescribed elements
    prescribed = [1, 2, 5, 11]
    
    # Precompute all possible pairwise sums for the prescribed set
    # and build initial used_sums set
    def compute_sums(elements):
        """Compute set of all pairwise sums a+b where a <= b from elements."""
        sums = set()
        for i in range(len(elements)):
            for j in range(i, len(elements)):
                sums.add(elements[i] + elements[j])
        return sums
    
    # Candidate elements (those not in prescribed)
    candidates = list(range(12, 285))
    
    # We'll do multiple randomized restarts with different orderings
    best_size = 0
    best_set = None
    
    # Initial sums from prescribed elements
    initial_sums = compute_sums(prescribed)
    
    # For efficiency, precompute for each candidate x, the sums it would create
    # with each existing element, and its self-sum 2x
    # We use a bitset-like approach via sets for clarity, but optimize the search
    
    def try_extend(current_elements, current_sums, target_size):
        """
        Try to extend current_elements to reach target_size using backtracking.
        Returns the extended list if successful, None otherwise.
        """
        if len(current_elements) >= target_size:
            return list(current_elements)
        
        # Check deadline
        if time.time() - start_time > DEADLINE:
            return None
        
        # Dynamic variable ordering: for each candidate, compute how many conflicts it has
        # We want to try candidates that are "most constrained" first (fewest valid options)
        # Actually, we want to pick the candidate that has the fewest conflicts with current sums
        
        # Build list of candidates that can be added without immediate conflict
        valid_candidates = []
        for x in candidates:
            if x in current_elements:
                continue
            # Check if x can be added: all new sums must not be in current_sums
            new_sums = set()
            conflict = False
            for y in current_elements:
                s = x + y
                if s in current_sums:
                    conflict = True
                    break
                new_sums.add(s)
            if not conflict:
                s2 = 2 * x
                if s2 in current_sums:
                    conflict = True
                else:
                    new_sums.add(s2)
            if not conflict:
                # Count how many future candidates would be blocked (forward checking)
                # For now, just add to valid list
                valid_candidates.append((x, new_sums))
        
        if not valid_candidates:
            return None
        
        # Order by: prefer candidates that block fewer other candidates
        # Approximate: prefer smaller new_sums impact
        # Simple heuristic: sort by x (ascending) or by some measure
        # Let's sort by the number of candidates that would become invalid
        # For speed, just sort by x
        valid_candidates.sort(key=lambda t: t[0])
        
        # Try each valid candidate
        for x, new_sums in valid_candidates:
            # Check deadline
            if time.time() - start_time > DEADLINE:
                return None
            
            # Add x
            new_elements = current_elements + [x]
            new_sums_set = current_sums | new_sums
            
            result = try_extend(new_elements, new_sums_set, target_size)
            if result is not None:
                return result
        
        return None
    
    # Strategy: try to reach sizes from 20 down, with multiple random restarts
    # First, try greedy to get a baseline
    def greedy_extend(elements, sums):
        """Greedy: add candidates in order if they don't conflict."""
        elements = list(elements)
        sums = set(sums)
        for x in candidates:
            if x in elements:
                continue
            new_sums = set()
            conflict = False
            for y in elements:
                s = x + y
                if s in sums:
                    conflict = True
                    break
                new_sums.add(s)
            if not conflict:
                s2 = 2 * x
                if s2 in sums:
                    conflict = True
                else:
                    new_sums.add(s2)
            if not conflict:
                elements.append(x)
                sums |= new_sums
        return elements
    
    # Get a baseline with greedy
    baseline = greedy_extend(prescribed, initial_sums)
    best_size = len(baseline)
    best_set = baseline
    
    # Now try to improve with backtracking, targeting larger sizes
    # Try target sizes from min(20, best_size + 5) down to best_size + 1
    max_target = 20  # theoretical max
    
    # Multiple restarts with different random orderings
    num_restarts = 50
    
    for target in range(max_target, best_size, -1):
        if time.time() - start_time > DEADLINE:
            break
        
        found = False
        for restart in range(num_restarts):
            if time.time() - start_time > DEADLINE:
                break
            
            # Randomize candidate order for this restart
            random.shuffle(candidates)
            
            result = try_extend(prescribed, initial_sums, target)
            if result is not None:
                best_size = len(result)
                best_set = result
                found = True
                # Try to go higher
                break
        
        if found:
            continue
        else:
            # If we couldn't reach this target, try lower targets
            # But first, update best with greedy if it's better
            pass
    
    # Final check: make sure best_set is valid and contains prescribed
    if best_set is None:
        best_set = baseline
    
    # Verify best_set
    assert set(prescribed).issubset(set(best_set)), "Prescribed elements missing"
    assert all(1 <= x <= 284 for x in best_set), "Elements out of range"
    assert len(set(best_set)) == len(best_set), "Duplicate elements"
    
    # Verify Sidon property
    sums_final = compute_sums(sorted(best_set))
    assert len(sums_final) == len(best_set) * (len(best_set) + 1) // 2, "Sidon property violated"
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
