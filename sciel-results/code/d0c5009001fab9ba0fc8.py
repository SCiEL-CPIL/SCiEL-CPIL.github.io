import numpy as np
import time

def construct():
    base = 51975
    start_time = time.time()
    
    # Get all divisors > 1 of base
    divisors = [d for d in range(2, base + 1) if base % d == 0]
    # Sort by size - smaller moduli cover more residues on average
    divisors.sort()
    
    def evaluate_classes(classes):
        """Count uncovered residues."""
        covered = np.zeros(base, dtype=bool)
        for m, r in classes:
            covered[r::m] = True
        return (~covered).sum()
    
    # Start with all divisors, using a good initial residue assignment
    # Greedy initialization: for each modulus, pick the residue that covers the most uncovered
    # But since we use all, order matters. Let's do a simple greedy first pass.
    
    # Actually, let's start with a simple initialization:
    # For each modulus, pick residue 0 or a random one
    # Then use local search to improve
    
    # Better initialization: greedy selection of residues
    # Process moduli in some order, for each pick the residue that maximizes new coverage
    
    # Let's try a few different initializations and pick the best
    
    best_classes = None
    best_uncovered = base  # worst case
    
    # Initialization 1: All residues = 0
    classes_init1 = [[m, 0] for m in divisors]
    uncovered1 = evaluate_classes(classes_init1)
    
    # Initialization 2: Greedy - process moduli in descending order (larger first)
    # Actually, smaller moduli are more "flexible" in some sense
    # Let's try: process in ascending order, pick best residue for each
    
    covered = np.zeros(base, dtype=bool)
    classes_init2 = []
    for m in divisors:
        if time.time() - start_time > 10:
            break
        best_r = 0
        best_count = -1
        for r in range(m):
            count = (~covered[r::m]).sum()
            if count > best_count:
                best_count = count
                best_r = r
        classes_init2.append([m, best_r])
        covered[best_r::m] = True
    uncovered2 = (~covered).sum()
    
    # Initialization 3: Greedy - process in descending order
    covered = np.zeros(base, dtype=bool)
    classes_init3 = []
    for m in sorted(divisors, reverse=True):
        if time.time() - start_time > 20:
            break
        best_r = 0
        best_count = -1
        for r in range(m):
            count = (~covered[r::m]).sum()
            if count > best_count:
                best_count = count
                best_r = r
        classes_init3.append([m, best_r])
        covered[best_r::m] = True
    uncovered3 = (~covered).sum()
    
    # Pick the best initialization
    candidates = [(uncovered1, classes_init1), (uncovered2, classes_init2), (uncovered3, classes_init3)]
    candidates.sort(key=lambda x: x[0])
    best_uncovered, best_classes = candidates[0]
    
    print(f"Initial best uncovered: {best_uncovered}", flush=True)
    
    # Now perform local search
    # State: residues for each modulus
    # Move: change the residue of one modulus
    # Score: number of uncovered residues
    
    # Precompute for each modulus and residue, the set of residues it covers
    # Actually, we can compute coverage incrementally
    
    # Use a cover count array: cover_count[x] = number of classes covering x
    # When we change a class from (m, r_old) to (m, r_new):
    # - Decrease cover_count for all x ≡ r_old (mod m)
    # - Increase cover_count for all x ≡ r_new (mod m)
    # - Uncovered count changes based on transitions
    
    # Let's implement local search with all moduli active
    
    # Current state
    current_residues = [r for m, r in best_classes]
    current_moduli = [m for m, r in best_classes]
    n_mod = len(current_moduli)
    
    # Compute initial cover count
    cover_count = np.zeros(base, dtype=np.int32)
    for i in range(n_mod):
        m = current_moduli[i]
        r = current_residues[i]
        cover_count[r::m] += 1
    
    current_uncovered = (~np.any(cover_count > 0, axis=0)).sum() if base > 0 else 0
    # Actually, simpler:
    current_uncovered = (cover_count == 0).sum()
    
    print(f"Local search start: uncovered = {current_uncovered}", flush=True)
    
    # Local search: iterate, try changing each modulus's residue
    # Keep track of best found
    
    best_residues = current_residues.copy()
    best_uncovered_ls = current_uncovered
    
    # Use simulated annealing or simple hill climbing with restarts
    # Given time constraints, let's do multiple rounds of hill climbing
    
    max_rounds = 100
    round = 0
    
    while time.time() - start_time < 100:
        round += 1
        improved = False
        
        # Try each modulus in random order
        order = np.random.permutation(n_mod)
        
        for i in order:
            if time.time() - start_time > 105:
                break
            
            m = current_moduli[i]
            old_r = current_residues[i]
            
            # Try all possible residues for this modulus
            best_r = old_r
            best_delta = 0  # We want to minimize uncovered, so we want negative delta
            
            for r in range(m):
                if r == old_r:
                    continue
                
                # Compute delta in uncovered count
                # Old coverage: x ≡ old_r (mod m)
                # New coverage: x ≡ r (mod m)
                
                # For x in old coverage but not new: cover_count decreases
                # For x in new coverage but not old: cover_count increases
                
                # The change in uncovered:
                # - x that were uncovered and become covered: uncovered decreases
                # - x that were covered only by this class and become uncovered: uncovered increases
                
                # Efficient computation:
                # Let's compute the delta directly
                
                # Residues that are in old but not new:
                # x ≡ old_r (mod m) and x not ≡ r (mod m)
                # Residues that are in new but not old:
                # x ≡ r (mod m) and x not ≡ old_r (mod m)
                
                # This is complex. Let's just recompute for simplicity if m is small,
                # or use the cover_count array
                
                # Actually, let's compute delta by iterating over the affected residues
                # Affected residues: all x ≡ old_r (mod m) or x ≡ r (mod m)
                # Union of these two sets
                
                # For efficiency, let's compute:
                # delta_uncovered = 0
                # For each x in old_coverage:
                #   if x not in new_coverage:
                #     cover_count[x] -= 1
                #     if cover_count[x] == 0: delta_uncovered += 1
                # For each x in new_coverage:
                #   if x not in old_coverage:
                #     cover_count[x] += 1
                #     if cover_count[x] == 1: delta_uncovered -= 1
                
                # But we need to be careful about the order and the current state.
                # Let's simulate the change.
                
                # This is O(N/m) per trial, which is acceptable for small m.
                # For large m, it's faster.
                
                # Let's compute the delta without modifying the global state first
                # by checking the current cover_count values.
                
                # Actually, the cleanest way is to temporarily apply the change,
                # compute the new uncovered, then revert. But that's O(N) per trial.
                
                # Alternative: compute delta analytically.
                # Let A = set of x ≡ old_r (mod m)
                # Let B = set of x ≡ r (mod m)
                # Let C = A ∩ B (residues covered by both)
                # Let D = A \ C (only in old)
                # Let E = B \ C (only in new)
                
                # For x in D: cover_count decreases by 1.
                #   If cover_count[x] was 1, it becomes 0: uncovered increases by 1.
                # For x in E: cover_count increases by 1.
                #   If cover_count[x] was 0, it becomes 1: uncovered decreases by 1.
                
                # So: delta = |{x in D : cover_count[x] == 1}| - |{x in E : cover_count[x] == 0}|
                
                # Compute D and E:
                # D: x ≡ old_r (mod m) and x not ≡ r (mod m)
                # E: x ≡ r (mod m) and x not ≡ old_r (mod m)
                
                # Note: A and B have the same size (base/m each), and |A ∩ B| depends on gcd.
                # Actually, if old_r != r, then A and B are disjoint? No!
                # x ≡ a (mod m) and x ≡ b (mod m) implies a ≡ b (mod m).
                # So if a != b, A and B are disjoint.
                
                # Wait, that's true! If old_r != r, then no x can satisfy both
                # x ≡ old_r (mod m) and x ≡ r (mod m) unless old_r ≡ r (mod m).
                # Since 0 <= old_r, r < m and old_r != r, they are distinct residues.
                # So A and B are disjoint.
                
                # Therefore, D = A, E = B, and C = empty.
                
                # So: delta = |{x in A : cover_count[x] == 1}| - |{x in B : cover_count[x] == 0}|
                
                # A = old_r::m, B = r::m
                
                old_coverage = cover_count[old_r::m]
                new_coverage = cover_count[r::m]
                
                # Count how many in old_coverage are exactly 1
                loss = (old_coverage == 1).sum()
                # Count how many in new_coverage are exactly 0
                gain = (new_coverage == 0).sum()
                
                delta = loss - gain
                
                if delta < best_delta:
                    best_delta = delta
                    best_r = r
            
            # Apply the best move if it improves
            if best_r != old_r and best_delta < 0:
                # Update cover_count
                m = current_moduli[i]
                old_r = current_residues[i]
                new_r = best_r
                
                # Remove old coverage
                cover_count[old_r::m] -= 1
                # Add new coverage
                cover_count[new_r::m] += 1
                
                current_residues[i] = new_r
                current_uncovered += best_delta
                improved = True
                
                if current_uncovered < best_uncovered_ls:
                    best_uncovered_ls = current_uncovered
                    best_residues = current_residues.copy()
        
        if not improved:
            # No improvement in this round
            # Try a random perturbation to escape local minimum
            if time.time() - start_time > 105:
                break
            
            # Pick a random modulus and change its residue randomly
            i = np.random.randint(n_mod)
            m = current_moduli[i]
            old_r = current_residues[i]
            new_r = np.random.randint(m)
            
            if new_r != old_r:
                # Update cover_count
                cover_count[old_r::m] -= 1
                cover_count[new_r::m] += 1
                current_residues[i] = new_r
                # Recompute uncovered
                current_uncovered = (cover_count == 0).sum()
        
        # Check time
        if time.time() - start_time > 105:
            break
    
    # Final answer
    final_classes = [[current_moduli[i], current_residues[i]] for i in range(n_mod)]
    final_uncovered = evaluate_classes(final_classes)
    
    # Also check the best found during local search
    ls_classes = [[current_moduli[i], best_residues[i]] for i in range(n_mod)]
    ls_uncovered = evaluate_classes(ls_classes)
    
    if ls_uncovered < final_uncovered:
        final_classes = ls_classes
        final_uncovered = ls_uncovered
    
    # Compare with initial best
    if final_uncovered < best_uncovered:
        best_uncovered = final_uncovered
        best_classes = final_classes
    
    print(f"Final uncovered: {best_uncovered}", flush=True)
    
    return best_classes
