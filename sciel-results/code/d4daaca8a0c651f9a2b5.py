"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time

def construct():
    """
    Construct a linear triple system on 16 points with small independence number.
    
    Strategy:
    1. Model as a graph where nodes are triples and edges represent compatibility (intersection <= 1).
    2. Use a randomized local search (Hill Climbing) to find a maximal linear system (clique)
       that minimizes the independence number.
    3. Independence number is computed by marking all supersets of the selected triples in a 2^16 array.
    """
    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    num_triples = len(all_triples)
    
    # Map triples to indices for faster lookup
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute compatibility matrix
    # compat[i][j] is True if triple i and triple j are compatible
    compat = [[False] * num_triples for _ in range(num_triples)]
    for i in range(num_triples):
        ti = set(all_triples[i])
        for j in range(i + 1, num_triples):
            tj = set(all_triples[j])
            if len(ti & tj) <= 1:
                compat[i][j] = True
                compat[j][i] = True
    # A triple is always compatible with itself? No, we select distinct triples.
    # But for clique logic, we just check pairs.

    def compute_independence_number(triple_indices):
        """
        Compute the independence number for a set of triple indices.
        Returns the size of the largest subset of points containing no triple from the set.
        """
        if not triple_indices:
            return 16
        
        # covered[mask] is True if the set of points represented by mask contains at least one triple
        covered = [False] * (1 << 16)
        
        for idx in triple_indices:
            t = all_triples[idx]
            mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            # Mark all supersets of this mask
            # Iterate over all subsets of the remaining 13 bits
            # There are 2^13 = 8192 supersets for each triple
            remaining_mask = ((1 << 16) - 1) ^ mask
            sub = remaining_mask
            while True:
                covered[mask | sub] = True
                if sub == 0:
                    break
                sub = (sub - 1) & remaining_mask
        
        # Find the largest mask that is NOT covered
        max_alpha = 0
        # We can iterate by popcount descending, but simple iteration is fast enough for 65536
        # To optimize, we can precompute popcounts or just check all.
        # 65536 checks is trivial.
        for mask in range(1 << 16):
            if not covered[mask]:
                popcount = bin(mask).count("1")
                if popcount > max_alpha:
                    max_alpha = popcount
                    # Early exit if we hit 16? Unlikely.
                    # If we find 16, we can break? No, we want the max.
        
        return max_alpha

    def is_compatible_with_all(new_idx, current_indices):
        """Check if new_idx is compatible with all indices in current_indices."""
        for idx in current_indices:
            if not compat[new_idx][idx]:
                return False
        return True

    def greedy_construct(rng):
        """
        Construct a maximal linear system using greedy method.
        """
        available = list(range(num_triples))
        rng.shuffle(available)
        
        selected = []
        selected_set = set()
        
        for idx in available:
            if idx in selected_set:
                continue
            if is_compatible_with_all(idx, selected):
                selected.append(idx)
                selected_set.add(idx)
                
        return selected

    def local_search(initial_indices, rng, time_limit=10):
        """
        Improve the solution using local search.
        """
        current_indices = set(initial_indices)
        current_alpha = compute_independence_number(list(current_indices))
        
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            # Pick a random triple to remove
            if not current_indices:
                break
            to_remove = rng.choice(list(current_indices))
            current_indices.remove(to_remove)
            
            # Try to add a random compatible triple
            # Sample a few candidates
            added = False
            for _ in range(20): # Try 20 random candidates
                candidate = rng.randint(0, num_triples - 1)
                if candidate in current_indices:
                    continue
                if is_compatible_with_all(candidate, current_indices):
                    new_indices = set(current_indices)
                    new_indices.add(candidate)
                    # Compute new alpha
                    # Optimization: Only compute if size changed or we suspect improvement
                    # But alpha is the objective.
                    new_alpha = compute_independence_number(list(new_indices))
                    if new_alpha <= current_alpha:
                        current_indices = new_indices
                        current_alpha = new_alpha
                        added = True
                        break
            
            # If we didn't add anything, we just kept the smaller set?
            # Usually we want maximal systems. If we can't add, we might want to restore the removed one
            # or try a different removal.
            # For simplicity, if no improvement found, revert to previous state?
            # Actually, removing a triple reduces the set of forbidden masks, so alpha can only increase or stay same.
            # So removing a triple is bad for alpha. We should only remove if we can replace it with something better.
            # My loop above: remove, then try to add. If add succeeds and alpha <= old, keep.
            # If add fails, we have a smaller set with larger alpha. We should revert.
            # Let's restructure:
            # 1. Pick candidate to add.
            # 2. If compatible, add it.
            # 3. Compute alpha.
            # 4. If alpha improved, keep. Else, remove it.
            # This ensures we never make the system "worse" in terms of alpha, but might get stuck in local optima.
            
            # Let's restart the loop logic for clarity in the actual code below.
            pass

    # Revised Local Search Logic
    def local_search_v2(initial_indices, rng, time_limit=10):
        current_indices = set(initial_indices)
        current_alpha = compute_independence_number(list(current_indices))
        
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            # Try to add a random triple
            candidate = rng.randint(0, num_triples - 1)
            if candidate in current_indices:
                continue
            if is_compatible_with_all(candidate, current_indices):
                current_indices.add(candidate)
                new_alpha = compute_independence_number(list(current_indices))
                if new_alpha > current_alpha:
                    # Adding a triple should never increase alpha?
                    # Wait. Adding a triple adds more constraints (more masks become "covered").
                    # So the set of independent sets shrinks.
                    # Thus, alpha can only decrease or stay the same.
                    # So new_alpha <= current_alpha is always true.
                    # So we should always accept a valid addition?
                    # Yes, adding a triple cannot hurt the objective (minimize alpha).
                    # It might help or be neutral.
                    current_alpha = new_alpha
                # Always keep the added triple as it maintains linearity and doesn't increase alpha.
            else:
                # Try to swap: remove one, add another
                if current_indices:
                    to_remove = rng.choice(list(current_indices))
                    current_indices.remove(to_remove)
                    # Try to find a replacement
                    found_replacement = False
                    for _ in range(5):
                        cand = rng.randint(0, num_triples - 1)
                        if cand in current_indices:
                            continue
                        if is_compatible_with_all(cand, current_indices):
                            current_indices.add(cand)
                            new_alpha = compute_independence_number(list(current_indices))
                            # Compare with the state BEFORE the swap (which had to_remove)
                            # We don't have the old alpha easily accessible here without re-computing or storing.
                            # But we know old_alpha was current_alpha before removal.
                            # Wait, removing to_remove increases alpha (or keeps same).
                            # Adding cand decreases alpha (or keeps same).
                            # We want the final alpha to be <= the alpha of the state with to_remove.
                            # Let's just compute alpha of the new state.
                            # If it's better than or equal to the alpha we had before the swap attempt, we keep.
                            # But we lost the "before" value.
                            # Let's store prev_alpha.
                            pass
                    
                    # This logic is getting messy. Let's simplify.
                    # Just do "Add" moves. This builds a maximal clique.
                    # Then do "Swap" moves carefully.
                    pass
        
        return list(current_indices), current_alpha

    # Simplest robust approach:
    # 1. Run many random greedy constructions.
    # 2. For each, try to add random compatible triples until maximal.
    # 3. Evaluate alpha.
    # 4. Keep best.
    # 5. Optionally, do a few local search swaps on the best one.

    best_alpha = 17
    best_triples = []
    rng = random.Random(42)
    
    # Number of iterations to fit in time
    # Each iteration: Greedy (fast) + Alpha calc (fast).
    # Let's do 1000 iterations.
    
    for i in range(1000):
        # 1. Greedy construction
        current = greedy_construct(rng)
        
        # 2. Try to add more triples (greedy might stop early if order is bad? No, it checks all available)
        # My greedy_construct checks all available triples in random order.
        # It adds if compatible. This results in a maximal clique (no more can be added).
        
        # 3. Evaluate
        alpha = compute_independence_number(current)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = current
            # Optional: Local search on this best
            # Let's do a quick local search: try to swap triples to see if we can reduce alpha further
            # Since adding triples is always good, we are already maximal.
            # Swapping might allow adding *more* triples?
            # If we swap A for B, and B allows C to be added, we might get a larger clique.
            # Larger clique -> more constraints -> lower alpha.
            
            # Quick local search
            for _ in range(50):
                if not current:
                    break
                to_swap = rng.choice(current)
                current.remove(to_swap)
                
                # Try to find a replacement that allows adding more
                best_new = None
                best_new_alpha = alpha # Current alpha is a lower bound for improvement? No, alpha can go up if we remove.
                # Actually, we want to minimize alpha.
                # Let's just try random swaps and keep if alpha improves.
                
                for _ in range(5):
                    cand = rng.randint(0, num_triples - 1)
                    if cand in current:
                        continue
                    if is_compatible_with_all(cand, current):
                        new_current = current + [cand]
                        new_alpha = compute_independence_number(new_current)
                        if new_alpha < best_alpha:
                            best_alpha = new_alpha
                            best_triples = new_current
                            current = new_current
                            to_swap = None # Break inner loop
                            break
                        elif new_alpha == best_alpha and len(new_current) > len(current):
                            # Same alpha, more triples? Might help future swaps.
                            # For now, just keep if alpha is better.
                            pass
                
                if to_swap is not None:
                    # If we didn't find a better swap, put to_swap back?
                    # If we didn't improve, we should revert the removal.
                    # But we might have modified 'current' in the loop above if we found a better one.
                    # If we didn't find a better one, 'current' is missing 'to_swap'.
                    # We should put it back to restore the maximal state.
                    current.append(to_swap)

        # Early exit if we reach theoretical minimum?
        # For 16 points, what is the min alpha?
        # If we have a Steiner Triple System (impossible for 16), alpha would be small.
        # A (16, 3, 1) packing covers each pair at most once.
        # Max number of triples in a linear system on 16 points:
        # Each triple uses 3 pairs. Total pairs 120. Max 40 triples.
        # If we have 40 triples, they cover all 120 pairs.
        # This is a Steiner Triple System STS(16)? No, STS(v) exists for v=1,3 mod 6.
        # So we cannot cover all pairs. Max packing size is less than 40.
        # The independence number will likely be around 6-8.
        
        if best_alpha <= 5:
            break

    # Convert indices back to triples
    if best_triples:
        return [list(all_triples[i]) for i in best_triples]
    else:
        # Fallback
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]

# EVOLVE-BLOCK-END
