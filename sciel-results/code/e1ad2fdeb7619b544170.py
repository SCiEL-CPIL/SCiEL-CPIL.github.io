import itertools
import time
import random
from collections import Counter

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0

    all_tsubsets = list(itertools.combinations(range(v), t))
    all_tsubsets_set = set(all_tsubsets)
    n_tsub = len(all_tsubsets_set)

    best_blocks = None
    best_len = float('inf')

    trial = 0
    while time.time() < deadline:
        trial += 1
        blocks = _greedy_cover(v, k, t, all_tsubsets_set, deadline, trial)
        if blocks is not None:
            # Phase 1: Remove redundant blocks (greedy pass)
            blocks = _remove_redundant(blocks, v, k, t, deadline)

            # Phase 2: Local search / refinement
            # We attempt to reduce the count further by replacing blocks
            blocks = _local_search(blocks, v, k, t, all_tsubsets_set, deadline, trial)

            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks

        # Early termination if we hit the known best
        if best_len <= 76:
            break

    if best_blocks is None:
        return []

    return [list(b) for b in best_blocks]


def _greedy_cover(v, k, t, all_tsubsets_set, deadline, seed_val):
    """
    Greedy set cover. Returns a list of blocks (tuples) or None if timeout.
    """
    rng = random.Random(seed_val)
    uncovered = set(all_tsubsets_set)
    blocks = []
    points = list(range(v))

    while uncovered:
        if time.time() > deadline:
            return None

        # Pick a random uncovered t-subset as the seed for the block
        # This helps diversify the search compared to always picking the first
        seed_tuple = tuple(rng.choice(list(uncovered)))
        current_block_set = set(seed_tuple)

        # Greedily add points to maximize coverage of *uncovered* t-subsets
        while len(current_block_set) < k:
            if time.time() > deadline:
                return None

            best_point = -1
            best_gain = -1
            best_tiebreak = []

            # Pre-compute (t-1) subsets of current block for fast gain calculation
            # C(6,2)=15 subsets. For each candidate x, check if (subset + x) is uncovered.
            t_minus_1_subsets = list(itertools.combinations(sorted(current_block_set), t - 1))

            for x in points:
                if x in current_block_set:
                    continue

                gain = 0
                for sub in t_minus_1_subsets:
                    # sub is a tuple of t-1 elements from current_block_set
                    # We need to check if {sub...} U {x} is a t-subset and is uncovered
                    # Since x is not in current_block_set, x is not in sub.
                    new_subset = tuple(sorted(sub + (x,)))
                    if new_subset in uncovered:
                        gain += 1

                if gain > best_gain:
                    best_gain = gain
                    best_point = x
                    best_tiebreak = [x]
                elif gain == best_gain and gain > 0:
                    best_tiebreak.append(x)

            if best_point == -1:
                # No point adds coverage? This shouldn't happen if uncovered is non-empty
                # and we are within v points, but handle safety.
                # If no gain, just pick a random unused point to fill the block?
                # Actually, if gain is 0 for all, it means no new t-subsets can be formed?
                # This is impossible if k > t and we have room to add, unless all potential
                # new t-subsets are already covered.
                # If we can't add coverage, we might as well stop adding points?
                # No, we must have size k.
                # Let's just pick a random one to satisfy the size constraint, though it's suboptimal.
                available = [x for x in points if x not in current_block_set]
                if not available:
                     return None
                best_point = rng.choice(available)
                best_gain = 0

            current_block_set.add(best_point)

        block = tuple(sorted(current_block_set))
        blocks.append(block)

        # Remove all t-subsets covered by this new block
        for sub in itertools.combinations(block, t):
            uncovered.discard(sub)

    return blocks


def _remove_redundant(blocks, v, k, t, deadline):
    """
    Iteratively removes blocks that are redundant (every t-subset in them is covered by others).
    """
    if not blocks:
        return blocks

    # Count how many times each t-subset is covered
    # We only care if count >= 1.
    # Optimization: Use a set of "critical" subsets?
    # Simple approach: Count coverage for all.
    cover_count = Counter()
    for block in blocks:
        for sub in itertools.combinations(block, t):
            cover_count[sub] += 1

    improved = list(blocks)
    changed = True

    while changed:
        if time.time() > deadline:
            break
        changed = False
        for i in range(len(improved) - 1, -1, -1):
            if time.time() > deadline:
                break
            block = improved[i]
            is_redundant = True
            for sub in itertools.combinations(block, t):
                if cover_count[sub] <= 1:
                    is_redundant = False
                    break

            if is_redundant:
                # Remove block
                for sub in itertools.combinations(block, t):
                    cover_count[sub] -= 1
                improved.pop(i)
                changed = True

    return improved


def _local_search(blocks, v, k, t, all_tsubsets_set, deadline, seed_val):
    """
    Attempts to improve the covering by replacing blocks.
    Strategy:
    1. Identify blocks that are "low value" (cover few unique/critical t-subsets).
    2. Try to replace them with a different block that covers the same criticals
       and potentially more, or allows the old one to be removed if it becomes redundant?
       No, 1-for-1 replacement doesn't change count.
    
    Wait, 1-for-1 replacement preserves the number of blocks.
    The goal is to REDUCE the number of blocks.
    
    How do we reduce the number?
    We can only reduce if a block becomes redundant after modifying OTHERS?
    Or if we can find a block that covers everything another block covers PLUS more,
    allowing us to remove the second one? That's a 2-opt move (add one, remove one).
    
    Current flow:
    1. Greedy finds a valid cover.
    2. _remove_redundant removes any that are strictly redundant.
    3. If no block is redundant, we are in a local minimum for the "removal" operator.
    
    To escape, we need to modify the set such that a block becomes redundant.
    This usually involves:
    - Replacing Block A with Block A' such that A' covers all t-subsets of A,
      and A' helps cover some other Block B's critical t-subsets, making B redundant?
      If A' covers A's t-subsets, A is removed (count -1).
      If A' also covers B's t-subsets, B might become redundant (count -1).
      Net change: -2 blocks.
      
    Or simpler:
    - Replace Block A with A' such that A' covers all t-subsets of A.
      Then A is redundant? No, we replace A with A'.
      If A' covers all t-subsets of A, then A is redundant *if* we keep A?
      But we are replacing A.
      
    Let's look at the "swap_improve" from the prompt.
    It replaced a block with a better one.
    Did it reduce count?
    The prompt's `_swap_improve` did not explicitly reduce count, it just swapped.
    However, if the new block covers more, it *might* make another block redundant.
    The prompt's `_swap_improve` didn't check for subsequent redundancy removals immediately within the loop,
    but the main loop in `construct` called `_remove_redundant` AFTER `_swap_improve`.
    
    So the mechanism is:
    1. Swap a block B_old with B_new.
    2. B_new covers all t-subsets of B_old (so validity is maintained for those).
    3. B_new might cover t-subsets of OTHER blocks.
    4. Run `_remove_redundant`. If B_old was the only cover for some subsets, validity breaks?
       No, B_new covers B_old's subsets. So validity is maintained.
       If B_new covers subsets of B_other, and those were the only cover for B_other,
       then B_other becomes redundant and is removed.
       
    So the key is:
    1. Generate B_new that covers B_old's t-subsets.
    2. Prefer B_new that covers *more* t-subsets of *other* blocks (especially those with low coverage count).
    
    Let's implement this logic.
    """
    if not blocks:
        return blocks

    rng = random.Random(seed_val)
    improved = list(blocks)
    
    # Pre-compute t-subsets for all blocks
    block_tsubs = []
    for b in improved:
        block_tsubs.append(list(itertools.combinations(b, t)))
    
    # Build cover count
    cover_count = Counter()
    for subs in block_tsubs:
        for sub in subs:
            cover_count[sub] += 1
    
    n_blocks = len(improved)
    points = list(range(v))
    
    # We want to try to find a replacement for a block that allows removal of another.
    # Heuristic: Target blocks that are "expensive" (cover many unique things? No, we want to cover the things they cover, 
    # so that they can be removed? No, we replace A with A'. A' covers A. 
    # We hope A' also covers B. Then B is removed.
    # So we should try to replace A with A' that covers A AND B.
    # This is a "merge" operation.
    
    # To find such A', we can:
    # 1. Take union of t-subsets of A and B.
    # 2. Try to find a k-set containing all these t-subsets?
    #    That's very hard (Hitting set / Set cover inside a set cover).
    
    # Alternative heuristic from the prompt:
    # Replace A with A' that covers A's critical t-subsets.
    # If A' covers more of the *global* uncovered-by-others t-subsets, it's better?
    # No, we want to make *another* block redundant.
    
    # Let's try a specific move:
    # For each pair (A, B), try to find a block A' that contains A's t-subsets AND B's t-subsets?
    # If such a k-set exists, we can replace A with A' and remove B.
    # Count change: -1 (remove B) + 0 (replace A) = -1.
    
    # How to find A'?
    # A' must contain the union of points in A's critical t-subsets?
    # No, A' must contain each t-subset of A.
    # This implies A' must be a hitting set for the family of (k-t+1) complements?
    # Too complex.
    
    # Let's stick to the prompt's approach but enhanced:
    # Replace A with A' that covers A's t-subsets.
    # Score A' by how many t-subsets of *other* blocks it covers that are currently "weak" (count=1).
    # If A' covers all "weak" t-subsets of B, then B becomes redundant.
    
    # Optimization:
    # Identify "weak" blocks: blocks that have at least one t-subset with cover_count == 1.
    # For each weak block B, let C_B be the set of t-subsets in B with count == 1.
    # If we replace A with A' such that A' covers all t-subsets of A (so A is still covered)
    # AND A' covers all t-subsets in C_B, then B becomes redundant (assuming A' is the new cover for C_B).
    # Wait, if A' covers C_B, and A' is the *only* cover for C_B (since they were count 1 before, and we haven't added any other block, just swapped A->A'),
    # then C_B is covered by A'.
    # But B still contains C_B.
    # Does B have *any other* t-subsets with count 1?
    # If B has *only* t-subsets that are covered by A', then B is redundant.
    # But B has C(t, t) t-subsets.
    # So we need A' to cover *all* t-subsets of B?
    # If A' covers all t-subsets of B, then B is redundant.
    # So we need A' to cover:
    # 1. All t-subsets of A.
    # 2. All t-subsets of B.
    
    # This is the "Union" condition.
    # A' must be a k-set that is a superset of the union of all points in any t-subset of A or B?
    # No, A' must *contain* each t-subset.
    # This is a very strong condition.
    
    # Let's relax:
    # We just want to improve the "slack" or "efficiency".
    # The prompt's `_swap_improve` just looked for a block that covers the *critical* t-subsets of the old block.
    # It did not try to cover another block.
    # But it might have *incidentally* covered another block's t-subsets, leading to redundancy removal.
    
    # Let's implement a more directed search for redundancy:
    # 1. Find a block B with minimum "redundancy slack" (most t-subsets with count=1).
    # 2. Find another block A.
    # 3. Try to replace A with A' such that A' covers all t-subsets of B.
    #    If successful, B is removed.
    #    But A' must also cover all t-subsets of A (to keep A's contribution valid, or rather, A is replaced, so A's t-subsets are gone, but they must be covered by the new set).
    #    Actually, if we replace A with A', the set of blocks is {A'} U (Old \ {A}).
    #    For the cover to remain valid, every t-subset of A must be covered by some block in the new set.
    #    The blocks in (Old \ {A}) might already cover some t-subsets of A.
    #    The ones that are NOT covered by (Old \ {A}) must be covered by A'.
    #    Let U_A = { s in T(A) | cover_count(s) == 1 } (only covered by A).
    #    A' must cover U_A.
    #    Additionally, if we want to remove B, A' must cover U_B = { s in T(B) | cover_count(s) == 1 } (only covered by B).
    #    So A' must cover U_A U U_B.
    
    # Algorithm:
    # 1. Compute cover_count.
    # 2. For each block B, compute U_B (unique t-subsets).
    # 3. If any U_B is empty, B is already redundant (handled by _remove_redundant).
    # 4. For each pair (A, B) where A != B:
    #      Let Target = U_A U U_B.
    #      Try to find a k-set A' that covers Target.
    #      If found, replace A with A'.
    #      Update cover_count.
    #      Check if B is now redundant (U_B' is empty? i.e., all t-subsets of B are covered by new set).
    #      If B is redundant, remove B.
    #      Count decreases by 1.
    
    # How to find A' covering Target?
    # Target is a set of t-subsets.
    # We need a k-set containing all elements that appear in *any* subset in Target?
    # No, containing *each* subset in Target.
    # This is equivalent to finding a k-set that is a hitting set for the complements?
    # Or simply: The union of all points in Target is the *minimum* size required.
    # Let UnionPts = Union(s for s in Target).
    # If len(UnionPts) > k, impossible.
    # If len(UnionPts) <= k, we can just take UnionPts and pad with random points?
    # Wait, does UnionPts cover each s in Target?
    # Yes, if s is a subset of UnionPts.
    # Since UnionPts is the union of all s in Target, every s is a subset of UnionPts.
    # So UnionPts is a valid candidate!
    
    # So:
    # 1. Compute UnionPts for Target = U_A U U_B.
    # 2. If len(UnionPts) <= k:
    #      A' = UnionPts + random_padding to size k.
    #      Replace A with A'.
    #      Check if B is redundant.
    #      If yes, remove B.
    
    # This is a very powerful move if such pairs exist.
    
    # Let's implement this.
    
    # To speed up, we don't check all pairs (A, B).
    # We focus on B with large |U_B| (hard to cover, so high value to remove).
    # And A that is "close" to B (shares points)?
    
    # Let's try a few iterations of this.
    
    max_iter = 50
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        # Recompute cover count and unique subsets
        cover_count = Counter()
        for i in range(len(improved)):
            for sub in block_tsubs[i]:
                cover_count[sub] += 1
        
        # Find blocks with unique subsets
        unique_subs = [None] * len(improved)
        for i in range(len(improved)):
            unique_subs[i] = [s for s in block_tsubs[i] if cover_count[s] == 1]
            
        # If any block has empty unique_subs, it's redundant
        for i in range(len(improved)):
            if not unique_subs[i]:
                # Remove block i
                # Update cover_count
                for sub in block_tsubs[i]:
                    cover_count[sub] -= 1
                del improved[i]
                del block_tsubs[i]
                del unique_subs[i]
                # Break to restart the loop with new state
                break
        
        # If we removed a block, continue to next iteration of max_iter
        # (The `break` above breaks the `for i` loop, then we hit the `continue`? No, `break` exits `for i`.
        #  Then we need to `continue` the `for _ in range(max_iter)` loop?
        #  Actually, if we removed a block, we should re-evaluate.
        
        # Let's restructure:
        # 1. Check for redundant blocks. If found, remove one, update, continue outer loop.
        # 2. If no redundant blocks, try to find a merge move.
        
        redundant_found = False
        for i in range(len(improved)):
            if not unique_subs[i]:
                for sub in block_tsubs[i]:
                    cover_count[sub] -= 1
                del improved[i]
                del block_tsubs[i]
                redundant_found = True
                break
        
        if redundant_found:
            continue # Restart outer loop
            
        # No redundant blocks. Try merge.
        # Sort blocks by number of unique subsets (descending) to target "important" blocks first?
        # Or just iterate.
        n = len(improved)
        if n == 0:
            break
            
        # Pick a target block B
        # Heuristic: Pick B with max len(unique_subs[B])?
        # Or random?
        b_idx = rng.randint(0, n - 1)
        b_unique = unique_subs[b_idx]
        
        # If b_unique is empty, it would have been removed. So it's not empty.
        
        # Try to find a block A to replace such that A' covers U_A U U_B.
        # We iterate over candidate A's.
        a_found = False
        # Shuffle A indices to randomize
        a_candidates = list(range(n))
        rng.shuffle(a_candidates)
        
        for a_idx in a_candidates:
            if a_idx == b_idx:
                continue
            
            a_unique = unique_subs[a_idx]
            
            # Target is U_A U U_B
            # We need to check if Union(U_A U U_B) size <= k
            # Optimization: Just compute the union of points.
            union_points = set()
            for s in a_unique:
                union_points.update(s)
            for s in b_unique:
                union_points.update(s)
            
            if len(union_points) <= k:
                # Construct A'
                # A' must contain union_points.
                # Pad with random points from v.
                a_prime = list(union_points)
                while len(a_prime) < k:
                    p = rng.randint(0, v - 1)
                    if p not in a_prime:
                        a_prime.append(p)
                
                a_prime = tuple(sorted(a_prime))
                
                # Perform replacement
                # 1. Remove A's contribution
                for sub in block_tsubs[a_idx]:
                    cover_count[sub] -= 1
                    if cover_count[sub] == 0:
                        del cover_count[sub]
                        
                # 2. Add A' contribution
                new_tsubs = list(itertools.combinations(a_prime, t))
                for sub in new_tsubs:
                    cover_count[sub] = cover_count.get(sub, 0) + 1
                
                # Update structures
                improved[a_idx] = a_prime
                block_tsubs[a_idx] = new_tsubs
                
                # Check if B is now redundant
                # Recalculate unique subsets for B?
                # Just check if all t-subsets of B have cover_count >= 2?
                # Or check if any has cover_count == 1.
                # If all are >= 2, then B is redundant?
                # No, B is redundant if ALL its t-subsets are covered by OTHER blocks.
                # Since we only changed A, and A' covers U_B (by construction),
                # the t-subsets in U_B now have count >= 2 (one from B, one from A').
                # The other t-subsets of B were already covered by others (count >= 2 originally).
                # So yes, B should be redundant.
                
                # Let's verify:
                b_redundant = True
                for sub in block_tsubs[b_idx]:
                    if cover_count.get(sub, 0) <= 1:
                        b_redundant = False
                        break
                        
                if b_redundant:
                    # Remove B
                    for sub in block_tsubs[b_idx]:
                        cover_count[sub] -= 1
                        if cover_count[sub] == 0:
                            del cover_count[sub]
                    del improved[b_idx]
                    del block_tsubs[b_idx]
                    # Note: block_tsubs indices are now messed up if b_idx < a_idx?
                    # We deleted b_idx.
                    # If a_idx > b_idx, a_idx in improved is now a_idx - 1?
                    # But we already updated improved[a_idx] before deletion.
                    # Wait, we modified improved[a_idx] and block_tsubs[a_idx].
                    # Then we delete index b_idx.
                    # If b_idx < a_idx, the object at a_idx moves to a_idx - 1.
                    # But we don't use the index anymore in this iteration.
                    # We just set a_found = True and break.
                    
                    a_found = True
                    break
            # If len(union_points) > k, this A is not a candidate for this B
            
        if a_found:
            continue # Restart outer loop to re-evaluate
            
        # If no merge move found in this iteration, we might be stuck.
        # To avoid infinite loop of doing nothing, we can stop or randomize more.
        # Let's just stop local search if we make no progress?
        # Or continue for a few more iterations with different random B.
        # Since we pick random B each time, it will eventually try all.
        # But if no merge is possible, we waste time.
        
        # Simple exit condition: if we haven't changed anything in the last X iterations?
        # For now, just let the max_iter limit handle it.
        
    return improved
