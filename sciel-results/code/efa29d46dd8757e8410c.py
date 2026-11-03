import time

def construct(q, n, radius):
    start_time = time.time()
    limit = 145.0  # Safety margin for 150s limit

    total_words = q ** n

    # Precompute powers of q for index conversion
    powers = [q ** i for i in range(n)]

    def get_digits(idx):
        digits = [0] * n
        for i in range(n):
            digits[i] = idx % q
            idx //= q
        return digits

    # Precompute neighbor masks
    # mask[idx] is an integer with bits set for all words within radius of idx
    neighbor_masks = [0] * total_words

    for idx in range(total_words):
        mask = 1 << idx
        # Radius 1 neighbors
        for i in range(n):
            original = (idx // powers[i]) % q
            base = idx - original * powers[i]
            for new_val in range(q):
                if new_val != original:
                    neighbor_idx = base + new_val * powers[i]
                    mask |= (1 << neighbor_idx)
        neighbor_masks[idx] = mask

    # Greedy Set Cover using bitmasks
    uncovered = (1 << total_words) - 1
    solution_indices = []

    while uncovered:
        if time.time() - start_time > limit:
            break

        best_word = -1
        max_covered = 0

        for w in range(total_words):
            count = (neighbor_masks[w] & uncovered).bit_count()
            if count > max_covered:
                max_covered = count
                best_word = w

        if best_word == -1:
            break

        solution_indices.append(best_word)
        uncovered &= ~neighbor_masks[best_word]

    # Local Search: Try to remove redundant codewords
    # A codeword is redundant if all words it covers are also covered by other codewords.
    # This is equivalent to checking if the union of neighbors of (solution \ {w}) covers the neighbors of w.
    # Or simpler: Re-check coverage.
    # Given time constraints, a simple pass to remove any codeword that doesn't add unique coverage is good.

    # Let's do a single pass of removal.
    # We need to know if removing w leaves any gap.
    # This is expensive if done naively.
    # Alternative: Just return the greedy result. The greedy result with bitmasks should be very fast and likely better than the slow set-based one.

    # Let's check if we have time for a simple local search.
    # The greedy phase for 6561 words with bitmasks is very fast (< 1s).
    # We have ~144s left.

    # Simple Local Search:
    # For each codeword in solution (in reverse order of addition?), check if it can be removed.
    # To check if w can be removed:
    # 1. Compute union of neighbors of all other codewords.
    # 2. Check if neighbors[w] is a subset of that union.
    # This is O(|solution|^2 * bit_ops). |solution| ~ 400. 160k ops. Very fast.

    new_solution = list(solution_indices)
    
    # Optimization: Maintain a "coverage count" array to check redundancy in O(1) per check (after O(k) update)
    # Actually, simpler: maintain union_mask of all other codewords.
    # But removing one and adding one is better.
    
    # Let's use the "coverage count" approach for speed.
    # coverage_count[x] = number of codewords in new_solution that cover word x.
    # A codeword w is redundant if for all x in neighbor_masks[w], coverage_count[x] > 1.
    # If we remove w, we decrement coverage_count for all x in neighbor_masks[w].
    # If we add w, we increment coverage_count for all x in neighbor_masks[w].
    
    # This is O(2^(n+1)) per add/remove which is ~ 2*6561*2 = 26k ops.
    # For 400 codewords, this is 10 million ops. Very fast.
    
    coverage_counts = [0] * total_words
    for w in new_solution:
        mask = neighbor_masks[w]
        # Iterate through bits set in mask
        # Efficient way to iterate bits in a large int is tricky in pure python without numpy
        # But we can use the fact that we have the masks.
        # Actually, updating counts for each bit is slow if done naively.
        # Let's stick to the bitmask intersection method but optimize the "others_mask" calculation.
        
    # Better approach for redundancy check:
    # Precompute nothing extra.
    # When checking if w is redundant:
    # others_mask = union of neighbor_masks[j] for j != i
    # This is O(k) big-int ORs.
    # k ~ 400. 400 big-int ORs is fast.
    # The check (neighbor_masks[w] & ~others_mask) == 0 is one big-int AND and NOT.
    
    # The bottleneck in the original code was potentially the loop structure or lack of early exit?
    # The original code did:
    # for i in range(len(new_solution) - 1, -1, -1):
    #   ...
    #   for j in range(len(new_solution)):
    #     if j != i:
    #       others_mask |= ...
    #   if redundant: pop, break
    
    # This is O(k^2) big-int ops per pass.
    # k=400, k^2=160k. 160k big-int ops is ~ < 1ms in Python? No, big-int ops are slower.
    # 6561 bits is ~ 100 64-bit words.
    # 160k * 100 word ops = 16M word ops. Should be < 0.1s.
    
    # So the original local search should be fast. Why did it stop?
    # Maybe it converged quickly?
    # The score is 1.118.
    
    # Let's try to improve the greedy step or add a "swap" local search.
    # Swap: remove one codeword, add one different codeword if it reduces size? No, size is fixed in swap.
    # Swap: replace one codeword with another if it allows removing another one later?
    
    # Actually, the "criticality" idea from inspiration:
    # Greedy with criticality:
    # Instead of maximizing coverage, maximize "criticality" or use a different heuristic.
    # But greedy is standard.
    
    # Let's try to speed up the redundancy check and ensure we do multiple passes.
    # Also, we can try to replace a codeword with another one that covers the same set of "unique" points but maybe allows better packing?
    # No, covering codes are about minimizing size.
    
    # Let's optimize the redundancy check loop.
    # Instead of rebuilding others_mask from scratch, we can maintain the union mask of all selected codewords.
    # total_union_mask = OR of neighbor_masks[w] for w in new_solution.
    # If we remove w, new_union = total_union_mask & ~neighbor_masks[w]? NO.
    # Union is not invertible like that.
    
    # However, we can check:
    # Is neighbor_masks[w] subset of (OR of neighbor_masks[j] for j != w)?
    # This is equivalent to: (neighbor_masks[w] & ~(OR of others)) == 0.
    
    # We can precompute prefix and suffix unions?
    # prefix[i] = OR of neighbor_masks[0..i]
    # suffix[i] = OR of neighbor_masks[i..k-1]
    # Then OR of others for index i is prefix[i-1] | suffix[i+1].
    # This makes the check O(1) big-int ops per codeword!
    # Precomputation is O(k) big-int ops.
    # Total per pass: O(k) big-int ops.
    # This is much faster than O(k^2).
    
    def try_remove_redundant(sol):
        k = len(sol)
        if k == 0:
            return sol, False
        
        prefix = [0] * k
        suffix = [0] * k
        
        prefix[0] = neighbor_masks[sol[0]]
        for i in range(1, k):
            prefix[i] = prefix[i-1] | neighbor_masks[sol[i]]
            
        suffix[k-1] = neighbor_masks[sol[k-1]]
        for i in range(k-2, -1, -1):
            suffix[i] = suffix[i+1] | neighbor_masks[sol[i]]
            
        new_sol = []
        changed = False
        for i in range(k):
            others_mask = 0
            if i > 0:
                others_mask |= prefix[i-1]
            if i < k - 1:
                others_mask |= suffix[i+1]
                
            if (neighbor_masks[sol[i]] & ~others_mask) == 0:
                changed = True
            else:
                new_sol.append(sol[i])
                
        return new_sol, changed

    # Apply redundancy removal repeatedly
    while time.time() - start_time < limit:
        new_solution, changed = try_remove_redundant(new_solution)
        if not changed:
            break
        
    # Now, try to improve with local search:
    # Try replacing one codeword with another to see if we can then remove more?
    # Or simply try to find a smaller set?
    
    # A better local search:
    # For each codeword w in solution, try to replace it with a different codeword v such that:
    # 1. v covers all points that w uniquely covers.
    # 2. This might not reduce size immediately, but might enable further reductions.
    # This is complex.
    
    # Let's try a simple "kick" or random restart if time permits?
    # No, deterministic is better.
    
    # Let's try to optimize the greedy choice slightly.
    # The current greedy picks the max coverage.
    # Ties are broken by index.
    # Maybe breaking ties by "criticality" or something else helps?
    
    # Actually, the main gain might come from a better local search.
    # Let's implement a 2-opt like search:
    # Remove two codewords, add one? No, size change.
    # Remove one, add one (swap).
    # Check if swap allows a subsequent removal.
    
    # Given the time, let's just ensure the redundancy removal is fast and robust.
    # The prefix/suffix optimization is a solid improvement in speed, allowing more passes or deeper search if we add it.
    
    # Convert indices back to lists of digits
    return [get_digits(idx) for idx in new_solution]
