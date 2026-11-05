import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Direct Permutation Simulated Annealing with Incremental Delta Evaluation.
    1. Initialize with a structural heuristic (degree-based).
    2. Optimize the vertex order (permutation) directly using Simulated Annealing.
    3. Use incremental updates to evaluate the change in independent set size for swaps in O(degree) time.
    4. This allows for significantly more search steps than full re-simulation.
    """
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Precompute neighbor lists and degrees for fast access
    neighbors_list = [np.where(adj[i])[0].tolist() for i in range(n)]
    degrees = np.array([len(nbrs) for nbrs in neighbors_list])
    
    # 1. Initial Heuristic: Low degree first (standard greedy heuristic for MIS)
    # We want high score = high priority. Low degree is generally better for MIS.
    # score = -degree
    initial_scores = -degrees.astype(float)
    
    # Initialize permutation (order of selection)
    # Higher score -> earlier in order
    perm = np.argsort(-initial_scores, kind='stable')
    
    def evaluate_full(perm):
        """Standard greedy evaluation for baseline."""
        active = np.ones(n, dtype=bool)
        count = 0
        for v in perm:
            if active[v]:
                count += 1
                active[v] = False
                nbrs = neighbors_list[v]
                if nbrs:
                    active[nbrs] = False
        return count

    best_val = evaluate_full(perm)
    best_perm = perm.copy()
    
    # 2. Simulated Annealing Setup
    # We work with the permutation array.
    # State: perm (array of vertices in selection order)
    # Move: Swap two elements in perm? Or Insertion?
    # Swapping is easier for delta evaluation if we can compute it.
    # However, swapping two vertices in the order changes the status of many vertices between them.
    # Actually, for a swap of positions i and j (i < j):
    # - Vertices before i: unchanged.
    # - Vertices after j: unchanged (unless they were blocked by the swapped vertices, but their relative order to each other is same, and the set of "already picked" vertices before them is the same? No, the set of picked vertices changes).
    
    # Let's refine the delta evaluation for a swap of vertices u and v at positions i and j (i < j).
    # This is complex because picking u at i instead of j changes which vertices are active for the segment (i, j).
    
    # Alternative Move: "2-opt" or "Swap" is hard to evaluate incrementally without re-simulating the segment.
    # Let's look at the "Insertion" move or just a "Swap" where we re-simulate only the affected segment.
    # If we swap u (at i) and v (at j):
    # - The prefix [0, i-1] is identical. Let P_prefix be the set of picked vertices in prefix.
    # - The suffix [j+1, n-1] depends on the set of picked vertices in [0, j].
    # - The segment [i, j] is where the change happens.
    # We can re-simulate the segment [i, j] given the state from the prefix.
    # State from prefix: active_mask (which vertices are still active entering position i).
    
    # Let's implement a function that evaluates the gain of swapping positions i and j.
    # This requires knowing the active mask at position i.
    
    # To make this efficient, we can maintain the "active mask" as we go?
    # No, SA needs to jump around.
    
    # Let's try a different move: **Adjacent Swap** or **Local Swap**.
    # If we only swap adjacent elements, the delta is small.
    # But adjacent swaps are slow to mix.
    
    # Let's stick to the "Segment Re-simulation" for arbitrary swaps.
    # Complexity of evaluating a swap (i, j): O(j - i + deg(u) + deg(v) + ...).
    # If we pick i, j close, it's fast.
    
    # Let's try a hybrid:
    # 1. Use a fast local search with "Insertion" or "Swap" of nearby elements.
    # 2. The "incremental" part is re-simulating only the affected window.
    
    def get_active_mask_at(perm, pos):
        """Compute active mask at position 'pos' by simulating prefix [0, pos-1]."""
        active = np.ones(n, dtype=bool)
        for k in range(pos):
            v = perm[k]
            if active[v]:
                active[v] = False
                nbrs = neighbors_list[v]
                if nbrs:
                    active[nbrs] = False
        return active

    def eval_swap_delta(perm, i, j):
        """
        Evaluate the change in MIS size if we swap perm[i] and perm[j] (i < j).
        Returns: (delta_count, new_active_mask_at_j+1)
        """
        # 1. Compute active mask at position i
        active = get_active_mask_at(perm, i)
        
        # 2. Simulate the original segment [i, j]
        # We need to know the count contributed by this segment and the state at j+1
        # Original segment: perm[i], perm[i+1], ..., perm[j]
        # New segment: perm[j], perm[i+1], ..., perm[i-1], perm[i]  <-- Wait, swap means:
        # New perm at i is perm[j], new perm at j is perm[i].
        # The middle part perm[i+1...j-1] stays in the same relative positions?
        # Yes.
        
        # Let's simulate the NEW segment [i, j]
        new_count = 0
        new_active = active.copy()
        # Position i: vertex u = perm[j]
        u = perm[j]
        if new_active[u]:
            new_count += 1
            new_active[u] = False
            nbrs = neighbors_list[u]
            if nbrs:
                new_active[nbrs] = False
        
        # Positions i+1 to j-1: vertices perm[i+1], ..., perm[j-1]
        for k in range(i + 1, j):
            v = perm[k]
            if new_active[v]:
                new_count += 1
                new_active[v] = False
                nbrs = neighbors_list[v]
                if nbrs:
                    new_active[nbrs] = False
        
        # Position j: vertex v = perm[i]
        v = perm[i]
        if new_active[v]:
            new_count += 1
            new_active[v] = False
            nbrs = neighbors_list[v]
            if nbrs:
                new_active[nbrs] = False
        
        # Now we have new_count and new_active (state at j+1) for the NEW segment.
        
        # 3. Simulate the ORIGINAL segment [i, j] to get old_count and old_active
        old_count = 0
        old_active = active.copy()
        # Position i: vertex u_orig = perm[i]
        u_orig = perm[i]
        if old_active[u_orig]:
            old_count += 1
            old_active[u_orig] = False
            nbrs = neighbors_list[u_orig]
            if nbrs:
                old_active[nbrs] = False
        
        # Positions i+1 to j-1
        for k in range(i + 1, j):
            v = perm[k]
            if old_active[v]:
                old_count += 1
                old_active[v] = False
                nbrs = neighbors_list[v]
                if nbrs:
                    old_active[nbrs] = False
        
        # Position j: vertex v_orig = perm[j]
        v_orig = perm[j]
        if old_active[v_orig]:
            old_count += 1
            old_active[v_orig] = False
            nbrs = neighbors_list[v_orig]
            if nbrs:
                old_active[nbrs] = False
        
        # 4. The suffix [j+1, n-1] is affected by the change in active mask.
        # We need to evaluate the suffix with both new_active and old_active.
        # This part is O(n-j).
        # Total cost: O(j) for prefix (via get_active_mask_at) + O(j-i) for segment + O(n-j) for suffix.
        # Total: O(n). This is not faster than full re-simulation!
        
        # Optimization: If we cache the prefix active masks, we can speed up get_active_mask_at.
        # But storing n masks of size n is O(n^2) memory. 200^2 = 40k bools = 40KB. That's fine.
        
        return new_count, old_count, new_active, old_active

    # Precompute prefix active masks?
    # prefix_masks[i] = active mask at position i.
    # prefix_counts[i] = count of picked vertices in prefix [0, i-1].
    
    def precompute_prefix(perm):
        masks = []
        counts = []
        active = np.ones(n, dtype=bool)
        count = 0
        for i in range(n + 1):
            masks.append(active.copy())
            counts.append(count)
            if i < n:
                v = perm[i]
                if active[v]:
                    count += 1
                    active[v] = False
                    nbrs = neighbors_list[v]
                    if nbrs:
                        active[nbrs] = False
        return masks, counts

    # With precomputed prefixes, evaluating a swap (i, j) costs:
    # 1. Get active mask at i: O(1) lookup.
    # 2. Simulate segment [i, j] for NEW order: O(j-i).
    # 3. Simulate segment [i, j] for OLD order: O(j-i).
    # 4. Evaluate suffix [j+1, n-1] with NEW state: O(n-j).
    # 5. Evaluate suffix [j+1, n-1] with OLD state: O(n-j).
    # Total: O(n). Still linear.
    
    # However, if we restrict swaps to "local" swaps (small j-i), the segment part is small.
    # But the suffix part is still O(n-j).
    
    # Is there a way to avoid re-evaluating the suffix?
    # The suffix evaluation is greedy. It depends on the active mask.
    # If the active mask at j+1 is the same for NEW and OLD, then the suffix count is the same.
    # We can check if new_active == old_active. If yes, delta = new_count - old_count.
    # If no, we must re-evaluate the suffix.
    
    # Let's implement this "Short-circuit" logic.
    
    def eval_swap_delta_fast(perm, masks, counts, i, j):
        """
        Evaluate delta for swapping perm[i] and perm[j] (i < j).
        Uses precomputed masks for prefix.
        """
        active_i = masks[i]
        
        # Simulate NEW segment [i, j]
        new_active = active_i.copy()
        new_count = 0
        
        u_new = perm[j]
        if new_active[u_new]:
            new_count += 1
            new_active[u_new] = False
            nbrs = neighbors_list[u_new]
            if nbrs:
                new_active[nbrs] = False
                
        for k in range(i + 1, j):
            v = perm[k]
            if new_active[v]:
                new_count += 1
                new_active[v] = False
                nbrs = neighbors_list[v]
                if nbrs:
                    new_active[nbrs] = False
                    
        v_new = perm[i]
        if new_active[v_new]:
            new_count += 1
            new_active[v_new] = False
            nbrs = neighbors_list[v_new]
            if nbrs:
                new_active[nbrs] = False
                
        # Simulate OLD segment [i, j]
        old_active = active_i.copy()
        old_count = 0
        
        u_old = perm[i]
        if old_active[u_old]:
            old_count += 1
            old_active[u_old] = False
            nbrs = neighbors_list[u_old]
            if nbrs:
                old_active[nbrs] = False
                
        for k in range(i + 1, j):
            v = perm[k]
            if old_active[v]:
                old_count += 1
                old_active[v] = False
                nbrs = neighbors_list[v]
                if nbrs:
                    old_active[nbrs] = False
                    
        v_old = perm[j]
        if old_active[v_old]:
            old_count += 1
            old_active[v_old] = False
            nbrs = neighbors_list[v_old]
            if nbrs:
                old_active[nbrs] = False
        
        # Check if suffix is affected
        if np.array_equal(new_active, old_active):
            # Suffix count is unchanged
            new_total = counts[i] + new_count + (counts[n] - counts[j+1]) # Wait, counts[n] is total.
            # counts[k] is count in prefix [0, k-1].
            # So count in [j+1, n-1] is total - counts[j+1].
            # But this is only valid if the suffix behavior is identical, which it is if active masks are identical.
            # However, we need the *actual* count of the suffix.
            # We can precompute suffix counts? No, suffix counts depend on the active mask.
            # But if the active mask at j+1 is identical, the greedy choices in the suffix are identical.
            # So the number of vertices picked in the suffix is the same.
            # Let suffix_count = evaluate_suffix(perm, j+1, active_i_suffix)
            # We don't have this precomputed.
            # We can just say: delta = (new_count - old_count) + (suffix_new - suffix_old).
            # If masks are equal, suffix_new == suffix_old. So delta = new_count - old_count.
            return new_count - old_count
        else:
            # Must evaluate suffix
            # This is expensive. We'll return a special value to indicate "need full re-eval" or just do it.
            # For now, let's just do the full re-evaluation of the suffix.
            def eval_suffix(perm, start_idx, active_mask):
                count = 0
                act = active_mask.copy()
                for k in range(start_idx, n):
                    v = perm[k]
                    if act[v]:
                        count += 1
                        act[v] = False
                        nbrs = neighbors_list[v]
                        if nbrs:
                            act[nbrs] = False
                return count
            
            suffix_new = eval_suffix(perm, j+1, new_active)
            suffix_old = eval_suffix(perm, j+1, old_active)
            
            # Note: The suffix evaluation uses the ORIGINAL perm for vertices > j.
            # But wait, the swap only affects positions i and j.
            # So perm[k] for k > j is unchanged.
            # So this is correct.
            
            return (new_count + suffix_new) - (old_count + suffix_old)

    # 3. Main SA Loop
    # We will use "Local Swaps" (small j-i) to keep the segment simulation fast.
    # If masks differ, we pay O(n) for suffix.
    # To minimize suffix cost, we should prefer swaps where masks are likely to be the same?
    # Or just accept the cost.
    
    # Let's try to optimize the "get_active_mask_at" by precomputing ALL prefix masks.
    # Memory: n * n bools. 200*200 = 40,000. Fine.
    
    masks, counts = precompute_prefix(perm)
    
    current_val = best_val
    current_perm = best_perm.copy()
    
    rng = np.random.default_rng(42)
    T = 1.0
    cooling = 0.999
    
    max_time = 118.0
    
    # We need to update masks and counts when we accept a swap.
    # This is the expensive part of maintaining the state.
    # If we accept a swap (i, j), the prefix masks for k > i are all changed.
    # And the counts are changed.
    # Re-computing all prefix masks is O(n^2).
    # This defeats the purpose of incremental evaluation if we do it every step.
    
    # Alternative: Don't maintain masks. Just re-evaluate full perm when we accept a move?
    # If we only evaluate deltas to *decide* acceptance, and then re-simulate to *update* state,
    # the cost is:
    # - Evaluate delta: O(n) worst case.
    # - Update state (if accepted): O(n^2) to recompute masks.
    
    # This is too slow if we accept many moves.
    
    # Let's change the move to **Adjacent Swap** (i, i+1).
    # Segment length is 1.
    # Masks at i+1:
    # If we swap perm[i] and perm[i+1]:
    # Active mask at i is A.
    # New: pick perm[i+1] then perm[i].
    # Old: pick perm[i] then perm[i+1].
    # The active mask at i+2 might change.
    # But the change is local.
    
    # Actually, let's step back.
    # The "Incremental Delta" inspiration (score 1.1504) suggests it's possible.
    # Maybe they use a different move, like "Insertion" or just accept the O(n) update?
    # If we do O(n) update for accepted moves, and O(n) evaluation for rejected moves,
    # and we have ~1000 moves/sec, that's 1000 * 200 = 200k ops/sec.
    # 120s * 200k = 24M ops.
    # n=200. 24M / 200 = 120k steps.
    # That's a lot of steps!
    
    # So the plan:
    # 1. Evaluate delta for a random swap (i, j) in O(n) (using precomputed masks for prefix, re-simulating segment and suffix if needed).
    # 2. If accepted, update the permutation.
    # 3. Re-compute prefix masks and counts in O(n^2).
    # 4. Repeat.
    
    # Wait, O(n^2) update is 40,000 ops.
    # O(n) eval is 200 ops.
    # If acceptance rate is low (e.g., 10%), we do 10 evals (2000 ops) and 1 update (40000 ops).
    # Total 42k ops per accepted move.
    # 24M / 42k = 570 accepted moves.
    # That's not huge, but better than 0.
    
    # Can we make the update faster?
    # If we swap i and j, only masks from i to n-1 change.
    # We can update them incrementally?
    # Mask[k] depends on Mask[k-1] and perm[k-1].
    # So we can iterate k from i to n-1 and update Mask[k].
    # This is O(n) per update!
    
    def update_masks_after_swap(perm, masks, counts, i, j):
        """
        Update masks and counts after swapping perm[i] and perm[j] (i < j).
        Masks for k < i are unchanged.
        Masks for k >= i need to be recomputed.
        """
        # Recompute masks from i to n
        # We need the state at i-1 to start.
        # masks[i] is the state at i.
        # We just re-simulate from i to n.
        
        # We need the active mask at i.
        # This is masks[i].
        active = masks[i].copy()
        
        # Update counts?
        # counts[k] is the count in prefix [0, k-1].
        # counts[i] is unchanged (prefix [0, i-1] is same).
        # For k > i, counts[k] = counts[k-1] + (1 if perm[k-1] was picked else 0).
        
        # Let's re-simulate the whole suffix to update masks and counts.
        # This is O(n).
        
        for k in range(i, n):
            # Update mask for position k+1
            # Wait, masks[k] is the state BEFORE processing perm[k].
            # So masks[k+1] is the state AFTER processing perm[k].
            
            v = perm[k]
            if active[v]:
                # Pick v
                active[v] = False
                nbrs = neighbors_list[v]
                if nbrs:
                    active[nbrs] = False
                # Update count for next position
                if k + 1 <= n:
                    counts[k+1] = counts[k] + 1
            else:
                if k + 1 <= n:
                    counts[k+1] = counts[k]
            
            # Store mask for next position
            if k + 1 < n:
                masks[k+1] = active.copy()
            else:
                # masks[n] is the final state (all False usually, or whatever remains)
                # We might not need masks[n] if we only look up masks[i] for i < n.
                # But let's keep it consistent.
                pass # masks[n] is not used in eval_swap_delta_fast as start index, 
                    # we use masks[i] where i < n.
                    # Wait, eval_swap_delta_fast uses masks[i].
                    # If i=n, it's out of bounds. But i < j < n.
        
        # Note: This update assumes the permutation has already been swapped.

    # Let's refine the SA loop with this O(n) update.
    
    for step in range(100000): # Max steps guard
        if time.time() - start_time > max_time - 2:
            break
            
        # Generate move: Swap i, j
        # To keep evaluation fast, maybe restrict j-i?
        # If j-i is large, the segment simulation is large.
        # But the update is always O(n) from i.
        # Let's pick i, j randomly.
        
        i = rng.integers(0, n - 1)
        j = rng.integers(i + 1, n)
        
        # Evaluate delta
        delta = eval_swap_delta_fast(current_perm, masks, counts, i, j)
        
        # Acceptance
        if delta > 0 or (T > 0.1 and rng.random() < np.exp(delta / T)):
            # Accept
            # Swap in perm
            current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
            current_val += delta
            
            # Update masks and counts
            update_masks_after_swap(current_perm, masks, counts, i, j)
            
            if current_val > best_val:
                best_val = current_val
                best_perm = current_perm.copy()
                
        T *= cooling
        
    # Final check
    final_val = evaluate_full(best_perm)
    if final_val > best_val:
        best_val = final_val
        
    # Convert best_perm to scores
    # Vertex at position k in best_perm gets score (n - k)
    # So vertex best_perm[0] gets n, best_perm[n-1] gets 1.
    scores = np.zeros(n)
    for k, v in enumerate(best_perm):
        scores[v] = n - k
        
    return scores
