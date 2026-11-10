import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    points = list(range(16))
    all_triples = list(combinations(points, 3))
    n_triples = len(all_triples)
    
    # Precompute masks for each triple
    triple_masks = [0] * n_triples
    for i, t in enumerate(all_triples):
        mask = 0
        for v in t:
            mask |= (1 << v)
        triple_masks[i] = mask
        
    # Precompute pair sets for linearity check
    # A triple is a set of 3 pairs. Two triples conflict if they share any pair.
    triple_pairs = [None] * n_triples
    for i, t in enumerate(all_triples):
        triple_pairs[i] = {
            (t[0], t[1]), (t[1], t[0]),
            (t[0], t[2]), (t[2], t[0]),
            (t[1], t[2]), (t[2], t[1])
        }

    def calc_alpha(triple_indices):
        """
        Calculate independence number using Bitmask Branch and Bound.
        Returns the size of the largest subset of vertices containing no triple.
        """
        if not triple_indices:
            return 16
            
        n = 16
        # active_masks contains the bitmask of each triple in the system
        active_masks = [triple_masks[i] for i in triple_indices]
        
        # Sort vertices by degree (number of triples containing them)
        # Higher degree vertices are more "constrained", good for pruning
        deg = [0] * n
        for mask in active_masks:
            for v in range(n):
                if mask & (1 << v):
                    deg[v] += 1
                    
        order = sorted(range(n), key=lambda v: -deg[v])
        
        best_alpha = [0]
        
        def search(idx, forbidden_mask, current_size):
            if time.time() > deadline - 1:
                return
                
            # Pruning: if max possible size (current + remaining vertices) <= best, return
            # Note: 'forbidden_mask' contains vertices that cannot be added.
            # Available vertices are those not in forbidden_mask.
            # However, the order is fixed. We just iterate through order.
            # A tighter bound: current_size + popcount(~forbidden_mask & ((1<<16)-1))
            # But since we iterate in 'order', we can just use n - idx as a loose bound
            # or precompute suffix counts of non-forbidden? 
            # Actually, the standard bound is: if current_size + (n - idx) <= best_alpha, return.
            # This is valid because even if we picked all remaining vertices, we couldn't beat best.
            if current_size + (n - idx) <= best_alpha[0]:
                return
            
            if current_size > best_alpha[0]:
                best_alpha[0] = current_size
                
            if idx == n:
                return
                
            v = order[idx]
            
            # Option 1: Exclude v
            search(idx + 1, forbidden_mask, current_size)
            
            # Option 2: Include v
            # Check if v is already forbidden
            if forbidden_mask & (1 << v):
                return
                
            # If we include v, any triple containing v becomes "activated" and forbids its other vertices.
            new_forbidden = forbidden_mask | (1 << v)
            
            # Check if including v creates a full triple
            # A triple is contained in the set if all its vertices are in the set.
            # We don't track the "set" directly, but we track forbidden.
            # Wait, the standard independent set logic:
            # If we pick v, we must ensure no triple is fully contained in {v} U (previous picks).
            # The 'forbidden_mask' approach usually tracks: "vertices that cannot be picked because picking them would complete a triple with already picked vertices".
            
            # Let's re-verify the logic.
            # State: (idx, forbidden_mask, current_size)
            # forbidden_mask: vertices that are in the "closed neighborhood" of the current independent set.
            # If v is in forbidden_mask, we cannot pick v.
            # If we pick v:
            # 1. v must not be in forbidden_mask.
            # 2. We add v to the set.
            # 3. Any vertex u such that there exists a triple {v, u, w} where w is already in the set... 
            #    Actually, it's simpler: 
            #    If we pick v, we must forbid all vertices that form a triple with v and ANY previously picked vertex?
            #    No. The condition is: No triple is fully contained in the final set.
            #    If we pick v, we cannot pick any pair {u, w} such that {v, u, w} is a triple.
            #    This means u and w cannot both be in the set.
            #    This is not a simple "forbid u" logic.
            
            # Alternative B&B for Independent Set in Hypergraph:
            # We want max set S such that no T in triples is subset of S.
            # This is equivalent to: For every T, S does not contain T.
            # Or: For every T, S intersects T^c.
            
            # Let's use a simpler, robust B&B for small N=16.
            # We can iterate through vertices in 'order'.
            # State: (idx, current_mask, current_size)
            # current_mask: bitmask of vertices currently in S.
            # Check: Does current_mask contain any triple?
            # If yes, prune.
            
            # Let's rewrite search with this logic. It's O(2^16 * |Triples|) worst case, but pruning is strong.
            
        # Let's use the "current_mask" approach for correctness and simplicity, optimized by early exit.
        
        def search_mask(idx, current_mask, current_size):
            if time.time() > deadline - 1:
                return
                
            if current_size + (n - idx) <= best_alpha[0]:
                return
            
            if current_size > best_alpha[0]:
                best_alpha[0] = current_size
                
            if idx == n:
                return
                
            v = order[idx]
            
            # Option 1: Exclude v
            search_mask(idx + 1, current_mask, current_size)
            
            # Option 2: Include v
            new_mask = current_mask | (1 << v)
            
            # Check if new_mask contains any triple
            # We can check if any active triple is a subset of new_mask
            # triple_mask & new_mask == triple_mask
            
            valid = True
            for am in active_masks:
                if am & new_mask == am:
                    valid = False
                    break
                    
            if valid:
                search_mask(idx + 1, new_mask, current_size + 1)
                
        search_mask(0, 0, 0)
        return best_alpha[0]

    def is_valid_addition(triple_idx, used_pairs):
        t = all_triples[triple_idx]
        pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
        for p in pairs:
            if p in used_pairs:
                return False
        return True

    def add_triple(triple_idx, current_triples, current_triple_set, used_pairs):
        t = all_triples[triple_idx]
        pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
        for p in pairs:
            used_pairs.add(p)
        current_triples.append(triple_idx)
        current_triple_set.add(triple_idx)

    def remove_triple(triple_idx, current_triples, current_triple_set, used_pairs):
        t = all_triples[triple_idx]
        pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
        for p in pairs:
            used_pairs.discard(p)
        current_triple_set.discard(triple_idx)
        # Remove from list
        for i, idx in enumerate(current_triples):
            if idx == triple_idx:
                current_triples.pop(i)
                break

    best_alpha = 17
    best_solution = None
    
    num_restarts = 100
    
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break
            
        random.seed(1234 + restart)
        
        used_pairs = set()
        current_triples = []
        current_triple_set = set()
        
        # Greedy Initialization
        indices = list(range(n_triples))
        random.shuffle(indices)
        
        for t_idx in indices:
            if is_valid_addition(t_idx, used_pairs):
                add_triple(t_idx, current_triples, current_triple_set, used_pairs)
                
        alpha = calc_alpha(current_triples)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_solution = list(current_triples)
            if best_alpha <= 4:
                break
            
        # Local Search
        # We will perform swaps that reduce alpha.
        # Since alpha is expensive to compute, we will only compute it for promising moves.
        # Strategy: 
        # 1. Find current witness (max independent set) - we need a function for this too, or just rely on alpha value?
        #    To guide the search, knowing the witness is helpful.
        #    Let's modify calc_alpha to also return the witness.
        
        # Re-defining calc_alpha to return witness is better for "witness breaking".
        # But for now, let's stick to alpha value for acceptance criterion, 
        # and use "random valid swap" for exploration.
        
        # To improve, we will bias the swap selection:
        # Remove a random triple.
        # Try to add a triple that shares vertices with the "likely" independent set.
        # Without witness, we just try random valid adds.
        
        steps = 0
        max_steps = 300
        while steps < max_steps:
            if time.time() > deadline - 5:
                break
            steps += 1
            
            if not current_triples:
                # If empty, add random
                add_idx = random.randint(0, n_triples - 1)
                if is_valid_addition(add_idx, used_pairs):
                    add_triple(add_idx, current_triples, current_triple_set, used_pairs)
                continue

            # Pick a triple to remove
            rem_idx = random.choice(current_triples)
            
            # Temporarily remove
            remove_triple(rem_idx, current_triples, current_triple_set, used_pairs)
            
            # Try to add a better triple
            # Candidates: all triples not in current set, valid
            # To speed up, we don't check all 5120. We check a random subset.
            # Or, we check triples that are "compatible" with the removed one's neighbors?
            
            # Let's try a few random candidates
            found_improvement = False
            old_alpha = alpha
            
            # Try 10 random candidates
            candidates = [random.randint(0, n_triples - 1) for _ in range(10)]
            candidates = list(set(candidates)) # unique
            
            best_cand = None
            best_cand_alpha = old_alpha
            
            for cand in candidates:
                if cand in current_triple_set:
                    continue
                if cand == rem_idx:
                    continue
                if is_valid_addition(cand, used_pairs):
                    # Tentatively add
                    add_triple(cand, current_triples, current_triple_set, used_pairs)
                    cand_alpha = calc_alpha(current_triples)
                    remove_triple(cand, current_triples, current_triple_set, used_pairs)
                    
                    if cand_alpha < best_cand_alpha:
                        best_cand_alpha = cand_alpha
                        best_cand = cand
                        if best_cand_alpha <= 4:
                            break
                
            if best_cand is not None:
                # Perform the swap
                add_triple(best_cand, current_triples, current_triple_set, used_pairs)
                alpha = best_cand_alpha
                found_improvement = True
                if alpha < best_alpha:
                    best_alpha = alpha
                    best_solution = list(current_triples)
                    if best_alpha <= 4:
                        break
            else:
                # Restore
                add_triple(rem_idx, current_triples, current_triple_set, used_pairs)
                # If no improvement, we might just stay put or do a random move later
                # For now, just restore.
                
            # Occasionally, try a pure addition if possible (to increase density)
            # But usually, we are maxed out on linearity.
            
            if found_improvement:
                steps = 0 # Reset step counter on improvement? Or keep it to force restart?
                # Keeping it is fine.
                
    if best_solution is None:
        return [[0, 1, 2]]
        
    return [list(all_triples[i]) for i in best_solution]
