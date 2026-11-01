import time
import numpy as np
from itertools import product
import random

def construct(q, n, radius):
    """
    Construct a covering code using Multi-Start Randomized Greedy + Local Search.
    
    Strategy:
    1. If N is small, use a bitset-based local search.
    2. Initialize with a good seed (e.g., Tetracode product for q=3, n=8).
    3. Run multiple restarts of Randomized Greedy.
    4. Apply Local Search (swap/add/remove) to minimize size.
    5. Return the best solution found within the time limit.
    """
    start_time = time.time()
    deadline = start_time + 115  # 115s to leave margin

    N = q ** n
    
    # For very small problems, we might be able to do better with exact methods, 
    # but for K_3(8,1), N=6561 is the target.
    
    # Generate all words
    all_words = np.array(list(product(range(q), repeat=n)), dtype=np.int32)
    num_words = N

    # Precompute balls as Python integers (bitsets) for fast coverage checks
    word_to_idx = {tuple(w): idx for idx, w in enumerate(all_words)}
    
    def get_idx(word_tuple):
        return word_to_idx.get(word_tuple, -1)

    balls = [0] * num_words
    for i in range(num_words):
        w = all_words[i]
        bits = 0
        bits |= (1 << i)
        if radius >= 1:
            for p in range(n):
                old_val = w[p]
                for new_val in range(q):
                    if new_val == old_val:
                        continue
                    new_w_list = list(w)
                    new_w_list[p] = new_val
                    j = get_idx(tuple(new_w_list))
                    if j != -1:
                        bits |= (1 << j)
        balls[i] = bits

    all_mask = (1 << num_words) - 1

    # --- Helper Functions ---

    def greedy_cover_randomized(seed=None):
        """
        Randomized Greedy Set Cover.
        At each step, pick the codeword that covers the most uncovered vertices.
        To introduce randomness, we can pick from the top-k best candidates.
        """
        if seed is not None:
            random.seed(seed)
        
        selected = []
        covered = 0
        remaining_indices = list(range(num_words))
        
        while covered != all_mask and remaining_indices:
            # Calculate gain for all remaining
            # To speed up, we can just iterate. 
            # Optimization: We can maintain a list of gains and update it? 
            # For N=6561, iterating 6561 times per step is ~500*6561 = 3.2M ops. 
            # Python bit_count is fast.
            best_gain = -1
            best_indices = []
            
            for i in remaining_indices:
                gain = (balls[i] & ~covered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_indices = [i]
                elif gain == best_gain and gain > 0:
                    best_indices.append(i)
            
            if not best_indices or best_gain == 0:
                break
            
            # Randomly pick from best candidates (or top 5 to add noise)
            # For now, just pick one randomly from the best
            choice = random.choice(best_indices)
            selected.append(choice)
            covered |= balls[choice]
            remaining_indices.remove(choice)
            
        return selected, covered

    def local_search(initial_selected, deadline):
        """
        Local Search: Try to swap selected words with unselected ones to reduce size.
        Or remove redundant words.
        """
        selected_set = set(initial_selected)
        selected_list = list(initial_selected)
        
        # 1. Redundancy Removal
        # Check if removing any word still covers everything.
        changed = True
        while changed:
            changed = False
            # Shuffle to avoid bias?
            # For each word in selected, check if it's critical.
            # A word is critical if there is some vertex covered ONLY by it.
            # i.e., intersection of balls of other selected words does not cover N[v] for some v in N[selected_word].
            
            # Efficient check:
            # For each v in selected:
            #   temp_covered = all_mask
            #   for u in selected, u != v: temp_covered &= ~balls[u]
            #   if (temp_covered & balls[v]) != 0: v is critical (cannot remove)
            #   else: v is redundant, remove it.
            
            # This is O(K^2 * bitops). K~500, K^2=250k. Bitops on 6561 bits are fast.
            # But doing this K times is K^3. Too slow.
            
            # Better: 
            # Compute union of balls of all selected except v.
            # This is still expensive.
            
            # Alternative: 
            # Just try to remove each one and check coverage?
            # For v in selected:
            #   new_selected = selected - {v}
            #   new_covered = 0
            #   for u in new_selected: new_covered |= balls[u]
            #   if new_covered == all_mask: remove v.
            
            # This is O(K^2 * bitops) per pass. 
            # With K=500, 250k * (6561/64) ~ 25M ops. Might be okay for a few passes.
            
            for v in list(selected_list):
                if time.time() > deadline:
                    return selected_list
                
                # Check if v is redundant
                # Compute coverage without v
                covered_without_v = 0
                for u in selected_list:
                    if u != v:
                        covered_without_v |= balls[u]
                
                if covered_without_v == all_mask:
                    selected_list.remove(v)
                    selected_set.discard(v)
                    changed = True
                    break # Restart loop since list changed
        
        # 2. Swap Search
        # Try to replace one selected word with one unselected word to allow further removal?
        # Or just try to find a smaller set?
        # This is hard. 
        # Instead, we can try "2-opt": remove 2, add 1? No, size must decrease.
        # So we only look for removals (handled above) or swaps that enable removals.
        
        # A common heuristic: 
        # While we can improve:
        #   Try to add a word w not in S.
        #   Then try to remove words from S U {w} that are now redundant.
        #   If size decreases, keep.
        
        # Given time constraints, the redundancy removal is the most impactful.
        # Let's add a simple swap: 
        # Pick a random unselected word w.
        # Add w.
        # Run redundancy removal on S U {w}.
        # If size < original size, accept.
        
        best_size = len(selected_list)
        best_list = selected_list[:]
        
        # Limit swaps based on time
        swaps_tried = 0
        while time.time() < deadline - 5: # Leave 5s for final check
            if not selected_list:
                break
            
            # Pick a random candidate to add
            # We want candidates that cover many uncovered? No, all are covered.
            # We want candidates that might allow removing multiple existing ones.
            # Heuristic: Pick a word that covers vertices that are "weakly covered" (low multiplicity)?
            # For now, random.
            
            # To make it faster, we can sample from a subset.
            # Let's just pick a random index.
            w = random.randint(0, num_words - 1)
            if w in selected_set:
                continue
            
            new_list = selected_list + [w]
            new_set = selected_set | {w}
            
            # Run redundancy removal on new_list
            # We can reuse the logic.
            # To save time, we only do one pass of removal?
            
            # Let's implement a fast removal check for the new set.
            # Since we added 1, we just need to check if any existing word becomes redundant.
            # Or if w is redundant (it isn't, we just added it and it covers something).
            
            # Actually, adding w might make some u in S redundant if N[u] subset N[S\{u} U {w}].
            
            # Simple check:
            # For each u in S:
            #   Check if u is redundant in S U {w}.
            
            # This is O(K * bitops).
            
            improved = False
            new_selected_list = new_list[:]
            new_selected_set = set(new_selected_list)
            
            # Try to remove from new_selected_list
            # We'll do a single pass to keep it fast
            for u in list(new_selected_list):
                if time.time() > deadline:
                    return best_list
                
                # Check if u is redundant
                covered_without_u = 0
                for v in new_selected_list:
                    if v != u:
                        covered_without_u |= balls[v]
                
                if covered_without_u == all_mask:
                    new_selected_list.remove(u)
                    new_selected_set.discard(u)
                    improved = True
            
            if len(new_selected_list) < len(selected_list):
                selected_list = new_selected_list
                selected_set = new_selected_set
                best_list = selected_list[:]
                best_size = len(selected_list)
            
            swaps_tried += 1

        return best_list

    # --- Main Loop ---

    best_solution = None
    best_size = num_words + 1

    # 1. Try Algebraic Seed if applicable
    if q == 3 and n == 8:
        # Tetracode product
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        seed_words = [h + list(tail) for h in tetracode for tail in product(range(3), repeat=n - 4)]
        seed_indices = [word_to_idx[tuple(w)] for w in seed_words]
        
        # Verify and optimize seed
        optimized_seed = local_search(seed_indices, deadline)
        if len(optimized_seed) < best_size:
            best_size = len(optimized_seed)
            best_solution = optimized_seed

    # 2. Multi-Start Randomized Greedy
    # Run as many restarts as possible
    # Each restart: Greedy + Local Search
    
    restart_count = 0
    while time.time() < deadline - 10:
        restart_count += 1
        seed = restart_count
        selected, covered = greedy_cover_randomized(seed)
        
        if covered != all_mask:
            # Should not happen if greedy works, but just in case
            # Fill gaps
            while covered != all_mask:
                # Find uncovered
                unc_mask = all_mask & ~covered
                if unc_mask == 0: break
                # Pick first uncovered
                lsb = unc_mask & -unc_mask
                v = lsb.bit_length() - 1
                # Find a neighbor
                cand_mask = balls[v]
                if cand_mask == 0: break
                lsb2 = cand_mask & -cand_mask
                u = lsb2.bit_length() - 1
                selected.append(u)
                covered |= balls[u]
        
        # Local Search
        optimized = local_search(selected, deadline)
        
        if len(optimized) < best_size:
            best_size = len(optimized)
            best_solution = optimized

    # Final Result
    if best_solution is None:
        # Fallback: Greedy
        selected, _ = greedy_cover_randomized(42)
        best_solution = selected
        best_size = len(selected)

    result = [all_words[i].tolist() for i in best_solution]
    return result


def _constructive_cover(q, n, radius):
    """Fallback constructive approach."""
    if q == 3 and n == 8:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in product(range(3), repeat=n - 4)]
    
    all_words = np.array(list(product(range(q), repeat=n)), dtype=np.int32)
    num_words = len(all_words)
    
    balls = np.zeros((num_words, num_words), dtype=bool)
    np.fill_diagonal(balls, True)
    for i in range(num_words):
        for j in range(num_words):
            if np.sum(all_words[i] != all_words[j]) <= radius:
                balls[i, j] = True
    
    selected = []
    covered = np.zeros(num_words, dtype=bool)
    while not covered.all():
        uncovered_count = balls[:, ~covered].sum(axis=1)
        best_idx = np.argmax(uncovered_count)
        if uncovered_count[best_idx] == 0:
            break
        selected.append(best_idx)
        covered |= balls[best_idx, :]
        
    return [all_words[i].tolist() for i in selected]
