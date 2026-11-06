import time
import random
from itertools import combinations

def construct():
    """
    Construct a triple system on 16 points maximizing the number of triples
    such that no three disjoint pairs have all eight transversal triples present
    (i.e., avoiding K(2,2,2) configurations).
    """
    start_time = time.time()
    # Evaluator kills at 150s. Leave 10s margin for safety.
    deadline = start_time + 140
    
    n = 16
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples) # 560
    
    # Map triple to index
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute all K(2,2,2) configurations (hyperedges)
    # A K(2,2,2) is a set of 8 triples formed by 3 disjoint pairs.
    # There are C(16,6) * 15 = 120120 such configurations.
    
    # For each triple, we store the indices of the K(2,2,2)s it belongs to.
    triple_to_k222s = [[] for _ in range(num_triples)]
    
    # We don't need to store the full list of K(2,2,2)s if we just want to count,
    # but for local search we might need to know if a specific K(2,2,2) is "full".
    # Actually, for the greedy step, we just need to know if adding a triple
    # completes any K(2,2,2).
    # For local search (swap), we need to check if a missing triple can be added
    # given the current set.
    
    # Let's precompute for each triple, the list of (k222_idx, other_triples_in_k222)
    # Or simpler: for each triple t, list of k222 indices.
    # And for each k222, the set of 8 triples.
    
    k222_list = [] # List of sets of triple indices
    k222_idx_counter = 0
    
    # Enumerate all 6-point subsets and their 15 pairings
    for points in combinations(range(n), 6):
        p0, p1, p2, p3, p4, p5 = points
        pairings = [
            ((p0,p1),(p2,p3),(p4,p5)),
            ((p0,p1),(p2,p4),(p3,p5)),
            ((p0,p1),(p2,p5),(p3,p4)),
            ((p0,p2),(p1,p3),(p4,p5)),
            ((p0,p2),(p1,p4),(p3,p5)),
            ((p0,p2),(p1,p5),(p3,p4)),
            ((p0,p3),(p1,p2),(p4,p5)),
            ((p0,p3),(p1,p4),(p2,p5)),
            ((p0,p3),(p1,p5),(p2,p4)),
            ((p0,p4),(p1,p2),(p3,p5)),
            ((p0,p4),(p1,p3),(p2,p5)),
            ((p0,p4),(p1,p5),(p2,p3)),
            ((p0,p5),(p1,p2),(p3,p4)),
            ((p0,p5),(p1,p3),(p2,p4)),
            ((p0,p5),(p1,p4),(p2,p3)),
        ]
        
        for pair1, pair2, pair3 in pairings:
            transversals = []
            for x in pair1:
                for y in pair2:
                    for z in pair3:
                        transversals.append((x, y, z))
            
            trans_sorted = [tuple(sorted(t)) for t in transversals]
            trans_indices = [triple_to_idx[t] for t in trans_sorted]
            
            k_idx = k222_idx_counter
            k222_idx_counter += 1
            k222_list.append(set(trans_indices))
            
            for idx in trans_indices:
                triple_to_k222s[idx].append(k_idx)
    
    num_k222s = len(k222_list)
    
    # Helper to check if adding triple_idx to current_present (set of indices)
    # would create a K(2,2,2).
    # We maintain a count of how many triples from each K(2,2,2) are in current_present.
    
    best_result = []
    best_count = 0
    
    # We will run a loop of restarts.
    # Each restart:
    # 1. Greedy construction with random order.
    # 2. Local search improvement.
    
    num_restarts = 0
    while time.time() < deadline - 5: # Leave 5s for final return
        num_restarts += 1
        
        # 1. Greedy Construction
        order = list(range(num_triples))
        random.shuffle(order)
        
        present = set()
        k222_counts = [0] * num_k222s
        
        for t_idx in order:
            if time.time() > deadline - 2:
                break
            
            # Check if adding t_idx completes any K(2,2,2)
            can_add = True
            for k_idx in triple_to_k222s[t_idx]:
                if k222_counts[k_idx] == 7:
                    can_add = False
                    break
            
            if can_add:
                present.add(t_idx)
                for k_idx in triple_to_k222s[t_idx]:
                    k222_counts[k_idx] += 1
        
        current_count = len(present)
        
        # 2. Local Search
        # Try to improve by 1-swaps.
        # Remove a random present triple, then try to add any missing triple.
        # If we can add >= 1, accept the swap (net gain >= 0, usually +1 if we add 1 and remove 1? 
        # Wait. If we remove 1 and add 1, count stays same. We want count to increase.
        # So we need to remove 1 and add > 1? Or remove 1 and add 1 that "unblocks" nothing?
        # Actually, removing a triple *relaxes* constraints. It might allow adding multiple triples.
        # So: Remove T_out. Check all missing T_in. Add all that are valid.
        # If added_count > 0, we have a new solution with count = current_count - 1 + added_count.
        # If added_count >= 1, we don't necessarily gain, but we might change the landscape.
        # However, usually we want to strictly increase count.
        # If added_count > 1, we gain.
        # If added_count == 1, we stay same, but maybe next step gains?
        # Let's try to find a swap that increases count.
        
        improved = True
        local_deadline = deadline - 2
        while improved and time.time() < local_deadline:
            improved = False
            
            # Try a few random swaps
            swaps_tried = 0
            max_swaps = 20
            present_list = list(present)
            if not present_list:
                break
            
            while swaps_tried < max_swaps and time.time() < local_deadline:
                swaps_tried += 1
                t_out = random.choice(present_list)
                
                # Remove t_out
                present.remove(t_out)
                for k_idx in triple_to_k222s[t_out]:
                    k222_counts[k_idx] -= 1
                
                # Try to add missing triples
                added_any = False
                added_indices = []
                
                # We can iterate over missing triples. To be faster, we can prioritize
                # triples that share K(2,2,2)s with t_out, as they are most likely to become valid.
                # But for simplicity and robustness, let's scan all missing.
                # 560 triples is small.
                
                for t_in in range(num_triples):
                    if t_in in present:
                        continue
                    
                    can_add = True
                    for k_idx in triple_to_k222s[t_in]:
                        if k222_counts[k_idx] == 7:
                            can_add = False
                            break
                    
                    if can_add:
                        present.add(t_in)
                        added_indices.append(t_in)
                        for k_idx in triple_to_k222s[t_in]:
                            k222_counts[k_idx] += 1
                        added_any = True
                
                if added_any:
                    # We added some triples.
                    # Net change: -1 (removed) + len(added_indices)
                    # If len(added_indices) >= 1, we might have same count or better.
                    # If len(added_indices) > 1, we definitely improved count.
                    # If len(added_indices) == 1, count is same.
                    # We accept the change if count increased OR if we want to explore.
                    # Let's accept if count increased.
                    if len(added_indices) > 1:
                        improved = True
                        break # Restart local search loop with new state
                    elif len(added_indices) == 1:
                        # Count is same. We might accept this to escape local optimum?
                        # Risky. Let's only accept strict improvements for now to ensure progress.
                        # Or accept with some probability?
                        # Let's stick to strict improvement for the "best" tracking,
                        # but we can accept equal improvements to move in the landscape.
                        # However, if we accept equal, we might loop.
                        # Let's only accept if len(added_indices) > 1.
                        # If len(added_indices) == 1, revert?
                        # Actually, if we revert, we wasted time.
                        # Let's keep the change if it allows further improvement?
                        # For now, let's only accept if count strictly increases.
                        pass
                else:
                    # Added nothing. Revert removal.
                    present.add(t_out)
                    for k_idx in triple_to_k222s[t_out]:
                        k222_counts[k_idx] += 1
                
                if not added_any:
                    continue
                
                # If we added exactly 1, we reverted logic above? No, we added it to present.
                # We need to decide whether to keep it.
                # If we keep it, count is same.
                # If we revert, we undo the addition and re-add t_out.
                
                # Let's simplify: Only accept swap if new_count > old_count.
                # old_count = current_count
                # new_count = current_count - 1 + len(added_indices)
                # So we need len(added_indices) > 1.
                
                if len(added_indices) <= 1:
                    # Revert the additions
                    for t_in in added_indices:
                        present.remove(t_in)
                        for k_idx in triple_to_k222s[t_in]:
                            k222_counts[k_idx] -= 1
                    # Re-add t_out
                    present.add(t_out)
                    for k_idx in triple_to_k222s[t_out]:
                        k222_counts[k_idx] += 1
        
        # Update best
        if len(present) > best_count:
            best_count = len(present)
            best_result = [list(all_triples[i]) for i in present]
            
    # Final check
    if not best_result:
        # Fallback: return empty or a trivial valid set
        # A single triple is always valid.
        best_result = [list(all_triples[0])]
        
    return best_result

if __name__ == "__main__":
    result = construct()
    print(f"Number of triples: {len(result)}")

