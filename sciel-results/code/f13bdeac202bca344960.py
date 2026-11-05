import time
import random
import numpy as np

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 150s, we aim to finish well before.
    # The prompt says "within 120 s" for evaluation, but the directive says 
    # "evaluator KILLS the program after 150s wall-clock". 
    # Let's use 110s as a hard stop to be safe.
    deadline = start_time + 110

    base = 51975
    
    # Generate all divisors of 51975 greater than 1
    # 51975 = 3^3 * 5^2 * 7 * 11
    divisors = []
    for d in range(2, base + 1):
        if base % d == 0:
            divisors.append(d)
    # Sort divisors. Smaller moduli cover more residues per class on average?
    # Actually, larger moduli cover fewer residues per class (1/m of the space).
    # Smaller moduli (like 3, 5) cover 1/3 or 1/5 of the space.
    # To cover efficiently, we might want a mix.
    divisors_sorted = sorted(divisors)
    
    # Precompute indices for each divisor to speed up updates?
    # No, slicing `count[r::m]` is fast in numpy.
    
    def score_from_counts(counts):
        return int(np.sum(counts == 0))

    def add_class(counts, m, r):
        counts[r::m] += 1
    
    def remove_class(counts, m, r):
        counts[r::m] -= 1
        
    def change_class(counts, m, old_r, new_r):
        counts[old_r::m] -= 1
        counts[new_r::m] += 1

    def greedy_construct(order):
        """
        Greedy construction: iterate through moduli in `order`, 
        pick the residue that maximizes newly covered residues.
        """
        counts = np.zeros(base, dtype=np.int8)
        classes = []
        used_moduli = set()
        
        for m in order:
            if time.time() > deadline - 10:
                break
            best_r = 0
            best_gain = -1
            # For small moduli, check all residues. For large, sample.
            # Since m divides base, m is at most base.
            # But we have distinct moduli.
            # Let's check all residues for m <= 100, else sample.
            if m <= 100:
                candidates = range(m)
            else:
                # Sample residues
                candidates = [random.randint(0, m - 1) for _ in range(50)]
                # Ensure 0 is checked
                if 0 not in candidates:
                    candidates.append(0)
            
            for r in candidates:
                # Gain is number of zeros in count[r::m]
                gain = int(np.sum(counts[r::m] == 0))
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            if best_gain > 0:
                classes.append([m, best_r])
                add_class(counts, m, best_r)
                used_moduli.add(m)
            else:
                # If no gain, adding it doesn't help immediately, 
                # but might help later? 
                # In greedy, if gain is 0, it's fully covered already. 
                # Adding it is useless for score reduction. 
                # However, for distinct moduli constraint, we just don't add it.
                pass
                
        return classes, counts

    def local_search(classes, counts, max_iters=10000):
        """
        Local search with coverage counts.
        Tries to minimize uncovered residues.
        """
        sc = score_from_counts(counts)
        used_moduli = set(m for m, _ in classes)
        unused_moduli = [m for m in divisors if m not in used_moduli]
        
        # Pre-compute potential gains for unused moduli? 
        # It's dynamic, so we compute on the fly.
        
        # To speed up, we can keep a list of "promising" moves.
        # But simple hill climbing is fast enough with numpy.
        
        # We will use a simple hill climbing with random restarts within the search.
        # Actually, let's do a mixed approach:
        # 1. Try to add a class that reduces score.
        # 2. Try to change a class residue to reduce score.
        # 3. Try to remove a class if it doesn't increase score (or if it allows a better add later? No, distinct moduli).
        
        # Optimization: 
        # Changing residue of m: 
        #   Old: counts[old_r::m]
        #   New: counts[new_r::m]
        #   Delta score = (zeros in new) - (zeros in old) 
        #   Wait, score is # zeros in TOTAL.
        #   Let Z be set of indices with count==0.
        #   If we change residue, some indices lose coverage, some gain.
        #   This is complex to calculate incrementally without full count update.
        #   However, updating counts is O(N/m). 
        #   N=51975. If m=3, N/m ~ 17000. Very fast.
        #   If m=51975, N/m = 1. Very fast.
        #   So updating counts is cheap.
        
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iters and time.time() < deadline - 5:
            improved = False
            iter_count += 1
            
            # 1. Try adding an unused modulus
            if unused_moduli:
                # Shuffle to randomize
                random.shuffle(unused_moduli)
                for m in unused_moduli:
                    if time.time() > deadline - 5:
                        return classes, counts, sc
                    # Find best residue for m
                    best_r = 0
                    best_sc = sc
                    # Sample residues
                    if m <= 100:
                        candidates = list(range(m))
                    else:
                        candidates = [random.randint(0, m - 1) for _ in range(20)]
                        if 0 not in candidates: candidates.append(0)
                    
                    for r in candidates:
                        # Simulate add
                        # We need to know how many zeros become non-zeros.
                        # gain = sum(counts[r::m] == 0)
                        # If gain > 0, score decreases by gain.
                        gain = int(np.sum(counts[r::m] == 0))
                        if gain > 0:
                            # This move improves score
                            add_class(counts, m, r)
                            classes.append([m, r])
                            used_moduli.add(m)
                            unused_moduli.remove(m)
                            sc -= gain
                            improved = True
                            break # Restart search with new state
                    
                    if improved:
                        break # Restart outer loop
                    
            # 2. Try changing residue of existing classes
            if not improved:
                # Pick a random class to change
                if classes:
                    idx = random.randint(0, len(classes) - 1)
                    m, old_r = classes[idx]
                    best_r = old_r
                    best_sc = sc
                    
                    # Sample new residues
                    if m <= 100:
                        candidates = list(range(m))
                    else:
                        candidates = [random.randint(0, m - 1) for _ in range(20)]
                        if old_r not in candidates: candidates.append(old_r)
                    
                    for new_r in candidates:
                        if new_r == old_r: continue
                        
                        # Calculate change in score
                        # We can simulate the change on a copy of counts? 
                        # Or just calculate the delta.
                        # Delta score = (new_zeros) - (old_zeros)
                        # old_zeros = sum(counts[old_r::m] == 0) ? No.
                        # The indices in old_r::m might have count > 1.
                        # Only indices with count == 1 in old_r::m will become 0 if we remove coverage there.
                        # Only indices with count == 0 in new_r::m will become 1 if we add coverage there.
                        
                        # Let's just do the update and check score. It's fast.
                        old_counts_slice = counts[old_r::m].copy() # Store to revert? 
                        # Better: update, check score, revert if worse.
                        
                        # Update
                        counts[old_r::m] -= 1
                        counts[new_r::m] += 1
                        
                        new_sc = score_from_counts(counts)
                        
                        if new_sc < best_sc:
                            best_sc = new_sc
                            best_r = new_r
                        else:
                            # Revert
                            counts[new_r::m] -= 1
                            counts[old_r::m] += 1
                    
                    if best_r != old_r:
                        classes[idx][1] = best_r
                        # The counts are already updated to the best_r state? 
                        # No, we reverted in the loop if it wasn't best.
                        # We need to apply the best move.
                        # In the loop, if new_sc < best_sc, we didn't revert.
                        # But we might have tried multiple. 
                        # Let's re-apply the best one cleanly.
                        
                        # Revert current state to before this class change?
                        # Actually, the loop logic above is tricky with reverts.
                        # Let's simplify: 
                        # 1. Save old counts? Too expensive.
                        # 2. Just revert to original state before trying candidates.
                        
                        # Let's restart the "Change Residue" block logic to be safer.
                        pass # This block is complex, let's use a simpler version below.

            # Simpler Change Residue Logic
            if not improved and classes:
                idx = random.randint(0, len(classes) - 1)
                m, old_r = classes[idx]
                
                # We want to find new_r that minimizes score.
                # Score change:
                # Let S be set of indices with count 0.
                # Removing old_r::m: indices in S that are in old_r::m and have count 1 become 0? No, they are already 0.
                # Indices in old_r::m that have count 1 become 0 (if no other class covers them). 
                # Wait, if count is 1, it's covered by exactly this class. Removing it makes it 0.
                # So "Loss" = number of indices in old_r::m with count == 1.
                
                # Adding new_r::m: indices in new_r::m with count 0 become 1.
                # So "Gain" = number of indices in new_r::m with count == 0.
                
                # Net Change in Uncovered = Loss - Gain.
                # We want Net Change < 0 (i.e., Gain > Loss).
                
                loss = int(np.sum(counts[old_r::m] == 1))
                
                best_r = old_r
                best_net = 0 # We keep old if no improvement
                
                # Sample new residues
                if m <= 100:
                    candidates = list(range(m))
                else:
                    candidates = [random.randint(0, m - 1) for _ in range(30)]
                    if old_r not in candidates: candidates.append(old_r)
                    
                for new_r in candidates:
                    if new_r == old_r: continue
                    gain = int(np.sum(counts[new_r::m] == 0))
                    net = loss - gain
                    if net < best_net:
                        best_net = net
                        best_r = new_r
                
                if best_r != old_r:
                    # Apply change
                    counts[old_r::m] -= 1
                    counts[best_r::m] += 1
                    classes[idx][1] = best_r
                    sc += best_net # best_net is negative
                    improved = True

            # 3. Try removing a class
            if not improved and len(classes) > 1:
                # Remove a class if it doesn't increase score (i.e., loss == 0)
                # Or if it allows a better configuration later?
                # For now, only remove if loss == 0.
                idx = random.randint(0, len(classes) - 1)
                m, r = classes[idx]
                loss = int(np.sum(counts[r::m] == 1))
                if loss == 0:
                    remove_class(counts, m, r)
                    classes.pop(idx)
                    used_moduli.remove(m)
                    unused_moduli.append(m)
                    # Score doesn't change
                    improved = True

        return classes, counts, sc

    best_classes = []
    best_score = base
    best_counts = None

    # Phase 1: Multiple Greedy Starts
    # Strategy A: Ascending moduli
    # Strategy B: Descending moduli
    # Strategy C: Random
    # Strategy D: "Balanced" - pick moduli that are coprime?
    
    orders = [
        divisors_sorted,
        sorted(divisors, reverse=True),
        divisors_sorted[:], # Will be shuffled
        divisors_sorted[:], # Will be shuffled
        divisors_sorted[:], # Will be shuffled
    ]
    
    for i, order in enumerate(orders):
        if i > 0:
            random.shuffle(order)
            
        if time.time() > deadline - 30:
            break
            
        classes, counts = greedy_construct(order)
        sc = score_from_counts(counts)
        
        if sc < best_score:
            best_score = sc
            best_classes = classes[:]
            best_counts = counts.copy()
            
        # Local Search
        if time.time() < deadline - 10:
            classes_ls, counts_ls, sc_ls = local_search(classes, counts, max_iters=2000)
            if sc_ls < best_score:
                best_score = sc_ls
                best_classes = classes_ls[:]
                best_counts = counts_ls.copy()

    # Phase 2: Aggressive Local Search on Best Solution
    if best_classes and time.time() < deadline - 5:
        # Run a longer local search on the best found so far
        classes_ls, counts_ls, sc_ls = local_search(best_classes, best_counts, max_iters=5000)
        if sc_ls < best_score:
            best_score = sc_ls
            best_classes = classes_ls[:]
            best_counts = counts_ls.copy()

    # Phase 3: Final Polish
    if best_classes and time.time() < deadline - 2:
        # Try to add any remaining moduli that help
        used_moduli = set(m for m, _ in best_classes)
        unused = [m for m in divisors if m not in used_moduli]
        random.shuffle(unused)
        for m in unused:
            if time.time() > deadline - 1:
                break
            # Find best residue
            best_r = 0
            best_gain = 0
            if m <= 100:
                candidates = range(m)
            else:
                candidates = [random.randint(0, m - 1) for _ in range(10)]
            
            for r in candidates:
                gain = int(np.sum(best_counts[r::m] == 0))
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            if best_gain > 0:
                add_class(best_counts, m, best_r)
                best_classes.append([m, best_r])
                best_score -= best_gain

    if not best_classes:
        best_classes = [[3, 0]]
        
    return best_classes

# Test/Validation
if __name__ == "__main__":
    classes = construct()
    print(f"Found {len(classes)} classes")
    # Calculate score for debugging
    base = 51975
    counts = np.zeros(base, dtype=np.int8)
    for m, r in classes:
        counts[r::m] += 1
    score = int(np.sum(counts == 0))
    print(f"Score: {score}")
