import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    N = 1000
    MAX_SUM = 2 * N  # 2000
    
    def make_counts():
        return [0] * (MAX_SUM + 1)
    
    def greedy_forward(counts, chosen):
        """Continue greedy from current state. Returns (chosen, counts)."""
        start = chosen[-1] + 1 if chosen else 1
        for x in range(start, N + 1):
            # Check if adding x keeps all sums <= 2
            ok = True
            for y in chosen:
                s = x + y
                if counts[s] >= 2:
                    ok = False
                    break
            if ok:
                s = 2 * x
                if counts[s] >= 2:
                    ok = False
            if ok:
                for y in chosen:
                    s = x + y
                    counts[s] += 1
                s = 2 * x
                counts[s] += 1
                chosen.append(x)
        return chosen, counts
    
    def greedy_with_skip(counts, chosen, skip_prob=0.1):
        """Greedy with occasional skips to explore different paths."""
        start = chosen[-1] + 1 if chosen else 1
        for x in range(start, N + 1):
            ok = True
            for y in chosen:
                s = x + y
                if counts[s] >= 2:
                    ok = False
                    break
            if ok:
                s = 2 * x
                if counts[s] >= 2:
                    ok = False
            if ok and random.random() > skip_prob:
                for y in chosen:
                    s = x + y
                    counts[s] += 1
                s = 2 * x
                counts[s] += 1
                chosen.append(x)
        return chosen, counts
    
    def rebuild_counts(chosen):
        """Rebuild sum counts array from chosen set."""
        counts = [0] * (MAX_SUM + 1)
        for i in range(len(chosen)):
            for j in range(i, len(chosen)):
                s = chosen[i] + chosen[j]
                counts[s] += 1
        return counts
    
    def try_add(chosen, counts, x):
        """Check if x can be added. Returns (ok, conflicts_list)."""
        conflicts = []
        for y in chosen:
            s = x + y
            if counts[s] >= 2:
                conflicts.append(y)
        s = 2 * x
        if counts[s] >= 2:
            return False, conflicts
        return len(conflicts) == 0, conflicts
    
    def local_search(chosen, counts, max_steps=200):
        """Try to improve by swapping elements. Uses array-based counts."""
        chosen_set = set(chosen)
        best_size = len(chosen)
        
        for _ in range(max_steps):
            if time.time() > deadline:
                break
            improved = False
            
            # Try to add a random element not in chosen
            candidates = [x for x in range(1, N + 1) if x not in chosen_set]
            if not candidates:
                break
            
            random.shuffle(candidates)
            for x in candidates[:80]:
                # Quick check: can we add x without conflicts?
                ok, conflicts = try_add(chosen, counts, x)
                
                if ok:
                    # Can add x directly
                    for y in chosen:
                        s = x + y
                        counts[s] += 1
                    s = 2 * x
                    counts[s] += 1
                    chosen.append(x)
                    chosen_set.add(x)
                    best_size = len(chosen)
                    improved = True
                    break
                elif conflicts:
                    # Try removing one conflicting element and adding x
                    for rem in conflicts:
                        # Simulate: remove rem, add x
                        # Check validity efficiently
                        valid = True
                        # Sums involving rem (to be removed) and x (to be added)
                        # A sum s is affected if it involves rem or x
                        # After swap: remove all pairs with rem, add all pairs with x
                        # Check that no sum exceeds 2 after the swap
                        
                        # Build a temporary delta
                        # For each sum s, compute delta = (contributions from x) - (contributions from rem)
                        # Then check if counts[s] + delta[s] <= 2 for all s, and >= 0
                        
                        # Efficient check: only sums that involve rem or x can change
                        # Sums involving rem: rem + y for y in chosen (y != rem), and 2*rem
                        # Sums involving x: x + y for y in chosen (y != rem after removal), and 2*x
                        
                        # Let's just do the full check but only on affected sums
                        affected = set()
                        for y in chosen:
                            if y != rem:
                                affected.add(rem + y)
                        affected.add(2 * rem)
                        for y in chosen:
                            if y != rem:
                                affected.add(x + y)
                        affected.add(2 * x)
                        
                        for s in affected:
                            # Compute new count for s
                            new_c = counts[s]
                            # Subtract contributions from rem
                            for y in chosen:
                                if y != rem:
                                    if rem + y == s:
                                        new_c -= 1
                            if 2 * rem == s:
                                new_c -= 1
                            # Add contributions from x
                            for y in chosen:
                                if y != rem:
                                    if x + y == s:
                                        new_c += 1
                            if 2 * x == s:
                                new_c += 1
                            if new_c < 0 or new_c > 2:
                                valid = False
                                break
                        
                        if valid:
                            # Apply swap
                            for y in chosen:
                                if y != rem:
                                    s = rem + y
                                    counts[s] -= 1
                            s = 2 * rem
                            counts[s] -= 1
                            for y in chosen:
                                if y != rem:
                                    s = x + y
                                    counts[s] += 1
                            s = 2 * x
                            counts[s] += 1
                            chosen.remove(rem)
                            chosen_set.discard(rem)
                            chosen.append(x)
                            chosen_set.add(x)
                            improved = True
                            break
                    if improved:
                        break
            if not improved:
                break
        
        return chosen, counts, best_size
    
    # Phase 1: Basic greedy
    counts1 = make_counts()
    chosen, counts1 = greedy_forward(counts1, [])
    best_chosen = list(chosen)
    best_size = len(chosen)
    
    if time.time() > deadline:
        return best_chosen
    
    # Phase 2: Multiple greedy restarts with different skip probabilities
    for trial in range(60):
        if time.time() > deadline:
            break
        random.seed(trial * 7 + 13)
        counts2 = make_counts()
        chosen2, counts2 = greedy_with_skip(counts2, [], skip_prob=0.03 + 0.008 * trial)
        if len(chosen2) > best_size:
            best_size = len(chosen2)
            best_chosen = list(chosen2)
    
    if time.time() > deadline:
        return best_chosen
    
    # Phase 3: Local search on best solution
    chosen_ls = list(best_chosen)
    counts_ls = rebuild_counts(chosen_ls)
    
    chosen_ls, counts_ls, best_size_ls = local_search(chosen_ls, counts_ls, max_steps=800)
    if best_size_ls > best_size:
        best_size = best_size_ls
        best_chosen = list(chosen_ls)
    
    if time.time() > deadline:
        return best_chosen
    
    # Phase 4: More greedy restarts with varying parameters + local search
    for trial in range(40):
        if time.time() > deadline:
            break
        random.seed(2000 + trial * 37)
        skip_p = random.uniform(0.0, 0.12)
        counts3 = make_counts()
        chosen3, counts3 = greedy_with_skip(counts3, [], skip_prob=skip_p)
        if len(chosen3) > best_size - 2:  # Only do local search if promising
            counts3 = rebuild_counts(chosen3)
            chosen3, counts3, best_size_ls2 = local_search(chosen3, counts3, max_steps=400)
            if best_size_ls2 > best_size:
                best_size = best_size_ls2
                best_chosen = list(chosen3)
    
    if time.time() > deadline:
        return best_chosen
    
    # Phase 5: Final polish - aggressive local search on best
    chosen_final = list(best_chosen)
    counts_final = rebuild_counts(chosen_final)
    chosen_final, counts_final, best_size_final = local_search(chosen_final, counts_final, max_steps=1000)
    if best_size_final > best_size:
        best_size = best_size_final
        best_chosen = list(chosen_final)
    
    return best_chosen
# EVOLVE-BLOCK-END
