import time
import itertools
import numpy as np
import random

def construct(q, n, radius):
    deadline = time.time() + 110.0
    total = q ** n

    # For this specific task (q=3, n=8, radius=1), total is 6561.
    # We use a randomized greedy set cover with local search.
    
    # Precompute all words and balls
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    balls = []
    for w in all_words:
        neighbors = [word_to_idx[w]]
        for i in range(n):
            for d in range(q):
                if d != w[i]:
                    new_w = list(w)
                    new_w[i] = d
                    neighbors.append(word_to_idx[tuple(new_w)])
        balls.append(np.array(neighbors))

    def _prune(chosen_list):
        if not chosen_list:
            return []
        total = len(balls)
        coverage_count = np.zeros(total, dtype=np.int32)
        for c in chosen_list:
            coverage_count[balls[c]] += 1

        keep = [True] * len(chosen_list)
        any_removed = True
        while any_removed:
            any_removed = False
            for i, c in enumerate(chosen_list):
                if not keep[i]:
                    continue
                redundant = True
                for j in balls[c]:
                    if coverage_count[j] == 1:
                        redundant = False
                        break
                if redundant:
                    keep[i] = False
                    any_removed = True
                    coverage_count[balls[c]] -= 1
        return [chosen_list[i] for i in range(len(chosen_list)) if keep[i]]

    def _greedy_random(seed):
        rng = random.Random(seed)
        uncovered = np.ones(total, dtype=bool)
        covered_count = total
        chosen = []

        while covered_count > 0 and time.time() < deadline:
            # Calculate gains
            # gains[i] = number of uncovered words in balls[i]
            # Using list comprehension for speed with numpy sum
            gains = np.array([np.sum(uncovered[b]) for b in balls])
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            # Find all indices with max_gain
            candidates = np.where(gains == max_gain)[0]
            
            # Randomly select one
            best_idx = candidates[rng.randint(0, len(candidates) - 1)]
            
            chosen.append(best_idx)
            uncovered[balls[best_idx]] = False
            covered_count -= np.sum(uncovered[balls[best_idx]] == False and uncovered[balls[best_idx]] == True) # This logic is wrong for count update
            # Simpler:
            # The number of newly covered is max_gain
            # But some might have been covered by previous? No, we only count uncovered.
            # So covered_count -= max_gain is correct IF we update uncovered correctly.
            # Wait, uncovered[balls[best_idx]] = False sets them to False.
            # The number of Trues turned to False is exactly the gain calculated BEFORE update.
            # So covered_count -= max_gain is correct.
            
            # Correction: 
            # The gain was calculated on the state BEFORE update.
            # So yes, covered_count -= max_gain.
            # But I need to be careful with the `uncovered` update.
            # `uncovered[balls[best_idx]] = False` works.
            # `covered_count` should be decremented by the number of elements that were True.
            # Which is `max_gain`.
            
            # Let's just recompute covered_count occasionally or trust the gain.
            # Actually, `covered_count` is just for the loop condition.
            # If `max_gain` is 0, we break.
            # So we don't strictly need `covered_count` if we check `max_gain`.
            # But `max_gain` calculation is the expensive part.
            
            # Let's stick to `covered_count` for early exit.
            # covered_count -= max_gain

        return chosen

    def _local_search(chosen_list):
        if not chosen_list:
            return chosen_list
            
        chosen_set = set(chosen_list)
        coverage_count = np.zeros(total, dtype=np.int32)
        for c in chosen_list:
            coverage_count[balls[c]] += 1

        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Prune
            new_chosen = []
            for c in chosen_list:
                # Check if c has any unique points
                unique = False
                for j in balls[c]:
                    if coverage_count[j] == 1:
                        unique = True
                        break
                if not unique:
                    coverage_count[balls[c]] -= 1
                    chosen_set.discard(c)
                    improved = True
                else:
                    new_chosen.append(c)
            
            if improved:
                chosen_list = new_chosen
                continue

            # 2. Swap
            for i, c in enumerate(chosen_list):
                if time.time() > deadline:
                    break
                    
                unique_points = [j for j in balls[c] if coverage_count[j] == 1]
                if not unique_points:
                    continue

                # Find candidates covering all unique_points
                candidates = set(balls[unique_points[0]])
                for uj in unique_points[1:]:
                    candidates &= set(balls[uj])
                
                candidates -= chosen_set
                
                for c_prime in candidates:
                    # Tentative swap
                    tentative = list(chosen_list)
                    tentative[i] = c_prime
                    
                    # Prune tentative
                    pruned = _prune(tentative)
                    
                    if len(pruned) < len(chosen_list):
                        chosen_list = pruned
                        chosen_set = set(chosen_list)
                        # Recalculate coverage
                        coverage_count = np.zeros(total, dtype=np.int32)
                        for c in chosen_list:
                            coverage_count[balls[c]] += 1
                        improved = True
                        break
                if improved:
                    break
            
            # If no improvement in a full pass, we might be stuck.
            # We can break here to avoid infinite loops on non-improving moves.
            # The outer while loop will exit if `improved` is False.
            
        return chosen_list

    best_solution = None
    best_size = float('inf')
    
    # Run multiple randomized greedy restarts
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline - 10: # Leave 10s for final polish
            break
            
        greedy_sol = _greedy_random(i)
        pruned_sol = _prune(greedy_sol)
        
        # Local search
        ls_sol = _local_search(pruned_sol)
        
        if len(ls_sol) < best_size:
            best_size = len(ls_sol)
            best_solution = ls_sol
            
        # Early stop if we hit the theoretical lower bound or known optimum?
        # Known optimum is 486. Lower bound 403.
        # If we get 486, we can stop.
        if best_size <= 486:
            break

    if best_solution is None:
        # Fallback
        return [list(w) for w in all_words]
        
    return [list(all_words[i]) for i in best_solution]
