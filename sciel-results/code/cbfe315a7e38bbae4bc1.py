import time
import random
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # 10s margin
    
    N = 511
    # Numbers are 1..511, indexed 0..510
    # Weight of number a (1-indexed) is 1/(512 - a)
    weights = [1.0 / (512 - a) for a in range(1, 512)]
    
    # Precompute conflict bitmask for each number
    # conflict[i] is a bitmask where bit j is set if number (i+1) conflicts with number (j+1)
    conflict = [0] * N
    for i in range(N):
        a = i + 1
        mask = 0
        for j in range(N):
            b = j + 1
            if a != b and gcd(a, b) > 1:
                mask |= (1 << j)
        conflict[i] = mask
    
    def score_of_mask(mask):
        total = 0.0
        for i in range(N):
            if mask & (1 << i):
                total += weights[i]
        return total
    
    def greedy_seed(weight_scale=None):
        """Greedy: iterate in order of (possibly perturbed) weights, add if coprime."""
        if weight_scale is None:
            order = list(range(N))
            order.sort(key=lambda i: -weights[i])
        else:
            # Perturb weights
            perturbed = [weights[i] * (1.0 + weight_scale * (random.random() - 0.5)) for i in range(N)]
            order = list(range(N))
            order.sort(key=lambda i: -perturbed[i])
        
        chosen_mask = 0
        for i in order:
            if conflict[i] & chosen_mask == 0:
                chosen_mask |= (1 << i)
        return chosen_mask
    
    def local_search(mask, max_iter=100):
        """Best-improvement local search with ADD, 1-1, 1-2, 2-1 moves."""
        best_mask = mask
        best_score = score_of_mask(mask)
        
        for iteration in range(max_iter):
            if time.time() - start_time > DEADLINE:
                break
            
            improved = False
            
            # Get list of chosen and not-chosen
            chosen = [i for i in range(N) if best_mask & (1 << i)]
            not_chosen = [i for i in range(N) if not (best_mask & (1 << i))]
            
            # ADD moves
            for i in not_chosen:
                if conflict[i] & best_mask == 0:
                    new_mask = best_mask | (1 << i)
                    new_score = best_score + weights[i]
                    if new_score > best_score + 1e-12:
                        best_mask = new_mask
                        best_score = new_score
                        improved = True
                        # Restart search with new mask
                        chosen = [i for i in range(N) if best_mask & (1 << i)]
                        not_chosen = [i for i in range(N) if not (best_mask & (1 << i))]
            
            if improved:
                continue
            
            # 1-1 swap moves
            for c in chosen:
                mask_without_c = best_mask & ~(1 << c)
                for x in not_chosen:
                    if conflict[x] & mask_without_c == 0:
                        new_mask = mask_without_c | (1 << x)
                        new_score = best_score - weights[c] + weights[x]
                        if new_score > best_score + 1e-12:
                            best_mask = new_mask
                            best_score = new_score
                            improved = True
                            chosen = [i for i in range(N) if best_mask & (1 << i)]
                            not_chosen = [i for i in range(N) if not (best_mask & (1 << i))]
                            break
                if improved:
                    break
            
            if improved:
                continue
            
            # 1-2 moves: remove one chosen, add two not-chosen
            for c in chosen:
                mask_without_c = best_mask & ~(1 << c)
                # Find pairs (x, y) in not_chosen that are coprime to each other and to mask_without_c
                candidates = [x for x in not_chosen if conflict[x] & mask_without_c == 0]
                for idx, x in enumerate(candidates):
                    for y in candidates[idx+1:]:
                        if x == y:
                            continue
                        if conflict[x] & (1 << y) == 0:  # x and y are coprime
                            new_mask = mask_without_c | (1 << x) | (1 << y)
                            new_score = best_score - weights[c] + weights[x] + weights[y]
                            if new_score > best_score + 1e-12:
                                best_mask = new_mask
                                best_score = new_score
                                improved = True
                                chosen = [i for i in range(N) if best_mask & (1 << i)]
                                not_chosen = [i for i in range(N) if not (best_mask & (1 << i))]
                                break
                    if improved:
                        break
                if improved:
                    break
            
            if improved:
                continue
            
            # 2-1 moves: remove two chosen, add one not-chosen
            for idx, c1 in enumerate(chosen):
                for c2 in chosen[idx+1:]:
                    mask_without_both = best_mask & ~(1 << c1) & ~(1 << c2)
                    for x in not_chosen:
                        if conflict[x] & mask_without_both == 0:
                            new_mask = mask_without_both | (1 << x)
                            new_score = best_score - weights[c1] - weights[c2] + weights[x]
                            if new_score > best_score + 1e-12:
                                best_mask = new_mask
                                best_score = new_score
                                improved = True
                                chosen = [i for i in range(N) if best_mask & (1 << i)]
                                not_chosen = [i for i in range(N) if not (best_mask & (1 << i))]
                                break
                    if improved:
                        break
                if improved:
                    break
            
            if improved:
                continue
            
            # No improvement found
            break
        
        return best_mask, best_score
    
    # Run multiple seeds
    best_mask = None
    best_score = -1.0
    
    # First, the deterministic greedy (highest weight first)
    mask = greedy_seed(None)
    mask, score = local_search(mask, max_iter=50)
    if score > best_score:
        best_score = score
        best_mask = mask
    
    # Then, random perturbation seeds
    n_seeds = 20
    for s in range(n_seeds):
        if time.time() - start_time > DEADLINE:
            break
        scale = random.uniform(0.1, 1.0)
        mask = greedy_seed(scale)
        mask, score = local_search(mask, max_iter=30)
        if score > best_score:
            best_score = score
            best_mask = mask
    
    # Final polish with more iterations
    if time.time() - start_time < DEADLINE:
        best_mask, best_score = local_search(best_mask, max_iter=100)
    
    # Convert mask to list of numbers
    result = [i + 1 for i in range(N) if best_mask & (1 << i)]
    return result
