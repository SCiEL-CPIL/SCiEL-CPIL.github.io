"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s kill
    
    def can_add(x, chosen, counts):
        """Check if x can be added to chosen without exceeding 2 representations."""
        if counts.get(2 * x, 0) >= 2:
            return False
        for y in chosen:
            if counts.get(x + y, 0) >= 2:
                return False
        return True

    def update_counts_add(x, chosen, counts):
        """Update counts after adding x (chosen does NOT contain x yet)."""
        new_counts = dict(counts)
        new_counts[2 * x] = new_counts.get(2 * x, 0) + 1
        for y in chosen:
            s = x + y
            new_counts[s] = new_counts.get(s, 0) + 1
        return new_counts

    def rebuild_counts(chosen):
        """Rebuild counts from scratch for a given chosen list."""
        counts = {}
        n = len(chosen)
        for i in range(n):
            a = chosen[i]
            counts[2 * a] = counts.get(2 * a, 0) + 1
            for j in range(i + 1, n):
                s = a + chosen[j]
                counts[s] = counts.get(s, 0) + 1
        return counts

    def complete(chosen, counts):
        """Greedy completion: iterate 1..1000, add if possible."""
        chosen_set = set(chosen)
        for x in range(1, 1001):
            if x in chosen_set:
                continue
            if can_add(x, chosen, counts):
                chosen.append(x)
                chosen_set.add(x)
                counts = update_counts_add(x, chosen, counts)
        return chosen, counts

    def greedy(order):
        """Greedy construction following a given order."""
        chosen = []
        counts = {}
        for x in order:
            if can_add(x, chosen, counts):
                chosen.append(x)
                counts = update_counts_add(x, chosen, counts)
        return chosen, counts

    # 1. Initial Standard Greedy (ascending order)
    order_std = list(range(1, 1001))
    best_chosen, best_counts = greedy(order_std)
    best_chosen, best_counts = complete(best_chosen, best_counts)
    best_size = len(best_chosen)
    
    random.seed(42)
    
    # 2. Randomized Greedy Restarts
    while time.time() < deadline:
        order_rand = list(range(1, 1001))
        random.shuffle(order_rand)
        
        cur_chosen, cur_counts = greedy(order_rand)
        cur_chosen, cur_counts = complete(cur_chosen, cur_counts)
        
        if len(cur_chosen) > best_size:
            best_size = len(cur_chosen)
            best_chosen = cur_chosen
            best_counts = cur_counts
        
        # 3. Local Search: Try removing one element and refilling
        # Pick a few random elements to try removing
        if best_chosen and len(best_chosen) > 0:
            # Try up to 5 random removals per iteration
            for _ in range(5):
                if time.time() > deadline:
                    break
                idx = random.randint(0, len(best_chosen) - 1)
                y = best_chosen[idx]
                
                # Create set without y
                new_chosen = [x for x in best_chosen if x != y]
                # Rebuild counts for new_chosen
                new_counts = rebuild_counts(new_chosen)
                
                # Complete
                new_chosen, new_counts = complete(new_chosen, new_counts)
                
                if len(new_chosen) > best_size:
                    best_size = len(new_chosen)
                    best_chosen = new_chosen
                    best_counts = new_counts
                    break  # Restart local search with new best
        
        if time.time() > deadline:
            break
    
    return best_chosen
# EVOLVE-BLOCK-END
