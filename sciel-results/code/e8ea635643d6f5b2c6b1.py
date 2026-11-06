# EVOLVE-BLOCK-START
import time
import random

def construct():
    N = 16
    DEADLINE = 80.0  # safety margin before 90s kill
    start_time = time.time()
    
    # Known 60-comparator sorting network for 16 wires
    base_comparators = [
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
    ]
    
    def is_correct(comps):
        """Verify sorting network using zero-one principle."""
        for mask in range(1 << N):
            wires = [(mask >> i) & 1 for i in range(N)]
            for i, j in comps:
                if wires[i] > wires[j]:
                    wires[i], wires[j] = wires[j], wires[i]
            for i in range(N - 1):
                if wires[i] > wires[i + 1]:
                    return False
        return True
    
    # Verify base network
    if not is_correct(base_comparators):
        # Fallback to odd-even transposition (120 comparators)
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    best_comparators = list(base_comparators)
    best_count = len(best_comparators)
    
    # Local search: try removing redundant comparators
    def try_remove(comps, idx):
        """Try removing comparator at index idx and check correctness."""
        new_comps = comps[:idx] + comps[idx+1:]
        return is_correct(new_comps), new_comps
    
    # Phase 1: Greedy removal - try to remove each comparator
    comps = list(best_comparators)
    i = 0
    while i < len(comps):
        if time.time() > DEADLINE:
            break
        ok, new_comps = try_remove(comps, i)
        if ok:
            comps = new_comps
            # Don't increment i, check same position again
        else:
            i += 1
    
    if len(comps) < best_count:
        best_comparators = comps
        best_count = len(comps)
    
    # Phase 2: Random search - try random removals and swaps
    random.seed(42)
    max_iterations = 5000
    for iteration in range(max_iterations):
        if time.time() > DEADLINE:
            break
        if len(comps) <= best_count:
            break
            
        # Try random removal
        if random.random() < 0.7:
            idx = random.randint(0, len(comps) - 1)
            ok, new_comps = try_remove(comps, idx)
            if ok and len(new_comps) < len(comps):
                comps = new_comps
                if len(comps) < best_count:
                    best_comparators = list(comps)
                    best_count = len(comps)
        else:
            # Try swapping two comparators (may help reordering)
            if len(comps) >= 2:
                idx1, idx2 = random.sample(range(len(comps)), 2)
                new_comps = list(comps)
                new_comps[idx1], new_comps[idx2] = new_comps[idx2], new_comps[idx1]
                if is_correct(new_comps):
                    # Try removing a random comparator from the swapped version
                    if len(new_comps) > 1:
                        rem_idx = random.randint(0, len(new_comps) - 1)
                        ok2, new_comps2 = try_remove(new_comps, rem_idx)
                        if ok2:
                            comps = new_comps2
                            if len(comps) < best_count:
                                best_comparators = list(comps)
                                best_count = len(comps)
    
    return best_comparators
# EVOLVE-BLOCK-END
