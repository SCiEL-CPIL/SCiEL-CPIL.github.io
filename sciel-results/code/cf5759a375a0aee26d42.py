import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    # Evaluator kills at 150s; leave safety margin
    deadline = start_time + 140.0

    total_space = q ** n

    # Precompute positional powers
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Precompute neighbor indices for all words (as numpy arrays for speed)
    # neighbors[i] is a list of word indices within distance 1 of i
    neighbors = [None] * total_space
    for i in range(total_space):
        nbrs = [i]
        for pos in range(n):
            digit_val = (i // powers[pos]) % q
            for new_val in range(q):
                if new_val == digit_val:
                    continue
                diff = new_val - digit_val
                nbrs.append(i + diff * powers[pos])
        neighbors[i] = np.array(nbrs, dtype=np.int32)

    # Also keep bitmasks for fast intersection checks
    masks = [0] * total_space
    for i in range(total_space):
        m = 0
        for j in neighbors[i]:
            m |= (1 << j)
        masks[i] = m

    full_mask = (1 << total_space) - 1

    best_size = total_space
    best_selected = list(range(total_space))

    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    def polish(selected, time_limit):
        """Remove redundant codewords iteratively."""
        selected = selected[:]
        if not selected:
            return selected

        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            counts[neighbors[w]] += 1

        changed = True
        while changed and time.time() < time_limit:
            changed = False
            indices = list(range(len(selected)))
            random.shuffle(indices)

            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                nbrs = neighbors[w]
                if np.all(counts[nbrs] > 1):
                    counts[nbrs] -= 1
                    selected.pop(idx)
                    changed = True

        return selected

    def local_search(selected, time_limit):
        """
        Local search with 1-for-1 swaps and limited 2-for-1 swaps.
        """
        selected = selected[:]
        if not selected:
            return selected

        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            counts[neighbors[w]] += 1

        selected_set = set(selected)
        
        # 1-for-1 swap
        for _ in range(100):
            if time.time() > time_limit:
                break
            improved = False
            indices = list(range(len(selected)))
            random.shuffle(indices)

            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                nbrs = neighbors[w]
                unique = nbrs[counts[nbrs] == 1]
                
                if len(unique) == 0:
                    # Can just remove
                    counts[nbrs] -= 1
                    selected.pop(idx)
                    selected_set.remove(w)
                    improved = True
                    continue

                # Find replacement C that covers unique
                best_gain = -1
                best_word = -1
                for w2 in range(total_space):
                    if w2 in selected_set:
                        continue
                    nbrs2 = neighbors[w2]
                    if np.all(np.isin(unique, nbrs2)):
                        gain = len(set(nbrs2) - set(nbrs))
                        if gain > best_gain:
                            best_gain = gain
                            best_word = w2

                if best_word >= 0 and best_gain > 0:
                    # Swap w with best_word
                    counts[nbrs] -= 1
                    counts[neighbors[best_word]] += 1
                    selected[idx] = best_word
                    selected_set.remove(w)
                    selected_set.add(best_word)
                    improved = True

            if not improved:
                break

        # 2-for-1 swap (limited)
        if time.time() < time_limit - 5:
            for _ in range(20):
                if time.time() > time_limit:
                    break
                improved = False
                indices = list(range(len(selected)))
                random.shuffle(indices)
                
                for i_idx in range(min(50, len(indices))):
                    idx1 = indices[i_idx]
                    if idx1 >= len(selected): continue
                    w1 = selected[idx1]
                    
                    idx2 = random.randint(0, len(selected)-1)
                    if idx2 == idx1: continue
                    w2 = selected[idx2]
                    
                    nbrs1 = neighbors[w1]
                    nbrs2 = neighbors[w2]
                    
                    # Words uniquely covered by {w1, w2}
                    u_ab = np.concatenate([
                        nbrs1[counts[nbrs1] == 1],
                        nbrs2[counts[nbrs2] == 1]
                    ])
                    u_ab = np.unique(u_ab)
                    
                    if len(u_ab) == 0:
                        # Can remove both
                        counts[nbrs1] -= 1
                        counts[nbrs2] -= 1
                        if idx1 > idx2:
                            idx1, idx2 = idx2, idx1
                        selected.pop(idx2)
                        selected.pop(idx1)
                        selected_set.remove(w1)
                        selected_set.remove(w2)
                        improved = True
                        break
                        
                    if len(u_ab) > 10:
                        continue

                    # Find C that covers u_ab
                    best_word = -1
                    for w3 in range(total_space):
                        if w3 in selected_set:
                            continue
                        nbrs3 = neighbors[w3]
                        if np.all(np.isin(u_ab, nbrs3)):
                            best_word = w3
                            break
                            
                    if best_word >= 0:
                        counts[nbrs1] -= 1
                        counts[nbrs2] -= 1
                        counts[neighbors[best_word]] += 1
                        
                        if idx1 > idx2:
                            idx1, idx2 = idx2, idx1
                        selected.pop(idx2)
                        selected.pop(idx1)
                        selected.append(best_word)
                        
                        selected_set.remove(w1)
                        selected_set.remove(w2)
                        selected_set.add(best_word)
                        
                        improved = True
                        break
                
                if improved:
                    selected = polish(selected, time_limit)
                    counts = np.zeros(total_space, dtype=np.int32)
                    for w in selected:
                        counts[neighbors[w]] += 1
                    selected_set = set(selected)

        return selected

    def run_greedy(randomized=False, alpha=1.0):
        """Greedy set cover."""
        covered = np.zeros(total_space, dtype=bool)
        selected = []
        selected_set = set()

        while not covered.all() and time.time() < deadline - 5:
            best_gain = -1
            best_word = -1

            for w in range(total_space):
                if w in selected_set:
                    continue
                nbrs = neighbors[w]
                gain = np.sum(~covered[nbrs])
                if randomized:
                    gain = gain + random.gauss(0, 1.0) * alpha
                if gain > best_gain:
                    best_gain = gain
                    best_word = w

            if best_word < 0:
                break

            selected.append(best_word)
            selected_set.add(best_word)
            covered[neighbors[best_word]] = True

        return selected

    # Strategy 1: Standard greedy + polish + local search
    if time.time() < deadline - 10:
        sol = run_greedy(randomized=False)
        polished = polish(sol[:], deadline - 10)
        ls = local_search(polished[:], deadline - 10)
        if len(ls) < best_size:
            best_size = len(ls)
            best_selected = ls[:]

    # Strategy 2: Multiple randomized greedy restarts
    num_restarts = 100
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break

        sol = run_greedy(randomized=True, alpha=random.uniform(0.1, 0.5))
        if len(sol) < best_size + 5:
            polished = polish(sol[:], deadline - 20)
            ls = local_search(polished[:], deadline - 20)
            if len(ls) < best_size:
                best_size = len(ls)
                best_selected = ls[:]

    # Final polish
    if time.time() < deadline - 2:
        polished = polish(best_selected[:], deadline - 2)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]

    best_words = [int_to_tuple(v) for v in best_selected]
    return best_words
