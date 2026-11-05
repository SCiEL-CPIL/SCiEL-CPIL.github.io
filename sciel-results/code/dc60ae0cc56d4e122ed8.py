import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Specific case for the task
    if q == 3 and n == 6 and radius == 1:
        total = 3 ** n
        powers = [3 ** p for p in range(n)]
        
        # Precompute neighbors
        neighbors = []
        for i in range(total):
            d = [(i // powers[p]) % q for p in range(n)]
            covered = [i]
            for p in range(n):
                for alt in range(q):
                    if alt != d[p]:
                        new_i = i + (alt - d[p]) * powers[p]
                        covered.append(new_i)
            neighbors.append(covered)
        
        # Precompute neighbor matrix for fast greedy (729x729, int8)
        N = np.zeros((total, total), dtype=np.int8)
        for i in range(total):
            for w in neighbors[i]:
                N[i, w] = 1
        
        # Precompute for each word, which codewords cover it (inverse)
        # coverers[w] = list of codeword indices that cover word w
        coverers = [[] for _ in range(total)]
        for i in range(total):
            for w in neighbors[i]:
                coverers[w].append(i)
        
        # Greedy Initialization using numpy matrix-vector multiply
        def greedy_init():
            uncovered_vec = np.ones(total, dtype=np.int8)
            codewords = []
            while uncovered_vec.sum() > 0:
                scores = N @ uncovered_vec
                best_cand = int(np.argmax(scores))
                if scores[best_cand] == 0:
                    break
                codewords.append(best_cand)
                uncovered_vec[neighbors[best_cand]] = 0
            return codewords

        codewords = greedy_init()
        if not codewords:
             return [[0]*n]
             
        codewords_set = set(codewords)
        
        # Helper to update counts
        def update_counts(cw, delta):
            for w in neighbors[cw]:
                counts[w] += delta

        counts = np.zeros(total, dtype=np.int32)
        for cw in codewords:
            update_counts(cw, 1)

        best_len = len(codewords)
        best_codewords = codewords[:]
        
        # Fast repair: greedily add codewords to cover uncovered words using numpy
        def fast_repair():
            """Repair uncovered words by greedily adding codewords."""
            while True:
                uncovered_vec = (counts == 0).astype(np.int8)
                if uncovered_vec.sum() == 0:
                    break
                scores = N @ uncovered_vec
                best_cand = int(np.argmax(scores))
                if scores[best_cand] == 0:
                    break
                if best_cand not in codewords_set:
                    update_counts(best_cand, 1)
                    codewords.append(best_cand)
                    codewords_set.add(best_cand)
                else:
                    # Shouldn't happen, but break to avoid infinite loop
                    break
        
        # Local Search
        while time.time() < deadline:
            improved = False
            
            # 1. Try to remove redundant codewords
            for i in range(len(codewords)):
                cw = codewords[i]
                # Check if all neighbors have count > 1
                redundant = True
                for w in neighbors[cw]:
                    if counts[w] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove
                    update_counts(cw, -1)
                    codewords.pop(i)
                    codewords_set.remove(cw)
                    improved = True
                    if len(codewords) < best_len:
                        best_len = len(codewords)
                        best_codewords = codewords[:]
                    break # Restart loop to check for new redundancies
            
            if improved:
                continue

            # 2. Try to swap codewords to enable further reduction
            swap_done = False
            for i in range(len(codewords)):
                cw_i = codewords[i]
                # Find essential words (covered only by cw_i)
                essential = []
                for w in neighbors[cw_i]:
                    if counts[w] == 1:
                        essential.append(w)
                
                if not essential:
                    # Should have been removed in step 1
                    continue

                # Find a replacement cw_j that covers all essential words
                # Use the coverers list for fast lookup
                # A candidate must cover ALL essential words
                # Start with coverers of the first essential word
                first_ess = essential[0]
                candidates = [j for j in coverers[first_ess] if j not in codewords_set]
                
                for cw_j in candidates:
                    # Check if cw_j covers all other essential words
                    valid = True
                    for w in essential[1:]:
                        if cw_j not in coverers[w]:
                            valid = False
                            break
                    if valid:
                        # Perform Swap
                        update_counts(cw_i, -1)
                        update_counts(cw_j, 1)
                        codewords[i] = cw_j
                        codewords_set.remove(cw_i)
                        codewords_set.add(cw_j)
                        swap_done = True
                        break
                if swap_done:
                    break
            
            if swap_done:
                continue

            # 3. Perturbation: Stuck, try to escape
            if time.time() > deadline - 5:
                break
                
            num_remove = random.randint(1, 2)
            to_remove = random.sample(codewords, min(num_remove, len(codewords)))
            for cw in to_remove:
                update_counts(cw, -1)
                codewords.remove(cw)
                codewords_set.remove(cw)
                
            # Repair using fast numpy-based greedy
            fast_repair()
            
            # After repair, loop back to check for redundancies

        # Final polish: try more swaps if time permits
        remaining = deadline - time.time()
        if remaining > 2:
            # Do additional swap passes
            for _ in range(int(remaining)):
                if time.time() > deadline - 1:
                    break
                # Remove redundancies
                changed = True
                while changed:
                    changed = False
                    for i in range(len(codewords)):
                        cw = codewords[i]
                        redundant = True
                        for w in neighbors[cw]:
                            if counts[w] == 1:
                                redundant = False
                                break
                        if redundant:
                            update_counts(cw, -1)
                            codewords.pop(i)
                            codewords_set.remove(cw)
                            changed = True
                            if len(codewords) < best_len:
                                best_len = len(codewords)
                                best_codewords = codewords[:]
                            break
                # Try one swap
                for i in range(len(codewords)):
                    cw_i = codewords[i]
                    essential = [w for w in neighbors[cw_i] if counts[w] == 1]
                    if not essential:
                        continue
                    first_ess = essential[0]
                    candidates = [j for j in coverers[first_ess] if j not in codewords_set]
                    for cw_j in candidates:
                        valid = all(cw_j in coverers[w] for w in essential[1:])
                        if valid:
                            update_counts(cw_i, -1)
                            update_counts(cw_j, 1)
                            codewords[i] = cw_j
                            codewords_set.remove(cw_i)
                            codewords_set.add(cw_j)
                            break
                    else:
                        continue
                    break

        # Return best found
        if len(best_codewords) <= len(codewords):
            final_codewords = best_codewords
        else:
            final_codewords = codewords
            
        # Convert to list of lists
        result = []
        for idx in final_codewords:
            word = [(idx // powers[p]) % q for p in range(n)]
            result.append(word)
            
        # Verify
        covered = np.zeros(total, dtype=bool)
        for cw in final_codewords:
            for w in neighbors[cw]:
                covered[w] = True
        if covered.all():
            return result
        else:
            # Fallback to greedy
            greedy_result = greedy_init()
            result = []
            for idx in greedy_result:
                word = [(idx // powers[p]) % q for p in range(n)]
                result.append(word)
            return result

    # Fallback for other cases
    return [[0] * n]
