"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools


def construct(q, n, radius):
    """
    Constructs a covering code of length n over alphabet size q with covering radius radius.
    Uses a greedy algorithm with local search refinement to minimize the number of codewords.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin (150s limit, leave 35s buffer)

    # Total number of words in the space
    total_words = q ** n

    # Helper to convert integer to list of digits
    def int_to_word(idx, n, q):
        word = [0] * n
        for i in range(n):
            word[i] = idx % q
            idx //= q
        return word

    # Helper to convert list of digits to integer
    def word_to_int(word, q):
        val = 0
        for i in range(len(word) - 1, -1, -1):
            val = val * q + word[i]
        return val

    # Precompute neighbors for each word (ball of radius 'radius')
    # For efficiency, we precompute the list of neighbors for each index
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        w = int_to_word(i, n, q)
        neighbors[i].append(i)  # self
        
        if radius >= 1:
            for pos in range(n):
                original_val = w[pos]
                for new_val in range(q):
                    if new_val != original_val:
                        w[pos] = new_val
                        neighbor_idx = word_to_int(w, q)
                        neighbors[i].append(neighbor_idx)
                w[pos] = original_val

        if radius >= 2:
            # For radius 2, add words at distance 2
            # This is more expensive but for small n,q it's okay
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    v1 = w[pos1]
                    v2 = w[pos2]
                    for nv1 in range(q):
                        if nv1 == v1:
                            continue
                        for nv2 in range(q):
                            if nv2 == v2:
                                continue
                            w[pos1] = nv1
                            w[pos2] = nv2
                            neighbor_idx = word_to_int(w, q)
                            neighbors[i].append(neighbor_idx)
                    w[pos1] = v1
                    w[pos2] = v2

    # Convert neighbors to sets for fast lookup
    neighbor_sets = [set(nb) for nb in neighbors]

    # Greedy Set Cover
    uncovered_set = set(range(total_words))
    codewords = []
    
    # Special seed for q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        # Tetracode direct sum: 9 * 9 = 81 words
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        for tail_a in range(3):
            for tail_b in range(3):
                for h in tetracode:
                    word = h + [tail_a, tail_b]
                    idx = word_to_int(word, q)
                    codewords.append(idx)
                    for nb in neighbors[idx]:
                        if nb in uncovered_set:
                            uncovered_set.remove(nb)

    # Greedy fill
    while uncovered_set:
        if time.time() > deadline - 10:  # Leave time for local search
            break
        
        best_center = -1
        best_count = -1
        
        # Optimization: If uncovered_set is small, pick a random uncovered word and find its best neighbor
        # Otherwise, standard greedy
        if len(uncovered_set) < 50:
            # Pick a random uncovered word
            target = random.choice(list(uncovered_set))
            # Find the neighbor of target that covers the most uncovered words
            # Actually, just pick the target itself or a neighbor?
            # Standard greedy: pick center that maximizes coverage of uncovered_set
            # But if we only have few uncovered, maybe we can be smarter?
            # Let's stick to standard greedy but maybe limit candidates?
            # No, standard greedy is fine.
            pass

        for candidate in range(total_words):
            count = 0
            # Count how many neighbors of candidate are in uncovered_set
            # Using intersection size
            count = len(neighbor_sets[candidate] & uncovered_set)
            
            if count > best_count:
                best_count = count
                best_center = candidate

        if best_center == -1:
            break

        codewords.append(best_center)
        # Update uncovered_set
        for nb in neighbors[best_center]:
            if nb in uncovered_set:
                uncovered_set.remove(nb)

    # Local Search Refinement
    # 1. Try to remove redundant codewords
    # 2. Try to swap a codeword with another to increase coverage (or maintain) and allow removal of another?
    # Simpler: Just try to remove codewords that are redundant.
    
    changed = True
    while changed:
        if time.time() > deadline - 2:
            break
        changed = False
        for i in range(len(codewords)):
            if time.time() > deadline - 2:
                break
            c_idx = codewords[i]
            
            # Check if c_idx is redundant
            # A codeword is redundant if every word in its ball is covered by at least one other codeword
            redundant = True
            for w_idx in neighbors[c_idx]:
                # Is w_idx covered by any other codeword?
                covered_by_other = False
                for j in range(len(codewords)):
                    if j == i:
                        continue
                    if w_idx in neighbor_sets[codewords[j]]:
                        covered_by_other = True
                        break
                if not covered_by_other:
                    redundant = False
                    break
            
            if redundant:
                codewords.pop(i)
                changed = True
                break # Restart loop as indices changed

    # 2. Try to swap codewords to reduce size?
    # If we can replace codeword A with codeword B, and then remove codeword C, that's a net reduction.
    # This is complex. Let's try a simpler improvement:
    # Try to replace each codeword with a different one that covers the same "essential" words?
    # Actually, the removal step above handles simple redundancy.
    # Let's try a 1-step swap:
    # For each codeword i, try replacing it with a non-codeword j.
    # If the new set (with j instead of i) is still a valid cover, we haven't reduced size.
    # But maybe it allows removal of another codeword k?
    # That's a 2-step move.
    
    # Let's try a different local search:
    # Randomly remove a codeword. If invalid, try to add a "best" codeword to fix it?
    # No, that increases size.
    
    # Let's try to find a codeword that can be replaced by another such that the new set is smaller?
    # No, replacement keeps size same.
    
    # The main way to reduce size is finding redundancy.
    # Maybe we can improve the greedy selection to be less likely to create redundancies?
    # Or do multiple random restarts of the greedy phase?
    
    # Given time constraints, let's stick to the current structure but ensure the local search is robust.
    # We can also try to "shake" the solution:
    # Randomly remove a few codewords, then greedily add back the best ones.
    # This might escape local minima of the removal process.
    
    best_codewords = list(codewords)
    
    # Try a few random perturbations if time allows
    for _ in range(5):
        if time.time() > deadline - 5:
            break
            
        # Make a copy
        current = list(best_codewords)
        
        # Remove 2 random codewords
        if len(current) > 2:
            idx1 = random.randint(0, len(current) - 1)
            idx2 = random.randint(0, len(current) - 1)
            if idx1 != idx2:
                current.pop(max(idx1, idx2))
                current.pop(min(idx1, idx2))
                
                # Greedily add back best codewords until covered
                # Rebuild uncovered_set
                covered_mask = [False] * total_words
                for c in current:
                    for nb in neighbors[c]:
                        covered_mask[nb] = True
                
                uncovered = set(i for i in range(total_words) if not covered_mask[i])
                
                # Greedy add
                while uncovered:
                    if time.time() > deadline - 2:
                        break
                    best_c = -1
                    best_cnt = -1
                    for cand in range(total_words):
                        cnt = len(neighbor_sets[cand] & uncovered)
                        if cnt > best_cnt:
                            best_cnt = cnt
                            best_c = cand
                    if best_c == -1:
                        break
                    current.append(best_c)
                    for nb in neighbors[best_c]:
                        if nb in uncovered:
                            uncovered.remove(nb)
                
                # Clean up redundancies
                changed = True
                while changed:
                    if time.time() > deadline - 1:
                        break
                    changed = False
                    for i in range(len(current)):
                        c_idx = current[i]
                        redundant = True
                        for w_idx in neighbors[c_idx]:
                            covered_by_other = False
                            for j in range(len(current)):
                                if j == i:
                                    continue
                                if w_idx in neighbor_sets[current[j]]:
                                    covered_by_other = True
                                    break
                            if not covered_by_other:
                                redundant = False
                                break
                        if redundant:
                            current.pop(i)
                            changed = True
                            break
                
                if len(current) < len(best_codewords):
                    best_codewords = current

    # Final check and cleanup
    # Ensure no duplicates
    best_codewords = list(set(best_codewords))
    
    # Final redundancy removal
    changed = True
    while changed:
        if time.time() > deadline - 0.5:
            break
        changed = False
        for i in range(len(best_codewords)):
            c_idx = best_codewords[i]
            redundant = True
            for w_idx in neighbors[c_idx]:
                covered_by_other = False
                for j in range(len(best_codewords)):
                    if j == i:
                        continue
                    if w_idx in neighbor_sets[best_codewords[j]]:
                        covered_by_other = True
                        break
                if not covered_by_other:
                    redundant = False
                    break
            if redundant:
                best_codewords.pop(i)
                changed = True
                break

    result = [int_to_word(c, n, q) for c in best_codewords]
    return result
# EVOLVE-BLOCK-END
