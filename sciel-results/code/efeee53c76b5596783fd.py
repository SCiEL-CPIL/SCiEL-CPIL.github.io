<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np
from collections import defaultdict, deque
import time

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Uses randomized local search with multiple restarts.
    """
    if n == 0:
        return [[]]
    
    total_words = q ** n
    
    def int_to_word(val, n, q):
        word = []
        for _ in range(n):
            word.append(val % q)
            val //= q
        return word
    
    def word_to_int(word, q):
        val = 0
        for i in range(len(word) - 1, -1, -1):
            val = val * q + word[i]
        return val
    
    all_words = list(range(total_words))
    
    # Precompute neighbors
    neighbors = {}
    for w in all_words:
        word = int_to_word(w, n, q)
        neigh = [w]
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val == word[pos]:
                        continue
                    new_word = word[:]
                    new_word[pos] = val
                    neigh.append(word_to_int(new_word, q))
        if radius >= 2:
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    for val1 in range(q):
                        if val1 == word[pos1]:
                            continue
                        for val2 in range(q):
                            if val2 == word[pos2]:
                                continue
                            new_word = word[:]
                            new_word[pos1] = val1
                            new_word[pos2] = val2
                            new_int = word_to_int(new_word, q)
                            if new_int not in neigh:
                                neigh.append(new_int)
        neighbors[w] = set(neigh)
    
    def greedy_init():
        covered = [False] * total_words
        codewords = []
        uncovered_count = total_words
        remaining = set(all_words)
        
        while uncovered_count > 0:
            best_word = -1
            best_new_coverage = -1
            # Randomize order to break ties
            rem_list = list(remaining)
            random.shuffle(rem_list)
            
            for w in rem_list:
                new_cov = 0
                for u in neighbors[w]:
                    if not covered[u]:
                        new_cov += 1
                if new_cov > best_new_coverage:
                    best_new_coverage = new_cov
                    best_word = w
                if new_cov == uncovered_count:
                    break
            
            if best_word == -1:
                break
            
            codewords.append(best_word)
            remaining.discard(best_word)
            for u in neighbors[best_word]:
                if not covered[u]:
                    covered[u] = True
                    uncovered_count -= 1
        
        return codewords
    
    def optimize(codewords, time_limit):
        coverage_count = [0] * total_words
        for cw in codewords:
            for u in neighbors[cw]:
                coverage_count[u] += 1
                
        cw_set = set(codewords)
        
        def try_remove_redundant():
            removed_any = False
            changed = True
            while changed:
                changed = False
                indices = list(range(len(codewords)))
                random.shuffle(indices)
                for i in indices:
                    cw = codewords[i]
                    can_remove = True
                    for u in neighbors[cw]:
                        if coverage_count[u] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        for u in neighbors[cw]:
                            coverage_count[u] -= 1
                        codewords.pop(i)
                        cw_set.discard(cw)
                        changed = True
                        removed_any = True
                        break
            return removed_any
        
        try_remove_redundant()
        
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            if try_remove_redundant():
                continue
                
            if len(codewords) == 0:
                break
                
            idx = random.randint(0, len(codewords) - 1)
            old_cw = codewords[idx]
            
            would_uncover = []
            for u in neighbors[old_cw]:
                if coverage_count[u] == 1:
                    would_uncover.append(u)
            
            if len(would_uncover) == 0:
                for u in neighbors[old_cw]:
                    coverage_count[u] -= 1
                codewords.pop(idx)
                cw_set.discard(old_cw)
                continue
            
            # Find best candidate from neighbors of the first uncovered word
            if would_uncover:
                first_uncovered = would_uncover[0]
                potential_candidates = list(neighbors[first_uncovered])
                random.shuffle(potential_candidates)
                
                best_new_cw = -1
                best_score = -1
                
                for new_cw in potential_candidates:
                    if new_cw in cw_set or new_cw == old_cw:
                        continue
                    
                    # Check if new_cw covers all would_uncover
                    valid = True
                    for u in would_uncover:
                        if u not in neighbors[new_cw]:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Score: Maximize redundancy gain
                    score = 0
                    for u in neighbors[new_cw]:
                        if u not in neighbors[old_cw]:
                            if coverage_count[u] >= 1:
                                score += 1
                    
                    if score > best_score:
                        best_score = score
                        best_new_cw = new_cw
                
                if best_new_cw != -1:
                    for u in neighbors[old_cw]:
                        coverage_count[u] -= 1
                    for u in neighbors[best_new_cw]:
                        coverage_count[u] += 1
                    codewords[idx] = best_new_cw
                    cw_set.discard(old_cw)
                    cw_set.add(best_new_cw)
        
        try_remove_redundant()
        return codewords
    
    # Multiple restarts
    start_time = time.time()
    time_limit = 115
    best_codewords = []
    
    while time.time() - start_time < time_limit:
        # Allocate time for this iteration
        # Greedy is fast, so give most time to optimize
        iter_time = min(20, time_limit - (time.time() - start_time))
        if iter_time <= 0:
            break
        
        cw = greedy_init()
        cw = optimize(cw, iter_time - 2) # Leave 2s for safety
        
        if len(cw) < len(best_codewords) or not best_codewords:
            best_codewords = cw
        
        # Check time
        if time.time() - start_time > time_limit - 5:
            break
    
    result = [int_to_word(cw, n, q) for cw in best_codewords]
    return result
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np
import time

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Uses vectorized greedy initialization and bitset-based local search optimization.
    """
    if n == 0:
        return [[]]
    
    total_words = q ** n
    
    def int_to_word(val, n, q):
        word = []
        for _ in range(n):
            word.append(val % q)
            val //= q
        return word
    
    def word_to_int(word, q):
        val = 0
        for i in range(len(word) - 1, -1, -1):
            val = val * q + word[i]
        return val
    
    # Precompute neighbor lists as sorted lists and bitsets
    neighbors_list = []  # neighbors_list[w] = sorted list of neighbor ints
    neighbors_bits = np.zeros(total_words, dtype=object)  # bitset for each word
    
    for w in range(total_words):
        word = int_to_word(w, n, q)
        neigh = {w}
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val == word[pos]:
                        continue
                    new_word = word[:]
                    new_word[pos] = val
                    neigh.add(word_to_int(new_word, q))
        if radius >= 2:
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    for val1 in range(q):
                        if val1 == word[pos1]:
                            continue
                        for val2 in range(q):
                            if val2 == word[pos2]:
                                continue
                            new_word = word[:]
                            new_word[pos1] = val1
                            new_word[pos2] = val2
                            neigh.add(word_to_int(new_word, q))
        neighbors_list.append(sorted(neigh))
        # Build bitset
        bs = np.zeros(total_words, dtype=bool)
        for u in neigh:
            bs[u] = True
        neighbors_bits[w] = bs
    
    # Also build reverse: for each word, which codewords cover it (inverse)
    # This is useful for finding candidates that cover a given word
    inverse_bits = np.zeros(total_words, dtype=object)
    for w in range(total_words):
        bs = np.zeros(total_words, dtype=bool)
        for u in neighbors_list[w]:
            bs[u] = True
        inverse_bits[w] = bs  # inverse_bits[w][u] = True iff w covers u (same as neighbors)
    
    def greedy_init_vectorized():
        """Vectorized greedy set cover using NumPy."""
        covered = np.zeros(total_words, dtype=bool)
        codewords = []
        uncovered_count = total_words
        
        # Precompute marginal coverage for all words initially
        # marginal[w] = number of uncovered words covered by w
        # Start: all uncovered, so marginal[w] = len(neighbors_list[w])
        marginal = np.array([len(nl) for nl in neighbors_list], dtype=np.int32)
        
        while uncovered_count > 0:
            if time.time() > greedy_deadline:
                break
            # Find word with max marginal coverage
            best_idx = int(np.argmax(marginal))
            if marginal[best_idx] <= 0:
                break
            
            codewords.append(best_idx)
            
            # Update: for each word u that best_idx covers, decrement marginal of all words that also cover u
            # Use bitset: for each u in neighbors_list[best_idx], find all v where neighbors_bits[v][u] is True
            # and decrement marginal[v]
            for u in neighbors_list[best_idx]:
                if not covered[u]:
                    covered[u] = True
                    uncovered_count -= 1
                # Find all v that cover u
                covering_v = np.where(neighbors_bits[:, u] if hasattr(neighbors_bits[:, u], '__len__') else np.array([nb[u] for nb in neighbors_bits]))[0]
                # Actually, let's use a precomputed inverse
                # inv[u] = list of words that cover u
                # Let's precompute this
                pass
            
            # Simpler approach: recompute marginal for remaining words
            # But that's O(n * total_words) per step which is too slow
            # Let's use the inverse mapping
            pass
        
        return codewords
    
    # Precompute inverse: inv[u] = set of words w such that u is in neighbors(w)
    inv = [set() for _ in range(total_words)]
    for w in range(total_words):
        for u in neighbors_list[w]:
            inv[u].add(w)
    
    def greedy_init():
        """Greedy set cover with efficient marginal updates."""
        covered = np.zeros(total_words, dtype=bool)
        codewords = []
        uncovered_count = total_words
        
        # marginal[w] = number of currently-uncovered words in neighbors(w)
        marginal = np.array([len(nl) for nl in neighbors_list], dtype=np.int32)
        
        while uncovered_count > 0:
            if time.time() > greedy_deadline:
                break
            best_idx = int(np.argmax(marginal))
            if marginal[best_idx] <= 0:
                break
            
            codewords.append(best_idx)
            
            # Update marginal: for each u in neighbors(best_idx), for each v in inv[u], marginal[v] -= 1
            for u in neighbors_list[best_idx]:
                if not covered[u]:
                    covered[u] = True
                    uncovered_count -= 1
                # For each v that covers u, decrement marginal[v]
                for v in inv[u]:
                    marginal[v] -= 1
                    if marginal[v] < 0:
                        marginal[v] = 0
        
        return codewords
    
    def optimize(codewords, time_limit):
        """Local search optimization using bitset operations."""
        # Build coverage count using bitset sum
        # coverage_count[u] = number of codewords that cover u
        coverage_count = np.zeros(total_words, dtype=np.int32)
        for cw in codewords:
            for u in neighbors_list[cw]:
                coverage_count[u] += 1
        
        cw_set = set(codewords)
        
        def try_remove_redundant():
            """Remove codewords that don't uniquely cover any word."""
            removed_any = False
            changed = True
            while changed:
                changed = False
                indices = list(range(len(codewords)))
                random.shuffle(indices)
                for i in indices:
                    cw = codewords[i]
                    # Check if any word is covered only by this codeword
                    can_remove = True
                    for u in neighbors_list[cw]:
                        if coverage_count[u] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        for u in neighbors_list[cw]:
                            coverage_count[u] -= 1
                        codewords.pop(i)
                        cw_set.discard(cw)
                        changed = True
                        removed_any = True
                        break
            return removed_any
        
        try_remove_redundant()
        
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            if try_remove_redundant():
                continue
            
            if len(codewords) == 0:
                break
            
            # Pick a random codeword to try removing/replacing
            idx = random.randint(0, len(codewords) - 1)
            old_cw = codewords[idx]
            
            # Find words that would become uncovered if we remove old_cw
            would_uncover = [u for u in neighbors_list[old_cw] if coverage_count[u] == 1]
            
            if len(would_uncover) == 0:
                # Can safely remove
                for u in neighbors_list[old_cw]:
                    coverage_count[u] -= 1
                codewords.pop(idx)
                cw_set.discard(old_cw)
                continue
            
            # Need to find a replacement that covers all would_uncover words
            # Candidate set: intersection of inv[u] for u in would_uncover, minus cw_set
            # Use bitset intersection
            candidate_bits = np.ones(total_words, dtype=bool)
            for u in would_uncover:
                # inv[u] is a set; build a boolean array
                tmp = np.zeros(total_words, dtype=bool)
                for v in inv[u]:
                    tmp[v] = True
                candidate_bits &= tmp
            
            # Exclude existing codewords and old_cw
            for cw in cw_set:
                candidate_bits[cw] = False
            candidate_bits[old_cw] = False
            
            candidates = np.where(candidate_bits)[0]
            
            if len(candidates) == 0:
                # Try random move: just pick a random codeword and try to replace it
                # Or skip this iteration
                continue
            
            # Score each candidate: how many new words does it cover (that aren't in old_cw's neighborhood)
            # that are already covered (redundancy gain)
            old_neigh_set = set(neighbors_list[old_cw])
            
            best_new_cw = -1
            best_score = -1
            
            # Sample candidates if too many
            if len(candidates) > 50:
                sampled = random.sample(list(candidates), 50)
            else:
                sampled = list(candidates)
            
            for new_cw in sampled:
                # Score: number of words in neighbors(new_cw) that are NOT in neighbors(old_cw)
                # and have coverage_count >= 1 (i.e., we're adding redundancy)
                score = 0
                for u in neighbors_list[new_cw]:
                    if u not in old_neigh_set and coverage_count[u] >= 1:
                        score += 1
                
                if score > best_score:
                    best_score = score
                    best_new_cw = new_cw
            
            if best_new_cw != -1 and best_score > 0:
                for u in neighbors_list[old_cw]:
                    coverage_count[u] -= 1
                for u in neighbors_list[best_new_cw]:
                    coverage_count[u] += 1
                codewords[idx] = best_new_cw
                cw_set.discard(old_cw)
                cw_set.add(best_new_cw)
        
        try_remove_redundant()
        return codewords
    
    # Set up deadlines
    global_start = time.time()
    total_time_limit = 115
    greedy_deadline = global_start + 5  # 5 seconds for greedy
    
    # Multiple restarts
    best_codewords = []
    
    while time.time() - global_start < total_time_limit - 10:
        # Reset greedy deadline for this iteration
        greedy_deadline = time.time() + 3
        
        cw = greedy_init()
        
        if len(cw) == 0:
            break
        
        remaining_time = total_time_limit - (time.time() - global_start) - 10
        if remaining_time < 2:
            break
        
        cw = optimize(cw, min(remaining_time, 15))
        
        if len(cw) < len(best_codewords) or not best_codewords:
            best_codewords = list(cw)
    
    # Final polish: try all single replacements in a deterministic pass
    if best_codewords and time.time() - global_start < total_time_limit - 3:
        polish_time = min(5, total_time_limit - (time.time() - global_start) - 2)
        # Re-optimize the best solution with more time
        best_codewords = optimize(list(best_codewords), polish_time)
    
    result = [int_to_word(cw, n, q) for cw in best_codewords]
    return result
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
