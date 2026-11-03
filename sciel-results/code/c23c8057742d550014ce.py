"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np
import time

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Uses structured initialization + iterated local search with simulated annealing.
    """
    if n == 0:
        return [[]]
    
    total_words = q ** n
    deadline = time.time() + 145  # Safety margin before 150s kill
    
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
    
    # Precompute neighbors
    neighbors = [None] * total_words
    
    if radius == 1:
        # Fast path: precompute digit decomposition
        digits = np.zeros((total_words, n), dtype=np.int32)
        for pos in range(n):
            for d in range(q):
                mask = (np.arange(total_words) // (q ** pos)) % q
                digits[mask == d, pos] = d
        
        for w in range(total_words):
            neigh = [w]
            word = digits[w].copy()
            for pos in range(n):
                cur = word[pos]
                base = w - cur * (q ** pos)
                for val in range(q):
                    if val == cur:
                        continue
                    neigh.append(base + val * (q ** pos))
            neighbors[w] = neigh
    elif radius == 0:
        for w in range(total_words):
            neighbors[w] = [w]
    else:
        for w in range(total_words):
            neigh = [w]
            word = int_to_word(w, n, q)
            if radius >= 1:
                for pos in range(n):
                    cur = word[pos]
                    for val in range(q):
                        if val == cur:
                            continue
                        word[pos] = val
                        neigh.append(word_to_int(word, q))
                    word[pos] = cur
            neighbors[w] = neigh
    
    # Convert to numpy arrays
    neighbor_arrays = [np.array(neighbors[w], dtype=np.int32) for w in range(total_words)]
    
    # Reverse neighbors
    rev_neighbors = [[] for _ in range(total_words)]
    for w in range(total_words):
        for u in neighbors[w]:
            rev_neighbors[u].append(w)
    
    def greedy_init():
        """Greedy set cover initialization."""
        coverage_count = np.zeros(total_words, dtype=np.int32)
        codewords = []
        uncovered_count = total_words
        
        while uncovered_count > 0:
            uncovered_mask = coverage_count == 0
            if not uncovered_mask.any():
                break
            
            uncov_indices = np.where(uncovered_mask)[0]
            if len(uncov_indices) > 100:
                sample_idx = np.random.choice(len(uncov_indices), size=100, replace=False)
                sample = uncov_indices[sample_idx]
            else:
                sample = uncov_indices
            
            candidates = set()
            for u in sample:
                candidates.update(rev_neighbors[u])
            
            if not candidates:
                u = uncov_indices[0]
                candidates = set(rev_neighbors[u])
            
            best_word = -1
            best_new_cov = -1
            
            for w in candidates:
                new_cov = np.sum(coverage_count[neighbor_arrays[w]] == 0)
                if new_cov > best_new_cov:
                    best_new_cov = new_cov
                    best_word = w
                    if best_new_cov >= len(neighbor_arrays[w]):
                        break
            
            if best_word == -1:
                best_word = candidates.pop()
                best_new_cov = 1
            
            codewords.append(best_word)
            coverage_count[neighbor_arrays[best_word]] += 1
            uncovered_count -= best_new_cov
        
        return codewords
    
    def compute_coverage(codewords):
        """Compute coverage count array from codeword list."""
        coverage_count = np.zeros(total_words, dtype=np.int32)
        for cw in codewords:
            coverage_count[neighbor_arrays[cw]] += 1
        return coverage_count
    
    def remove_redundant(codewords, coverage_count, cw_set):
        """Remove all redundant codewords."""
        changed = True
        while changed:
            changed = False
            indices = list(range(len(codewords)))
            random.shuffle(indices)
            for i in indices:
                cw = codewords[i]
                if np.any(coverage_count[neighbor_arrays[cw]] == 1):
                    continue
                coverage_count[neighbor_arrays[cw]] -= 1
                codewords.pop(i)
                cw_set.discard(cw)
                changed = True
                break
        return changed
    
    def local_search(codewords, coverage_count, cw_set, time_limit):
        """Local search with swap moves."""
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            if remove_redundant(codewords, coverage_count, cw_set):
                continue
            
            if not codewords:
                break
            
            idx = random.randrange(len(codewords))
            old_cw = codewords[idx]
            
            old_neigh = neighbor_arrays[old_cw]
            private_mask = coverage_count[old_neigh] == 1
            private_words = old_neigh[private_mask]
            
            if len(private_words) == 0:
                coverage_count[old_neigh] -= 1
                codewords.pop(idx)
                cw_set.discard(old_cw)
                continue
            
            if len(private_words) > 3:
                continue
            
            potential = set(rev_neighbors[private_words[0]])
            for pw in private_words[1:]:
                potential.intersection_update(rev_neighbors[pw])
                if not potential:
                    break
            
            if not potential:
                continue
            
            valid_candidates = [c for c in potential if c not in cw_set and c != old_cw]
            
            if not valid_candidates:
                continue
            
            random.shuffle(valid_candidates)
            valid_candidates = valid_candidates[:50]
            
            best_new_cw = -1
            best_score = -1
            
            for new_cw in valid_candidates:
                new_neigh = neighbor_arrays[new_cw]
                score = np.sum(np.maximum(coverage_count[new_neigh] - 1, 0))
                if score > best_score:
                    best_score = score
                    best_new_cw = new_cw
            
            if best_new_cw != -1:
                coverage_count[old_neigh] -= 1
                coverage_count[neighbor_arrays[best_new_cw]] += 1
                codewords[idx] = best_new_cw
                cw_set.discard(old_cw)
                cw_set.add(best_new_cw)
        
        remove_redundant(codewords, coverage_count, cw_set)
        return codewords
    
    def simulated_annealing(codewords, coverage_count, cw_set, time_limit):
        """Simulated annealing on the codeword set."""
        start_time = time.time()
        current_energy = float(len(codewords))
        best_codewords = codewords[:]
        best_energy = current_energy
        temperature = 1.0
        cooling = 0.995
        
        while time.time() - start_time < time_limit and temperature > 0.01:
            if not codewords:
                break
            
            # Random move: add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.3 and len(codewords) < total_words:
                # Add a random word
                new_cw = random.randrange(total_words)
                if new_cw in cw_set:
                    continue
                old_energy = current_energy
                coverage_count[neighbor_arrays[new_cw]] += 1
                codewords.append(new_cw)
                cw_set.add(new_cw)
                new_energy = float(len(codewords))
                delta = new_energy - old_energy
                if delta < 0 or random.random() < np.exp(-delta / temperature):
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_codewords = codewords[:]
                else:
                    # Reject
                    codewords.pop()
                    cw_set.discard(new_cw)
                    coverage_count[neighbor_arrays[new_cw]] -= 1
                    
            elif move_type < 0.5 and len(codewords) > 0:
                # Remove a random word
                idx = random.randrange(len(codewords))
                old_cw = codewords[idx]
                old_energy = current_energy
                coverage_count[neighbor_arrays[old_cw]] -= 1
                codewords.pop(idx)
                cw_set.discard(old_cw)
                new_energy = float(len(codewords))
                # Check if still valid
                if np.any(coverage_count == 0):
                    # Invalid, reject
                    codewords.insert(idx, old_cw)
                    cw_set.add(old_cw)
                    coverage_count[neighbor_arrays[old_cw]] += 1
                else:
                    delta = new_energy - old_energy
                    if delta < 0 or random.random() < np.exp(-delta / temperature):
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_codewords = codewords[:]
                    else:
                        codewords.insert(idx, old_cw)
                        cw_set.add(old_cw)
                        coverage_count[neighbor_arrays[old_cw]] += 1
                        
            else:
                # Swap
                if len(codewords) == 0:
                    continue
                idx = random.randrange(len(codewords))
                old_cw = codewords[idx]
                new_cw = random.randrange(total_words)
                if new_cw == old_cw or new_cw in cw_set:
                    continue
                
                old_energy = current_energy
                coverage_count[neighbor_arrays[old_cw]] -= 1
                coverage_count[neighbor_arrays[new_cw]] += 1
                codewords[idx] = new_cw
                cw_set.discard(old_cw)
                cw_set.add(new_cw)
                new_energy = float(len(codewords))
                
                # Check validity
                if np.any(coverage_count == 0):
                    # Invalid, reject
                    codewords[idx] = old_cw
                    cw_set.discard(new_cw)
                    cw_set.add(old_cw)
                    coverage_count[neighbor_arrays[new_cw]] -= 1
                    coverage_count[neighbor_arrays[old_cw]] += 1
                else:
                    delta = new_energy - old_energy
                    if delta < 0 or random.random() < np.exp(-delta / temperature):
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_codewords = codewords[:]
                    else:
                        codewords[idx] = old_cw
                        cw_set.discard(new_cw)
                        cw_set.add(old_cw)
                        coverage_count[neighbor_arrays[new_cw]] -= 1
                        coverage_count[neighbor_arrays[old_cw]] += 1
            
            temperature *= cooling
        
        # Restore best
        codewords[:] = best_codewords
        cw_set = set(codewords)
        coverage_count = compute_coverage(codewords)
        remove_redundant(codewords, coverage_count, cw_set)
        return codewords
    
    # Main search loop
    start_time = time.time()
    best_codewords = []
    best_len = float('inf')
    
    # First run: greedy + local search + SA
    cw = greedy_init()
    if time.time() < deadline - 10:
        coverage_count = compute_coverage(cw)
        cw_set = set(cw)
        ls_time = min(20, deadline - time.time() - 30)
        if ls_time > 0:
            cw = local_search(cw, coverage_count, cw_set, ls_time)
        if time.time() < deadline - 10:
            coverage_count = compute_coverage(cw)
            cw_set = set(cw)
            sa_time = min(30, deadline - time.time() - 5)
            if sa_time > 0:
                cw = simulated_annealing(cw, coverage_count, cw_set, sa_time)
    
    if len(cw) < best_len:
        best_len = len(cw)
        best_codewords = cw[:]
    
    # Iterated local search with restarts
    while time.time() < deadline - 15:
        remaining = deadline - time.time() - 5
        if remaining < 10:
            break
        
        # Perturb best solution
        cw = best_codewords[:]
        cw_set = set(cw)
        coverage_count = compute_coverage(cw)
        
        # Random perturbation: remove some codewords and re-add greedily
        num_remove = max(1, len(cw) // 10)
        indices = random.sample(range(len(cw)), min(num_remove, len(cw)))
        for i in sorted(indices, reverse=True):
            cw.pop(i)
        cw_set = set(cw)
        coverage_count = compute_coverage(cw)
        
        # Re-greedy to fix coverage
        uncovered_mask = coverage_count == 0
        if uncovered_mask.any():
            uncov_indices = np.where(uncovered_mask)[0]
            if len(uncov_indices) > 50:
                sample_idx = np.random.choice(len(uncov_indices), size=50, replace=False)
                sample = uncov_indices[sample_idx]
            else:
                sample = uncov_indices
            
            candidates = set()
            for u in sample:
                candidates.update(rev_neighbors[u])
            
            for u in uncov_indices:
                for c in rev_neighbors[u]:
                    if c not in cw_set:
                        candidates.add(c)
            
            for w in list(candidates):
                if coverage_count[neighbor_arrays[w]].any() == 0:
                    continue
                new_cov = np.sum(coverage_count[neighbor_arrays[w]] == 0)
                if new_cov > 0:
                    cw.append(w)
                    cw_set.add(w)
                    coverage_count[neighbor_arrays[w]] += 1
        
        # Local search
        ls_time = min(8, remaining * 0.3)
        if ls_time > 0 and time.time() < deadline - 5:
            coverage_count = compute_coverage(cw)
            cw_set = set(cw)
            cw = local_search(cw, coverage_count, cw_set, ls_time)
        
        # SA
        if time.time() < deadline - 10:
            coverage_count = compute_coverage(cw)
            cw_set = set(cw)
            sa_time = min(10, remaining * 0.2)
            if sa_time > 0:
                cw = simulated_annealing(cw, coverage_count, cw_set, sa_time)
        
        if len(cw) < best_len:
            best_len = len(cw)
            best_codewords = cw[:]
    
    # Final polish
    if time.time() < deadline - 3 and best_codewords:
        coverage_count = compute_coverage(best_codewords)
        cw_set = set(best_codewords)
        ls_time = min(5, deadline - time.time() - 2)
        if ls_time > 0:
            polished = local_search(best_codewords[:], coverage_count, cw_set, ls_time)
            if len(polished) < best_len:
                best_len = len(polished)
                best_codewords = polished[:]
    
    result = [int_to_word(cw, n, q) for cw in best_codewords]
    return result
# EVOLVE-BLOCK-END
