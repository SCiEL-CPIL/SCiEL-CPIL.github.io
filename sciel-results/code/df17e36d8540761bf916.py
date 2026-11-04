"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def _word_to_int(word, q):
    """Convert a word (list of ints 0..q-1) to an integer index."""
    idx = 0
    for v in word:
        idx = idx * q + v
    return idx


def _int_to_word(idx, q, n):
    """Convert an integer index to a word (list of ints 0..q-1)."""
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = idx % q
        idx //= q
    return word


def _generate_ball_indices(word, q, n, radius):
    """Generate all word indices within Hamming distance `radius` from `word`."""
    ball = {tuple(word)}
    for r in range(1, radius + 1):
        for positions in itertools.combinations(range(n), r):
            for values in itertools.product(range(1, q), repeat=r):
                new_word = list(word)
                for pos, val in zip(positions, values):
                    new_word[pos] = (word[pos] + val) % q
                ball.add(tuple(new_word))
    return ball


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Leave safety margin before 150s kill
    
    total_words = q ** n
    
    # Precompute all words as tuples
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    
    # Precompute ball index sets
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    ball_sets = []
    for w in all_words:
        ball = _generate_ball_indices(w, q, n, radius)
        ball_sets.append({word_to_idx[b] for b in ball})
    
    # Precompute M: M[i][j] = 1 if j is in ball_i
    M = np.zeros((total_words, total_words), dtype=np.int8)
    ball_size = np.zeros(total_words, dtype=np.int32)
    for i in range(total_words):
        for j in ball_sets[i]:
            M[i][j] = 1
        ball_size[i] = len(ball_sets[i])
    
    # Precompute ball member lists for fast access
    ball_members = [np.where(M[i] == 1)[0] for i in range(total_words)]
    
    best_solution = None
    best_size = float('inf')
    
    def greedy_set_cover(random_seed=0):
        """Greedy set cover using precomputed matrices with random tie-breaking."""
        rng = random.Random(random_seed)
        covered_vec = np.zeros(total_words, dtype=np.int32)
        solution = []
        
        while True:
            if time.time() > deadline - 30:
                break
            
            # gain[i] = ball_size[i] - M[i] @ covered_vec
            gains = ball_size - M @ covered_vec
            
            # Find all indices with max gain
            max_gain = np.max(gains)
            if max_gain <= 0:
                break
            
            # Random tie-breaking among max-gain candidates
            candidates = np.where(gains == max_gain)[0]
            best_idx = int(rng.choice(candidates))
            
            solution.append(best_idx)
            # Update covered_vec
            covered_vec[ball_members[best_idx]] += 1
        
        # Convert to boolean coverage
        covered = covered_vec > 0
        return solution, covered
    
    def systematic_deletion_fast(solution):
        """Fast systematic deletion using coverage count vector."""
        if not solution:
            return solution, np.zeros(total_words, dtype=bool)
        
        # Build coverage count vector
        cov_count = np.zeros(total_words, dtype=np.int32)
        for w_idx in solution:
            cov_count[ball_members[w_idx]] += 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(solution) - 1, -1, -1):
                if time.time() > deadline - 30:
                    break
                w_idx = solution[i]
                # Check if any word would become uncovered
                ball = ball_members[w_idx]
                if np.any(cov_count[ball] == 1):
                    continue
                # Safe to remove
                cov_count[ball] -= 1
                solution.pop(i)
                changed = True
        
        covered = cov_count > 0
        return solution, covered
    
    def local_search_fast(solution, rng, max_iters=2000):
        """Fast local search using coverage count vector."""
        if not solution:
            return solution, np.zeros(total_words, dtype=bool)
        
        cov_count = np.zeros(total_words, dtype=np.int32)
        for w_idx in solution:
            cov_count[ball_members[w_idx]] += 1
        
        temp = 8.0
        cooling = 0.995
        min_temp = 0.5
        
        for sa_iter in range(max_iters):
            if time.time() > deadline - 30:
                break
            if len(solution) <= 1:
                break
            
            idx_to_remove = rng.randint(0, len(solution) - 1)
            w_rm = solution[idx_to_remove]
            ball_rm = ball_members[w_rm]
            
            # Words uniquely covered by w_rm
            unique_mask = cov_count[ball_rm] == 1
            n_unique = int(np.sum(unique_mask))
            
            if n_unique == 0:
                # Safe to remove
                cov_count[ball_rm] -= 1
                solution.pop(idx_to_remove)
                temp = max(temp * cooling, min_temp)
            else:
                # Try to swap: find word that covers most unique words
                unique_indices = ball_rm[unique_mask]
                if len(unique_indices) > 0:
                    # For each candidate, count how many unique words it covers
                    cov_counts = np.array([np.sum(cov_count[unique_indices] == 0) if False else 0])  # placeholder
                    # Actually: for candidate c, it covers unique word j if M[c][j]==1
                    # And j is currently only covered by w_rm, so after removing w_rm, j is uncovered
                    # Candidate c helps if M[c][j]==1 for some j in unique_indices
                    # Also need: after adding c and removing w_rm, no new uncovered words
                    # Simplified: just count overlap with unique set
                    candidate_scores = np.zeros(total_words, dtype=np.int32)
                    for j in unique_indices:
                        candidate_scores[:, j] += 1  # This is wrong dimension
                    # Better approach:
                    # For each candidate c, score = |ball_c ∩ unique_indices|
                    # = sum over j in unique_indices of M[c][j]
                    # = M[:, unique_indices].sum(axis=1)
                    scores = M[:, unique_indices].sum(axis=1)
                    # Exclude current solution members (except w_rm which we're removing)
                    for w_idx in solution:
                        if w_idx != w_rm:
                            scores[w_idx] = 0
                    
                    best_add = int(np.argmax(scores))
                    best_score = scores[best_add]
                    
                    if best_score > 0:
                        # Check if swap is valid: after removing w_rm and adding best_add
                        # No word should become uncovered
                        # Words that lose coverage: those in ball_rm but not in ball_best_add
                        # and currently only covered by w_rm
                        # Words that gain coverage: those in ball_best_add
                        # Net: words in ball_rm \ ball_best_add that were only covered by w_rm become uncovered
                        # These are exactly: unique_indices that are NOT in ball_best_add
                        lost = unique_indices[~(M[best_add, unique_indices] == 1)]
                        if len(lost) == 0:
                            # Valid swap
                            cov_count[ball_rm] -= 1
                            cov_count[ball_members[best_add]] += 1
                            solution[idx_to_remove] = best_add
                            temp = max(temp * cooling, min_temp)
                            continue
                        
                        # SA: accept with probability based on penalty
                        penalty = len(lost)
                        if rng.random() < np.exp(-penalty / max(temp, 0.1)):
                            # Do the swap anyway (will be fixed later)
                            cov_count[ball_rm] -= 1
                            cov_count[ball_members[best_add]] += 1
                            solution[idx_to_remove] = best_add
                            temp = max(temp * cooling, min_temp)
                            continue
                
                temp = max(temp * cooling, min_temp)
        
        covered = cov_count > 0
        return solution, covered
    
    # Try multiple random restarts
    n_restarts = 50
    for attempt in range(n_restarts):
        if time.time() > deadline - 35:
            break
        
        # Phase 1: Greedy set cover with different random seeds
        solution, covered = greedy_set_cover(random_seed=42 + attempt)
        
        if not np.all(covered):
            continue
        
        # Phase 2: Systematic deletion
        solution, covered = systematic_deletion_fast(solution)
        
        # Phase 3: Local search
        rng = random.Random(1000 + attempt)
        solution, covered = local_search_fast(solution, rng, max_iters=1500)
        
        # Phase 4: Another systematic deletion pass
        solution, covered = systematic_deletion_fast(solution)
        
        # Validate and update best
        if np.all(covered):
            sol_set = set(solution)
            if len(sol_set) == len(solution):
                if len(solution) < best_size:
                    best_size = len(solution)
                    best_solution = [list(_int_to_word(w, q, n)) for w in solution]
                    # Early termination if we hit known optimum
                    if best_size <= 73:
                        break
    
    if best_solution is None:
        # Fallback: use direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return best_solution
# EVOLVE-BLOCK-END
