"""Improved program: BMM/GF(2) sign search + structured position search + local polish."""
import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin within 90s limit

    def count_nonzero(poly):
        return int(np.count_nonzero(poly))

    def terms_to_dense(terms):
        if not terms:
            return np.array([0], dtype=np.int64)
        max_deg = max(pos for pos, _ in terms)
        dense = np.zeros(max_deg + 1, dtype=np.int64)
        for pos, val in terms:
            dense[pos] = val
        return dense

    def evaluate_terms(terms):
        """Evaluate: return number of nonzero terms in square."""
        if len(terms) != 12:
            return 1000
        positions = [pos for pos, _ in terms]
        if len(set(positions)) != 12:
            return 1000
        for pos, val in terms:
            if val == 0 or abs(val) > 64 or pos > 40:
                return 1000
        dense = terms_to_dense(terms)
        sq = np.convolve(dense, dense, mode='full')
        return count_nonzero(sq)

    def sum_collision_count(positions):
        """Count distinct sum positions p_i + p_j for i <= j."""
        sums = set()
        n = len(positions)
        for i in range(n):
            for j in range(i, n):
                sums.add(positions[i] + positions[j])
        return len(sums)

    def gf2_sign_search(positions, time_limit=1.0):
        """
        Use GF(2) linear algebra to find sign assignments that zero out
        as many cross-term coefficients as possible in the square.
        
        For a coefficient at position s in P^2:
          - If s = 2*p_i for some i (diagonal): coeff = a_i^2 (always nonzero for |a_i|>=1)
          - Otherwise: coeff = 2 * sum of a_i*a_j over pairs (i,j) with p_i+p_j=s, i<j
            This is zero iff sum of a_i*a_j = 0 over those pairs.
        
        With |a_i|=1, a_i*a_j = s_i*s_j where s_i in {+1,-1}.
        So we need: sum of s_i*s_j over pairs in each collision group = 0.
        
        This is a system of equations over the reals, but we can use GF(2)
        as a heuristic: work mod 2 where +1 and -1 both become 1, so
        the constraint becomes: number of pairs in each group must be even
        (necessary but not sufficient). For the actual sign search, we use
        a more direct approach.
        
        Strategy: For each collision group with k pairs, we need the sum
        of k products s_i*s_j to be 0. With k=2: s_i*s_j + s_k*s_l = 0
        means s_i*s_j = -s_k*s_l, i.e., s_i/s_j = -s_k/s_l.
        
        We'll do a randomized search biased by the GF(2) structure.
        """
        start_t = time.time()
        
        # Build sum-to-pairs mapping
        sum_to_pairs = defaultdict(list)
        n = len(positions)
        for i in range(n):
            for j in range(i, n):
                s = positions[i] + positions[j]
                sum_to_pairs[s].append((i, j))
        
        # Identify which sums are "diagonal" (only one pair, i==j)
        # These are always nonzero (a_i^2), so they contribute to the count.
        diag_sums = set()
        cross_groups = {}
        for s, pairs in sum_to_pairs.items():
            if len(pairs) == 1:
                i, j = pairs[0]
                if i == j:
                    diag_sums.add(s)
            else:
                cross_groups[s] = pairs
        
        # Minimum possible nonzero count = len(diag_sums) + number of cross groups we can't zero
        # Each cross group with k pairs: can we make sum of s_i*s_j = 0?
        # For k=2: yes (set s_i*s_j = -s_k*s_l)
        # For k=3: need s_i*s_j + s_k*s_l + s_m*s_n = 0, possible with right signs
        # For k=4: possible
        # Generally: for k>=2, it's often possible with ±1 values.
        
        # Count how many cross groups have even number of pairs (necessary for GF(2) feasibility)
        even_groups = 0
        odd_groups = 0
        for s, pairs in cross_groups.items():
            if len(pairs) % 2 == 0:
                even_groups += 1
            else:
                odd_groups += 1
        
        # Theoretical lower bound: diag_sums + odd_groups (odd groups can't be zeroed with ±1)
        # But actually even groups can also fail. Let's do a smart search.
        
        # Build constraint graph: variables are signs s_0..s_11
        # Each cross group gives a constraint: sum of s_i*s_j over pairs = 0
        
        # Strategy: random sign assignments with local improvement
        best_score = 1000
        best_signs = None
        
        n_trials = 0
        while time.time() - start_t < time_limit and n_trials < 50000:
            n_trials += 1
            
            # Random signs
            signs = [random.choice([1, -1]) for _ in range(12)]
            
            # Quick local improvement: flip one sign at a time
            improved = True
            while improved and time.time() - start_t < time_limit:
                improved = False
                for idx in range(12):
                    signs[idx] = -signs[idx]
                    # Evaluate
                    dense = np.zeros(max(positions) + 1, dtype=np.int64)
                    for p, v in zip(positions, signs):
                        dense[p] = v
                    sq = np.convolve(dense, dense, mode='full')
                    score = count_nonzero(sq)
                    if score < best_score:
                        best_score = score
                        best_signs = signs[:]
                    if score >= best_score and signs[idx] != -signs[idx]:
                        # Didn't improve, revert
                        pass
                    # Check if this flip helped vs previous
                    # Actually let's just track best
                    signs[idx] = -signs[idx]  # revert
                    # Try the flip
                    signs[idx] = -signs[idx]  # this is the flip
                    dense2 = np.zeros(max(positions) + 1, dtype=np.int64)
                    for p, v in zip(positions, signs):
                        dense2[p] = v
                    sq2 = np.convolve(dense2, dense2, mode='full')
                    score2 = count_nonzero(sq2)
                    if score2 < best_score:
                        best_score = score2
                        best_signs = signs[:]
                        improved = True
                    else:
                        signs[idx] = -signs[idx]  # revert
                    if best_score <= len(diag_sums) + odd_groups:
                        break
                if best_score <= len(diag_sums) + odd_groups:
                    break
            
            if best_score <= len(diag_sums) + odd_groups:
                break
        
        return best_score, best_signs

    def gf2_magnitude_search(positions, time_limit=2.0):
        """
        Search over magnitudes and signs to minimize square nonzero count.
        Uses the structure from GF(2) analysis to guide the search.
        """
        start_t = time.time()
        
        sum_to_pairs = defaultdict(list)
        for i in range(12):
            for j in range(i, 12):
                s = positions[i] + positions[j]
                sum_to_pairs[s].append((i, j))
        
        best_score = 1000
        best_vals = None
        
        # First try pure ±1 with local search
        score1, signs = gf2_sign_search(positions, min(1.0, time_limit * 0.4))
        if signs is not None:
            best_score = score1
            best_vals = signs[:]
        
        if best_score <= 13:
            return best_score, best_vals
        
        # Now try with small magnitudes
        n_trials = 0
        while time.time() - start_t < time_limit and n_trials < 30000:
            n_trials += 1
            
            vals = []
            for _ in range(12):
                sign = random.choice([1, -1])
                r = random.random()
                if r < 0.30:
                    mag = 1
                elif r < 0.55:
                    mag = random.randint(1, 2)
                elif r < 0.75:
                    mag = random.randint(1, 4)
                elif r < 0.90:
                    mag = random.randint(1, 8)
                else:
                    mag = random.randint(1, 64)
                vals.append(sign * mag)
            
            dense = np.zeros(max(positions) + 1, dtype=np.int64)
            for p, v in zip(positions, vals):
                dense[p] = v
            sq = np.convolve(dense, dense, mode='full')
            score = count_nonzero(sq)
            
            if score < best_score:
                best_score = score
                best_vals = vals[:]
                if best_score <= 12:
                    break
        
        # Local search on best found
        if best_vals is not None and time.time() - start_t < time_limit * 0.3:
            for _ in range(200):
                if time.time() - start_t > time_limit:
                    break
                idx = random.randint(0, 11)
                old_val = best_vals[idx]
                new_val = random.choice([1, -1]) * random.randint(1, 64)
                best_vals[idx] = new_val
                dense = np.zeros(max(positions) + 1, dtype=np.int64)
                for p, v in zip(positions, best_vals):
                    dense[p] = v
                sq = np.convolve(dense, dense, mode='full')
                score = count_nonzero(sq)
                if score < best_score:
                    best_score = score
                else:
                    best_vals[idx] = old_val
                if best_score <= 12:
                    break
        
        return best_score, best_vals

    def generate_position_candidates():
        """Generate diverse position sets sorted by collision density."""
        candidates = []
        
        # Arithmetic progressions
        for start in range(0, 41):
            for step in range(1, 8):
                pos = [start + i * step for i in range(12)]
                if pos[-1] <= 40:
                    candidates.append(tuple(pos))
        
        # Two interleaved APs of 6
        for s1 in range(0, 25):
            for s2 in range(s1 + 1, 35):
                for step in range(1, 6):
                    pos = []
                    for i in range(6):
                        pos.append(s1 + i * step)
                        pos.append(s2 + i * step)
                    pos.sort()
                    if len(set(pos)) == 12 and pos[-1] <= 40:
                        candidates.append(tuple(pos))
        
        # Three interleaved APs of 4
        for s1 in range(0, 18):
            for s2 in range(s1 + 1, 25):
                for s3 in range(s2 + 1, 32):
                    for step in range(1, 5):
                        pos = []
                        for i in range(4):
                            pos.extend([s1 + i * step, s2 + i * step, s3 + i * step])
                        pos.sort()
                        if len(set(pos)) == 12 and pos[-1] <= 40:
                            candidates.append(tuple(pos))
        
        # Four interleaved APs of 3
        for s1 in range(0, 15):
            for s2 in range(s1 + 1, 20):
                for s3 in range(s2 + 1, 25):
                    for s4 in range(s3 + 1, 30):
                        for step in range(1, 4):
                            pos = []
                            for i in range(3):
                                pos.extend([s1 + i * step, s2 + i * step, s3 + i * step, s4 + i * step])
                            pos.sort()
                            if len(set(pos)) == 12 and pos[-1] <= 40:
                                candidates.append(tuple(pos))
        
        # Structured: pairs with same sum (to create collision groups)
        # e.g., positions {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11} gives many collisions
        for offset in range(0, 30):
            pos = tuple(offset + i for i in range(12))
            if pos[-1] <= 40:
                candidates.append(pos)
        
        # Random sets
        for _ in range(2000):
            pos = tuple(sorted(random.sample(range(41), 12)))
            candidates.append(pos)
        
        # Clustered sets
        for _ in range(1000):
            n_clusters = random.choice([2, 3, 4, 6])
            if 12 % n_clusters != 0:
                continue
            cluster_size = 12 // n_clusters
            bases = sorted(random.sample(range(0, 41 - cluster_size + 1), n_clusters))
            pos = []
            valid = True
            for b in bases:
                offsets = random.sample(range(0, cluster_size + 4), cluster_size)
                pos.extend(b + o for o in offsets)
            pos = sorted(set(pos))
            if len(pos) == 12 and pos[-1] <= 40:
                candidates.append(tuple(pos))
        
        # Deduplicate
        candidates = list(set(candidates))
        
        # Sort by collision count (ascending = more collisions = better)
        candidates.sort(key=lambda p: sum_collision_count(p))
        
        return candidates

    def local_search_terms(terms, time_limit=2.0):
        """Local search on full (position, value) pairs."""
        start_t = time.time()
        current_terms = [(pos, val) for pos, val in terms]
        current_score = evaluate_terms(current_terms)
        
        def neighbor(terms):
            terms = list(terms)
            move_type = random.random()
            if move_type < 0.30:
                # Change position
                idx = random.randint(0, 11)
                new_pos = random.randint(0, 40)
                if new_pos == terms[idx][0]:
                    return terms
                for i, (pos, val) in enumerate(terms):
                    if i != idx and pos == new_pos:
                        return terms
                terms[idx] = (new_pos, terms[idx][1])
            elif move_type < 0.60:
                # Change value
                idx = random.randint(0, 11)
                new_val = random.choice([1, -1]) * random.randint(1, 64)
                terms[idx] = (terms[idx][0], new_val)
            elif move_type < 0.80:
                # Flip sign
                idx = random.randint(0, 11)
                terms[idx] = (terms[idx][0], -terms[idx][1])
            else:
                # Swap positions
                i, j = random.sample(range(12), 2)
                terms[i], terms[j] = (terms[j][0], terms[i][1]), (terms[i][0], terms[j][1])
            return terms
        
        no_improve_count = 0
        while time.time() - start_t < time_limit:
            improved = False
            for _ in range(50):
                if time.time() - start_t > time_limit:
                    break
                new_terms = neighbor(current_terms)
                new_score = evaluate_terms(new_terms)
                if new_score < current_score:
                    current_terms = new_terms
                    current_score = new_score
                    improved = True
                    no_improve_count = 0
                    if current_score <= 12:
                        return current_terms, current_score
            if not improved:
                no_improve_count += 1
                if no_improve_count >= 5:
                    break
        
        return current_terms, current_score

    def main_search():
        best_terms = None
        best_score = 1000
        
        # Phase 1: Structured position search with GF(2)-guided value optimization
        candidates = generate_position_candidates()
        
        # Process top candidates (most collisions first)
        n_processed = 0
        for idx, positions in enumerate(candidates):
            if time.time() > deadline - 20:
                break
            
            # Adaptive time per candidate
            remaining = deadline - time.time() - 20
            n_remaining = max(len(candidates) - idx, 1)
            time_per = max(0.05, min(1.5, remaining / max(n_remaining, 1)))
            
            # Skip if collision count is too high (not enough collision groups)
            coll_count = sum_collision_count(positions)
            if coll_count > 78:  # 78 = 12*11/2, max possible
                continue
            
            score, values = gf2_magnitude_search(list(positions), time_limit=time_per)
            n_processed += 1
            
            if values is not None and score < best_score:
                best_score = score
                best_terms = list(zip(positions, values))
                if best_score <= 12:
                    break
            
            # Process top 50 candidates with more time
            if idx >= 80 and time.time() > deadline - 30:
                break
        
        # Phase 2: Local search refinement
        if best_terms is not None and time.time() < deadline - 5:
            ls_time = min(5.0, max(1.0, deadline - time.time() - 3))
            refined_terms, refined_score = local_search_terms(best_terms, time_limit=ls_time)
            if refined_score < best_score:
                best_score = refined_score
                best_terms = refined_terms
        
        # Phase 3: Random restarts with local search
        restart_count = 0
        while time.time() < deadline - 3 and restart_count < 20:
            restart_count += 1
            # Random initial
            positions = sorted(random.sample(range(41), 12))
            values = [random.choice([1, -1]) * random.randint(1, 64) for _ in range(12)]
            terms = list(zip(positions, values))
            
            # Quick local search
            ls_time = min(1.0, max(0.3, (deadline - time.time() - 3) / max(20 - restart_count, 1)))
            refined_terms, refined_score = local_search_terms(terms, time_limit=ls_time)
            
            if refined_score < best_score:
                best_score = refined_score
                best_terms = refined_terms
                if best_score <= 12:
                    break
        
        # Phase 4: Final polish
        if best_terms is not None and time.time() < deadline - 1:
            polish_time = max(0.5, deadline - time.time() - 1)
            refined_terms, refined_score = local_search_terms(best_terms, time_limit=polish_time)
            if refined_score < best_score:
                best_score = refined_score
                best_terms = refined_terms
        
        return best_terms, best_score

    # Run main search
    best_terms, best_score = main_search()
    
    # Fallback
    if best_terms is None or len(best_terms) != 12:
        positions = list(range(12))
        values = [1] * 12
        best_terms = list(zip(positions, values))
        best_score = evaluate_terms(best_terms)
    
    # Convert to coefficient list
    max_pos = max(pos for pos, _ in best_terms)
    coeff_list = [0] * (max_pos + 1)
    for pos, val in best_terms:
        coeff_list[pos] = int(val)
    
    # Ensure constraints
    if len(coeff_list) > 41:
        coeff_list = coeff_list[:41]
    while len(coeff_list) > 1 and coeff_list[-1] == 0:
        coeff_list.pop()
    if not coeff_list or coeff_list[-1] == 0:
        coeff_list = [1]
    
    return coeff_list
