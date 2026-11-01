import itertools
import time
import numpy as np
from scipy import sparse
from scipy.optimize import LinearConstraint, milp, Bounds
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

def _generate_words(q, n):
    return [list(word) for word in itertools.product(range(q), repeat=n)]

def _hamming_distance(w1, w2):
    return sum(1 for a, b in zip(w1, w2) if a != b)

def _seed_solution(q, n, radius):
    # Direct sum of ternary tetracode
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other cases: simple greedy
        words = _generate_words(q, n)
        covered = [False] * len(words)
        code = []
        # Map word to index
        word_to_idx = {tuple(w): i for i, w in enumerate(words)}
        
        # Precompute coverage
        cover_sets = []
        for c in words:
            cov = []
            for u in words:
                if _hamming_distance(c, u) <= radius:
                    cov.append(word_to_idx[tuple(u)])
            cover_sets.append(cov)
            
        # Greedy
        while any(not c for c in covered):
            best_idx = -1
            best_gain = -1
            for i, c in enumerate(words):
                gain = sum(1 for j in cover_sets[i] if not covered[j])
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_gain == 0: break # Should not happen if radius >= 0
            code.append(words[best_idx])
            for j in cover_sets[best_idx]:
                covered[j] = True
        return code

    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

def _build_conflict_graph(q, n):
    """Build adjacency bitsets for conflict graph (distance <= 2)."""
    words = _generate_words(q, n)
    N = len(words)
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    # Precompute neighbors at distance 0, 1, 2
    adj = [0] * N  # Python int as bitset
    
    for i, w in enumerate(words):
        # Distance 0: self
        adj[i] |= (1 << i)
        
        # Distance 1
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                w2 = w.copy()
                w2[pos] = new_val
                j = word_to_idx[tuple(w2)]
                adj[i] |= (1 << j)
        
        # Distance 2
        for pos1 in range(n):
            for pos2 in range(pos1 + 1, n):
                orig1, orig2 = w[pos1], w[pos2]
                for v1 in range(q):
                    for v2 in range(q):
                        if v1 == orig1 and v2 == orig2:
                            continue
                        if v1 == orig1 and v2 == orig2:
                            continue
                        # Skip if both same as original
                        if v1 == orig1 and v2 == orig2:
                            continue
                        # Count changes
                        changes = 0
                        if v1 != orig1:
                            changes += 1
                        if v2 != orig2:
                            changes += 1
                        if changes != 2:
                            continue
                        w2 = w.copy()
                        w2[pos1] = v1
                        w2[pos2] = v2
                        j = word_to_idx[tuple(w2)]
                        adj[i] |= (1 << j)
    
    return words, adj, word_to_idx

def _greedy_color_bound(adj, candidates_bitset, N):
    """Upper bound on independent set size using greedy coloring."""
    if candidates_bitset == 0:
        return 0
    
    # Get candidate indices
    candidates = []
    b = candidates_bitset
    while b:
        lsb = b & (-b)
        idx = lsb.bit_length() - 1
        candidates.append(idx)
        b ^= lsb
    
    if not candidates:
        return 0
    
    # Greedy coloring: assign colors to get upper bound
    color_class = [0] * N
    num_colors = 0
    uncolored = set(candidates)
    
    while uncolored:
        num_colors += 1
        # Pick a vertex with max degree in remaining subgraph
        best_v = -1
        best_deg = -1
        for v in uncolored:
            deg = bin(adj[v] & candidates_bitset).count('1') - 1  # exclude self
            if deg > best_deg:
                best_deg = deg
                best_v = v
        
        if best_v == -1:
            break
        
        # Assign this color to best_v and all its non-neighbors
        color_class[best_v] = num_colors
        uncolored.discard(best_v)
        
        # Find vertices that can share this color (non-adjacent to best_v)
        non_neighbors = ~adj[best_v] & candidates_bitset
        # Remove best_v from non_neighbors consideration
        non_neighbors &= ~(1 << best_v)
        
        # Add all non-adjacent uncolored vertices to this color class
        to_add = []
        for v in list(uncolored):
            if (non_neighbors >> v) & 1:
                to_add.append(v)
        
        for v in to_add:
            color_class[v] = num_colors
            uncolored.discard(v)
    
    return num_colors

def _max_independent_set_bnb(adj, N, initial_code, deadline, best_size_ref):
    """Branch and bound for maximum independent set."""
    words = _generate_words(3, 6)
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    # initial_code is a set of indices that form a valid code
    initial_code_set = set(initial_code)
    
    # The independent set we're looking for is the complement of the code
    # in the conflict graph. Wait, let me reconsider.
    # 
    # Actually, let's think differently. We want a minimum dominating set.
    # Let's use a different approach: local search on the code itself.
    
    # For now, let's do a simple local search to improve the seed solution.
    best_code = list(initial_code)
    best_size = len(best_code)
    
    # Convert to set for fast lookup
    best_set = set(best_code)
    
    # Precompute which words are covered by each codeword
    def get_covered(codeword_idx):
        """Return set of word indices covered by codeword at codeword_idx."""
        w = words[codeword_idx]
        covered = set()
        covered.add(codeword_idx)
        for pos in range(6):
            orig = w[pos]
            for new_val in range(3):
                if new_val == orig:
                    continue
                w2 = w.copy()
                w2[pos] = new_val
                j = word_to_idx[tuple(w2)]
                covered.add(j)
        return covered
    
    # Compute coverage for current code
    def compute_coverage(code_set):
        """Return set of all covered word indices."""
        all_covered = set()
        for c in code_set:
            all_covered |= get_covered(c)
        return all_covered
    
    # Initial coverage
    current_covered = compute_coverage(best_set)
    
    # Local search: try to remove codewords and add better ones
    improvements = 0
    while time.time() < deadline - 5:
        improved = False
        
        # Try removing each codeword
        for c in list(best_set):
            if time.time() > deadline - 5:
                break
            
            new_set = best_set - {c}
            new_covered = compute_coverage(new_set)
            
            # Check if still valid
            if len(new_covered) == 729:
                best_set = new_set
                best_code = list(best_set)
                current_covered = new_covered
                improved = True
                improvements += 1
                break
        
        if improved:
            continue
        
        # Try adding one and removing another
        if time.time() > deadline - 10:
            break
        
        # Find uncovered or poorly covered words
        # Try all possible additions
        for add_idx in range(729):
            if time.time() > deadline - 5:
                break
            
            if add_idx in best_set:
                continue
            
            new_set = best_set | {add_idx}
            new_covered = compute_coverage(new_set)
            
            # Try to remove one codeword from new_set
            for rem_idx in list(new_set):
                if rem_idx == add_idx:
                    continue
                
                trial_set = new_set - {rem_idx}
                if len(trial_set) >= len(best_set):
                    continue
                
                trial_covered = compute_coverage(trial_set)
                if len(trial_covered) == 729:
                    best_set = trial_set
                    best_code = list(best_set)
                    current_covered = trial_covered
                    improved = True
                    improvements += 1
                    break
            
            if improved:
                break
        
        if not improved:
            # Try random restarts
            if time.time() > deadline - 5:
                break
            
            # Perturb: remove 2-3 random codewords and re-optimize
            if len(best_set) > 5:
                import random
                to_remove = random.sample(list(best_set), min(3, len(best_set) - 5))
                perturbed = best_set - set(to_remove)
                
                # Greedy fill
                perturbed_covered = compute_coverage(perturbed)
                while len(perturbed_covered) < 729:
                    best_gain = -1
                    best_add = -1
                    for idx in range(729):
                        if idx in perturbed:
                            continue
                        gain = len(get_covered(idx) - perturbed_covered)
                        if gain > best_gain:
                            best_gain = gain
                            best_add = idx
                    
                    if best_add == -1 or best_gain == 0:
                        break
                    
                    perturbed.add(best_add)
                    perturbed_covered = compute_coverage(perturbed)
                
                if len(perturbed) < len(best_set):
                    best_set = perturbed
                    best_code = list(best_set)
                    current_covered = compute_coverage(best_set)
                    improved = True
        
        if not improved and time.time() > deadline - 20:
            break
    
    return best_code, best_size

def _solve_q3_n6_r1(deadline):
    """Solve the specific case using local search."""
    # Start with seed solution
    seed = _seed_solution(3, 6, 1)
    seed_set = set()
    words = _generate_words(3, 6)
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    for w in seed:
        seed_set.add(word_to_idx[tuple(w)])
    
    # Local search
    best_code, best_size = _max_independent_set_bnb(None, 729, list(seed_set), deadline, [len(seed_set)])
    
    # Verify
    def is_valid(code):
        covered = set()
        for c in code:
            w = words[c]
            covered.add(c)
            for pos in range(6):
                orig = w[pos]
                for new_val in range(3):
                    if new_val == orig:
                        continue
                    w2 = w.copy()
                    w2[pos] = new_val
                    covered.add(word_to_idx[tuple(w2)])
        return len(covered) == 729
    
    if not is_valid(best_code):
        return seed
    
    return [words[i] for i in best_code]

def _generic_bitset_construct(q, n, radius, deadline):
    """Generic construction using bitsets."""
    words = _generate_words(q, n)
    N = len(words)
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    # Precompute coverage sets
    cover_sets = []
    for c in words:
        cov = set()
        cov.add(word_to_idx[tuple(c)])
        # Distance 1
        for pos in range(n):
            orig = c[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                c2 = c.copy()
                c2[pos] = new_val
                cov.add(word_to_idx[tuple(c2)])
        cover_sets.append(cov)
    
    # Greedy
    covered = set()
    code = []
    
    while len(covered) < N:
        best_idx = -1
        best_gain = -1
        for i in range(N):
            gain = len(cover_sets[i] - covered)
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_idx == -1 or best_gain == 0:
            break
        
        code.append(words[best_idx])
        covered |= cover_sets[best_idx]
    
    return code

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    if q == 3 and n == 6 and radius == 1:
        try:
            result = _solve_q3_n6_r1(deadline)
            return result
        except Exception:
            return _seed_solution(q, n, radius)
    else:
        try:
            result = _generic_bitset_construct(q, n, radius, deadline)
            return result
        except Exception:
            return _seed_solution(q, n, radius)
