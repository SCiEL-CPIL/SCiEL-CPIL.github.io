import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        return solve_3_6_1()
    # Fallback: return all words (valid but large)
    return [list(w) for w in itertools.product(range(q), repeat=n)]

def solve_3_6_1():
    q = 3
    n = 6
    N = q ** n  # 729
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    
    # Precompute spheres: for each center, list of indices within Hamming distance 1
    # Sphere size = 1 + n*(q-1) = 1 + 6*2 = 13
    sphere_indices = np.zeros((N, 13), dtype=np.int32)
    for i in range(N):
        w = all_words[i]
        sphere_indices[i, 0] = i
        idx = 1
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = w.copy()
                new_w[pos] = val
                # Convert to base-3 index
                idx_val = 0
                for j in range(n):
                    idx_val = idx_val * 3 + new_w[j]
                sphere_indices[i, idx] = idx_val
                idx += 1
    
    # Coverage matrix: cov[c, p] = 1 if center c covers point p
    # Shape: (N, N) = (729, 729)
    cov = np.zeros((N, N), dtype=np.int8)
    for c in range(N):
        for p in sphere_indices[c]:
            cov[c, p] = 1
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s limit
    
    best_size = N
    best_codewords = None
    rng = np.random.RandomState(42)
    
    # Precompute for each point, which centers cover it (inverse coverage)
    # point_coverers[p] = list of centers that cover point p
    point_coverers = []
    for p in range(N):
        centers = [c for c in range(N) if cov[c, p]]
        point_coverers.append(centers)
    
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Greedy set cover initialization
        is_covered = np.zeros(N, dtype=np.int8)
        uncovered_count = N
        selected = []
        
        while uncovered_count > 0:
            gain = cov @ (1 - is_covered)
            max_gain = np.max(gain)
            candidates = np.where(gain == max_gain)[0]
            
            if len(candidates) == 0 or max_gain == 0:
                break
            
            c = rng.choice(candidates)
            selected.append(c)
            
            newly_covered = cov[c] & (1 - is_covered)
            new_count = np.sum(newly_covered)
            is_covered |= cov[c]
            uncovered_count -= new_count
        
        selected = list(set(selected))
        
        # Simulated annealing with 1-1 swaps and removals
        current_size = len(selected)
        selected_set = set(selected)
        # Track coverage counts: how many selected centers cover each point
        cover_count = np.zeros(N, dtype=np.int32)
        for c in selected:
            cover_count += cov[c]
        
        current_uncovered = np.sum(cover_count == 0)
        
        # SA parameters
        T = 1.0
        T_min = 0.01
        cooling = 0.995
        max_sa_steps = 5000
        
        sa_step = 0
        while sa_step < max_sa_steps and T > T_min:
            if time.time() > deadline:
                break
            
            # Choose move type: 70% swap, 30% removal attempt
            move_type = rng.random()
            
            if move_type < 0.7 and len(selected) > 1:
                # 1-1 swap: remove one, add one
                idx_out = rng.randint(0, len(selected) - 1)
                c_out = selected[idx_out]
                
                # Choose c_in: prefer centers that cover currently uncovered points
                # or centers with high coverage of low-coverage points
                uncovered_mask = (cover_count == 0).astype(np.int8)
                low_cover_mask = (cover_count == 1).astype(np.int8)
                
                # Score each candidate center
                # Gain from covering uncovered points, penalty for redundancy
                gains = cov @ uncovered_mask  # covers uncovered
                redundancy = cov @ low_cover_mask  # adds to already-covered-by-1 points
                
                # Weight: prefer covering uncovered, avoid adding redundancy
                scores = gains * 10 - redundancy * 2
                # Exclude current selected centers
                for c in selected:
                    scores[c] = -np.inf
                
                # Also exclude c_out (we're removing it)
                # Pick from positive score candidates if any, else random
                valid_mask = scores > -np.inf
                if np.any(valid_mask):
                    # Weighted random selection
                    valid_scores = np.where(valid_mask, scores, 0)
                    # Shift to positive for softmax-like selection
                    valid_scores = valid_scores - np.min(valid_scores[valid_mask])
                    valid_scores = np.exp(valid_scores)
                    valid_scores[~valid_mask] = 0
                    total = np.sum(valid_scores)
                    if total > 0:
                        probs = valid_scores / total
                        c_in = rng.choice(N, p=probs)
                    else:
                        c_in = rng.choice(N)
                else:
                    c_in = rng.choice(N)
                
                if c_in == c_out or c_in in selected_set:
                    continue
                
                # Evaluate swap
                new_cover_count = cover_count - cov[c_out] + cov[c_in]
                new_uncovered = np.sum(new_cover_count == 0)
                
                delta = new_uncovered - current_uncovered
                # Size stays same, so only uncovered matters
                if delta <= 0 or rng.random() < np.exp(-delta / T):
                    # Accept
                    selected[idx_out] = c_in
                    selected_set.remove(c_out)
                    selected_set.add(c_in)
                    cover_count = new_cover_count
                    current_uncovered = new_uncovered
                    
                    if current_uncovered == 0 and current_size < best_size:
                        best_size = current_size
                        best_codewords = [all_words[c].tolist() for c in selected]
            else:
                # Removal: try removing a center that doesn't uncover any points
                # Find centers where removing them doesn't create uncovered points
                removable = []
                for i, c in enumerate(selected):
                    # Check if all points covered by c are also covered by others
                    c_covered = cov[c]
                    without_c = cover_count - cov[c]
                    # Points that would become uncovered
                    would_uncover = np.sum((without_c == 0) & (c_covered == 1))
                    if would_uncover == 0:
                        removable.append(i)
                
                if removable:
                    idx_out = rng.choice(removable)
                    c_out = selected[idx_out]
                    
                    # Accept removal
                    selected.pop(idx_out)
                    selected_set.remove(c_out)
                    cover_count -= cov[c_out]
                    current_size -= 1
                    current_uncovered = np.sum(cover_count == 0)
                    
                    if current_uncovered == 0 and current_size < best_size:
                        best_size = current_size
                        best_codewords = [all_words[c].tolist() for c in selected]
            
            sa_step += 1
            T *= cooling
        
        # After SA, try simple removal pass
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for i in range(len(selected)):
                c = selected[i]
                without_c = cover_count - cov[c]
                would_uncover = np.sum((without_c == 0) & (cov[c] == 1))
                if would_uncover == 0:
                    selected.pop(i)
                    selected_set.remove(c)
                    cover_count -= cov[c]
                    current_size -= 1
                    improved = True
                    break
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_codewords = [all_words[c].tolist() for c in selected]
    
    # Fallback if no improvement found
    if best_codewords is None:
        # Use a known construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        best_codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
    
    return best_codewords

def is_valid_covering(codeword_indices, cov, N):
    """Check if the given codeword indices cover all N points."""
    if len(codeword_indices) == 0:
        return False
    covered = np.zeros(N, dtype=np.int8)
    for c in codeword_indices:
        covered |= cov[c]
    return np.all(covered)
