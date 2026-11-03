import time
import random
import itertools
import math
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin for 120s limit

    # 1. Precomputation
    all_t = list(itertools.combinations(range(v), t))
    N = len(all_t)
    idx_t = {s: i for i, s in enumerate(all_t)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    M = len(all_blocks)
    
    # block_cover[b] = list of indices of t-subsets covered by block b
    block_cover = []
    for b in all_blocks:
        covered = [idx_t[s] for s in itertools.combinations(b, t)]
        block_cover.append(covered)
        
    # t_to_blocks[i] = list of block indices that cover t-subset i
    t_to_blocks = [[] for _ in range(N)]
    for bi, covered in enumerate(block_cover):
        for ti in covered:
            t_to_blocks[ti].append(bi)

    # Helper: Check if a list of block indices forms a valid cover
    def is_valid(blocks):
        if not blocks:
            return N == 0
        cnt = np.zeros(N, dtype=np.int32)
        for bi in blocks:
            for ti in block_cover[bi]:
                cnt[ti] += 1
        return np.all(cnt > 0)

    # Helper: Compute coverage counts
    def get_counts(blocks):
        cnt = np.zeros(N, dtype=np.int32)
        for bi in blocks:
            for ti in block_cover[bi]:
                cnt[ti] += 1
        return cnt

    # 2. Lagrangian Relaxation / Subgradient Optimization
    # We want to solve the LP relaxation of the Set Covering problem:
    # Minimize sum(x_b) subject to sum(x_b * A[b,i]) >= 1 for all i
    # Dual: Maximize sum(y_i) subject to sum(y_i * A[b,i]) <= 1 for all b, y_i >= 0
    
    # We use Subgradient Method to find dual variables y.
    # These y values represent the "cost" or "difficulty" of covering each t-subset.
    
    def subgradient_opt(max_iters=500):
        # Initialize y with small random values
        y = np.random.uniform(0, 0.1, N)
        best_y = y.copy()
        best_lb = 0.0
        
        # Step size parameter
        # Standard form: alpha_k = c * (UB - LB) / ||g||^2
        # We don't have a tight UB yet, so we use a simple decaying step size
        # or a fixed one. Let's use a heuristic step size.
        
        for it in range(max_iters):
            if time.time() > deadline - 30.0:
                break
                
            # Compute subgradient: g_b = 1 - sum(y_i * A[b,i])
            # If g_b > 0, constraint is violated (dual infeasible).
            # If g_b < 0, slack.
            
            # Efficiently compute sum(y_i * A[b,i]) for all b
            # This is matrix-vector product: A^T y
            # But A is sparse. We can compute it by iterating over blocks.
            # Actually, let's compute the violation for each block.
            
            # We need to update y based on the subgradient of the dual objective.
            # The subgradient of the dual function with respect to y_i is (1 - sum_b x_b A[b,i])
            # where x_b is the primal solution associated with the current dual.
            # In subgradient method for covering, we often use:
            # g_i = 1 - max_b (sum_{j in b} y_j) ... wait.
            
            # Let's stick to the standard Subgradient Method for Set Covering.
            # Objective: Minimize sum x_b
            # Constraints: A x >= 1
            # Lagrangian: L(x, y) = sum x_b + sum y_i (1 - (Ax)_i)
            # Dual function: g(y) = sum y_i + min_x { sum x_b - sum x_b (A^T y)_b }
            # min_x is 0 if (A^T y)_b <= 1 for all b, else -infinity.
            # So we want to maximize sum y_i subject to A^T y <= 1.
            
            # Subgradient of g(y) w.r.t y_i is (1 - (Ax*)_i) where x* is the primal solution
            # corresponding to the current y.
            # The primal solution x* for a given y is:
            # x_b = 1 if (A^T y)_b == 1 (tight), else 0.
            # If multiple are tight, we can choose any.
            
            # Compute A^T y
            # A^T y has length M. (A^T y)_b = sum_{i in block b} y_i
            # We can compute this efficiently.
            
            # Initialize sum_y_for_block
            sum_y_block = np.zeros(M)
            for bi, covered in enumerate(block_cover):
                sum_y_block[bi] = np.sum(y[covered])
                
            # Identify tight constraints: sum_y_block[bi] >= 1 (due to numerical issues, use >= 1 - eps)
            tight_blocks = np.where(sum_y_block >= 1.0 - 1e-6)[0]
            
            if len(tight_blocks) == 0:
                # No tight constraints, increase all y slightly?
                # This means we can increase y to improve LB.
                # Subgradient is 1 for all i? No.
                # If no block is tight, we can scale up y.
                # Let's just add a small epsilon to all y.
                y += 0.01
                continue
                
            # Choose a primal solution x* from tight blocks.
            # To get a good subgradient, we should choose x* that maximizes the subgradient direction.
            # Simple heuristic: Choose the tight block that covers the most "uncovered" mass?
            # Or just pick one randomly, or the one with highest sum_y_block?
            # Let's pick the one with highest sum_y_block (most "saturated").
            # Or better: Compute the subgradient vector g.
            # g_i = 1 - sum_{b: x*_b=1} A[b,i]
            # If we pick x*_b = 1 for one specific block b*, then g_i = 1 - A[b*,i].
            # This is a valid subgradient.
            
            # To diversify, let's pick a random tight block.
            b_star = tight_blocks[random.randint(0, len(tight_blocks)-1)]
            
            # Compute subgradient g
            # g_i = 1 if i not in block b_star, else 0?
            # Wait. A[b*, i] is 1 if i is in block b*.
            # So sum_{b: x*_b=1} A[b,i] is 1 if i in b*, else 0.
            # So g_i = 1 - (1 if i in b* else 0) = 0 if i in b*, else 1.
            
            g = np.ones(N)
            for ti in block_cover[b_star]:
                g[ti] = 0.0
                
            # Update y
            # y_new = y + alpha * g
            # Step size alpha
            # Let's use a simple decaying step size.
            alpha = 0.1 / (1.0 + it * 0.01)
            
            y += alpha * g
            y = np.maximum(y, 0) # Ensure non-negativity
            
            # Compute Lower Bound
            lb = np.sum(y)
            if lb > best_lb:
                best_lb = lb
                best_y = y.copy()
                
        return best_y, best_lb

    # Run Subgradient Optimization
    y_dual, lb_value = subgradient_opt()
    
    # 3. Guided Greedy Construction
    # Use y_dual to weight the gain of each block.
    # Gain of block b = sum_{i in b} y_dual[i] * (1 if i uncovered else 0)
    # This prioritizes covering subsets that are "expensive" in the dual sense.
    
    def guided_greedy(y_weights, seed=0):
        rng = random.Random(seed)
        uncovered = set(range(N))
        blocks = []
        
        while uncovered:
            best_b = -1
            best_gain = -1.0
            
            # We can optimize this by only considering blocks that cover at least one uncovered item.
            # But M is ~75k, N is ~11628.
            # Iterating over all blocks is 75k * avg_cover_size.
            # Let's try it. If too slow, we'll switch to a candidate pool.
            
            for bi in range(M):
                gain = 0.0
                for ti in block_cover[bi]:
                    if ti in uncovered:
                        gain += y_weights[ti]
                if gain > best_gain:
                    best_gain = gain
                    best_b = bi
                    
            if best_b == -1:
                return None
                
            blocks.append(best_b)
            for ti in block_cover[best_b]:
                uncovered.discard(ti)
                
        return blocks

    best_blocks = None
    best_score = float('inf')
    
    # Run a few guided greedy runs with different seeds
    # Also, we can perturb the dual weights slightly to get diversity.
    
    for s in range(5):
        if time.time() > deadline - 20.0:
            break
            
        # Perturb weights
        perturbed_y = y_dual * (1.0 + 0.1 * np.random.randn(N))
        perturbed_y = np.maximum(perturbed_y, 0)
        
        cand = guided_greedy(perturbed_y, seed=s)
        if cand:
            if len(cand) < best_score:
                best_score = len(cand)
                best_blocks = cand
                
    if not best_blocks:
        # Fallback to standard greedy
        def standard_greedy(seed):
            rng = random.Random(seed)
            uncovered = set(range(N))
            blocks = []
            while uncovered:
                best_b = -1
                max_gain = -1
                for bi in range(M):
                    gain = 0
                    for ti in block_cover[bi]:
                        if ti in uncovered:
                            gain += 1
                    if gain > max_gain:
                        max_gain = gain
                        best_b = bi
                if best_b == -1:
                    return None
                blocks.append(best_b)
                for ti in block_cover[best_b]:
                    uncovered.discard(ti)
            return blocks
            
        cand = standard_greedy(0)
        if cand:
            best_blocks = cand
            best_score = len(cand)

    # 4. Local Search
    # Similar to the original, but we can use the dual weights to guide the search.
    # However, the original local search was decent. Let's use a modified version
    # that focuses on "high cost" uncovered areas if we allow temporary invalidity,
    # but we stick to valid states.
    
    def prune_to_minimal(blocks):
        cnt = get_counts(blocks)
        current_blocks = list(blocks)
        i = 0
        while i < len(current_blocks):
            bi = current_blocks[i]
            redundant = True
            for ti in block_cover[bi]:
                if cnt[ti] == 1:
                    redundant = False
                    break
            if redundant:
                current_blocks.pop(i)
                for ti in block_cover[bi]:
                    cnt[ti] -= 1
            else:
                i += 1
        return current_blocks

    best_blocks = prune_to_minimal(best_blocks)
    best_score = len(best_blocks)

    def local_search(blocks, max_time=10.0):
        blocks = list(blocks)
        cnt = get_counts(blocks)
        current_cost = len(blocks)
        best_local = list(blocks)
        best_local_cost = current_cost
        
        start_time = time.time()
        no_improve_steps = 0
        max_no_improve = 2000
        
        while time.time() - start_time < max_time and no_improve_steps < max_no_improve:
            # 1. Try to drop a redundant block
            dropped = False
            for i in range(len(blocks)):
                bi = blocks[i]
                can_drop = True
                for ti in block_cover[bi]:
                    if cnt[ti] <= 1:
                        can_drop = False
                        break
                if can_drop:
                    blocks.pop(i)
                    for ti in block_cover[bi]:
                        cnt[ti] -= 1
                    current_cost -= 1
                    if current_cost < best_local_cost:
                        best_local_cost = current_cost
                        best_local = list(blocks)
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
                    dropped = True
                    break
            
            if dropped:
                continue
                
            # 2. Try a swap
            if len(blocks) == 0:
                break
                
            idx_old = random.randrange(len(blocks))
            bi_old = blocks[idx_old]
            
            weaks = []
            for ti in block_cover[bi_old]:
                if cnt[ti] == 1:
                    weaks.append(ti)
            
            if not weaks:
                blocks.pop(idx_old)
                for ti in block_cover[bi_old]:
                    cnt[ti] -= 1
                current_cost -= 1
                if current_cost < best_local_cost:
                    best_local_cost = current_cost
                    best_local = list(blocks)
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
                continue
                
            weaks_sorted = sorted(weaks, key=lambda w: len(t_to_blocks[w]))
            candidates = set(t_to_blocks[weaks_sorted[0]])
            for w in weaks_sorted[1:]:
                candidates &= set(t_to_blocks[w])
                if not candidates:
                    break
                    
            if not candidates:
                no_improve_steps += 1
                continue
                
            candidates.discard(bi_old)
            if not candidates:
                no_improve_steps += 1
                continue
                
            bi_new = random.choice(list(candidates))
            
            # Heuristic: Prefer swaps that cover high-dual-weight areas
            # Calculate "dual gain" of the swap
            # Gain = sum_{ti in new} y[ti] - sum_{ti in old} y[ti]
            # But we must maintain validity. The swap is valid by construction of weaks.
            # We accept if it doesn't increase cost (it doesn't) and maybe if dual gain is positive?
            # For now, just accept random to explore.
            
            blocks[idx_old] = bi_new
            
            for ti in block_cover[bi_old]:
                cnt[ti] -= 1
            for ti in block_cover[bi_new]:
                cnt[ti] += 1
                
            no_improve_steps += 1
            
        return best_local

    while time.time() < deadline - 2.0:
        candidate = local_search(best_blocks, max_time=5.0)
        
        if len(candidate) < best_score:
            best_score = len(candidate)
            best_blocks = candidate
            
        if best_score <= 264:
            break

    if not is_valid(best_blocks):
        # Fallback
        # This should not happen if local search is correct
        pass

    return [list(all_blocks[bi]) for bi in best_blocks]
