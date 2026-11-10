import numpy as np
import time
import random
import itertools

N = 10
X_GRID, Y_GRID = np.meshgrid(np.arange(N), np.arange(N), indexing='xy')
X_FLAT = X_GRID.flatten()
Y_FLAT = Y_GRID.flatten()

# Pre-compute projection indices
# U: x - y
# V: x
# W: x + y
# Z: x + 2y
U_VALS = X_FLAT - Y_FLAT
V_VALS = X_FLAT
W_VALS = X_FLAT + Y_FLAT
Z_VALS = X_FLAT + 2 * Y_FLAT

U_UNIQ, U_INV = np.unique(U_VALS, return_inverse=True)
V_UNIQ, V_INV = np.unique(V_VALS, return_inverse=True)
W_UNIQ, W_INV = np.unique(W_VALS, return_inverse=True)
Z_UNIQ, Z_INV = np.unique(Z_VALS, return_inverse=True)

def _entropy_from_counts(counts):
    """
    Calculate Shannon entropy in bits from integer counts.
    Assumes total sum is fixed (100).
    Ignores zero counts.
    """
    total = 100
    if total == 0:
        return 0.0
    # Filter out zeros
    non_zero = counts[counts > 0]
    if len(non_zero) == 0:
        return 0.0
    probs = non_zero / total
    return -np.sum(probs * np.log2(probs))

def _compute_ratio_from_counts(counts):
    """
    Compute D / M from integer count vector of length 100 summing to 100.
    """
    # Project counts
    c_u = np.bincount(U_INV, weights=counts, minlength=len(U_UNIQ))
    c_v = np.bincount(V_INV, weights=counts, minlength=len(V_UNIQ))
    c_w = np.bincount(W_INV, weights=counts, minlength=len(W_UNIQ))
    c_z = np.bincount(Z_INV, weights=counts, minlength=len(Z_UNIQ))
    
    D = _entropy_from_counts(c_u)
    M = max(_entropy_from_counts(c_v), _entropy_from_counts(c_w), _entropy_from_counts(c_z))
    
    if M <= 1e-9:
        return 0.0
    return D / M

def _get_support(counts):
    """Return list of indices where count > 0."""
    return [i for i in range(100) if counts[i] > 0]

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    rng = random.Random(42)
    
    best_counts = None
    best_ratio = -np.inf
    
    # Initial diverse starting points
    # 1. Uniform
    # 2. Diagonal (x=y) -> high x-y=0 concentration? No, x-y=0 is one bin.
    #    We want HIGH D (x-y spread) and LOW M (x, x+y, x+2y concentrated).
    #    To maximize x-y spread, we need x and y to be uncorrelated or negatively correlated?
    #    x-y range is -9 to 9.
    #    To minimize x entropy, concentrate x.
    #    To minimize x+y entropy, concentrate x+y.
    #    To minimize x+2y entropy, concentrate x+2y.
    
    # Let's try to concentrate on a specific structure.
    # If we pick points where x is constant, H_x = 0.
    # If x is constant, x-y = c-y. As y varies 0..9, x-y varies c..c-9. 10 distinct values.
    # H_{x-y} = log2(10) ~ 3.32.
    # H_{x+y} = log2(10) ~ 3.32.
    # H_{x+2y} = log2(10) ~ 3.32.
    # Ratio = 1.0. This is the baseline.
    
    # Can we do better?
    # We need D > M.
    # D is entropy of x-y. M is max(H_x, H_{x+y}, H_{x+2y}).
    
    # Strategy: Spread x-y as much as possible while keeping x, x+y, x+2y concentrated.
    
    # Start with a few random sparse distributions
    start_candidates = []
    
    # Candidate 1: Uniform
    start_candidates.append(np.ones(100, dtype=int))
    
    # Candidate 2: Diagonal x=y
    c = np.zeros(100, dtype=int)
    for i in range(10):
        c[i*10+i] = 10
    start_candidates.append(c)
    
    # Candidate 3: Anti-diagonal x+y=9
    c = np.zeros(100, dtype=int)
    for i in range(10):
        c[(9-i)*10+i] = 10
    start_candidates.append(c)
    
    # Candidate 4: Vertical line x=0
    c = np.zeros(100, dtype=int)
    for i in range(10):
        c[i*10+0] = 10
    start_candidates.append(c)
    
    # Candidate 5: Horizontal line y=0
    c = np.zeros(100, dtype=int)
    for i in range(10):
        c[0*10+i] = 10
    start_candidates.append(c)
        
    # Candidate 6: Random sparse
    for _ in range(5):
        c = np.zeros(100, dtype=int)
        indices = rng.sample(range(100), 10)
        for i in indices:
            c[i] = 10
        start_candidates.append(c)

    # Iterated Local Search
    for start_c in start_candidates:
        if time.time() > deadline - 5:
            break
            
        current_counts = start_c.copy()
        current_ratio = _compute_ratio_from_counts(current_counts)
        
        if current_ratio > best_ratio:
            best_ratio = current_ratio
            best_counts = current_counts.copy()
            
        # Local Search Phase
        # We perform 1-swap moves: move 1 unit from i to j
        # Or more efficiently, move k units.
        # Given 100 cells, 100*100 = 10000 moves. 
        # Evaluating each move requires recomputing entropies.
        # We can optimize by only updating the bins affected by i and j.
        
        # Pre-compute which bins are affected by each cell index
        # cell i affects U_INV[i], V_INV[i], W_INV[i], Z_INV[i]
        
        # For efficiency in local search, we'll just do a full recompute for simplicity 
        # first, and optimize if too slow. 10000 evals * ~10us/eval = 0.1s per iteration.
        # 100 iterations = 10s. This is acceptable.
        
        max_iters = 200
        no_improve_count = 0
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            improved = False
            support = _get_support(current_counts)
            if len(support) == 0:
                break
                
            # Randomly pick a source cell with count > 0
            # And a target cell (can be same, but that's no-op, so pick different)
            # Actually, we can pick any source with count>0 and any target.
            
            # To speed up, we can sample a subset of moves or use a smarter heuristic.
            # Let's try all moves from a random source to all targets.
            # Or random source, random target.
            
            # Heuristic: Try moving mass from cells that contribute to low entropy bins 
            # to cells that spread high entropy bins?
            # For now, random search over moves.
            
            moves_checked = 0
            # Check a random subset of moves to save time if needed, 
            # but 10000 is small enough to check all from a single source.
            
            # Pick a random source with mass
            src = rng.choice(support)
            
            # Try moving 1 unit to all other cells
            # Also try moving more if count is high? 
            # Let's stick to moving 1 unit at a time for fine-grained control, 
            # or moving all mass from src to dst?
            # Moving 1 unit allows exploring the boundary.
            
            best_move_ratio = current_ratio
            best_dst = -1
            best_amount = 0
            
            # Try moving 1 unit to all j != src
            # Also try moving all mass from src to j?
            # Let's try moving 1 unit to all j.
            
            for dst in range(100):
                if dst == src:
                    continue
                
                # Move 1 unit from src to dst
                # Only valid if current_counts[src] >= 1
                if current_counts[src] < 1:
                    break # Should not happen if src in support
                
                # Quick check: if moving 1 unit doesn't change the bin counts 
                # significantly, skip? No, entropy is non-linear.
                
                # Perform move
                current_counts[src] -= 1
                current_counts[dst] += 1
                
                ratio = _compute_ratio_from_counts(current_counts)
                
                # Undo move
                current_counts[src] += 1
                current_counts[dst] -= 1
                
                if ratio > best_move_ratio + 1e-9:
                    best_move_ratio = ratio
                    best_dst = dst
                    best_amount = 1
                    improved = True
            
            # Also try moving more mass if src has a lot
            if current_counts[src] > 1:
                # Try moving all mass from src to a random few dsts
                # This helps escape local minima where 1-step moves are flat
                for _ in range(10):
                    dst = rng.randint(0, 99)
                    if dst == src:
                        continue
                    
                    amount = current_counts[src]
                    if amount < 1:
                        continue
                    
                    current_counts[src] -= amount
                    current_counts[dst] += amount
                    
                    ratio = _compute_ratio_from_counts(current_counts)
                    
                    current_counts[src] += amount
                    current_counts[dst] -= amount
                    
                    if ratio > best_move_ratio + 1e-9:
                        best_move_ratio = ratio
                        best_dst = dst
                        best_amount = amount
                        improved = True
            
            if improved:
                # Apply the best move found
                current_counts[src] -= best_amount
                current_counts[dst] += best_amount
                current_ratio = best_move_ratio
                no_improve_count = 0
                
                if current_ratio > best_ratio:
                    best_ratio = current_ratio
                    best_counts = current_counts.copy()
            else:
                no_improve_count += 1
                # If stuck, perturb
                if no_improve_count > 5:
                    # Random perturbation: move mass from a random cell to another
                    src_p = rng.choice(support)
                    dst_p = rng.randint(0, 99)
                    if dst_p != src_p and current_counts[src_p] > 0:
                        current_counts[src_p] -= 1
                        current_counts[dst_p] += 1
                        current_ratio = _compute_ratio_from_counts(current_counts)
                    no_improve_count = 0
                    # If perturbation makes it worse, it's okay, we explore.
                    # But if it makes it invalid (negative count), we handled that.
                    
        # End Local Search Phase

    # Convert best_counts to P
    if best_counts is None:
        P_final = np.ones((10, 10)) / 100.0
    else:
        P_final = best_counts.reshape((10, 10)).astype(float) / 100.0
        
    # Validation
    P_final = np.clip(P_final, 0, None)
    total = np.sum(P_final)
    if total <= 1e-9:
        P_final = np.ones((10, 10)) / 100.0
    else:
        P_final = P_final / total
        
    return P_final

# EVOLVE-BLOCK-END
