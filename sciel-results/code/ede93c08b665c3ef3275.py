"""
Strategy: Binary Spin System with Swendsen-Wang Clustering and Combinatorial Refinement.

The problem is to find h in [0,1]^N with sum N/2 minimizing max_k overlap_k.
Insight: The objective is likely minimized by a binary sequence (h_i in {0,1}) 
with exactly N/2 ones. This is because the overlap function is convex in h, 
and the minimum of a convex function over a convex polytope often occurs at 
extreme points (vertices), which for the box constraints and sum constraint 
are binary vectors.

We treat h as a spin configuration s_i in {-1, 1} (mapping 0->-1, 1->1).
The overlap for shift k is related to the correlation of spins.
Specifically, if h_i = (1+s_i)/2, then:
overlap_k = sum_i h_i (1-h_{i-k}) = sum_i h_i - sum_i h_i h_{i-k}
Since sum h_i = N/2 is constant, minimizing max overlap is equivalent to 
maximizing min_k (sum_i h_i h_{i-k}).
In spin terms, h_i h_{i-k} = (1+s_i)/2 * (1+s_{i-k})/2 = (1 + s_i + s_{i-k} + s_i s_{i-k})/4.
The term sum_i s_i s_{i-k} is the spin-spin correlation at lag k.
We want to maximize the minimum of these correlations across all lags.

We use Swendsen-Wang cluster algorithm to flip large domains of identical spins.
This is efficient for binary systems and avoids the "metastability" of local 
search methods like Nelder-Mead or simple coordinate descent.
After finding a good binary configuration, we refine using combinatorial swaps.
"""

import numpy as np
import time
import random

N = 100

# EVOLVE-BLOCK-START

def _max_overlap_binary(s):
    """
    Compute max overlap for a binary spin configuration s (values -1, 1).
    h = (1+s)/2.
    overlap_k = N/2 - sum_i h_i h_{i-k}
    sum_i h_i h_{i-k} = N/4 + (1/4) sum_i s_i + (1/4) sum_i s_{i-k} + (1/4) sum_i s_i s_{i-k}
    Since sum s_i = 0 (exactly N/2 ones and N/2 -1s), this simplifies to:
    sum_i h_i h_{i-k} = N/4 + (1/4) C_k, where C_k = sum_i s_i s_{i-k}.
    So overlap_k = N/2 - (N/4 + C_k/4) = N/4 - C_k/4.
    Max overlap = N/4 - min_k(C_k)/4.
    We want to minimize max overlap, so we want to maximize min_k(C_k).
    """
    # Compute circular autocorrelation of spins
    S = np.fft.fft(s)
    autocorr = np.real(np.fft.ifft(S * np.conj(S)))
    # autocorr[k] = sum_i s[i] * s[(i-k)%N]
    min_corr = np.min(autocorr)
    max_overlap = N / 4.0 - min_corr / 4.0
    return max_overlap

def _project_to_binary(h):
    """
    Convert a real-valued h to a binary sequence with exactly N/2 ones.
    This is a heuristic projection: threshold at 0.5, then adjust count.
    """
    # Threshold
    s = np.where(h >= 0.5, 1.0, -1.0)
    
    # Count ones
    n_ones = np.sum(s == 1.0)
    target_ones = N // 2
    
    if n_ones == target_ones:
        return s
    
    if n_ones > target_ones:
        # Need to flip some 1s to -1s
        # Prioritize flipping 1s that are 'less stable' (lower h value? or random?)
        # For simplicity, pick random indices where s==1
        indices = np.where(s == 1.0)[0]
        random.shuffle(indices)
        to_flip = indices[:int(n_ones - target_ones)]
        s[to_flip] = -1.0
    else:
        # Need to flip some -1s to 1s
        indices = np.where(s == -1.0)[0]
        random.shuffle(indices)
        to_flip = indices[:int(target_ones - n_ones)]
        s[to_flip] = 1.0
        
    return s

def _swendsen_wang_step(s, p=0.5):
    """
    Perform one Swendsen-Wang cluster flip step on binary spin configuration s.
    1. Find clusters of connected same-spin spins (on a ring).
    2. Flip each cluster with probability p.
    """
    n_clusters = 0
    visited = np.zeros(N, dtype=bool)
    
    # Identify clusters
    # A cluster is a maximal contiguous sequence of same spins (circularly).
    # Since it's a ring, we need to be careful.
    
    # Simple approach: iterate and find boundaries.
    # Or use connected components logic.
    
    # For a 1D ring, clusters are just blocks of identical spins.
    # Let's find all blocks.
    
    # Find indices where spin changes
    changes = np.where(s != np.roll(s, -1))[0]
    
    if len(changes) == 0:
        # All spins are the same. This is a valid cluster.
        # Flip with prob p
        if random.random() < p:
            s = -s
        return s, 1
    
    # The clusters are the segments between changes.
    # changes[i] is the index where s[changes[i]] != s[changes[i]-1] (circularly).
    # Actually, let's just iterate.
    
    # Better: use a label array.
    labels = -np.ones(N, dtype=int)
    current_label = 0
    i = 0
    while i < N:
        if labels[i] != -1:
            i += 1
            continue
            
        # Start a new cluster at i
        spin_val = s[i]
        j = i
        while j < N and s[j] == spin_val:
            labels[j] = current_label
            j += 1
        # Handle wrap-around
        if j == N:
            # Check if it connects to start
            if s[0] == spin_val:
                # Merge with cluster at 0
                # Find label of 0
                start_label = labels[0]
                if start_label != current_label:
                    # Relabel current cluster to start_label
                    mask = labels == current_label
                    labels[mask] = start_label
                    current_label -= 1 # Decrement since we merged
            # If it doesn't connect, it's a separate cluster ending at N-1
        i = j
        current_label += 1
    
    # Now flip each unique label with probability p
    unique_labels = np.unique(labels)
    for lab in unique_labels:
        if random.random() < p:
            mask = labels == lab
            s[mask] = -s[mask]
            
    return s, len(unique_labels)

def _construct_binary_initial():
    """Generate a random binary configuration with N/2 ones."""
    s = np.zeros(N)
    # Randomly choose N/2 indices to be 1
    indices = np.random.choice(N, N//2, replace=False)
    s[indices] = 1.0
    s[np.where(s == 0)[0]] = -1.0
    return s

def _refine_binary_swaps(s, max_iters=5000, time_limit=10.0):
    """
    Refine a binary configuration using local swaps (2-opt style).
    Try swapping a 1 and a -1 to see if it improves the worst-case overlap.
    """
    s = s.copy()
    best_score = _max_overlap_binary(s)
    start_time = time.time()
    
    ones = np.where(s == 1.0)[0]
    minuses = np.where(s == -1.0)[0]
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        if len(ones) == 0 or len(minuses) == 0:
            break
            
        i = random.choice(ones)
        j = random.choice(minuses)
        
        # Swap
        s[i], s[j] = s[j], s[i]
        
        new_score = _max_overlap_binary(s)
        
        if new_score < best_score:
            best_score = new_score
            # Update lists
            ones = np.where(s == 1.0)[0]
            minuses = np.where(s == -1.0)[0]
        else:
            # Revert
            s[i], s[j] = s[j], s[i]
            
    return s, best_score

def construct():
    start_time = time.time()
    deadline = 145.0 # 5s safety margin
    
    random.seed(42)
    np.random.seed(42)
    
    best_s = _construct_binary_initial()
    best_score = _max_overlap_binary(best_s)
    
    # Phase 1: Swendsen-Wang Monte Carlo to find a good binary structure
    # We run multiple restarts or a long chain.
    # Given the 150s limit, we can afford a decent amount of SW steps.
    
    # Estimate cost of one SW step: O(N) for cluster finding, O(N) for flip.
    # FFT cost: O(N log N).
    # N=100 is small.
    
    # Let's do a simulated annealing-like approach with SW.
    # Or just run SW at p=0.5 for a while, keeping track of best.
    
    s = _construct_binary_initial()
    current_score = _max_overlap_binary(s)
    
    # Adaptive p? Start with higher p for mixing, lower for refinement?
    # Or just fixed p=0.5 for exploration.
    
    steps_per_second_target = 5000 # Optimistic
    
    while time.time() < deadline - 20: # Reserve 20s for refinement
        # Perform a batch of SW steps
        batch_size = 100
        for _ in range(batch_size):
            if time.time() > deadline - 20:
                break
            s, _ = _swendsen_wang_step(s, p=0.5)
            score = _max_overlap_binary(s)
            if score < best_score:
                best_score = score
                best_s = s.copy()
        
        # Occasionally restart if stuck?
        # For now, just continue.
        
    # Phase 2: Combinatorial Refinement
    # Use the best binary structure found as a starting point for local search.
    # Also try refining a few other good candidates found during SW?
    # For simplicity, refine the global best.
    
    # We can also try refining the initial random one if it was good?
    # No, SW should have beaten it.
    
    remaining_time = deadline - time.time()
    
    # Split time among a few candidates for refinement
    # Candidate 1: best_s
    # Candidate 2: A fresh random binary
    # Candidate 3: Another fresh random binary
    
    candidates = [best_s, _construct_binary_initial(), _construct_binary_initial()]
    
    time_per_candidate = remaining_time / len(candidates)
    
    for idx, s_init in enumerate(candidates):
        if time.time() > deadline - 5:
            break
            
        s_ref, score_ref = _refine_binary_swaps(
            s_init, 
            max_iters=100000, 
            time_limit=time_per_candidate
        )
        
        if score_ref < best_score:
            best_score = score_ref
            best_s = s_ref
    
    # Convert best binary spin config to h
    # h = (1+s)/2
    best_h = (1.0 + best_s) / 2.0
    
    # Ensure exact sum and bounds (should be exact for binary)
    # Sum should be exactly N/2
    # Bounds [0,1] are satisfied
    
    # Final check
    # The evaluator might check sum(h) == N/2 exactly.
    # With binary, sum is integer. N/2 = 50.
    # Sum of h = sum((1+s)/2) = (N + sum(s))/2 = (100 + 0)/2 = 50. Correct.
    
    return best_h

# EVOLVE-BLOCK-END
