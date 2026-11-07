"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(demands, capacity):
    start_time = time.time()
    DEADLINE = 320.0  # Safety margin before 330s kill

    n = len(demands)
    pair_sum = demands[:, None] + demands[None, :]
    gap = capacity - pair_sum

    # Initialize with very small value for invalid pairs
    H = np.full((n, n), 1e-9)

    # Valid pairs are those where both items fit together
    valid = gap >= 0

    # Build a cumulative count of items by demand size
    demand_ints = np.round(demands).astype(int)
    max_demand = int(np.max(demand_ints))

    # Create a frequency array
    freq = np.zeros(max_demand + 1, dtype=np.int64)
    for d in demand_ints:
        freq[d] += 1

    # Cumulative sum: cumsum[k] = number of items with demand <= k
    cumsum = np.cumsum(freq)

    # Get indices of valid pairs
    valid_indices = np.where(valid)
    i_idx = valid_indices[0]
    j_idx = valid_indices[1]

    gap_valid = gap[valid]

    # Clip gap to valid range for cumsum lookup
    gap_clipped = np.clip(gap_valid, 0, max_demand).astype(int)

    # Count of items with demand <= gap
    fit_counts = cumsum[gap_clipped]

    # Check if item i fits in gap[i,j]: 2*d_i + d_j <= capacity
    i_fits = (2.0 * demands[i_idx] + demands[j_idx]) <= capacity
    # Check if item j fits in gap[i,j]: d_i + 2*d_j <= capacity
    j_fits = (demands[i_idx] + 2.0 * demands[j_idx]) <= capacity

    # Adjust fit_counts: subtract 1 for each item that fits and is i or j
    adjusted_counts = fit_counts - i_fits.astype(np.int64) - j_fits.astype(np.int64)

    # Ensure counts are non-negative
    adjusted_counts = np.maximum(adjusted_counts, 0)

    # The heuristic should reward pairs with FEWER fitting items in the residual
    # Use a smooth penalty: pairs with 0 fitting items get highest score
    H_valid = 1.0 / (adjusted_counts + 1.0)

    # Add a bonus for smaller gaps (tighter packing)
    gap_normalized = gap_valid / capacity
    H_valid = H_valid * (1.0 - 0.5 * gap_normalized)

    H[valid] = H_valid

    # Boost exact fits even more
    exact_fit = valid & (gap == 0)
    H[exact_fit] = 15.0

    # Check time budget
    if time.time() - start_time > DEADLINE:
        return H

    # SPECTRAL EMBEDDING APPROACH:
    # Build a compatibility graph where edge weights represent how well
    # items i and j complement each other (negative residual space).
    # Use spectral embedding to capture global structural affinities.

    # Build the adjacency/compatibility matrix
    # Weight = how tightly items fit together (smaller gap = better)
    # Only consider valid pairs
    W = np.zeros((n, n), dtype=np.float64)
    W[valid] = gap_valid

    # Invert: we want higher weight for tighter fits
    # Use a smooth transformation: w = 1 / (1 + gap/capacity)
    # This gives w close to 1 for exact fits, close to 0.5 for half-capacity gaps
    W_inv = np.zeros((n, n), dtype=np.float64)
    W_inv[valid] = 1.0 / (1.0 + gap_valid / capacity)

    # Normalize by degree for spectral embedding
    degree = W_inv.sum(axis=1)
    degree[degree == 0] = 1.0  # Avoid division by zero
    D_inv_sqrt = 1.0 / np.sqrt(degree)

    # Symmetrize and normalize
    W_sym = 0.5 * (W_inv + W_inv.T)
    W_norm = W_sym * D_inv_sqrt[:, None] * D_inv_sqrt[None, :]

    # Compute spectral embedding using power iteration for top few eigenvectors
    # This captures community structure in the compatibility graph
    try:
        # Use scipy for eigendecomposition if available, otherwise fallback
        try:
            from scipy.sparse.linalg import eigsh
            from scipy.sparse import csc_matrix

            W_sparse = csc_matrix(W_norm)
            # Get top 5 eigenvectors (largest magnitude)
            k = min(5, n - 1)
            if k > 0:
                eigenvalues, eigenvectors = eigsh(W_sparse, k=k, which='LM', return_eigenvectors=True)
            else:
                eigenvectors = np.zeros((n, 0))
        except ImportError:
            # Fallback to numpy eigendecomposition for smaller matrices
            eigenvalues, eigenvectors = np.linalg.eigh(W_norm)
            # Take top 5
            k = min(5, n)
            idx = np.argsort(eigenvalues)[::-1][:k]
            eigenvectors = eigenvectors[:, idx]

        # Compute pairwise distances in spectral embedding space
        # This captures global structural affinities
        U = eigenvectors  # Shape: (n, k)

        # Squared Euclidean distances in embedding space
        U_norm_sq = np.sum(U ** 2, axis=1)
        dist_sq = U_norm_sq[:, None] + U_norm_sq[None, :] - 2.0 * (U @ U.T)
        dist_sq = np.maximum(dist_sq, 0.0)  # Numerical stability
        dist = np.sqrt(dist_sq)

        # Convert distances to affinities: closer in embedding = more compatible
        # Use a smooth kernel: affinity = exp(-dist^2 / sigma^2)
        # Choose sigma based on median distance
        median_dist = np.median(dist[dist > 0]) if np.any(dist > 0) else 1.0
        sigma = max(median_dist, 0.1)
        affinity = np.exp(-dist_sq / (2.0 * sigma ** 2))

        # Combine with local heuristic: blend spectral and local signals
        # Weight the spectral component to enhance but not override local fit
        alpha = 0.3  # Blend factor

        H_combined = (1.0 - alpha) * H + alpha * affinity * 10.0  # Scale affinity to comparable range

        # Only apply to valid pairs
        H_combined[~valid] = 1e-9

        # Preserve exact fit boost
        H_combined[exact_fit] = 15.0

        # Ensure all values are positive and finite
        H_combined = np.where(np.isfinite(H_combined) & (H_combined > 0), H_combined, 1e-9)

        H = H_combined

    except Exception:
        # If spectral embedding fails, fall back to original heuristic
        pass

    # Final check for finite values
    H = np.where(np.isfinite(H), H, 1e-9)
    H = np.maximum(H, 1e-9)

    return H
# EVOLVE-BLOCK-END
