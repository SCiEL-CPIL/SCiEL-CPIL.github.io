import numpy as np

def heuristics(demands, capacity):
       demands = np.asarray(demands, dtype=float)
       n = len(demands)
       
       # Calculate pair sums
       sums = demands[:, None] + demands[None, :]
       
       # Identify valid pairs (sum <= capacity)
       valid = sums <= capacity
       
       # Calculate gap for valid pairs
       gaps = capacity - sums
       
       # Initialize heuristic with a very small value
       eta = np.full((n, n), 1e-9, dtype=float)
       
       # Strategy: Reward pairs that create "dead gaps" (residual space too small for any remaining item)
       # This encourages tight packing by making it "expensive" (low attractiveness) to leave large voids.
       
       # For each valid pair (i, j), calculate the residual space:
       # residual = capacity - demands[i] - demands[j]
       # Count how many items k (k != i, k != j) fit into this residual space.
       # If count is 0, it's a "dead gap" -> high attractiveness.
       # If count is high, it's a "live gap" -> low attractiveness.
       
       # We can vectorize this somewhat, but n=120 is small enough that a loop or simple broadcasting might work.
       # Let's try to be efficient.
       
       # For each pair (i, j), we want count of k such that demands[k] <= gaps[i, j] and k != i and k != j.
       # Note: The task says "count of items in the demand set that fit into the residual space".
       # It doesn't explicitly exclude i and j from the count, but logically, i and j are already placed.
       # However, since demands are positive and gaps[i,j] < demands[i] (usually, unless demands[i] is very small and j is very small? No, sum <= cap, so gap = cap - sum. If sum is small, gap is large. If sum is close to cap, gap is small.
       # Actually, if gap is large, it might fit i or j again? But i and j are already in the bin.
       # The heuristic is for placing j AFTER i. So i is already there. j is being considered.
       # The "remaining items" are the ones not yet placed.
       # But the heuristic is computed ONCE before the colony runs. It doesn't know the current state of the bin (which items are already in it besides i).
       # The heuristic only depends on the pair (i, j).
       # So "count of items in the demand set that fit into the residual space" likely means:
       # How many items k in the original set of demands fit into (capacity - d_i - d_j)?
       # Should we exclude i and j?
       # If we include them, it's just counting how many items from the whole pool fit.
       # If we exclude them, it's counting how many *other* items fit.
       # Given the context of "dead gaps", we want to know if there's *any* other item that could have been placed there.
       # So excluding i and j makes more sense.
       
       # Let's compute this count for all valid pairs.
       # n=120.
       # For each i, j:
       #   residual = capacity - demands[i] - demands[j]
       #   count = sum(1 for k in range(n) if k != i and k != j and demands[k] <= residual)
       
       # This is O(n^3) if done naively with a loop over k. 120^3 = 1.7M. That's fine in Python if vectorized or even in a tight loop?
       # Let's try to vectorize the count calculation.
       
       # demands is (n,).
       # residuals is (n, n).
       # We want for each (i, j), the number of k such that demands[k] <= residuals[i, j] AND k != i AND k != j.
       
       # Let's create a matrix `fits` where fits[i, j, k] is True if demands[k] <= residuals[i, j].
       # But that's n^3 memory. 120^3 = 1.7M booleans. That's ~1.7MB. Acceptable.
       # However, we can do it more cleverly.
       
       # For a fixed residual value R, the number of items fitting is the count of demands <= R.
       # Let's sort demands.
       # sorted_demands = np.sort(demands)
       # For each residual R, count = np.searchsorted(sorted_demands, R, side='right')
       # But we need to exclude i and j.
       # If demands[i] <= R, we subtract 1.
       # If demands[j] <= R, we subtract 1.
       
       # So:
       # 1. Compute residuals matrix (n, n).
       # 2. For each residual, find total count of items in full set that fit.
       # 3. Subtract 1 if demands[i] fits.
       # 4. Subtract 1 if demands[j] fits.
       
       # Step 1:
       residuals = capacity - sums
       
       # Step 2:
       sorted_demands = np.sort(demands)
       # We need to map each residual to the count.
       # residuals is (n, n).
       # We can use searchsorted on the flattened residuals?
       # Or just loop over rows? n=120 is small.
       
       # Let's use a loop over i to be safe and clear, or vectorize with searchsorted on 2D array?
       # np.searchsorted works on 1D.
       # We can flatten residuals, find counts, then reshape.
       
       flat_residuals = residuals.flatten()
       # Count of items <= R
       total_counts = np.searchsorted(sorted_demands, flat_residuals, side='right')
       total_counts = total_counts.reshape(n, n)
       
       # Step 3 & 4:
       # For pair (i, j), residual = residuals[i, j].
       # Does item i fit? demands[i] <= residuals[i, j]
       # Does item j fit? demands[j] <= residuals[i, j]
       
       i_fits = demands[:, None] <= residuals  # shape (n, n)
       j_fits = demands[None, :] <= residuals  # shape (n, n)
       
       # Subtract counts for i and j if they fit in the residual
       dead_gap_counts = total_counts.astype(float)
       dead_gap_counts -= i_fits.astype(float)
       dead_gap_counts -= j_fits.astype(float)
       
       # Ensure non-negative
       dead_gap_counts = np.maximum(dead_gap_counts, 0)
       
       # Sharper nonlinear decay: strongly reward dead gaps (count=0),
       # penalize live gaps more aggressively.
       # Use exp(-alpha * count) so that count=0 -> 1.0, count=1 -> exp(-alpha), etc.
       # alpha controls how sharply we prefer dead gaps.
       alpha = 2.0
       scores = np.exp(-alpha * dead_gap_counts)
       
       # Apply only to valid pairs
       eta[valid] = scores[valid]
       
       # For invalid pairs (sum > capacity), keep the floor value 1e-9
       # Diagonal
       np.fill_diagonal(eta, 1e-9)
       
       # Ensure all values are positive and finite
       eta[~np.isfinite(eta)] = 1e-9
       eta[eta <= 0] = 1e-9
       
       return eta
       
       # Diagonal
       np.fill_diagonal(eta, 1e-9)
       
       return eta
