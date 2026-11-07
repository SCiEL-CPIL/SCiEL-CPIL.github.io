import numpy as np

def acquisition(mu, sigma, f_best):
            N = len(mu)
            # Use MC-KG
            S = 500 # Number of MC samples
            
            # Generate samples
            # mu, sigma shape (N,)
            # Z shape (S, N)
            Z = np.random.randn(S, N)
            F = mu[None, :] + sigma[None, :] * Z
            
            # Find min and 2nd min in each row
            # argsort is O(N log N). For 2000, it's fast.
            # Or use partition.
            # np.argpartition(F, 1, axis=1) gives indices of 2 smallest.
            # But we need the values.
            
            # Efficient way to get min and 2nd min:
            # Sort is easier to write correctly.
            # 500 * 2000 log 2000 ~ 500 * 2000 * 11 = 11M ops. Fine.
            
            sorted_F = np.sort(F, axis=1)
            min1 = sorted_F[:, 0]
            min2 = sorted_F[:, 1]
            
            # Indices of min1
            # We need to know which index was min1 to replace it with min2 in the "excluded" matrix
            # argsort gives the indices
            idx_sorted = np.argsort(F, axis=1)
            idx_min1 = idx_sorted[:, 0]
            
            # Initialize MinExcl with min1
            MinExcl = np.tile(min1[:, None], (1, N))
            
            # For each row s, at index idx_min1[s], the min excluding that index is min2[s]
            # Vectorized assignment:
            # rows = np.arange(S)
            # MinExcl[rows, idx_min1] = min2
            
            rows = np.arange(S)
            MinExcl[rows, idx_min1] = min2
            
            # Now MinExcl[s, i] is min_{j != i} F[s, j]
            
            # Val_i[s] = min(F[s, i], MinExcl[s, i])
            Val = np.minimum(F, MinExcl)
            
            # KG_i = mean(Val[:, i]) - mean(min1)
            ExpectedNewMin = np.mean(Val, axis=0)
            ExpectedOldMin = np.mean(min1)
            
            kg_scores = ExpectedNewMin - ExpectedOldMin
            
            # kg_scores will be negative or zero (since new min <= old min usually? No, E[min] is not monotonic like that per sample, but on average, adding a point can only decrease or keep the min. So E[min_new] <= E[min_old]. So KG <= 0.)
            # We want "higher meaning evaluate next".
            # So we should return -kg_scores? Or just the magnitude of improvement?
            # Improvement = OldMin - NewMin.
            # Score = -kg_scores.
            
            scores = -kg_scores
            
            # Ensure finite
            scores = np.where(np.isfinite(scores), scores, 0.0)
            
            return scores
