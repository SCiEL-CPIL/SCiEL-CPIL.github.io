import numpy as np

def heuristics(prizes, distance_matrix, max_len):
        prizes = np.asarray(prizes, dtype=np.float64)
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        
        # Add small epsilon to avoid division by zero
        eps = 1e-9
        
        # Heuristic: Prize / sqrt(Distance)
        # This reduces the myopia of the 1/Distance heuristic,
        # allowing ants to consider slightly further nodes with higher prizes.
        eta = (prizes[None, :] + eps) / (distance_matrix + eps)**0.5
        
        # Ensure no negative or zero values if possible, though prizes >= 0
        # The problem says non-positive become 1e-9.
        # Our formula produces positive values for positive prizes.
        # For prize 0 (node 0), eta is 0.
        # The ACO loop handles node 0 specially (start).
        # But just in case, replace 0 with 1e-9?
        # The prompt says: "Heuristic entries that are not positive become 1e-9".
        # So we should do that.
        
        eta[eta <= 0] = 1e-9
        
        return eta
