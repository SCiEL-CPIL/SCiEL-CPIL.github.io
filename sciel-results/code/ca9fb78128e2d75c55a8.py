"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix):
    # Ensure input is a numpy array for consistent operations
    d = np.asarray(distance_matrix)
    
    # Use a sharper heuristic: 1/d^2
    # This makes short edges significantly more attractive than long ones,
    # guiding the ants towards a more greedy, nearest-neighbor-like behavior
    # which can help find good tours quickly in the limited iteration count.
    # We add a small epsilon to avoid division by zero (though diagonal is 0, 
    # ants don't pick visited nodes, but the matrix should be well-formed).
    # The task says "Heuristic entries that are not positive become 1e-9".
    # 1/d^2 is positive for d > 0.
    
    # Calculate 1 / (d^2 + epsilon)
    # Using 1e-9 as the floor for non-positive or zero distances if any exist (though d_ii=0)
    # Actually, the evaluator replaces non-positive with 1e-9.
    # We just need to return a matrix.
    
    # To be safe against d=0 on diagonal (which might be 0), we use a large number 
    # or let the evaluator handle it? 
    # The task says: "Heuristic entries that are not positive become 1e-9".
    # So if I return 0 for diagonal, it becomes 1e-9.
    # If I return inf, it might cause issues if not handled? "non-finite values ... make the whole result invalid".
    # So I must return finite values.
    
    # 1 / (d^2 + 1e-9) ensures finite values.
    # For d=0, value is 1e9. This is very attractive.
    # But ants don't pick visited nodes. Node 0 is visited initially.
    # So diagonal values don't really matter for the choice, but they must be valid.
    
    eta = 1.0 / (d**2 + 1e-9)
    
    # Ensure no negative values (d**2 is non-negative, so eta is positive)
    # Ensure no NaNs or Infs.
    # If d is very large, eta is small but finite.
    # If d is 0, eta is 1e9.
    
    return eta
# EVOLVE-BLOCK-END
