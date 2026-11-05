"""Program for low discrepancy on all arithmetic progressions.
Uses a quadratic minimization approach with a precomputed sparse matrix.
"""
import numpy as np
import scipy.sparse as sp

def construct():
    n = 256
    
    # 1. Generate all arithmetic progressions
    # An AP is defined by start index s, step d, and end index e.
    # The indices are s, s+d, ..., e.
    # We store these as rows in a sparse matrix A.
    
    row_indices = []
    col_indices = []
    data = []
    
    row_id = 0
    for d in range(1, n + 1):
        for s in range(0, n):
            # Find all valid end indices e such that e >= s and (e - s) % d == 0
            # e = s, s+d, s+2d, ...
            for e in range(s, n, d):
                # The AP is s, s+d, ..., e
                # Indices are s, s+d, ..., e
                # We can generate these indices
                ap_indices = list(range(s, e + 1, d))
                
                # Add to sparse matrix data
                # Row is row_id, Cols are ap_indices, Data is 1
                row_indices.extend([row_id] * len(ap_indices))
                col_indices.extend(ap_indices)
                data.extend([1] * len(ap_indices))
                
                row_id += 1
                
    # Convert to CSR matrix for fast row slicing and matrix-vector multiplication
    A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(row_id, n))
    
    # 2. Initialize signs
    # Start with alternating signs
    signs = np.array([1 if i % 2 == 0 else -1 for i in range(n)], dtype=np.int8)
    # Maintain a real-valued vector for gradient descent
    x_real = signs.astype(np.float64)
    
    # 3. Gradient Descent on Quadratic Loss ||A x||^2
    # Loss L = x^T A^T A x
    # Gradient g = 2 A^T A x
    # We can compute y = A x, then g = 2 A^T y
    
    # Learning rate
    # Since we project to +/-1, the effective step is binary.
    # We want to update x_real such that signs flip if the gradient suggests it.
    # A simple approach: x_real -= eta * g
    # Then signs = sign(x_real)
    
    # We need to tune eta. 
    # The magnitude of g depends on the number of APs containing each index.
    # Each index i is in roughly O(n log n) APs? No, O(n^2) APs?
    # For index i, it's in APs starting at i, i-d, i-2d...
    # Sum of 1/d is log n. So O(n log n) APs contain a given index?
    # Actually, for each step d, there are ~n/d starts.
    # Total APs is n^2 log n.
    # Average row sum is n/2? No, average AP length is small.
    
    # Let's just try a fixed eta and run a few iterations.
    # Or adapt eta.
    
    # To ensure we finish within 60s, we limit iterations.
    # Matrix-vector mult is fast.
    
    eta = 0.1 # Initial guess
    max_iters = 100
    
    current_max_disc = 0
    # Compute initial max discrepancy to track improvement
    # We can compute this from y = A @ signs
    y = A @ signs
    current_max_disc = np.max(np.abs(y))
    
    best_signs = signs.copy()
    best_max_disc = current_max_disc
    
    for _ in range(max_iters):
        # Compute gradient
        y = A @ x_real
        g = A.T @ y
        
        # Update
        x_real = x_real - eta * g
        
        # Project to signs
        new_signs = np.sign(x_real)
        # Handle zeros (shouldn't happen often with float)
        new_signs[new_signs == 0] = 1
        
        # Check if signs changed
        if np.array_equal(new_signs, signs):
            # No change, maybe break or adjust eta?
            # If no change, the gradient is not strong enough to flip any sign.
            # We can try increasing eta or breaking.
            # Let's break to save time if stuck.
            break
        
        signs = new_signs
        
        # Evaluate max discrepancy
        y = A @ signs
        max_disc = np.max(np.abs(y))
        
        if max_disc < best_max_disc:
            best_max_disc = max_disc
            best_signs = signs.copy()
            
        # Adaptive learning rate?
        # If max_disc increased, we might have overshot.
        # But since we project, it's discrete.
        
    # Return the best signs found
    return [int(s) for s in best_signs]
