"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def synthesize(matrix, depth_limit):
    import numpy as np
    
    # Parse input matrix
    # matrix is expected to be a 2D array-like of binary values
    # We need to handle various input formats (list of lists, numpy array, etc.)
    try:
        A = np.array(matrix, dtype=int)
    except Exception:
        A = np.array(matrix)
        A = (A != 0).astype(int)
    
    if A.ndim == 1:
        A = A.reshape(1, -1)
    
    m, n = A.shape
    
    # Input wires: 0..n-1
    # Constant-zero wire: n
    # Gate k (0-based) produces wire n+1+k
    
    # We want to compute each output row as XOR of some subset of inputs.
    # The "baseline" is the number of independent rows (rank) or sum of weights?
    # The task says: "Minimize mean gate-count / independent-row baseline"
    # This suggests we want to minimize the total number of gates used.
    # A naive implementation would use one gate per 1 in the matrix, but that's not optimal.
    # We can use Gaussian elimination to find a basis, then express each row in terms of the basis.
    # However, each basis vector might require multiple gates if it's not a single input.
    
    # Let's think about the circuit structure:
    # - Inputs are wires 0..n-1
    # - We can create new wires by XORing existing wires.
    # - Each new wire costs one gate.
    # - We need to produce m output wires.
    
    # Strategy:
    # 1. Find a basis for the row space using Gaussian elimination over GF(2).
    # 2. For each basis vector, if it's a single input, use that input directly.
    #    Otherwise, create a new wire for it using gates.
    # 3. For each output row, express it as a combination of basis vectors and create the necessary XORs.
    
    # But wait, we can be smarter. We can share intermediate results.
    # A better approach: use the rank to determine the minimum number of gates.
    # The minimum number of gates to synthesize a linear transformation is related to the rank.
    
    # Let's use a simple but effective strategy:
    # 1. Perform Gaussian elimination to get the row echelon form.
    # 2. For each pivot row, if it has more than one 1, create a new wire for the XOR of those inputs.
    # 3. For each non-pivot row, express it in terms of pivot rows and create the necessary XORs.
    
    # Actually, let's just use a direct approach:
    # For each output, create a wire that is the XOR of all the inputs that are 1 in that row.
    # This uses sum(weights) gates, which is the naive approach.
    
    # To do better, we can use the fact that if two rows share many 1s, we can factor them.
    # But for simplicity and to ensure correctness, let's start with the naive approach
    # and then optimize.
    
    # Naive approach:
    # For each row i, if the row is all zeros, output is wire n (constant zero).
    # Otherwise, create a wire that is the XOR of all inputs j where A[i][j] == 1.
    
    gates = []
    outputs = []
    next_wire = n + 1  # First new wire
    
    for i in range(m):
        row = A[i]
        ones = np.where(row == 1)[0]
        
        if len(ones) == 0:
            outputs.append(n)  # Constant zero
        elif len(ones) == 1:
            outputs.append(int(ones[0]))  # Single input
        else:
            # Create a chain of XORs
            current_wire = int(ones[0])
            for j in range(1, len(ones)):
                gates.append([current_wire, int(ones[j])])
                current_wire = next_wire
                next_wire += 1
            outputs.append(current_wire)
    
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
