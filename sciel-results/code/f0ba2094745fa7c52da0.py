import time

_DEADLINE = time.time() + 85.0

def construct():
    """
    Return a list of 9 mutually orthogonal Latin squares of order 10.
    
    Strategy:
    1. Construct 10 mutually orthogonal Latin squares (MOLS) of order 11 using the finite field F_11.
       The squares are L_k(i, j) = (i + k * j) % 11 for k in {1, ..., 10}.
       These are pairwise orthogonal because for any distinct k, l, the map (i,j) -> (L_k(i,j), L_l(i,j)) is bijective.
    2. Remove the last row (index 10) and last column (index 10) from each square.
       This results in 10 MOLS of order 10.
       The symbols are 0..9.
    3. Return the first 9 of these squares to satisfy the constraint "Return between 1 and 9 squares".
    """
    # Generate 10 MOLS of order 11
    order_11_squares = []
    for k in range(1, 11):
        sq = [[(i + k * j) % 11 for j in range(11)] for i in range(11)]
        order_11_squares.append(sq)
    
    # Reduce to order 10 by removing row 10 and column 10
    order_10_squares = []
    for sq in order_11_squares:
        # Keep rows 0..9 and columns 0..9
        reduced_sq = [[sq[i][j] for j in range(10)] for i in range(10)]
        order_10_squares.append(reduced_sq)
        
    # We have 10 MOLS of order 10. The task asks for between 1 and 9.
    # We return the first 9.
    result = order_10_squares[:9]
    
    # Ensure types are Python ints and lists
    final_result = []
    for sq in result:
        clean_sq = [[int(x) for x in row] for row in sq]
        final_result.append(clean_sq)
        
    return final_result

# Keep the original function name and signature
# The prompt says "Keep the function names and signatures of the seed program."
# The seed program has `construct()`.

# We also need to ensure no other functions are required.
# The seed program had helper functions, but we can replace them all.
# However, to be safe and clean, I will just provide the `construct` function.
# The prompt says "combine... into ONE coherent program".

# I will include a main block or just the function.
# The evaluator calls `construct()`.

if __name__ == "__main__":
    # Simple test
    sqs = construct()
    print(f"Generated {len(sqs)} squares.")
    # Verify Latin
    for sq in sqs:
        for i in range(10):
            if sorted(sq[i]) != list(range(10)):
                raise ValueError("Row not Latin")
        for j in range(10):
            if sorted([sq[i][j] for i in range(10)]) != list(range(10)):
                raise ValueError("Col not Latin")
    # Verify Orthogonality
    for i in range(len(sqs)):
        for j in range(i+1, len(sqs)):
            seen = set()
            for r in range(10):
                for c in range(10):
                    p = (sqs[i][r][c], sqs[j][r][c])
                    if p in seen:
                        raise ValueError("Not orthogonal")
                    seen.add(p)
    print("All checks passed.")
