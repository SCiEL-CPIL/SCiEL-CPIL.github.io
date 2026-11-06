import random

N = 100

# EVOLVE-BLOCK-START
def construct():
    # Generate a random 100-bit string, then apply a simple hill-climbing
    # local search to maximize the number of '1' bits.
    # Start with a random initial population of size 1 (the candidate itself).
    candidate = ['0' if random.random() < 0.5 else '1' for _ in range(N)]
    
    # Simple hill climbing: repeatedly flip bits if it improves the score
    # until no single-bit flip improves the score.
    improved = True
    while improved:
        improved = False
        score = sum(1 for b in candidate if b == '1')
        for i in range(N):
            if candidate[i] == '0':
                # Flipping 0 to 1 always improves score by 1
                candidate[i] = '1'
                improved = True
                break  # restart scan from beginning for simplicity
        # If we never broke, no improvement possible
        if not improved:
            # Check if we can improve by flipping 1 to 0? No, that decreases score.
            # So we're at local optimum (which is global max for this objective)
            break
    
    return ''.join(candidate)
# EVOLVE-BLOCK-END
