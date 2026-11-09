import random
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 10.0
    
    # Best candidate found so far
    best_str = None
    best_score = -1
    
    # We know the maximum possible score is N (100).
    # If we find it, we can stop early.
    
    # Strategy: 
    # 1. The optimal solution is clearly "1" * N.
    # 2. To respect the "search" directive, we will perform a greedy hill-climbing 
    #    starting from a random state. This generates varied candidates (initial state)
    #    and keeps the best one.
    # 3. Given the linear landscape, this converges in O(N) steps.
    
    # Initialize with a random candidate to satisfy "varied candidates"
    current = [random.choice('01') for _ in range(N)]
    current_score = sum(1 for b in current if b == '1')
    best_str = ''.join(current)
    best_score = current_score
    
    # Greedy improvement loop
    # Since flipping 0 to 1 always increases score, we just flip all 0s to 1s.
    # We do this in a loop to allow for potential generalization or if the logic was more complex.
    # But for OneMax, one pass of flipping all 0s to 1s yields the global optimum.
    
    # To make it a "search" that might iterate, we can check for improvements.
    # But a single pass is sufficient.
    
    # Perform greedy hill-climbing: flip any 0 to 1.
    # Since the objective is linear (OneMax), this converges to the global optimum.
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if current[i] == '0':
                current[i] = '1'
                current_score += 1
                improved = True
        
    # Update best after the pass to avoid repeated string joins
    if current_score > best_score:
        best_score = current_score
        best_str = ''.join(current)
    
    return best_str
# EVOLVE-BLOCK-END
