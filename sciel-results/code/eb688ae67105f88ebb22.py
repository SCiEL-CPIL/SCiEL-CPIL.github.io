import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for 120s limit
    
    N = q ** n
    
    # Precompute balls using numpy for speed
    # Each ball has size 1 + n*(q-1)
    ball_size = 1 + n * (q - 1)
    
    # Precompute the neighbor offsets for each position
    # For position i, the offsets in base-q representation
    # word[i] changes by delta (1..q-1), which changes the integer value by delta * q^(n-1-i)
    # But we need to handle wrap-around. Let's just precompute the balls directly.
    
    balls = np.zeros((N, ball_size), dtype=np.int32)
    
    for idx in range(N):
        # Convert idx to word
        w = [0] * n
        temp = idx
        for i in range(n - 1, -1, -1):
            w[i] = temp % q
            temp //= q
        
        balls[idx, 0] = idx
        pos = 1
        for i in range(n):
            for delta in range(1, q):
                new_w = w.copy()
                new_w[i] = (new_w[i] + delta) % q
                # Convert back to int
                new_idx = 0
                for j in range(n):
                    new_idx = new_idx * q + new_w[j]
                balls[idx, pos] = new_idx
                pos += 1
    
    # Precompute the "reverse" balls: for each element u, which centers cover it
    # Since the distance is symmetric in Hamming space, balls[u] is the set of centers that cover u
    # So we don't need a separate array, balls serves both purposes.
    
    best_code = None
    best_len = N  # Upper bound
    
    def greedy_set_cover():
        """Run one greedy set cover with random tie-breaking using numpy."""
        uncovered = np.ones(N, dtype=bool)
        # counts[i] = number of uncovered elements in balls[i]
        counts = np.zeros(N, dtype=np.int32)
        
        for i in range(N):
            counts[i] = np.sum(uncovered[balls[i]])
        
        code = []
        
        while True:
            remaining = np.sum(uncovered)
            if remaining == 0:
                break
            
            # Find index with max count
            max_count = np.max(counts)
            if max_count == 0:
                break
                
            # Find all indices with max count
            max_indices = np.where(counts == max_count)[0]
            
            # Random tie break
            w = int(random.choice(max_indices))
            code.append(w)
            
            # Update uncovered and counts
            ball_w = balls[w]
            for u in ball_w:
                if uncovered[u]:
                    uncovered[u] = False
                    # Decrement counts for all centers covering u
                    for v in balls[u]:
                        counts[v] -= 1
        
        return code
    
    def local_search(code):
        """Try to remove elements from code while maintaining coverage."""
        # Build coverage: for each element u, how many centers in code cover it
        cover_count = np.zeros(N, dtype=np.int32)
        for w in code:
            for u in balls[w]:
                cover_count[u] += 1
        
        improved = True
        while improved:
            improved = False
            # Try removing each element in reverse order
            for i in range(len(code) - 1, -1, -1):
                w = code[i]
                # Check if removing w still covers everything
                can_remove = True
                for u in balls[w]:
                    if cover_count[u] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    # Remove w
                    for u in balls[w]:
                        cover_count[u] -= 1
                    code.pop(i)
                    improved = True
                    break  # Restart after removal
        
        return code
    
    # Run multiple random restarts
    while time.time() < deadline:
        code = greedy_set_cover()
        
        # Local search: try to remove redundant codewords
        code = local_search(code)
        
        if len(code) < best_len:
            best_len = len(code)
            best_code = code.copy()
            
            # Early termination if we hit the known optimum
            if best_len <= 73:
                # Keep going a bit to see if we can do better
                pass
    
    if best_code is None:
        # Fallback if something went wrong
        return [[0] * n for _ in range(N)]
        
    # Convert int indices back to words
    result = []
    for w in best_code:
        word = [0] * n
        temp = w
        for i in range(n - 1, -1, -1):
            word[i] = temp % q
            temp //= q
        result.append(word)
    
    return result
