import itertools
import numpy as np
import time
import random

def construct(q, n, radius):
    # Parameters for the specific task: q=3, n=6, radius=1
    # General handling if parameters differ, though the task is fixed.
    if q != 3 or n != 6 or radius != 1:
        # Fallback to a simple greedy or product construction for other params
        # For robustness, we return a valid (though maybe large) code
        # Product construction: use Hamming code for first m coords if possible, else brute force
        # Here we just return all zeros repeated? No, must cover.
        # For general case, a simple greedy set cover is safe.
        return _greedy_cover(q, n, radius)

    # Specific optimization for q=3, n=6, radius=1
    total_words = 3 ** 6
    # Precompute neighbors for all 729 words
    # A word covers itself and its 6*2 = 12 neighbors
    neighbors = []
    for i in range(total_words):
        w = [0] * n
        rem = i
        for j in range(n):
            w[j] = rem % 3
            rem //= 3
        w = tuple(w)
        nbrs = [i]
        for pos in range(n):
            for val in range(3):
                if w[pos] != val:
                    new_w = list(w)
                    new_w[pos] = val
                    new_idx = 0
                    for k in range(n):
                        new_idx = new_idx * 3 + new_w[k]
                    nbrs.append(new_idx)
        neighbors.append(nbrs)
    
    # Initial Code: Direct product of Tetracode (9 words, len 4) and all words (len 2)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            word = h + list(tail)
            idx = 0
            for val in word:
                idx = idx * 3 + val
            initial_code.append(idx)
            
    # State
    current_code = list(initial_code)
    cover_count = np.zeros(total_words, dtype=int)
    
    # Initialize cover count
    for idx in current_code:
        for nbr in neighbors[idx]:
            cover_count[nbr] += 1
            
    best_code = list(current_code)
    best_size = len(current_code)
    
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    def is_valid():
        return np.all(cover_count > 0)
        
    def add_word(idx):
        for nbr in neighbors[idx]:
            cover_count[nbr] += 1
        current_code.append(idx)
        
    def remove_word(idx):
        for nbr in neighbors[idx]:
            cover_count[nbr] -= 1
        # Remove from list
        for i, w in enumerate(current_code):
            if w == idx:
                current_code.pop(i)
                break

    # Local Search
    # We alternate between trying to remove words and swapping words.
    # If we get stuck, we restart the loop with a different random seed/order.
    
    # To speed up, we can check validity less frequently if we are careful, 
    # but np.all is fast.
    
    stuck_counter = 0
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Try to remove words
        # Shuffle current code to try removals in random order
        random.shuffle(current_code)
        
        for idx in list(current_code):
            if time.time() > deadline: break
            
            # Try removing idx
            for nbr in neighbors[idx]:
                cover_count[nbr] -= 1
            
            if is_valid():
                # Keep removed
                # Remove from list (already shuffled, but list order doesn't matter for set logic)
                current_code.remove(idx)
                improved = True
                # Update best if smaller
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)
                # Break to restart the removal phase with the new smaller list
                break
            else:
                # Restore
                for nbr in neighbors[idx]:
                    cover_count[nbr] += 1
        
        if improved:
            stuck_counter = 0
            continue
        else:
            stuck_counter += 1
            
        # Phase 2: Try swaps if stuck on removals
        # We want to change the configuration to allow future removals
        # Number of swap attempts proportional to stuck_counter or fixed
        swap_attempts = 500 + stuck_counter * 100
        if stuck_counter > 100:
            # If stuck for a long time, restart from best known
            if random.random() < 0.1:
                current_code = list(best_code)
                cover_count = np.zeros(total_words, dtype=int)
                for idx in current_code:
                    for nbr in neighbors[idx]:
                        cover_count[nbr] += 1
                stuck_counter = 0
                continue

        for _ in range(swap_attempts):
            if time.time() > deadline: break
            
            if len(current_code) == 0: break
            
            idx_old = random.choice(current_code)
            idx_new = random.choice(range(total_words))
            
            if idx_new == idx_old:
                continue
            if idx_new in current_code:
                continue
                
            # Perform swap
            # Remove old
            for nbr in neighbors[idx_old]:
                cover_count[nbr] -= 1
            # Add new
            for nbr in neighbors[idx_new]:
                cover_count[nbr] += 1
            
            if is_valid():
                # Accept swap
                current_code.remove(idx_old)
                current_code.append(idx_new)
                improved = True
                # Don't update best size here as size is same
                break
            else:
                # Revert
                for nbr in neighbors[idx_new]:
                    cover_count[nbr] -= 1
                for nbr in neighbors[idx_old]:
                    cover_count[nbr] += 1
        
        if not improved:
            # If we couldn't even do a valid swap, we are in a very tight spot
            # or the code is locally optimal.
            # Increase stuck counter to trigger restarts
            stuck_counter += 1

    # Convert best_code indices back to word lists
    result = []
    for idx in best_code:
        w = [0] * n
        rem = idx
        for j in range(n):
            w[j] = rem % 3
            rem //= 3
        result.append(w)
        
    return result

def _greedy_cover(q, n, radius):
    """Fallback greedy set cover for non-standard parameters."""
    total_words = q ** n
    all_words = [list(map(int, str(i).zfill(n))) if q==10 else None for i in range(total_words)] # This is wrong for base q
    # Proper generation
    all_words = []
    for i in range(total_words):
        w = []
        rem = i
        for _ in range(n):
            w.append(rem % q)
            rem //= q
        all_words.append(w)
        
    # Precompute neighbors
    neighbors = []
    for i, w in enumerate(all_words):
        nbrs = [i]
        for pos in range(n):
            for val in range(q):
                if w[pos] != val:
                    new_w = w[:]
                    new_w[pos] = val
                    # Find index
                    idx = 0
                    for k in range(n):
                        idx = idx * q + new_w[k]
                    nbrs.append(idx)
        neighbors.append(nbrs)
        
    uncovered = set(range(total_words))
    code = []
    
    while uncovered:
        # Pick word covering most uncovered
        best_idx = -1
        max_covered = 0
        # Sample if large
        candidates = list(uncovered)
        if len(candidates) > 1000:
            candidates = random.sample(candidates, 1000)
            
        for c in candidates:
            covered = 0
            for nbr in neighbors[c]:
                if nbr in uncovered:
                    covered += 1
            if covered > max_covered:
                max_covered = covered
                best_idx = c
                
        if best_idx == -1:
            break
            
        code.append(all_words[best_idx])
        for nbr in neighbors[best_idx]:
            uncovered.discard(nbr)
            
    return code
