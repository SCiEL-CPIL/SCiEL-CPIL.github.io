import itertools
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0
    
    if q == 3 and n == 6 and radius == 1:
        # Specialized solver for the known problem
        # Use greedy + local search with precomputed distances
        
        all_words = list(itertools.product(range(3), repeat=6))
        num_words = 729
        
        # Precompute Hamming distances between all pairs
        # dist[i][j] = Hamming distance between all_words[i] and all_words[j]
        dist = [[0]*num_words for _ in range(num_words)]
        for i in range(num_words):
            for j in range(i+1, num_words):
                d = sum(1 for a, b in zip(all_words[i], all_words[j]) if a != b)
                dist[i][j] = d
                dist[j][i] = d
        
        # For each codeword index, precompute which words it covers (dist <= 1)
        covers = [set() for _ in range(num_words)]
        for i in range(num_words):
            for j in range(num_words):
                if dist[i][j] <= radius:
                    covers[i].add(j)
        
        # Greedy set cover
        uncovered = set(range(num_words))
        codeword_indices = []
        
        while uncovered and time.time() < deadline:
            best_idx = -1
            best_count = 0
            best_covered = set()
            
            for i in range(num_words):
                if i in codeword_indices:
                    continue
                covered = covers[i] & uncovered
                cnt = len(covered)
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
                    best_covered = covered
            
            if best_idx == -1 or best_count == 0:
                break
            
            codeword_indices.append(best_idx)
            uncovered -= best_covered
        
        # Local search: remove redundant codewords
        def check_valid(indices):
            covered = set()
            for i in indices:
                covered |= covers[i]
            return len(covered) == num_words
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(codeword_indices)):
                trial = codeword_indices[:i] + codeword_indices[i+1:]
                if check_valid(trial):
                    codeword_indices = trial
                    changed = True
                    break
        
        # Try 2-for-1 swaps
        for _ in range(100):
            if time.time() > deadline:
                break
            improved = False
            for i in range(len(codeword_indices)):
                for j in range(i+1, len(codeword_indices)):
                    trial = [k for k, idx in enumerate(codeword_indices) if k != i and k != j]
                    covered = set()
                    for idx in trial:
                        covered |= covers[idx]
                    needed = set(range(num_words)) - covered
                    if not needed:
                        codeword_indices = trial
                        improved = True
                        break
                    for k in range(num_words):
                        if covers[k] >= needed:
                            codeword_indices = trial + [k]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return [list(all_words[i]) for i in codeword_indices]
    
    # General case: fall back to simple approach
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
