"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct(n):
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s limit

    if n <= 6:
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    if n == 7:
        # Build S_6 standard construction
        word = [1]
        for m in range(2, 7):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new

        # Extract all 720 permutations of 1..6 in order of appearance
        perms, seen = [], set()
        for i in range(len(word) - 5):
            window = tuple(word[i:i + 6])
            if len(set(window)) == 6 and window not in seen:
                seen.add(window)
                perms.append(window)

        N = len(perms)

        # Precompute overlap matrix
        # overlap[i][j] = longest suffix of perms[i] that is prefix of perms[j]
        overlap_mat = [[0] * N for _ in range(N)]
        for i in range(N):
            pi = perms[i]
            for j in range(N):
                if i == j:
                    overlap_mat[i][j] = 6
                    continue
                pj = perms[j]
                ov = 0
                # Max possible overlap is 5 (since they are distinct, 6 is impossible)
                for k in range(5, 0, -1):
                    if pi[-k:] == pj[:k]:
                        ov = k
                        break
                overlap_mat[i][j] = ov

        def calc_total_overlap(p):
            total = 0
            for i in range(len(p) - 1):
                total += overlap_mat[p[i]][p[i + 1]]
            return total

        # Initial solution: Nearest Neighbor
        random.seed(42)
        unvisited = list(range(N))
        random.shuffle(unvisited)
        path = [unvisited.pop()]
        
        while unvisited:
            last = path[-1]
            best_idx = -1
            best_ov = -1
            for idx in unvisited:
                ov = overlap_mat[last][idx]
                if ov > best_ov:
                    best_ov = ov
                    best_idx = idx
            path.append(best_idx)
            unvisited.remove(best_idx)

        current_score = calc_total_overlap(path)
        best_score = current_score
        best_path = path[:]

        # Simulated Annealing
        # Initial temperature and cooling rate
        T = 1.0
        cooling = 0.9995
        min_T = 1e-6
        
        # Time management
        t_start = time.time()
        
        while time.time() < deadline - 10:
            if T < min_T:
                break
            
            # Move: Swap two random elements (Or-Opt 1 variant / 2-opt of length 1)
            # We use adjacent swap for simplicity and O(1) delta calculation
            # Actually, let's use a more powerful move: Swap of two arbitrary elements?
            # No, adjacent swap is often effective in TSP-like problems if combined with SA.
            # But the previous analysis showed adjacent swap might be limited.
            # Let's stick to the robust "Swap adjacent" which is effectively 2-opt on a segment of length 2.
            # Or better, let's implement a proper 2-opt reversal with exact delta?
            # Exact delta for 2-opt reversal of segment [i+1, j] involves summing internal edges.
            # For N=720, if we pick random i, j, the expected length is N/3 ~ 240.
            # Summing 240 terms is fast.
            # Let's implement 2-opt reversal with exact delta.
            
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            
            # Segment to reverse: path[i+1] ... path[j]
            # Left neighbor: L = path[i]
            # Right neighbor: R = path[j+1] (if j < N-1)
            
            # Old edges:
            # 1. (L, path[i+1])
            # 2. Internal: (path[k], path[k+1]) for k from i+1 to j-1
            # 3. (path[j], R) if j < N-1
            
            # New edges:
            # 1. (L, path[j])
            # 2. Internal: (path[k+1], path[k]) for k from i+1 to j-1
            # 3. (path[i+1], R) if j < N-1
            
            old_delta = 0
            new_delta = 0
            
            # Boundary 1
            old_delta += overlap_mat[path[i]][path[i+1]]
            new_delta += overlap_mat[path[i]][path[j]]
            
            # Boundary 2
            if j < N - 1:
                old_delta += overlap_mat[path[j]][path[j+1]]
                new_delta += overlap_mat[path[i+1]][path[j+1]]
            
            # Internal edges
            # Sum of w(p[k], p[k+1]) for k in [i+1, j-1]
            # vs Sum of w(p[k+1], p[k]) for k in [i+1, j-1]
            # We can compute this in O(length)
            for k in range(i + 1, j):
                old_delta += overlap_mat[path[k]][path[k+1]]
                new_delta += overlap_mat[path[k+1]][path[k]]
            
            delta = new_delta - old_delta
            
            # Accept move
            if delta > 0 or random.random() < math.exp(delta / T):
                # Perform reversal
                path[i+1:j+1] = reversed(path[i+1:j+1])
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_path = path[:]
            
            T *= cooling

        # Use best path found
        path = best_path

        # Construct S_7 using the optimized path
        new_word = []
        for idx in path:
            p = perms[idx]
            block = list(p) + [7] + list(p)
            # Find overlap with current new_word
            overlap = 0
            min_len = min(len(new_word), len(block))
            for o in range(min_len, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]

        return "".join(map(str, new_word))

    # Fallback for n > 7
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
