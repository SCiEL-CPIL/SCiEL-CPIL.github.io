import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    start = time.time()
    deadline = start + 112.0  # Safety margin before 120s wall-clock

    # Fallback for non-(3,7,1): simple product construction (valid, not optimal)
    if q != 3 or n != 7 or radius != 1:
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q]
                     for a in range(q) for b in range(q)]
        return [h + list(tail) for h in tetracode
                for tail in itertools.product(range(q), repeat=n - 4)]

    N = 3 ** n  # 2187
    
    # Precompute all words
    all_words = np.zeros((N, n), dtype=np.int8)
    for idx in range(N):
        t = idx
        for pos in range(n - 1, -1, -1):
            all_words[idx, pos] = t % 3
            t //= 3

    # Precompute neighbors (balls) for each word
    # neighbors[i] is a list of indices j such that dist(i, j) <= 1
    # We also need a fast way to check coverage.
    # Let's create a list of lists for neighbors.
    neighbors = [[] for _ in range(N)]
    for i in range(N):
        neighbors[i].append(i)
        w = all_words[i]
        for pos in range(n):
            base = 3 ** (n - 1 - pos)
            for val in range(3):
                if val == w[pos]:
                    continue
                j = i + (val - int(w[pos])) * base
                neighbors[i].append(j)
        
    # Convert to numpy arrays for faster access if needed, but lists are okay for iteration
    # However, for vectorized operations in greedy, we might want a matrix or sparse structure.
    # Let's stick to lists for local search and use a matrix for greedy.
    
    # Ball matrix B[i, j] = 1 iff dist(i, j) <= radius
    B = np.zeros((N, N), dtype=np.uint8)
    for i in range(N):
        for j in neighbors[i]:
            B[i, j] = 1

    def is_valid(code):
        if not code:
            return False
        # Check if every word is covered
        covered = np.zeros(N, dtype=bool)
        for c in code:
            covered |= B[c].astype(bool)
        return bool(covered.all())

    def greedy_fill(code, covered):
        """Add codewords until every word is covered. Returns (code, covered)."""
        code = list(code)
        covered = covered.copy()
        while not covered.all():
            if time.time() > deadline - 6:
                break
            # Calculate gain: number of newly covered words
            gain_vec = (~covered).astype(np.int32)
            gains = B @ gain_vec
            best = int(np.argmax(gains))
            if gains[best] == 0:
                break
            code.append(best)
            covered |= B[best].astype(bool)
        return code, covered

    def local_search(code):
        """
        Perform local search: deletion and swaps.
        code: list of codeword indices
        """
        code = list(code)
        cs = set(code)
        
        # counts[i] = number of codewords whose ball covers word i
        counts = np.zeros(N, dtype=np.int32)
        for c in code:
            for p in neighbors[c]:
                counts[p] += 1

        improved = True
        while improved and time.time() < deadline - 6:
            improved = False

            # ---- Deletion pass ----
            # A codeword c is redundant if for all p in neighbors[c], counts[p] > 1
            to_remove = []
            for c in code:
                if all(counts[p] > 1 for p in neighbors[c]):
                    to_remove.append(c)
            
            if to_remove:
                improved = True
                for c in to_remove:
                    code.remove(c)
                    cs.discard(c)
                    for p in neighbors[c]:
                        counts[p] -= 1

            # ---- 1-Swap pass ----
            # Try to replace a codeword c with a new word w
            # such that the new configuration is valid and potentially better (fewer critical points)
            if time.time() > deadline - 6:
                break
            
            shuffled = list(code)
            random.shuffle(shuffled)
            for c in shuffled:
                if time.time() > deadline - 6:
                    break
                if c not in cs:
                    continue
                
                # Find critical points for c: points covered only by c
                crit = [p for p in neighbors[c] if counts[p] == 1]
                if not crit:
                    continue # Would be removed by deletion
                
                # We need to find a new position w such that:
                # 1. w is not in cs
                # 2. w covers all critical points of c
                # 3. Ideally, w covers other critical points of other codewords to allow their removal later?
                #    Or simply, the swap doesn't increase the number of critical points.
                
                # Candidate w must be in the intersection of balls of all critical points
                cand_set = set(neighbors[crit[0]])
                for p in crit[1:]:
                    cand_set &= set(neighbors[p])
                
                candidates = [w for w in cand_set if w not in cs]
                if not candidates:
                    continue

                # Score the candidates
                # A good swap reduces the total number of critical points or keeps it same while allowing future deletions.
                # Let's count how many critical points of OTHER codewords are covered by w.
                best_w = -1
                best_score = -1000
                for w in candidates:
                    # If we swap c -> w:
                    # counts for neighbors[c] decrease by 1
                    # counts for neighbors[w] increase by 1
                    
                    # Critical points of c are covered by w, so they won't become uncovered.
                    # What about other critical points?
                    # A point p is critical for codeword x if counts[p] == 1 and p in neighbors[x].
                    # After swap, counts change.
                    
                    # Simplified heuristic:
                    # Count how many points p in neighbors[w] are currently critical (counts[p]==1) 
                    # AND are NOT in neighbors[c].
                    # If w covers such a point, that point becomes count 2, reducing criticality.
                    # Also, if w covers a point p that was count 2, it becomes 3 (bad? no, just less flexible).
                    
                    score = 0
                    # Benefit: covering critical points of others
                    for p in neighbors[w]:
                        if counts[p] == 1 and p not in neighbors[c]:
                            score += 1
                    # Penalty: making non-critical points less critical? No, that's good.
                    # Penalty: losing coverage of points that were count 2?
                    # If p in neighbors[c] and counts[p] == 2, after swap counts[p] becomes 1.
                    # This creates a new critical point (for c's other covering codewords).
                    # So we should penalize this.
                    for p in neighbors[c]:
                        if counts[p] == 2 and p not in neighbors[w]:
                            score -= 1
                            
                    if score > best_score:
                        best_score = score
                        best_w = w

                if best_w >= 0 and best_score > 0: # Only swap if it improves the "criticality" metric
                    # Perform swap
                    for p in neighbors[c]:
                        counts[p] -= 1
                    for p in neighbors[best_w]:
                        counts[p] += 1
                    code.remove(c)
                    cs.discard(c)
                    code.append(best_w)
                    cs.add(best_w)
                    improved = True

        return code

    # ---- Initial greedy construction ----
    covered = np.zeros(N, dtype=bool)
    code = greedy_fill([], covered)[0]
    code = local_search(code)
    if not is_valid(code):
        code, _ = greedy_fill(code, np.zeros(N, dtype=bool))
        code = local_search(code)
    
    best = code if is_valid(code) else []
    best_len = len(best) if best else float('inf')

    # ---- Restart loop with kicking ----
    # We will try to kick out some codewords, re-greedy, and local search.
    # If we find a smaller valid code, we update best.
    
    kick_size = 1
    no_improve_count = 0
    max_kick_size = 5
    
    while time.time() < deadline - 8:
        cur = list(best)
        if not cur:
            break
            
        # Choose kick size
        # If stuck, increase kick size
        if no_improve_count > 20:
            kick_size = min(max_kick_size, kick_size + 1)
            no_improve_count = 0
        
        # Randomly remove kick_size codewords
        to_remove = set(random.sample(cur, min(kick_size, len(cur))))
        kicked = [c for c in cur if c not in to_remove]
        
        # Re-greedy fill
        covered = np.zeros(N, dtype=bool)
        if kicked:
            # Efficiently compute covered by kicked
            # Using matrix multiplication or summing rows
            # B[kicked_indices].any(axis=0)
            kicked_arr = np.array(kicked, dtype=np.intp)
            covered = B[kicked_arr].any(axis=0).astype(bool)
        
        new_code, _ = greedy_fill(kicked, covered)
        new_code = local_search(new_code)
        
        if is_valid(new_code):
            if len(new_code) < best_len:
                best = new_code
                best_len = len(new_code)
                no_improve_count = 0
                # Reset kick size if we improved significantly? Or keep increasing?
                # Let's keep increasing to explore further.
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1

    if not best or not is_valid(best):
        # Safety fallback
        covered = np.zeros(N, dtype=bool)
        code = greedy_fill([], covered)[0]
        code = local_search(code)
        best = code

    if not is_valid(best):
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3]
                     for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode
                for tail in itertools.product(range(3), repeat=n - 4)]

    return [list(all_words[c]) for c in best]
