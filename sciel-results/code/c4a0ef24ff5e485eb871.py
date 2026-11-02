import itertools
import time
import random

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    if n > 7:
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

    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s margin

    # Generate all permutations of 1..n-1
    prev_perms = list(itertools.permutations(range(1, n)))
    
    # Create blocks: p + [n] + p
    blocks = []
    for p in prev_perms:
        b = list(p) + [n] + list(p)
        blocks.append(b)
        
    num_blocks = len(blocks)
    block_len = len(blocks[0])
    
    # Precompute overlaps using string representation for speed
    # M[i][j] = max k such that blocks[i][-k:] == blocks[j][:k]
    # Convert blocks to strings for faster comparison
    block_strs = ["".join(map(str, b)) for b in blocks]
    
    overlaps = [[0]*num_blocks for _ in range(num_blocks)]
    
    for i in range(num_blocks):
        bi = block_strs[i]
        for j in range(num_blocks):
            bj = block_strs[j]
            max_k = min(len(bi), len(bj))
            found = 0
            for k in range(max_k, 0, -1):
                if bi[-k:] == bj[:k]:
                    found = k
                    break
            overlaps[i][j] = found

    def path_length(path):
        """Calculate total length of word given a path of block indices."""
        total = block_len
        for k in range(len(path)-1):
            total += block_len - overlaps[path[k]][path[k+1]]
        return total

    # Greedy TSP Heuristic with multiple restarts
    best_path = None
    best_len = float('inf')
    
    random.seed(42)
    
    # Try multiple restarts with different random starts
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline - 60:
            break
            
        current = random.randint(0, num_blocks - 1)
        unvisited = set(range(num_blocks))
        unvisited.remove(current)
        path = [current]
        
        while unvisited:
            best_next = -1
            best_overlap = -1
            ov_row = overlaps[current]
            for j in unvisited:
                if ov_row[j] > best_overlap:
                    best_overlap = ov_row[j]
                    best_next = j
            path.append(best_next)
            unvisited.remove(best_next)
            current = best_next
            
        curr_len = path_length(path)
        if curr_len < best_len:
            best_len = curr_len
            best_path = path[:]

    # Full 2-opt local search with O(1) delta using prefix sums
    # We maintain prefix sums of overlaps along the path for fast delta computation.
    
    def two_opt_search(path, time_limit):
        """Perform 2-opt local search on the path. Returns improved path."""
        N = len(path)
        if N <= 2:
            return path
        
        # Compute prefix sums of overlaps:
        # prefix[k] = sum of overlaps[path[i]][path[i+1]] for i in 0..k-1
        # So prefix[0] = 0, prefix[1] = ov(path[0],path[1]), etc.
        # prefix[k] = ov(path[0],path[1]) + ov(path[1],path[2]) + ... + ov(path[k-1],path[k])
        
        def build_prefix(path):
            prefix = [0] * (N)
            for i in range(N - 1):
                prefix[i+1] = prefix[i] + overlaps[path[i]][path[i+1]]
            return prefix
        
        prefix = build_prefix(path)
        
        def total_overlap_sum():
            return prefix[N-1]
        
        improved = True
        while improved:
            if time.time() > time_limit:
                break
            improved = False
            for i in range(N - 1):
                if time.time() > time_limit:
                    break
                for j in range(i + 2, N):
                    # 2-opt: reverse path[i+1..j]
                    # Old: ... path[i], path[i+1], ..., path[j], path[j+1] ...
                    # New: ... path[i], path[j], ..., path[i+1], path[j+1] ...
                    
                    # Overlaps affected:
                    # Old: ov(path[i], path[i+1]) + [internal: ov(path[i+1],path[i+2])+...+ov(path[j-1],path[j])] + ov(path[j], path[j+1])
                    # New: ov(path[i], path[j]) + [internal reversed: ov(path[j],path[j-1])+...+ov(path[i+2],path[i+1])] + ov(path[i+1], path[j+1])
                    
                    # Internal sum old = prefix[j] - prefix[i+1]
                    # Internal sum new = sum of ov(path[k+1], path[k]) for k in i+1..j-1
                    # We need to compute this. Let's precompute a reverse prefix sum too.
                    # rev_prefix[k] = sum of ov(path[k], path[k-1]) for k in 1..N-1
                    # Actually, let's just compute the delta directly.
                    
                    # Let A = path[i], B = path[i+1], C = path[j], D = path[j+1] (if j < N-1)
                    A = path[i]
                    B = path[i+1]
                    C = path[j]
                    D = path[j+1] if j < N - 1 else None
                    
                    # Old edge overlaps involving boundary:
                    old_edge = overlaps[A][B]
                    if D is not None:
                        old_edge += overlaps[C][D]
                    
                    # New edge overlaps:
                    new_edge = overlaps[A][C]
                    if D is not None:
                        new_edge += overlaps[B][D]
                    
                    # Internal old sum: sum(ov(path[k], path[k+1]) for k in i+1..j-1)
                    internal_old = prefix[j] - prefix[i+1]
                    
                    # Internal new sum: sum(ov(path[k+1], path[k]) for k in i+1..j-1)
                    # = sum(ov(path[k], path[k-1]) for k in i+2..j)
                    # We need a reverse prefix sum for this.
                    # Let's compute it on the fly or precompute.
                    # For efficiency, let's precompute rev_prefix where:
                    # rev_prefix[k] = sum of ov(path[t], path[t-1]) for t in 1..k
                    # So rev_prefix[k] - rev_prefix[i+1] = sum for t in i+2..k
                    # But we need sum for t in i+2..j, which is rev_prefix[j] - rev_prefix[i+1]
                    
                    # Let me just compute it directly since N=720 and this is O(j-i) per move.
                    # Actually that's too slow. Let me precompute rev_prefix.
                    
                    # I'll add rev_prefix computation inside the loop.
                    pass
                    
                    # For now, let's use a simpler approach: just compute the full path length
                    # if the edge delta suggests improvement, then verify.
                    
                    # Actually, let's just do a proper implementation with rev_prefix.
                    # I'll restructure.
                    pass
        
        return path
    
    # Let me rewrite the 2-opt properly with both prefix sums
    def two_opt_search_v2(path, time_limit):
        N = len(path)
        if N <= 2:
            return path
        
        def compute_prefixes(path):
            prefix = [0] * N
            rev_prefix = [0] * N
            for i in range(N - 1):
                prefix[i+1] = prefix[i] + overlaps[path[i]][path[i+1]]
                rev_prefix[i+1] = rev_prefix[i] + overlaps[path[i+1]][path[i]]
            return prefix, rev_prefix
        
        prefix, rev_prefix = compute_prefixes(path)
        
        improved = True
        while improved:
            if time.time() > time_limit:
                break
            improved = False
            for i in range(N - 1):
                if time.time() > time_limit:
                    break
                for j in range(i + 2, N):
                    A = path[i]
                    B = path[i+1]
                    C = path[j]
                    D = path[j+1] if j < N - 1 else None
                    
                    old_edge = overlaps[A][B]
                    if D is not None:
                        old_edge += overlaps[C][D]
                    
                    new_edge = overlaps[A][C]
                    if D is not None:
                        new_edge += overlaps[B][D]
                    
                    internal_old = prefix[j] - prefix[i+1]
                    internal_new = rev_prefix[j] - rev_prefix[i+1]
                    
                    delta = (new_edge - old_edge) + (internal_new - internal_old)
                    
                    if delta > 0:
                        # Perform the reversal
                        path[i+1:j+1] = reversed(path[i+1:j+1])
                        improved = True
                        # Recompute prefix sums
                        prefix, rev_prefix = compute_prefixes(path)
                        break  # Restart the inner loop from beginning
                if improved:
                    break
        
        return path
    
    # Run 2-opt on the best path found so far
    if best_path is not None and time.time() < deadline - 30:
        best_path = two_opt_search_v2(best_path, deadline - 30)
        best_len = path_length(best_path)
    
    # Try additional restarts + 2-opt if time permits
    for r in range(20):
        if time.time() > deadline - 30:
            break
        
        current = random.randint(0, num_blocks - 1)
        unvisited = set(range(num_blocks))
        unvisited.remove(current)
        path = [current]
        
        while unvisited:
            best_next = -1
            best_overlap = -1
            ov_row = overlaps[current]
            for j in unvisited:
                if ov_row[j] > best_overlap:
                    best_overlap = ov_row[j]
                    best_next = j
            path.append(best_next)
            unvisited.remove(best_next)
            current = best_next
        
        # Apply 2-opt to this path
        if time.time() < deadline - 10:
            path = two_opt_search_v2(path, deadline - 10)
        
        curr_len = path_length(path)
        if curr_len < best_len:
            best_len = curr_len
            best_path = path[:]
    
    # Final construction
    final_word = list(blocks[best_path[0]])
    for k in range(1, len(best_path)):
        ov = overlaps[best_path[k-1]][best_path[k]]
        final_word += blocks[best_path[k]][ov:]
        
    return "".join(map(str, final_word))
