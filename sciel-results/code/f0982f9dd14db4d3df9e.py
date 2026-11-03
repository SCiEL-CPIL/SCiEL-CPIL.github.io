import time
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    if n < 7:
        return _classic_construct(n)
    
    if n == 7:
        # Generate permutations of 1..6
        perms = list(permutations(range(1, 7)))
        # Map perm to index for fast lookup? Or just use tuples.
        # We need overlap matrix.
        # Overlap(A, B): longest suffix of A matching prefix of B.
        
        # Precompute overlaps
        # N = 720
        # Matrix 720x720 is ~500k entries. Small.
        overlaps = [[0]*720 for _ in range(720)]
        
        # Optimization: 
        # For each perm, compute its prefix hashes?
        # Or just simple string conversion?
        # Tuples are fast enough for 720x720.
        
        for i, p1 in enumerate(perms):
            for j, p2 in enumerate(perms):
                # Find max k < 6 such that p1[-k:] == p2[:k]
                # Since p1 != p2, k < 6.
                # Check k=5, 4, 3, 2, 1
                # p1[-5:] == p2[:5]
                if p1[-5:] == p2[:5]:
                    overlaps[i][j] = 5
                elif p1[-4:] == p2[:4]:
                    overlaps[i][j] = 4
                elif p1[-3:] == p2[:3]:
                    overlaps[i][j] = 3
                elif p1[-2:] == p2[:2]:
                    overlaps[i][j] = 2
                elif p1[-1:] == p2[:1]:
                    overlaps[i][j] = 1
                else:
                    overlaps[i][j] = 0
                    
        # Now find a permutation of indices 0..719 that maximizes sum of overlaps.
        # Use 2-opt.
        
        best_score = -1
        best_order = None
        
        # Initial order: classic order?
        # The classic order for n=6 is the order of permutations in the classic superpermutation of length 6.
        # We can generate it.
        classic_order = _get_classic_order(6)
        
        # Run local search
        current_order = list(range(720))
        random.shuffle(current_order)
        
        # It might be better to start with classic_order
        current_order = classic_order[:]
        
        # Calculate current score
        def calc_score(order):
            s = 0
            for k in range(len(order)-1):
                s += overlaps[order[k]][order[k+1]]
            return s
        
        current_score = calc_score(current_order)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # 2-opt pass
            for i in range(len(current_order) - 1):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(current_order)):
                    # Reverse segment i+1 to j
                    # Edges affected:
                    # (order[i], order[i+1]) and (order[j], order[j+1])
                    # New edges: (order[i], order[j]) and (order[i+1], order[j+1])
                    # Note: if j is last, there is no (order[j], order[j+1])
                    
                    u = current_order[i]
                    v = current_order[i+1]
                    w = current_order[j]
                    
                    old_score_part = overlaps[u][v]
                    if j < len(current_order) - 1:
                        z = current_order[j+1]
                        old_score_part += overlaps[w][z]
                        new_score_part = overlaps[u][w] + overlaps[v][z]
                    else:
                        # w is the last element
                        # Old: u->v ... w
                        # New: u->w ... v
                        # Edges removed: (u,v)
                        # Edges added: (u,w)
                        # The rest of the chain is reversed, but internal edges are just reversed.
                        # Overlap is not symmetric! L(A,B) != L(B,A).
                        # So reversing the segment changes all internal edges too.
                        # This makes 2-opt complex for directed graphs.
                        # We need to recalculate the score for the segment or use a different heuristic.
                        # Given the size (720), maybe simple recalculation of the whole score is too slow if done for every move.
                        # But 720 is small.
                        # Let's stick to swaps or simple moves if directed 2-opt is tricky.
                        # Or, just recalculate the score of the whole path?
                        # 720 steps per evaluation.
                        # If we do 1000 evaluations, 720k ops. Very fast.
                        pass

        # Given the complexity of directed 2-opt, let's try a simpler approach:
        # Random restarts with a simple hill climbing (swap neighbors).
        
        best_score = -1
        best_order = None
        
        for restart in range(50):
            if time.time() > deadline - 10:
                break
            
            if restart == 0:
                order = classic_order[:]
            else:
                order = list(range(720))
                random.shuffle(order)
            
            current_score = calc_score(order)
            
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                # Try swapping adjacent elements
                for i in range(len(order) - 1):
                    # Swap order[i] and order[i+1]
                    u = order[i]
                    v = order[i+1]
                    
                    # Old edges: (prev, u), (u, v), (v, next)
                    # New edges: (prev, v), (v, u), (u, next)
                    
                    prev = order[i-1] if i > 0 else None
                    next_ = order[i+2] if i+2 < len(order) else None
                    
                    old_local = 0
                    new_local = 0
                    
                    if prev is not None:
                        old_local += overlaps[prev][u]
                        new_local += overlaps[prev][v]
                    
                    old_local += overlaps[u][v]
                    new_local += overlaps[v][u]
                    
                    if next_ is not None:
                        old_local += overlaps[v][next_]
                        new_local += overlaps[u][next_]
                    
                    if new_local > old_local:
                        order[i], order[i+1] = order[i+1], order[i]
                        current_score += new_local - old_local
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_order = order[:]
                        # Restart the scan from beginning or continue?
                        # Continue, but maybe restart to avoid local minima?
                        # Let's just continue.
            
            if current_score > best_score:
                best_score = current_score
                best_order = order[:]
        
        if best_order is None:
            best_order = classic_order
            best_score = calc_score(best_order)
            
        # Construct the word
        word = ""
        for idx in range(len(best_order)):
            p = perms[best_order[idx]]
            block = "".join(map(str, p)) + "7" + "".join(map(str, p))
            if idx == 0:
                word = block
            else:
                # Find overlap
                # We know the overlap length from the matrix?
                # Yes, overlaps[best_order[idx-1]][best_order[idx]]
                ov = overlaps[best_order[idx-1]][best_order[idx]]
                # Append block[ov:]
                word += block[ov:]
        
        return word

    else:
        return _classic_construct(n)

def _classic_construct(n):
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
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))

def _get_classic_order(n):
    # Returns the order of permutations of 1..n as they appear in the classic superpermutation
    # For n=6, this is the order of windows of length 6 that are permutations.
    word = _classic_construct(n)
    word_list = list(map(int, word))
    perms, seen = [], set()
    for i in range(len(word_list) - n + 1):
        window = tuple(word_list[i:i + n])
        if len(set(window)) == n and window not in seen:
            seen.add(window)
            perms.append(window)
    # Map these permutations to indices in the full list of permutations
    all_perms = list(permutations(range(1, n+1)))
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    order = [perm_to_idx[p] for p in perms]
    return order
