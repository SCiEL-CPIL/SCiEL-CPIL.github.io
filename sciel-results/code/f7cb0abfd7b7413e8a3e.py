"""Improved program: construct a 5-colouring of K_n with no monochromatic C4."""
import time
import random

def _index(i, j, n):
    """Lexicographic index of pair (i,j) with i<j in the flat list."""
    return i * (2 * n - i - 1) // 2 + (j - i - 1)

def _build_adj(colors, n):
    """Build adjacency bitsets for each color. adj[c][v] is a bitset of neighbors of v in color c."""
    adj = [[0] * n for _ in range(5)]
    for i in range(n):
        for j in range(i + 1, n):
            idx = _index(i, j, n)
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
    return adj

def _count_c4s(adj, n):
    """Count total number of C4 violations across all colors."""
    count = 0
    for c in range(5):
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[c][u] & adj[c][v]
                if common & (common - 1):
                    count += 1
    return count

def _is_valid(colors, n):
    """Check if coloring is valid (no monochromatic C4)."""
    adj = _build_adj(colors, n)
    return _count_c4s(adj, n) == 0

def construct():
    deadline = time.time() + 145  # safety margin before 150s kill
    random.seed(42)
    
    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # Try to find larger valid colorings via local search
    # We aim for n=25, 26, etc.
    for target_n in range(29, 11, -1):
        if time.time() > deadline - 10:
            break
        
        num_edges = target_n * (target_n - 1) // 2
        found = False
        
        # Precompute edge list for fast lookup
        edge_list = []
        for i in range(target_n):
            for j in range(i + 1, target_n):
                edge_list.append((i, j))
        
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            # Random initial coloring
            colors = [random.randint(0, 4) for _ in range(num_edges)]
            adj = _build_adj(colors, target_n)
            
            # Initialize violation counts per pair
            # viol[c][u][v] = number of common neighbors in color c
            # We only care if it is >= 2.
            # To save memory and time, we can just maintain the count of pairs with >= 2 common neighbors.
            # Let's use a dictionary or 2D array for counts.
            # n is up to 29. 29x29 is small.
            counts = [[[0] * target_n for _ in range(target_n)] for _ in range(5)]
            total_violations = 0
            
            # Calculate initial counts
            for c in range(5):
                for u in range(target_n):
                    for v in range(u + 1, target_n):
                        common = adj[c][u] & adj[c][v]
                        cnt = common.bit_count()
                        counts[c][u][v] = cnt
                        counts[c][v][u] = cnt
                        if cnt >= 2:
                            total_violations += 1
            
            max_iters = 10000
            for it in range(max_iters):
                if time.time() > deadline - 2:
                    break
                
                if total_violations == 0:
                    found = True
                    break
                
                # Pick a random violation
                # We need a list of violations to pick from.
                # Rebuilding it is O(n^2).
                # Alternatively, pick a random pair and check if it's a violation.
                # If not, try another.
                # Since we want to escape local minima, we can just pick a random edge and recolor it?
                # No, that's slow to converge.
                # Let's pick a random edge involved in a violation.
                
                # Find a violation
                # We can keep a list of violating pairs.
                # Updating the list is expensive.
                # Let's just pick a random edge (a, b) and try to recolor it.
                # This is a random walk.
                # To be smarter, pick an edge (a, b) such that it is in a violation.
                # Let's find one.
                
                # To avoid O(n^2) search for a violation every step, we can store violating edges.
                # But updating them is tricky.
                # Let's just pick a random edge and recolor it.
                # With incremental updates, this is fast.
                
                idx = random.randint(0, num_edges - 1)
                a, b = edge_list[idx]
                old_c = colors[idx]
                
                best_new_c = -1
                best_count = total_violations
                
                for new_c in range(5):
                    if new_c == old_c:
                        continue
                    
                    # Simulate change
                    # Remove (a, b) from old_c
                    adj[old_c][a] &= ~(1 << b)
                    adj[old_c][b] &= ~(1 << a)
                    
                    # Update counts for old_c
                    delta = 0
                    for u in range(target_n):
                        if u == a or u == b:
                            continue
                        # Update counts[old_c][a][u]
                        if adj[old_c][b] & (1 << u):
                            # b was a neighbor of a in old_c, and u is neighbor of b?
                            # Wait. Removing edge (a, b).
                            # For any u, count(a, u) decreases by 1 if b is in adj[old_c][u] AND a is in adj[old_c][u]?
                            # No.
                            # count(a, u) = |adj[old_c][a] & adj[old_c][u]|.
                            # We removed b from adj[old_c][a].
                            # So if b was in adj[old_c][u], then b was in the intersection.
                            # So count(a, u) decreases by 1 if b in adj[old_c][u].
                            
                            # Check if b in adj[old_c][u]
                            if adj[old_c][u] & (1 << b):
                                old_cnt = counts[old_c][a][u]
                                new_cnt = old_cnt - 1
                                counts[old_c][a][u] = new_cnt
                                counts[old_c][u][a] = new_cnt
                                if old_cnt >= 2 and new_cnt < 2:
                                    delta -= 1
                                elif old_cnt < 2 and new_cnt >= 2: # Impossible
                                    delta += 1
                            
                            # Similarly for count(b, u)
                            if adj[old_c][u] & (1 << a):
                                old_cnt = counts[old_c][b][u]
                                new_cnt = old_cnt - 1
                                counts[old_c][b][u] = new_cnt
                                counts[old_c][u][b] = new_cnt
                                if old_cnt >= 2 and new_cnt < 2:
                                    delta -= 1
                                elif old_cnt < 2 and new_cnt >= 2:
                                    delta += 1
                    
                    # Add (a, b) to new_c
                    adj[new_c][a] |= (1 << b)
                    adj[new_c][b] |= (1 << a)
                    
                    # Update counts for new_c
                    for u in range(target_n):
                        if u == a or u == b:
                            continue
                        if adj[new_c][u] & (1 << b):
                            old_cnt = counts[new_c][a][u]
                            new_cnt = old_cnt + 1
                            counts[new_c][a][u] = new_cnt
                            counts[new_c][u][a] = new_cnt
                            if old_cnt < 2 and new_cnt >= 2:
                                delta += 1
                            elif old_cnt >= 2 and new_cnt < 2: # Impossible
                                delta -= 1
                        
                        if adj[new_c][u] & (1 << a):
                            old_cnt = counts[new_c][b][u]
                            new_cnt = old_cnt + 1
                            counts[new_c][b][u] = new_cnt
                            counts[new_c][u][b] = new_cnt
                            if old_cnt < 2 and new_cnt >= 2:
                                delta += 1
                            elif old_cnt >= 2 and new_cnt < 2:
                                delta -= 1
                    
                    new_total = total_violations + delta
                    
                    # Revert
                    adj[new_c][a] &= ~(1 << b)
                    adj[new_c][b] &= ~(1 << a)
                    adj[old_c][a] |= (1 << b)
                    adj[old_c][b] |= (1 << a)
                    
                    # Revert counts
                    for u in range(target_n):
                        if u == a or u == b:
                            continue
                        if adj[old_c][u] & (1 << b):
                            counts[old_c][a][u] += 1
                            counts[old_c][u][a] += 1
                        if adj[old_c][u] & (1 << a):
                            counts[old_c][b][u] += 1
                            counts[old_c][u][b] += 1
                        
                        if adj[new_c][u] & (1 << b):
                            counts[new_c][a][u] -= 1
                            counts[new_c][u][a] -= 1
                        if adj[new_c][u] & (1 << a):
                            counts[new_c][b][u] -= 1
                            counts[new_c][u][b] -= 1
                    
                    if new_total < best_count:
                        best_count = new_total
                        best_new_c = new_c
                        best_delta = delta
                    
                    if best_count == 0:
                        break
                
                if best_new_c != -1:
                    # Apply the best change
                    adj[old_c][a] &= ~(1 << b)
                    adj[old_c][b] &= ~(1 << a)
                    adj[best_new_c][a] |= (1 << b)
                    adj[best_new_c][b] |= (1 << a)
                    colors[idx] = best_new_c
                    total_violations = best_count
                else:
                    # No improvement. Random move.
                    new_c = random.randint(0, 4)
                    if new_c != old_c:
                        # Apply random move
                        adj[old_c][a] &= ~(1 << b)
                        adj[old_c][b] &= ~(1 << a)
                        adj[new_c][a] |= (1 << b)
                        adj[new_c][b] |= (1 << a)
                        colors[idx] = new_c
                        
                        # Update counts
                        delta = 0
                        for u in range(target_n):
                            if u == a or u == b:
                                continue
                            if adj[new_c][u] & (1 << b):
                                old_cnt = counts[new_c][a][u]
                                new_cnt = old_cnt + 1
                                counts[new_c][a][u] = new_cnt
                                counts[new_c][u][a] = new_cnt
                                if old_cnt < 2 and new_cnt >= 2:
                                    delta += 1
                            if adj[new_c][u] & (1 << a):
                                old_cnt = counts[new_c][b][u]
                                new_cnt = old_cnt + 1
                                counts[new_c][b][u] = new_cnt
                                counts[new_c][u][b] = new_cnt
                                if old_cnt < 2 and new_cnt >= 2:
                                    delta += 1
                            
                            # Revert old_c counts
                            if adj[old_c][u] & (1 << b):
                                old_cnt = counts[old_c][a][u]
                                new_cnt = old_cnt - 1
                                counts[old_c][a][u] = new_cnt
                                counts[old_c][u][a] = new_cnt
                                if old_cnt >= 2 and new_cnt < 2:
                                    delta -= 1
                            if adj[old_c][u] & (1 << a):
                                old_cnt = counts[old_c][b][u]
                                new_cnt = old_cnt - 1
                                counts[old_c][b][u] = new_cnt
                                counts[old_c][u][b] = new_cnt
                                if old_cnt >= 2 and new_cnt < 2:
                                    delta -= 1
                        
                        total_violations += delta
            
            if found:
                if _is_valid(colors, target_n):
                    best_n = target_n
                    best_colors = colors[:]
                    break
        
        if found:
            break
    
    return {'n': best_n, 'colors': best_colors}
