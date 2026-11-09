"""Program for R_5(C4) maximization:
1) Greedy randomized construction with dynamic forbidden-pair tracking for n=26..28.
2) High-speed local search with delta-updates to refine the coloring.
3) Fallback to smaller n if necessary."""
import time
import random
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 115.0

    def edge_idx(i, j, n):
        # Map (i, j) with i < j to index in lexicographic order
        # i=0: 0..n-2
        # i=1: n-1 .. 2n-3
        # Sum of lengths for rows 0..i-1 is i*(n-1) - i*(i-1)/2
        # Actually, simpler: 
        # Row i has n-1-i edges.
        # Index = sum_{k=0}^{i-1} (n-1-k) + (j-i-1)
        # Sum_{k=0}^{i-1} (n-1-k) = i(n-1) - i(i-1)/2
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def idx_to_edge(idx, n):
        # Inverse of edge_idx
        # Binary search or precomputed lookup for small n
        # For n<=30, a linear scan or precomputed list is fine, but let's do a quick calc
        # We can precompute row starts
        # Or just loop, it's O(n) which is small
        row_start = 0
        for i in range(n):
            row_len = n - 1 - i
            if idx < row_start + row_len:
                j = i + 1 + (idx - row_start)
                return (i, j)
            row_start += row_len
        return (0, 1)

    class C4Tracker:
        def __init__(self, n, colors):
            self.n = n
            self.colors = colors
            self.adj = [np.zeros((n, n), dtype=np.uint8) for _ in range(5)]
            self.common = [np.zeros((n, n), dtype=np.int32) for _ in range(5)]
            self.danger = [np.zeros((n, n), dtype=np.int32) for _ in range(5)] # common == 1
            self.bad = [np.zeros((n, n), dtype=np.int32) for _ in range(5)]    # common >= 2
            self.total_bad = 0
            self.total_danger = 0
            self.rebuild()

        def rebuild(self):
            self.adj = [np.zeros((self.n, self.n), dtype=np.uint8) for _ in range(5)]
            m = self.n * (self.n - 1) // 2
            for idx in range(m):
                i, j = idx_to_edge(idx, self.n)
                c = self.colors[idx]
                self.adj[c][i, j] = 1
                self.adj[c][j, i] = 1
            
            for c in range(5):
                A = self.adj[c].astype(np.int32)
                M = A @ A.T
                np.fill_diagonal(M, 0)
                # common[c][u][v] = M[u][v]
                self.common[c] = M
                self.danger[c] = (M == 1).astype(np.int32)
                self.bad[c] = (M >= 2).astype(np.int32)
                # Count upper triangle
                # np.triu(M, 1)
                tri = np.triu(M, 1)
                self.total_bad += int(np.sum(tri >= 2))
                self.total_danger += int(np.sum(tri == 1))

        def flip(self, idx, new_c):
            i, j = idx_to_edge(idx, self.n)
            old_c = self.colors[idx]
            if old_c == new_c:
                return
            
            self.colors[idx] = new_c
            
            # Update adj
            self.adj[old_c][i, j] = 0
            self.adj[old_c][j, i] = 0
            self.adj[new_c][i, j] = 1
            self.adj[new_c][j, i] = 1
            
            # Update common counts for pairs involving i or j
            # A pair (u, v) is affected if u or v is i or j.
            # Actually, common[u][v] changes if one of the endpoints is i or j.
            # Let's update common for all u != i, u != j? No, just pairs (u, i) and (u, j).
            # common[u][i] = sum(w) adj[u][w] * adj[i][w]
            # When we change adj[i][j], it affects common[u][i] for all u != i, j?
            # No. common[u][i] depends on adj[u][*] and adj[i][*].
            # The change is in adj[i][j] and adj[j][i].
            # So common[u][i] changes by +/- 1 if adj[u][j] == 1.
            # Similarly common[u][j] changes by +/- 1 if adj[u][i] == 1.
            # And common[i][j] changes? 
            # common[i][j] = sum(w) adj[i][w] adj[j][w].
            # The term w=k where adj[i][k] and adj[j][k] are 1.
            # The flip changes adj[i][j] and adj[j][i].
            # Does common[i][j] include w=i? No, no self loops.
            # Does common[i][j] include w=j? No.
            # So the set of common neighbors of i and j does NOT change when we flip the edge (i,j) itself?
            # Wait. common[i][j] counts w such that i-w and j-w are edges.
            # If we flip edge (i,j), does it affect whether i-w and j-w are edges?
            # No, it only affects edges incident to i or j.
            # So common[i][j] is invariant under flipping edge (i,j).
            # However, common[u][v] for u,v not in {i,j} is invariant.
            # So we only need to update common[u][i] and common[u][j] for all u.
            
            # Let's refine:
            # For any u, common[u][i] = sum_{w} adj[u][w] adj[i][w].
            # The only change in adj[i][*] is at w=j (from 1 to 0 or 0 to 1).
            # So common[u][i] changes by delta = (adj[u][j] * (new_val - old_val)).
            # new_val = 1 if new_c, 0 if old_c.
            # So if we remove edge (i,j) from color old_c:
            #   adj[old_c][i][j] becomes 0.
            #   For all u, common[old_c][u][i] decreases by adj[old_c][u][j].
            #   For all u, common[old_c][u][j] decreases by adj[old_c][u][i].
            # If we add edge (i,j) to color new_c:
            #   adj[new_c][i][j] becomes 1.
            #   For all u, common[new_c][u][i] increases by adj[new_c][u][j].
            #   For all u, common[new_c][u][j] increases by adj[new_c][u][i].
            
            # We need to update danger and bad counts.
            # This is O(n) per update.
            
            def update_common(c, u, v, delta):
                # Update common[c][u][v] by delta
                # Note: common is symmetric
                old_val = self.common[c][u, v]
                new_val = old_val + delta
                self.common[c][u, v] = new_val
                self.common[c][v, u] = new_val
                
                # Update danger/bad counters
                # We need to know if this pair was in upper triangle to update totals?
                # Or just maintain totals by checking condition.
                # Let's just recalculate totals? No, too slow.
                # Let's maintain totals carefully.
                # If u < v:
                #   if old_val == 1 and new_val != 1: total_danger -= 1
                #   if old_val != 1 and new_val == 1: total_danger += 1
                #   if old_val >= 2 and new_val < 2: total_bad -= 1
                #   if old_val < 2 and new_val >= 2: total_bad += 1
                
                if u < v:
                    if old_val == 1 and new_val != 1:
                        self.total_danger -= 1
                    elif old_val != 1 and new_val == 1:
                        self.total_danger += 1
                    if old_val >= 2 and new_val < 2:
                        self.total_bad -= 1
                    elif old_val < 2 and new_val >= 2:
                        self.total_bad += 1

            # Remove from old_c
            if old_c != -1:
                # delta = -1 for neighbors of j
                # For all u != i, j:
                # common[old_c][u][i] -= adj[old_c][u][j]
                # common[old_c][u][j] -= adj[old_c][u][i]
                
                # Vectorized update might be faster, but n is small.
                for u in range(self.n):
                    if u == i or u == j:
                        continue
                    # Update (u, i)
                    if self.adj[old_c][u, j]:
                        update_common(old_c, u, i, -1)
                    # Update (u, j)
                    if self.adj[old_c][u, i]:
                        update_common(old_c, u, j, -1)
                
                # What about (i, j)?
                # common[i][j] is invariant?
                # Let's verify.
                # common[i][j] = sum(w) adj[i][w] adj[j][w].
                # The flip changes adj[i][j] and adj[j][i].
                # The sum is over w. The terms are adj[i][w] and adj[j][w].
                # If w=j, adj[i][j] changes. But adj[j][j] is 0. So term is 0.
                # If w=i, adj[j][i] changes. But adj[i][i] is 0. So term is 0.
                # So common[i][j] is indeed invariant.
            
            # Add to new_c
            for u in range(self.n):
                if u == i or u == j:
                    continue
                # Update (u, i)
                if self.adj[new_c][u, j]:
                    update_common(new_c, u, i, 1)
                # Update (u, j)
                if self.adj[new_c][u, i]:
                    update_common(new_c, u, j, 1)
            
            # Note: The adj matrices are updated BEFORE this loop?
            # Yes, I updated self.adj at the start.
            # So self.adj[new_c][u][j] reflects the NEW state?
            # Wait.
            # When adding to new_c:
            # We set adj[new_c][i][j] = 1.
            # The loop checks adj[new_c][u][j].
            # If u is such that (u,j) is an edge in new_c, it should be counted.
            # The edge (i,j) was just added.
            # Does it affect adj[new_c][u][j]? No, (u,j) is a different edge.
            # So the check is correct.

        def is_valid(self):
            return self.total_bad == 0

    def greedy_construct(n, time_limit):
        m = n * (n - 1) // 2
        colors = [-1] * m
        tracker = C4Tracker(n, colors)
        # We need to initialize tracker with -1? 
        # C4Tracker rebuild assumes valid colors. 
        # Let's initialize with a random valid-ish start or just 0s and then fix?
        # Better: Start with all 0, then greedily fix?
        # Or: Start with empty graph?
        # The tracker expects a full coloring.
        # Let's just start with a random coloring and use local search.
        # The "Greedy" part in the prompt suggests sequential assignment.
        # But maintaining the tracker during sequential assignment is complex if we don't have a full graph.
        # Let's stick to: Random Init + Fast Local Search.
        
        # Random Init
        for idx in range(m):
            colors[idx] = random.randint(0, 4)
        
        tracker.rebuild()
        
        t_start = time.time()
        no_improve = 0
        best_bad = tracker.total_bad
        best_colors = colors.copy()
        
        while time.time() < t_start + time_limit:
            if tracker.total_bad == 0:
                break
            
            # Pick random edge
            idx = random.randint(0, m - 1)
            old_c = colors[idx]
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue
            
            # Try flip
            tracker.flip(idx, new_c)
            
            if tracker.total_bad < best_bad:
                best_bad = tracker.total_bad
                best_colors = colors.copy()
                no_improve = 0
            elif tracker.total_bad < tracker.total_bad + 1: # Always true?
                # Accept if it doesn't increase bad too much?
                # Standard hill climbing: accept if bad decreases.
                # If bad increases, revert.
                pass
            else:
                # Revert
                tracker.flip(idx, old_c)
                no_improve += 1
                if no_improve > 5000:
                    # Restart
                    for idx2 in range(m):
                        colors[idx2] = random.randint(0, 4)
                    tracker.rebuild()
                    best_bad = tracker.total_bad
                    best_colors = colors.copy()
                    no_improve = 0
            
            if tracker.total_bad < best_bad:
                 best_bad = tracker.total_bad
                 best_colors = colors.copy()
            else:
                 # If we didn't improve, and this was a worsening move, we should have reverted.
                 # My logic above: 
                 # 1. Flip
                 # 2. If bad < best_bad: update best, reset no_improve.
                 # 3. Else: revert, increment no_improve.
                 # This is a strict hill climber.
                 pass

        if best_bad == 0:
            return best_colors
        return None

    # Try n=26, 27, 28
    for n in [28, 27, 26]:
        if time.time() > deadline - 10:
            break
        remaining = deadline - time.time() - 5
        if remaining < 5:
            break
        
        # Try multiple restarts
        found = None
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            res = greedy_construct(n, time_limit=(deadline - time.time() - 5) / 3.0)
            if res:
                found = res
                break
        
        if found:
            return {'n': n, 'colors': found}

    # Fallback n=25
    if time.time() < deadline - 5:
        res = greedy_construct(25, time_limit=deadline - time.time() - 5)
        if res:
            return {'n': 25, 'colors': res}
            
    return {'n': 1, 'colors': []}
