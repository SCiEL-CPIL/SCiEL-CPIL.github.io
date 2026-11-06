"""Maximum Cut via Spectral Initialization, Fast 1-flip + 2-flip Local Search, and Simulated Annealing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
import math

def partition(adjacency):
    """
    Computes a maximum cut partition for the given adjacency matrix.
    Uses spectral initialization, fast local search with 1-flip and 2-flip moves,
    and simulated annealing for exploration.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    adj = np.asarray(adjacency, dtype=np.int8)
    n = len(adj)
    
    if n == 0:
        return np.array([], dtype=int)
    
    # Precompute neighbor lists
    neighbors = [list(np.nonzero(adj[i])[0]) for i in range(n)]
    degrees = np.array([len(neighbors[i]) for i in range(n)])
    
    # Helper to calculate cut size
    def cut_size(labels):
        # Count edges where endpoints have different labels
        total = 0
        for i in range(n):
            for j in neighbors[i]:
                if i < j and labels[i] != labels[j]:
                    total += 1
        return total
    
    # Compute 1-flip gain for vertex v given current labels
    def flip1_gain(v, labels):
        lv = labels[v]
        gain = 0
        for u in neighbors[v]:
            if labels[u] == lv:
                gain += 1  # flipping v moves this edge into cut
            else:
                gain -= 1  # flipping v moves this edge out of cut
        return gain
    
    # Compute 2-flip gain for pair (u, v) given current labels and 1-flip gains
    def flip2_gain(u, v, labels, g1):
        if u == v:
            return 0
        lu = labels[u]
        lv = labels[v]
        auv = adj[u, v]
        # Formula: g2 = g1[u] + g1[v] - 2 * auv * sign
        # where sign = +1 if same label, -1 if different
        if lu == lv:
            sign = 1
        else:
            sign = -1
        return g1[u] + g1[v] - 2 * auv * sign
    
    # Fast local search with 1-flip and 2-flip moves
    def local_search(labels, time_budget=10.0):
        labels = labels.copy()
        best_labels = labels.copy()
        best_cut = cut_size(best_labels)
        
        # Precompute 1-flip gains
        g1 = np.array([flip1_gain(v, labels) for v in range(n)])
        
        # Build heap for 1-flips
        heap1 = []
        for v in range(n):
            if g1[v] > 0:
                heapq.heappush(heap1, (-g1[v], v))
        
        # Build heap for 2-flips (only consider pairs with positive gain)
        heap2 = []
        in_heap2 = set()
        for u in range(n):
            for v in neighbors[u]:
                if u < v:
                    g2 = flip2_gain(u, v, labels, g1)
                    if g2 > 0:
                        key = (u, v)
                        if key not in in_heap2:
                            heapq.heappush(heap2, (-g2, u, v))
                            in_heap2.add(key)
        
        local_start = time.time()
        max_iter = 100000
        iter_count = 0
        
        while time.time() - local_start < time_budget and iter_count < max_iter:
            iter_count += 1
            
            # Try 1-flip first if available
            best_move = None
            best_gain = 0
            
            # Check 1-flip
            while heap1:
                neg_g, v = heapq.heappop(heap1)
                g = g1[v]
                if g == -neg_g and g > 0:
                    best_move = ('flip1', v, 0)
                    best_gain = g
                    break
                # Stale entry, skip
                if g > 0:
                    heapq.heappush(heap1, (-g, v))
                    break
            else:
                heap1 = []
            
            # Check 2-flip if no positive 1-flip
            if best_move is None:
                while heap2:
                    neg_g, u, v = heapq.heappop(heap2)
                    key = (u, v)
                    if key not in in_heap2:
                        continue
                    in_heap2.discard(key)
                    g2 = flip2_gain(u, v, labels, g1)
                    if g2 == -neg_g and g2 > 0:
                        best_move = ('flip2', u, v)
                        best_gain = g2
                        break
                    elif g2 > 0:
                        heapq.heappush(heap2, (-g2, u, v))
                        in_heap2.add(key)
                        break
                else:
                    heap2 = []
            
            if best_move is None:
                break
            
            # Apply the move
            if best_move[0] == 'flip1':
                v = best_move[1]
                labels[v] = 1 - labels[v]
                old_g1_v = g1[v]
                g1[v] = -old_g1_v  # flipping changes sign of own gain
                
                # Update g1 for neighbors
                affected = [v]
                for u in neighbors[v]:
                    # g1[u] changes by ±2 depending on whether v and u were same/diff
                    if labels[u] != labels[v]:  # after flip, they are different
                        # before flip, they were same
                        g1[u] += 2
                    else:
                        # before flip, they were different
                        g1[u] -= 2
                    affected.append(u)
                
                best_cut += best_gain
                
            else:  # flip2
                u, v = best_move[1], best_move[2]
                labels[u] = 1 - labels[u]
                labels[v] = 1 - labels[v]
                
                # Update g1 for u, v and their neighbors
                # This is complex; recompute g1 for affected vertices
                affected = set([u, v])
                for x in [u, v]:
                    for y in neighbors[x]:
                        affected.add(y)
                
                for x in affected:
                    g1[x] = flip1_gain(x, labels)
                
                best_cut += best_gain
            
            if best_cut > cut_size(best_labels):
                best_labels = labels.copy()
            
            # Rebuild heaps from scratch for correctness (simpler but slower)
            # For better performance, we could update incrementally, but rebuilding
            # is safe. Given n=100, this is acceptable.
            heap1 = []
            for w in range(n):
                if g1[w] > 0:
                    heapq.heappush(heap1, (-g1[w], w))
            
            heap2 = []
            in_heap2 = set()
            for w1 in range(n):
                for w2 in neighbors[w1]:
                    if w1 < w2:
                        g2 = flip2_gain(w1, w2, labels, g1)
                        if g2 > 0:
                            key = (w1, w2)
                            if key not in in_heap2:
                                heapq.heappush(heap2, (-g2, w1, w2))
                                in_heap2.add(key)
        
        # Final polish
        final_cut = cut_size(labels)
        if final_cut > best_cut:
            best_labels = labels.copy()
            best_cut = final_cut
        
        return best_labels, best_cut
    
    # Spectral initialization using Fiedler vector
    def spectral_init():
        degrees = np.sum(adj, axis=1)
        D = np.diag(degrees)
        L = D - adj.astype(float)
        
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            fiedler = eigenvectors[:, 1]
            labels = (fiedler >= 0).astype(int)
            if np.all(labels == labels[0]):
                labels = np.random.randint(0, 2, n)
            return labels
        except Exception:
            return np.random.randint(0, 2, n)
    
    # GW-style rounding with random hyperplane in eigenspace
    def gw_init():
        degrees = np.sum(adj, axis=1)
        D = np.diag(degrees)
        L = D - adj.astype(float)
        
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            k = min(5, n - 1)
            if k < 1:
                return np.random.randint(0, 2, n)
            
            weights = np.random.randn(k)
            weights /= np.linalg.norm(weights)
            
            v = np.zeros(n)
            for i in range(k):
                v += weights[i] * eigenvectors[:, 1 + i]
            
            labels = (v >= 0).astype(int)
            if np.all(labels == labels[0]):
                labels = np.random.randint(0, 2, n)
            return labels
        except Exception:
            return np.random.randint(0, 2, n)
    
    # 1. Multiple initialization strategies
    best_labels = None
    best_cut = -1
    
    # Strategy 1: Spectral (Fiedler)
    labels_spectral = spectral_init()
    ls_labels, ls_cut = local_search(labels_spectral, time_budget=8.0)
    if ls_cut > best_cut:
        best_cut = ls_cut
        best_labels = ls_labels.copy()
    
    # Strategy 2: GW rounding
    if time.time() < deadline - 30.0:
        labels_gw = gw_init()
        ls_labels, ls_cut = local_search(labels_gw, time_budget=8.0)
        if ls_cut > best_cut:
            best_cut = ls_cut
            best_labels = ls_labels.copy()
    
    # Strategy 3: Random restarts
    if time.time() < deadline - 25.0:
        for _ in range(4):
            if time.time() >= deadline - 20.0:
                break
            labels_rand = np.random.randint(0, 2, n)
            ls_labels, ls_cut = local_search(labels_rand, time_budget=6.0)
            if ls_cut > best_cut:
                best_cut = ls_cut
                best_labels = ls_labels.copy()
    
    # 2. Simulated Annealing with 1-flip and 2-flip moves
    if time.time() < deadline - 15.0:
        current_labels = best_labels.copy()
        current_cut = best_cut
        g1 = np.array([flip1_gain(v, current_labels) for v in range(n)])
        
        T_start = 15.0
        T_end = 0.1
        cooling_rate = 0.98
        T = T_start
        sa_deadline = deadline - 8.0
        
        while T > T_end and time.time() < sa_deadline:
            # Randomly choose between 1-flip and 2-flip
            if np.random.rand() < 0.6:
                # 1-flip
                v = np.random.randint(0, n)
                gain = g1[v]
                
                if gain > 0 or (T > 0 and np.random.rand() < math.exp(-gain / T)):
                    current_labels[v] = 1 - current_labels[v]
                    current_cut += gain
                    g1[v] = -gain
                    for u in neighbors[v]:
                        if current_labels[u] != current_labels[v]:
                            g1[u] += 2
                        else:
                            g1[u] -= 2
                    
                    if current_cut > best_cut:
                        best_cut = current_cut
                        best_labels = current_labels.copy()
            else:
                # 2-flip
                u = np.random.randint(0, n)
                v = np.random.randint(0, n)
                if u == v:
                    continue
                gain = flip2_gain(u, v, current_labels, g1)
                
                if gain > 0 or (T > 0 and np.random.rand() < math.exp(-gain / T)):
                    current_labels[u] = 1 - current_labels[u]
                    current_labels[v] = 1 - current_labels[v]
                    current_cut += gain
                    
                    # Recompute g1 for affected vertices
                    affected = set([u, v])
                    for x in [u, v]:
                        for y in neighbors[x]:
                            affected.add(y)
                    for x in affected:
                        g1[x] = flip1_gain(x, current_labels)
                    
                    if current_cut > best_cut:
                        best_cut = current_cut
                        best_labels = current_labels.copy()
            
            T *= cooling_rate
        
        # Final local search polish
        if time.time() < deadline - 5.0:
            ls_labels, ls_cut = local_search(best_labels, time_budget=5.0)
            if ls_cut > best_cut:
                best_cut = ls_cut
                best_labels = ls_labels.copy()
    
    # Final deadline check
    if time.time() > deadline - 2.0 and best_labels is not None:
        # Quick final local search if time permits
        pass
    
    return best_labels

# EVOLVE-BLOCK-END
