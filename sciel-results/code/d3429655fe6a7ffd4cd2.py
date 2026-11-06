import time
import random
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 20
    total_pairs = n * (n - 1) // 2
    
    def is_valid_addition(adj, u, v):
        if adj[u] & (1 << v):
            return False
        
        # Check triples containing u: {u, a, b}
        # If v is in N(a) and N(b), then v is added to N({u, a, b}).
        # We need old |N({u, a, b})| <= 1.
        for a in range(n):
            if a == u or a == v:
                continue
            if not (adj[a] & (1 << v)):
                continue
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                if not (adj[b] & (1 << v)):
                    continue
                
                count = (adj[u] & adj[a] & adj[b]).bit_count()
                if count > 1:
                    return False
                        
        # Check triples containing v: {v, a, b}
        # If u is in N(a) and N(b), then u is added to N({v, a, b}).
        for a in range(n):
            if a == u or a == v:
                continue
            if not (adj[a] & (1 << u)):
                continue
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                if not (adj[b] & (1 << u)):
                    continue
                
                count = (adj[v] & adj[a] & adj[b]).bit_count()
                if count > 1:
                    return False
                        
        return True

    def count_edges(adj):
        return sum(x.bit_count() for x in adj) // 2

    def greedy_construct(random_seed):
        random.seed(random_seed)
        adj = [0] * n
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if is_valid_addition(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
        return adj

    def spectral_construct(random_seed):
        """Initialize graph using spectral heuristic on a random symmetric matrix.
        Threshold the top eigenvector to create a dense, structured starting point."""
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Generate a random symmetric matrix
        A = np.random.randn(n, n)
        A = (A + A.T) / 2.0
        
        # Compute eigenvalues and eigenvectors
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(A)
        except:
            return greedy_construct(random_seed)
        
        # Use the top eigenvector (corresponding to the largest eigenvalue)
        top_eigvec = eigenvectors[:, -1]
        
        # Threshold at the mean of the squared values to create a balanced density
        # This creates a dense random-like starting point
        threshold = np.mean(top_eigvec ** 2)
        
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                # Use the product of components as a score
                score = top_eigvec[i] * top_eigvec[j]
                # Threshold: if score is above a certain level, add edge
                # We want roughly 60-70% density for a good start
                if score > -0.1:  # Adjusted threshold for density
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Clean up: remove invalid edges by checking validity
        # We do this by starting from empty and re-adding valid edges from the spectral candidate
        clean_adj = [0] * n
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if adj[u] & (1 << v):
                if is_valid_addition(clean_adj, u, v):
                    clean_adj[u] |= (1 << v)
                    clean_adj[v] |= (1 << u)
                    
        return clean_adj

    def local_search(adj, max_steps=200, use_kick=False):
        current_adj = adj[:]
        current_edges = count_edges(current_adj)
        best_adj = current_adj[:]
        best_edges = current_edges
        
        # Phase 1: Greedy Add
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            pairs = [(i, j) for i in range(n) for j in range(i+1, n) if not (current_adj[i] & (1 << j))]
            random.shuffle(pairs)
            
            for u, v in pairs:
                if time.time() > deadline:
                    break
                if is_valid_addition(current_adj, u, v):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_edges += 1
                    improved = True
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = current_adj[:]
            
        # Phase 2: Edge Swaps (1 in, 2 out)
        steps = 0
        while steps < max_steps:
            if time.time() > deadline:
                break
            steps += 1
            
            edges = [(i, j) for i in range(n) for j in range(i+1, n) if current_adj[i] & (1 << j)]
            if not edges:
                break
            random.shuffle(edges)
            
            found_swap = False
            for u, v in edges:
                if time.time() > deadline:
                    break
                if found_swap:
                    break
                
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
                
                added = []
                missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (current_adj[i] & (1 << j)) and (i != u or j != v) and (i != v or j != u)]
                random.shuffle(missing)
                
                for a, b in missing:
                    if time.time() > deadline:
                        break
                    if is_valid_addition(current_adj, a, b):
                        current_adj[a] |= (1 << b)
                        current_adj[b] |= (1 << a)
                        added.append((a, b))
                        if len(added) == 2:
                            break
                
                if len(added) == 2:
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = current_adj[:]
                    found_swap = True
                elif len(added) == 1:
                    current_edges += 0
                    found_swap = True
                else:
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    
        # Phase 3: Kick + Greedy Fill (if enabled)
        if use_kick:
            for _ in range(3):
                if time.time() > deadline:
                    break
                
                # Kick: randomly remove and add some edges
                kick_adj = current_adj[:]
                kick_edges = current_edges
                
                # Remove 2-4 random edges
                kick_edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if kick_adj[i] & (1 << j)]
                random.shuffle(kick_edges_list)
                for u, v in kick_edges_list[:random.randint(2, 4)]:
                    kick_adj[u] &= ~(1 << v)
                    kick_adj[v] &= ~(1 << u)
                    kick_edges -= 1
                
                # Greedy fill
                improved = True
                while improved:
                    if time.time() > deadline:
                        break
                    improved = False
                    pairs = [(i, j) for i in range(n) for j in range(i+1, n) if not (kick_adj[i] & (1 << j))]
                    random.shuffle(pairs)
                    
                    for u, v in pairs:
                        if time.time() > deadline:
                            break
                        if is_valid_addition(kick_adj, u, v):
                            kick_adj[u] |= (1 << v)
                            kick_adj[v] |= (1 << u)
                            kick_edges += 1
                            improved = True
                            if kick_edges > best_edges:
                                best_edges = kick_edges
                                best_adj = kick_adj[:]
                
                if kick_edges >= current_edges:
                    current_adj = kick_adj
                    current_edges = kick_edges
        
        return best_adj, best_edges

    def perturb_adj(adj, num_remove, num_add):
        """Perturb the graph by removing some edges and adding random valid ones."""
        perturbed = adj[:]
        
        # Remove edges
        edges = [(i, j) for i in range(n) for j in range(i+1, n) if perturbed[i] & (1 << j)]
        random.shuffle(edges)
        for u, v in edges[:num_remove]:
            perturbed[u] &= ~(1 << v)
            perturbed[v] &= ~(1 << u)
        
        # Add random valid edges
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (perturbed[i] & (1 << j))]
        random.shuffle(missing)
        
        added = 0
        for u, v in missing:
            if added >= num_add:
                break
            if is_valid_addition(perturbed, u, v):
                perturbed[u] |= (1 << v)
                perturbed[v] |= (1 << u)
                added += 1
                
        return perturbed

    best_adj = [0] * n
    best_edges = 0
    
    # Run multiple restarts with mixed initialization strategies
    num_restarts = 300
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Alternate between greedy and spectral initialization
        if i % 3 == 0:
            adj = spectral_construct(i * 11 + 1)
        else:
            adj = greedy_construct(i * 13 + 1)
        
        # Local search with kick
        adj, edges = local_search(adj, max_steps=150, use_kick=True)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
            
    # Final polish
    if time.time() < deadline - 5:
        for _ in range(10):
            if time.time() > deadline - 2:
                break
            adj, edges = local_search(best_adj, max_steps=300, use_kick=True)
            if edges > best_edges:
                best_edges = edges
                best_adj = adj[:]
            else:
                break

    result_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                result_edges.append([i, j])
                
    return result_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
