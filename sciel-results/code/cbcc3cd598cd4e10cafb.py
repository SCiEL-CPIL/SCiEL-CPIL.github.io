import random
import time
import itertools
from collections import Counter

def construct():
    """
    Constructs a graph on n vertices with no regular induced subgraph on k >= 6 vertices.
    Uses a local search approach with efficient violation counting.
    """
    start_time = time.time()
    time_limit = 145.0  # 5s buffer for 150s limit

    def check_violations(adj, n):
        """
        Counts regular induced subgraphs of size k >= 6.
        Returns (count, list_of_masks) - only first 50 masks stored for memory.
        """
        count = 0
        masks = []
        # For each subset of size >= 6, check if regular
        # Optimize by checking k=6 first (most common), then larger
        for k in range(6, n + 1):
            for combo in itertools.combinations(range(n), k):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                
                # Check if regular: all vertices in mask have same degree within mask
                # Get degrees of all vertices in mask
                degrees = []
                m = mask
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    degrees.append((adj[v] & mask).bit_count())
                    m ^= lsb
                
                if len(set(degrees)) == 1:
                    count += 1
                    if len(masks) < 50:
                        masks.append(mask)
                    if count > 200:  # Early exit for performance
                        return count, masks
        
        return count, masks

    def local_search(n, initial_adj, deadline):
        """
        Local search: minimize violation count by flipping edges.
        Returns (success, final_adj)
        """
        current_adj = initial_adj[:]
        best_adj = current_adj[:]
        
        # Initial count
        best_count, _ = check_violations(current_adj, n)
        
        if best_count == 0:
            return True, current_adj
        
        stuck_count = 0
        max_stuck = 500
        
        while time.time() < deadline:
            count, masks = check_violations(current_adj, n)
            
            if count == 0:
                return True, current_adj
            
            if count < best_count:
                best_adj = current_adj[:]
                best_count = count
                stuck_count = 0
            else:
                stuck_count += 1
            
            if stuck_count > max_stuck:
                break
            
            # Pick a random violation
            if not masks:
                break
            mask = random.choice(masks)
            
            # Get vertices in violation
            verts = []
            m = mask
            while m:
                lsb = m & -m
                verts.append(lsb.bit_length() - 1)
                m ^= lsb
            
            if len(verts) < 2:
                continue
            
            # Strategy: flip an edge between two vertices in the violation
            # Or flip an edge from a violation vertex to a non-violation vertex
            # Try both strategies randomly
            if random.random() < 0.7:
                # Flip edge within violation
                u, v = random.sample(verts, 2)
            else:
                # Flip edge from violation vertex to outside
                u = random.choice(verts)
                outside = [i for i in range(n) if i not in set(verts)]
                if not outside:
                    u, v = random.sample(verts, 2)
                else:
                    v = random.choice(outside)
            
            # Flip edge u-v
            if (current_adj[u] >> v) & 1:
                current_adj[u] ^= (1 << v)
                current_adj[v] ^= (1 << u)
            else:
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
        
        # Return best found
        if best_count == 0:
            return True, best_adj
        return False, best_adj

    def greedy_extend(base_adj, base_n, target_n, deadline):
        """
        Try to extend a valid graph from base_n to target_n by adding vertices one at a time.
        Returns (success, final_adj)
        """
        current_adj = base_adj[:]
        current_n = base_n
        
        while current_n < target_n:
            if time.time() > deadline:
                break
            
            found = False
            for attempt in range(200):
                if time.time() > deadline:
                    break
                
                # Create new vertex with random connections
                new_adj = current_adj[:] + [0]
                for i in range(current_n):
                    if random.random() < 0.45:  # Slightly sparse
                        new_adj[i] |= (1 << current_n)
                        new_adj[current_n] |= (1 << i)
                
                count, _ = check_violations(new_adj, current_n + 1)
                if count == 0:
                    current_adj = new_adj
                    current_n += 1
                    found = True
                    break
            
            if not found:
                # Try with different random seed / restart
                continue
        
        if current_n == target_n:
            count, _ = check_violations(current_adj, target_n)
            if count == 0:
                return True, current_adj
        
        return False, current_adj

    best_result = {'n': 1, 'edges': []}
    
    # Try to find valid graphs for n = 20, 21, 22
    for target_n in [20, 21, 22]:
        if time.time() > time_limit - 15:
            break
        
        success = False
        
        # Strategy 1: Random initialization + local search
        max_restarts = 15 if target_n <= 20 else 8
        for r in range(max_restarts):
            if time.time() > time_limit - 10:
                break
            
            # Random graph with p=0.5
            adj = [0] * target_n
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            ok, final_adj = local_search(target_n, adj, time_limit - 5)
            
            if ok:
                count, _ = check_violations(final_adj, target_n)
                if count == 0:
                    edges = []
                    for i in range(target_n):
                        for j in range(i + 1, target_n):
                            if (final_adj[i] >> j) & 1:
                                edges.append([i, j])
                    best_result = {'n': target_n, 'edges': edges}
                    success = True
                    break
        
        if success and target_n == 22:
            break
        
        # Strategy 2: If we have a smaller valid graph, try greedy extension
        if not success and best_result['n'] >= 15 and time.time() < time_limit - 10:
            # Build a valid graph up to best_result['n'] first
            base_adj = [0] * best_result['n']
            for i in range(best_result['n']):
                for j in range(i + 1, best_result['n']):
                    if [i, j] in best_result['edges'] or [j, i] in best_result['edges']:
                        base_adj[i] |= (1 << j)
                        base_adj[j] |= (1 << i)
            
            ok, final_adj = greedy_extend(base_adj, best_result['n'], target_n, time_limit - 5)
            if ok:
                edges = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if (final_adj[i] >> j) & 1:
                            edges.append([i, j])
                best_result = {'n': target_n, 'edges': edges}
                success = True
                if target_n == 22:
                    break
    
    # Fallback: ensure we return at least n=1
    if best_result['n'] < 1:
        best_result = {'n': 1, 'edges': []}
    
    return best_result

if __name__ == "__main__":
    construct()
