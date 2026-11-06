<<<<<<< SEARCH
import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def get_edges(adj, n):
        """Extract edge list from bitmask adjacency."""
        edges = []
        for i in range(n):
            mask = adj[i]
            j = 0
            while mask:
                if mask & 1:
                    if i < j:
                        edges.append([i, j])
                mask >>= 1
                j += 1
        return edges

    def check_valid_fast(adj, n, min_deg):
        """Check if graph is C4-free and has min degree >= min_deg. Optimized."""
        # Check min degree
        for i in range(n):
            if adj[i].bit_count() < min_deg:
                return False
        # Check C4: any two vertices share at most 1 common neighbor
        for i in range(n):
            for j in range(i + 1, n):
                common = adj[i] & adj[j]
                if common.bit_count() > 1:
                    return False
        return True

    def polarity_graph_46():
        """
        Construct a C4-free graph on 46 vertices using a polarity graph approach.
        Based on finite field construction.
        """
        n = 46
        adj = [0] * n
        # Use a construction based on quadratic residues mod a prime
        # Or use a known C4-free graph structure
        
        # Simpler: use a random greedy approach seeded to find good structures
        # But let's try a structured approach:
        # Divide vertices into groups and connect based on algebraic properties
        
        # For now, use a well-known construction:
        # Take vertices as elements of Z_46, connect i-j if (i-j) is a quadratic residue mod 43 or similar
        # This is complex, so let's use a different strategy:
        # Start with a known small C4-free graph and extend
        
        # Actually, let's use a greedy approach with specific ordering for n=46
        # The key insight: for n=46, min_deg needed is 7
        # This is relatively low, so greedy should work well
        
        return None  # We'll use greedy_build for this

    def greedy_build(n, target_deg, seed=None):
        """
        Greedily construct a C4-free graph on n vertices aiming for min degree >= target_deg.
        Returns adj list of bitmasks or None if failed.
        Uses priority-based approach: always connect the lowest-degree vertex first.
        """
        if seed is not None:
            random.seed(seed)
        
        adj = [0] * n
        deg = [0] * n
        
        # Priority-based greedy: repeatedly pick the vertex with minimum degree
        # and try to add edges to it
        for _ in range(n * n * 2):  # Upper bound on iterations
            if time.time() > deadline:
                break
            
            # Find vertex with minimum degree
            u = min(range(n), key=lambda i: deg[i])
            
            if deg[u] >= target_deg:
                # Check if all vertices meet target
                if min(deg) >= target_deg:
                    return adj
                # Otherwise continue to boost others
                # Find next lowest
                u = min(range(n), key=lambda i: deg[i])
            
            # Find candidates for u
            candidates = []
            for v in range(n):
                if u == v:
                    continue
                if adj[u] & (1 << v):
                    continue
                common = adj[u] & adj[v]
                if common == 0:
                    candidates.append(v)
            
            if candidates:
                # Sort by degree (ascending) to balance
                candidates.sort(key=lambda x: deg[x])
                # Add edges to all valid candidates (safe for C4-freeness)
                for v in candidates:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    deg[u] += 1
                    deg[v] += 1
            else:
                # No candidates for u, try next vertex
                continue
        
        if min(deg) >= target_deg:
            return adj
        return None

    def local_search_with_repair(adj, n, target_deg, max_time=10.0):
        """
        Improve min degree using edge additions and C4 repair via edge removal.
        Returns (adj, min_deg).
        """
        deg = [0] * n
        for i in range(n):
            deg[i] = adj[i].bit_count()
        
        best_min_deg = min(deg)
        best_adj = adj[:]
        
        start = time.time()
        no_improve = 0
        
        while time.time() - start < max_time:
            if no_improve > 300:
                break
            
            # First, check for and fix C4 violations
            fixed = True
            while fixed:
                fixed = False
                for i in range(n):
                    for j in range(i + 1, n):
                        common = adj[i] & adj[j]
                        if common.bit_count() > 1:
                            # Find common neighbors
                            common_list = []
                            mask = common
                            k = 0
                            while mask:
                                if mask & 1:
                                    common_list.append(k)
                                mask >>= 1
                                k += 1
                            
                            # Remove edges to reduce common neighbors to 1
                            # Remove from the vertex with higher degree to preserve min degree
                            for w in common_list[1:]:
                                if deg[i] >= deg[j]:
                                    adj[i] &= ~(1 << w)
                                    adj[w] &= ~(1 << i)
                                    deg[i] -= 1
                                else:
                                    adj[j] &= ~(1 << w)
                                    adj[w] &= ~(1 << j)
                                    deg[j] -= 1
                            fixed = True
                            break
                    if fixed:
                        break
            
            # Now try to add edges to boost min degree
            u = min(range(n), key=lambda i: deg[i])
            
            if deg[u] >= target_deg:
                if best_min_deg >= target_deg and check_valid_fast(adj, n, target_deg):
                    return adj, best_min_deg
                if min(deg) >= target_deg and check_valid_fast(adj, n, target_deg):
                    return adj, best_min_deg
            
            # Find candidates for u
            candidates = []
            for v in range(n):
                if u == v:
                    continue
                if adj[u] & (1 << v):
                    continue
                common = adj[u] & adj[v]
                if common == 0:
                    candidates.append(v)
            
            if candidates:
                candidates.sort(key=lambda x: deg[x])
                v = candidates[0]  # Pick lowest degree candidate
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
                
                new_min = min(deg)
                if new_min > best_min_deg:
                    best_min_deg = new_min
                    best_adj = adj[:]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Try edge swap: remove (u,w) and add (u,v)
                neighbors = []
                mask = adj[u]
                k = 0
                while mask:
                    if mask & 1:
                        neighbors.append(k)
                    mask >>= 1
                    k += 1
                
                if neighbors:
                    w = random.choice(neighbors)
                    adj[u] &= ~(1 << w)
                    adj[w] &= ~(1 << u)
                    deg[u] -= 1
                    deg[w] -= 1
                    
                    candidates2 = []
                    for v in range(n):
                        if u == v:
                            continue
                        if adj[u] & (1 << v):
                            continue
                        common = adj[u] & adj[v]
                        if common == 0:
                            candidates2.append(v)
                    
                    if candidates2:
                        candidates2.sort(key=lambda x: deg[x])
                        v = candidates2[0]
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        deg[u] += 1
                        deg[v] += 1
                        
                        new_min = min(deg)
                        if new_min > best_min_deg:
                            best_min_deg = new_min
                            best_adj = adj[:]
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        # Revert
                        adj[u] |= (1 << w)
                        adj[w] |= (1 << u)
                        deg[u] += 1
                        deg[w] += 1
                        no_improve += 1
                else:
                    no_improve += 1
        
        return best_adj, best_min_deg

    # Try n from 47 down to 43
    for n in range(47, 42, -1):
        min_deg = n - 39
        if min_deg < 1:
            continue
        
        # Strategy 2: Priority-based Greedy Build (multiple attempts)
        for attempt in range(500):
            if time.time() > deadline:
                break
            
            adj = greedy_build(n, min_deg, seed=attempt * 7919 + 13)
            
            if adj is not None:
                if check_valid_fast(adj, n, min_deg):
                    edges = get_edges(adj, n)
                    return {'n': n, 'edges': edges}
        
        # Strategy 2: Local search with repair from greedy results
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            adj = greedy_build(n, min_deg, seed=attempt * 104729 + 7)
            
            if adj is not None:
                adj_ls, min_deg_ls = local_search_with_repair(adj, n, min_deg, max_time=3.0)
                if min_deg_ls >= min_deg and check_valid_fast(adj_ls, n, min_deg):
                    edges = get_edges(adj_ls, n)
                    return {'n': n, 'edges': edges}
        
        # Strategy 3: Random Initialization + Repair
        for attempt in range(30):
            if time.time() > deadline:
                break
            
            random.seed(attempt * 15485863 + 1)
            
            # Initialize with random graph
            p = min(min_deg / n * 1.5, 0.6)
            adj = [0] * n
            
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Repair phase
            for _ in range(200):
                if time.time() > deadline:
                    break
                
                # Check for C4s and fix
                fixed_c4 = False
                for i in range(n):
                    for j in range(i + 1, n):
                        common = adj[i] & adj[j]
                        if common.bit_count() > 1:
                            common_list = []
                            mask = common
                            k = 0
                            while mask:
                                if mask & 1:
                                    common_list.append(k)
                                mask >>= 1
                                k += 1
                            
                            for w in common_list[1:]:
                                deg_i = adj[i].bit_count()
                                deg_j = adj[j].bit_count()
                                if deg_i >= deg_j:
                                    adj[i] &= ~(1 << w)
                                    adj[w] &= ~(1 << i)
                                else:
                                    adj[j] &= ~(1 << w)
                                    adj[w] &= ~(1 << j)
                            fixed_c4 = True
                            break
                    if fixed_c4:
                        break
                
                if not fixed_c4:
                    # No C4s, check min degree
                    min_deg_actual = min(adj[i].bit_count() for i in range(n))
                    if min_deg_actual >= min_deg:
                        if check_valid_fast(adj, n, min_deg):
                            edges = get_edges(adj, n)
                            return {'n': n, 'edges': edges}
                        break
                    else:
                        # Need to add edges
                        low_verts = [i for i in range(n) if adj[i].bit_count() < min_deg]
                        if not low_verts:
                            break
                        u = min(low_verts, key=lambda i: adj[i].bit_count())
                        candidates = []
                        for v in range(n):
                            if v == u:
                                continue
                            if not (adj[u] & (1 << v)):
                                common = adj[u] & adj[v]
                                if common == 0:
                                    candidates.append(v)
                        
                        if candidates:
                            candidates.sort(key=lambda x: adj[x].bit_count())
                            v = candidates[0]
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                        else:
                            pass

    # Fallback: Return a valid small graph
    n = 40
    min_deg = 1
    adj = [0] * n
    for i in range(n):
        adj[i] |= (1 << ((i + 1) % n))
        adj[(i + 1) % n] |= (1 << i)
    
    if check_valid_fast(adj, n, min_deg):
        edges = get_edges(adj, n)
        return {'n': n, 'edges': edges}
    
    return {'n': 39, 'edges': []}
=======
import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def get_edges(adj, n):
        """Extract edge list from bitmask adjacency."""
        edges = []
        for i in range(n):
            mask = adj[i]
            j = 0
            while mask:
                if mask & 1:
                    if i < j:
                        edges.append([i, j])
                mask >>= 1
                j += 1
        return edges

    def check_valid_fast(adj, n, min_deg):
        """Check if graph is C4-free and has min degree >= min_deg. Optimized."""
        # Check min degree
        for i in range(n):
            if adj[i].bit_count() < min_deg:
                return False
        # Check C4: any two vertices share at most 1 common neighbor
        for i in range(n):
            for j in range(i + 1, n):
                common = adj[i] & adj[j]
                if common.bit_count() > 1:
                    return False
        return True

    def algebraic_c4_free(n, seed=0):
        """
        Construct a C4-free graph using algebraic/structured approach.
        For n=46, min_deg needed is 7. We use a block design approach:
        Partition vertices into groups and connect based on finite geometry.
        """
        adj = [0] * n
        
        # Strategy: use a "polarity" type construction
        # For small n like 46, we can use a greedy structured approach
        # where we build the graph vertex by vertex with careful neighbor selection
        
        # Alternative: use quadratic residues modulo a prime
        # For p=43 (prime), connect i to j if (i-j) mod p is a QR
        # This gives a C4-free graph on p+1 or p vertices
        
        # Let's try: vertices 0..n-1, connect i to j if some algebraic condition
        # Simple approach: use a known C4-free construction
        
        # For n=46, we need min degree 7. Let's use a greedy with structure:
        # Assign each vertex a "type" and connect based on type compatibility
        
        # Actually, let's use a more direct approach:
        # Start with empty graph, add edges greedily but with structured ordering
        
        deg = [0] * n
        
        # Process vertices in a specific order for better balance
        order = list(range(n))
        random.seed(seed)
        random.shuffle(order)
        
        for u in order:
            # Find vertices that can be connected to u without creating C4
            candidates = []
            for v in range(n):
                if v == u:
                    continue
                if adj[u] & (1 << v):
                    continue
                common = adj[u] & adj[v]
                if common == 0:
                    candidates.append(v)
            
            # Sort candidates by current degree (lowest first for balance)
            candidates.sort(key=lambda x: deg[x])
            
            # Add edges to boost degree of u
            for v in candidates:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
        
        return adj

    def greedy_build(n, target_deg, seed=None):
        """
        Greedily construct a C4-free graph on n vertices aiming for min degree >= target_deg.
        Returns adj list of bitmasks or None if failed.
        Uses priority-based approach: always connect the lowest-degree vertex first.
        """
        if seed is not None:
            random.seed(seed)
        
        adj = [0] * n
        deg = [0] * n
        
        # Priority-based greedy: repeatedly pick the vertex with minimum degree
        # and try to add edges to it
        for _ in range(n * n * 3):  # Upper bound on iterations
            if time.time() > deadline:
                break
            
            # Find vertex with minimum degree
            min_d = min(deg)
            if min_d >= target_deg:
                return adj
            
            # Find all vertices below target
            low_verts = [i for i in range(n) if deg[i] < target_deg]
            u = min(low_verts, key=lambda i: deg[i])
            
            # Find candidates for u
            candidates = []
            for v in range(n):
                if u == v:
                    continue
                if adj[u] & (1 << v):
                    continue
                common = adj[u] & adj[v]
                if common == 0:
                    candidates.append(v)
            
            if candidates:
                # Sort by degree (ascending) to balance
                candidates.sort(key=lambda x: deg[x])
                # Add edges to all valid candidates (safe for C4-freeness)
                for v in candidates:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    deg[u] += 1
                    deg[v] += 1
            else:
                # No candidates for u, try next vertex
                continue
        
        if min(deg) >= target_deg:
            return adj
        return None

    def local_search_with_repair(adj, n, target_deg, max_time=10.0):
        """
        Improve min degree using edge additions and C4 repair via edge removal.
        Returns (adj, min_deg).
        """
        deg = [0] * n
        for i in range(n):
            deg[i] = adj[i].bit_count()
        
        best_min_deg = min(deg)
        best_adj = adj[:]
        
        start = time.time()
        no_improve = 0
        
        while time.time() - start < max_time:
            if no_improve > 500:
                break
            
            # First, check for and fix C4 violations
            fixed = True
            while fixed:
                fixed = False
                for i in range(n):
                    for j in range(i + 1, n):
                        common = adj[i] & adj[j]
                        if common.bit_count() > 1:
                            # Find common neighbors
                            common_list = []
                            mask = common
                            k = 0
                            while mask:
                                if mask & 1:
                                    common_list.append(k)
                                mask >>= 1
                                k += 1
                            
                            # Remove edges to reduce common neighbors to 1
                            # Remove from the vertex with higher degree to preserve min degree
                            for w in common_list[1:]:
                                if deg[i] >= deg[j]:
                                    adj[i] &= ~(1 << w)
                                    adj[w] &= ~(1 << i)
                                    deg[i] -= 1
                                else:
                                    adj[j] &= ~(1 << w)
                                    adj[w] &= ~(1 << j)
                                    deg[j] -= 1
                            fixed = True
                            break
                    if fixed:
                        break
            
            # Now try to add edges to boost min degree
            min_d = min(deg)
            if min_d >= target_deg:
                if check_valid_fast(adj, n, target_deg):
                    return adj, min_d
            
            u = min(range(n), key=lambda i: deg[i])
            
            # Find candidates for u
            candidates = []
            for v in range(n):
                if u == v:
                    continue
                if adj[u] & (1 << v):
                    continue
                common = adj[u] & adj[v]
                if common == 0:
                    candidates.append(v)
            
            if candidates:
                candidates.sort(key=lambda x: deg[x])
                v = candidates[0]  # Pick lowest degree candidate
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
                
                new_min = min(deg)
                if new_min > best_min_deg:
                    best_min_deg = new_min
                    best_adj = adj[:]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Try edge swap: remove (u,w) and add (u,v)
                neighbors = []
                mask = adj[u]
                k = 0
                while mask:
                    if mask & 1:
                        neighbors.append(k)
                    mask >>= 1
                    k += 1
                
                if neighbors:
                    w = random.choice(neighbors)
                    adj[u] &= ~(1 << w)
                    adj[w] &= ~(1 << u)
                    deg[u] -= 1
                    deg[w] -= 1
                    
                    candidates2 = []
                    for v in range(n):
                        if u == v:
                            continue
                        if adj[u] & (1 << v):
                            continue
                        common = adj[u] & adj[v]
                        if common == 0:
                            candidates2.append(v)
                    
                    if candidates2:
                        candidates2.sort(key=lambda x: deg[x])
                        v = candidates2[0]
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        deg[u] += 1
                        deg[v] += 1
                        
                        new_min = min(deg)
                        if new_min > best_min_deg:
                            best_min_deg = new_min
                            best_adj = adj[:]
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        # Revert
                        adj[u] |= (1 << w)
                        adj[w] |= (1 << u)
                        deg[u] += 1
                        deg[w] += 1
                        no_improve += 1
                else:
                    no_improve += 1
        
        return best_adj, best_min_deg

    # Try n from 47 down to 43
    for n in range(47, 42, -1):
        min_deg = n - 39
        if min_deg < 1:
            continue
        
        # Strategy 1: Algebraic/structured construction
        for attempt in range(50):
            if time.time() > deadline:
                break
            adj = algebraic_c4_free(n, seed=attempt * 1000003 + 7)
            if check_valid_fast(adj, n, min_deg):
                edges = get_edges(adj, n)
                return {'n': n, 'edges': edges}
        
        # Strategy 2: Priority-based Greedy Build (multiple attempts)
        for attempt in range(800):
            if time.time() > deadline:
                break
            
            adj = greedy_build(n, min_deg, seed=attempt * 7919 + 13)
            
            if adj is not None:
                if check_valid_fast(adj, n, min_deg):
                    edges = get_edges(adj, n)
                    return {'n': n, 'edges': edges}
        
        # Strategy 3: Local search with repair from greedy results
        for attempt in range(80):
            if time.time() > deadline:
                break
            
            adj = greedy_build(n, min_deg, seed=attempt * 104729 + 7)
            
            if adj is not None:
                adj_ls, min_deg_ls = local_search_with_repair(adj, n, min_deg, max_time=2.0)
                if min_deg_ls >= min_deg and check_valid_fast(adj_ls, n, min_deg):
                    edges = get_edges(adj_ls, n)
                    return {'n': n, 'edges': edges}
        
        # Strategy 4: Random Initialization + Repair
        for attempt in range(40):
            if time.time() > deadline:
                break
            
            random.seed(attempt * 15485863 + 1)
            
            # Initialize with random graph
            p = min(min_deg / n * 2.0, 0.5)
            adj = [0] * n
            
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Repair phase
            for _ in range(300):
                if time.time() > deadline:
                    break
                
                # Check for C4s and fix
                fixed_c4 = False
                for i in range(n):
                    for j in range(i + 1, n):
                        common = adj[i] & adj[j]
                        if common.bit_count() > 1:
                            common_list = []
                            mask = common
                            k = 0
                            while mask:
                                if mask & 1:
                                    common_list.append(k)
                                mask >>= 1
                                k += 1
                            
                            for w in common_list[1:]:
                                deg_i = adj[i].bit_count()
                                deg_j = adj[j].bit_count()
                                if deg_i >= deg_j:
                                    adj[i] &= ~(1 << w)
                                    adj[w] &= ~(1 << i)
                                else:
                                    adj[j] &= ~(1 << w)
                                    adj[w] &= ~(1 << j)
                            fixed_c4 = True
                            break
                    if fixed_c4:
                        break
                
                if not fixed_c4:
                    # No C4s, check min degree
                    min_deg_actual = min(adj[i].bit_count() for i in range(n))
                    if min_deg_actual >= min_deg:
                        if check_valid_fast(adj, n, min_deg):
                            edges = get_edges(adj, n)
                            return {'n': n, 'edges': edges}
                        break
                    else:
                        # Need to add edges
                        low_verts = [i for i in range(n) if adj[i].bit_count() < min_deg]
                        if not low_verts:
                            break
                        u = min(low_verts, key=lambda i: adj[i].bit_count())
                        candidates = []
                        for v in range(n):
                            if v == u:
                                continue
                            if not (adj[u] & (1 << v)):
                                common = adj[u] & adj[v]
                                if common == 0:
                                    candidates.append(v)
                        
                        if candidates:
                            candidates.sort(key=lambda x: adj[x].bit_count())
                            v = candidates[0]
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                        else:
                            pass

    # Fallback: Return a valid small graph
    n = 40
    min_deg = 1
    adj = [0] * n
    for i in range(n):
        adj[i] |= (1 << ((i + 1) % n))
        adj[(i + 1) % n] |= (1 << i)
    
    if check_valid_fast(adj, n, min_deg):
        edges = get_edges(adj, n)
        return {'n': n, 'edges': edges}
    
    return {'n': 39, 'edges': []}
>>>>>>> REPLACE
