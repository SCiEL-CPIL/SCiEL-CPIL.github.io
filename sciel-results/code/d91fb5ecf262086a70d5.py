import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s limit to be safe
    
    def check_all_subsets(adj, n):
        """Check all subsets of size >= 6 for regularity. Returns violation or None."""
        for k in range(6, n + 1):
            for subset in combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                degs = []
                for v in subset:
                    degs.append((adj[v] & mask).bit_count())
                if len(set(degs)) == 1:
                    return subset
        return None
    
    def count_violations_fast(adj, n):
        """Count violations of size 6 (fastest check)."""
        count = 0
        for subset in combinations(range(n), 6):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            degs = []
            for v in subset:
                degs.append((adj[v] & mask).bit_count())
            if len(set(degs)) == 1:
                count += 1
        return count
    
    def local_search(adj, n, time_limit):
        """Local search to eliminate regular induced subgraphs."""
        deadline_local = time.time() + time_limit
        while time.time() < deadline_local:
            violation = check_all_subsets(adj, n)
            if violation is None:
                return True
            # Pick random pair in violation and flip edge
            u = random.choice(violation)
            v = random.choice([x for x in violation if x != u])
            if adj[u] & (1 << v):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return False
    
    # Try deterministic algebraic constructions first
    # Construction 1: Paley-like graph on 21 vertices
    # Use a quadratic residue graph variant
    for n in range(21, 19, -1):
        if time.time() > deadline:
            break
        
        # Try multiple deterministic constructions
        constructions = []
        
        # Construction A: Based on quadratic residues mod p (adapted for n vertices)
        # For n=21, use Z_21 with quadratic residues
        if n == 21:
            # Quadratic residues mod 21: {1, 4, 9, 16, 0} but 0 is special
            # Let's use a different approach: take vertices 0..20, connect i-j if (i-j) is a QR mod some prime
            # Actually, let's use the Clebsch graph extension or other known constructions
            
            # Try: connect i and j if (i-j) mod 21 is in a carefully chosen set
            # For a (21, 10) strongly regular graph: the complement of the line graph of K_7
            # Or use the Hoffman-Singleton graph variant
            
            # Simple approach: use a 21-vertex graph from known constructions
            # The (21, 10, 4, 5) strongly regular graph exists (complement of Clebsch-like)
            
            # Let me try: vertices 0..20, edge if (i-j) mod 21 is in {1, 2, 4, 8, 16} (powers of 2 mod 21)
            # Powers of 2 mod 21: 1, 2, 4, 8, 16, 11, 1, ... so {1, 2, 4, 8, 16, 11}
            # That gives degree 6, which is too low.
            
            # Try a denser construction: use all quadratic residues mod 23, then remove 2 vertices
            # QR mod 23: {1, 2, 3, 4, 6, 8, 9, 12, 13, 16, 18}
            # That's 11 residues, so degree 11 in the 23-vertex Paley graph.
            # Remove vertices 21 and 22, get 21-vertex graph.
            
            # Build Paley graph on 23 vertices, then take first 21
            adj_23 = [0] * 23
            for i in range(23):
                for j in range(i + 1, 23):
                    diff = (j - i) % 23
                    # Check if diff is a QR mod 23
                    is_qr = False
                    for x in range(1, 23):
                        if (x * x) % 23 == diff:
                            is_qr = True
                            break
                    if is_qr:
                        adj_23[i] |= (1 << j)
                        adj_23[j] |= (1 << i)
            
            # Take first 21 vertices
            adj = [0] * 21
            for i in range(21):
                for j in range(i + 1, 21):
                    if adj_23[i] & (1 << j):
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            constructions.append(adj.copy())
            
            # Construction B: Different vertex removal
            adj2 = [0] * 21
            # Remove vertices 0 and 22 instead
            for i in range(1, 22):
                for j in range(i + 1, 22):
                    if adj_23[i] & (1 << j):
                        adj2[i - 1] |= (1 << (j - 1))
                        adj2[j - 1] |= (1 << (i - 1))
            constructions.append(adj2.copy())
            
            # Construction C: Use a different prime
            # Paley on 19 vertices (QR mod 19: {1,4,5,6,7,9,11,16,17}), then add 2 vertices
            # Or use 29-vertex Paley, remove 8 vertices
            # Let's try 29: QR mod 29 = {1,4,5,6,7,9,13,16,20,22,23,24,25,28} (14 residues)
            adj_29 = [0] * 29
            for i in range(29):
                for j in range(i + 1, 29):
                    diff = (j - i) % 29
                    is_qr = False
                    for x in range(1, 29):
                        if (x * x) % 29 == diff:
                            is_qr = True
                            break
                    if is_qr:
                        adj_29[i] |= (1 << j)
                        adj_29[j] |= (1 << i)
            
            # Take first 21 vertices from 29-vertex Paley
            adj3 = [0] * 21
            for i in range(21):
                for j in range(i + 1, 21):
                    if adj_29[i] & (1 << j):
                        adj3[i] |= (1 << j)
                        adj3[j] |= (1 << i)
            constructions.append(adj3.copy())
            
            # Construction D: Random but seeded
            random.seed(42)
            adj4 = [0] * 21
            for i in range(21):
                for j in range(i + 1, 21):
                    if random.random() < 0.5:
                        adj4[i] |= (1 << j)
                        adj4[j] |= (1 << i)
            constructions.append(adj4.copy())
        
        elif n == 20:
            # Known good: use 23-vertex Paley, remove 3 vertices
            adj_23 = [0] * 23
            for i in range(23):
                for j in range(i + 1, 23):
                    diff = (j - i) % 23
                    is_qr = False
                    for x in range(1, 23):
                        if (x * x) % 23 == diff:
                            is_qr = True
                            break
                    if is_qr:
                        adj_23[i] |= (1 << j)
                        adj_23[j] |= (1 << i)
            
            adj = [0] * 20
            for i in range(20):
                for j in range(i + 1, 20):
                    if adj_23[i] & (1 << j):
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            constructions.append(adj.copy())
            
            # Also try random
            random.seed(123)
            adj2 = [0] * 20
            for i in range(20):
                for j in range(i + 1, 20):
                    if random.random() < 0.5:
                        adj2[i] |= (1 << j)
                        adj2[j] |= (1 << i)
            constructions.append(adj2.copy())
        
        elif n == 19:
            # Paley on 19 vertices
            adj = [0] * 19
            for i in range(19):
                for j in range(i + 1, 19):
                    diff = (j - i) % 19
                    is_qr = False
                    for x in range(1, 19):
                        if (x * x) % 19 == diff:
                            is_qr = True
                            break
                    if is_qr:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            constructions.append(adj.copy())
        
        # Try each construction
        for adj in constructions:
            if time.time() > deadline:
                break
            
            # Check if already valid
            if check_all_subsets(adj, n) is None:
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
            
            # Try local search
            remaining_time = deadline - time.time()
            if remaining_time > 5:
                if local_search(adj, n, remaining_time - 2):
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
        
        # If all constructions failed, try pure random search
        for restart in range(20):
            if time.time() > deadline:
                break
            
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            remaining_time = deadline - time.time()
            if remaining_time > 5:
                if local_search(adj, n, remaining_time - 2):
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
    
    # Fallback: return a small valid graph
    return {'n': 5, 'edges': []}
