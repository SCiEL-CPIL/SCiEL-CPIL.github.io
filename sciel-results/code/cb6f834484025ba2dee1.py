import random
import time
import sys

def _adj_bitsets(n, edges):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _find_k4s(n, adj):
    k4s = []
    for a in range(n):
        for b in range(a+1, n):
            if adj[a] & (1 << b):
                for c in range(b+1, n):
                    if (adj[a] & (1 << c)) and (adj[b] & (1 << c)):
                        for d in range(c+1, n):
                            if (adj[a] & (1 << d)) and (adj[b] & (1 << d)) and (adj[c] & (1 << d)):
                                k4s.append((a, b, c, d))
    return k4s

def _find_i4s(n, adj):
    i4s = []
    # Independent set of size 4: a,b,c,d such that no edges between them
    # This is a K4 in the complement graph
    comp_adj = [0] * n
    for i in range(n):
        comp_adj[i] = ((1 << n) - 1) ^ adj[i] ^ (1 << i)
        
    for a in range(n):
        for b in range(a+1, n):
            if comp_adj[a] & (1 << b):
                for c in range(b+1, n):
                    if (comp_adj[a] & (1 << c)) and (comp_adj[b] & (1 << c)):
                        for d in range(c+1, n):
                            if (comp_adj[a] & (1 << d)) and (comp_adj[b] & (1 << d)) and (comp_adj[c] & (1 << d)):
                                i4s.append((a, b, c, d))
    return i4s

def _valid_adjacency(S_bits, n, k4s, i4s):
    # S_bits is an integer bitmask representing the neighbors of the new vertex
    # Check K4 condition: No K4 is a subset of S
    for k4 in k4s:
        mask = (1 << k4[0]) | (1 << k4[1]) | (1 << k4[2]) | (1 << k4[3])
        if (S_bits & mask) == mask:
            return False
            
    # Check I4 condition: No I4 is a subset of V\S
    # V\S is ~S_bits & ((1<<n)-1)
    non_S = (~S_bits) & ((1 << n) - 1)
    for i4 in i4s:
        mask = (1 << i4[0]) | (1 << i4[1]) | (1 << i4[2]) | (1 << i4[3])
        if (non_S & mask) == mask:
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # 1. Construct Paley Graph on 37 vertices
    n = 37
    edges = []
    # Paley graph: edge (u,v) if (v-u) is a quadratic residue mod p
    # p=37 is prime. 37 = 1 mod 4, so it's self-complementary.
    # Quadratic residues mod 37:
    residues = set()
    for x in range(1, 37):
        residues.add((x*x) % 37)
    
    for u in range(n):
        for v in range(u+1, n):
            if ((v - u) % 37) in residues or ((u - v) % 37) in residues:
                edges.append((u, v))
                
    adj = _adj_bitsets(n, edges)
    k4s = _find_k4s(n, adj)
    i4s = _find_i4s(n, adj)
    
    # 2. Try to extend
    # We want to add vertices one by one.
    # For each new vertex, we search for a valid S (neighbors).
    
    while n < 46: # Upper bound is 46
        if time.time() > deadline:
            break
            
        found = False
        # Try random sampling for S
        # S is a bitmask of length n
        # We can bias the sampling.
        # A good heuristic: S should be roughly half the vertices?
        # Or we can use local search.
        
        # Let's try a few random restarts
        for _ in range(5000):
            if time.time() > deadline:
                break
            
            # Generate random S
            # To make it more likely to be valid, we can start with a random set
            # and then repair violations.
            
            S = 0
            # Random initialization
            for v in range(n):
                if random.random() < 0.5:
                    S |= (1 << v)
                    
            # Local search / Repair
            # We want to minimize violations.
            # Violations:
            # 1. K4s contained in S
            # 2. I4s contained in ~S
            
            # Simple repair:
            # If a K4 is in S, remove one of its vertices from S.
            # If an I4 is in ~S, add one of its vertices to S.
            
            improved = True
            while improved:
                improved = False
                # Check K4s
                for k4 in k4s:
                    mask = (1 << k4[0]) | (1 << k4[1]) | (1 << k4[2]) | (1 << k4[3])
                    if (S & mask) == mask:
                        # Remove a random vertex from this K4 from S
                        idx = random.choice(k4)
                        S &= ~(1 << idx)
                        improved = True
                        break # Restart check? Or continue? Breaking and restarting loop is safer for convergence logic simplicity, but might be slow.
                        # Let's just break inner and restart while loop to re-evaluate all constraints.
                
                # Check I4s
                if improved:
                    continue
                    
                non_S = (~S) & ((1 << n) - 1)
                for i4 in i4s:
                    mask = (1 << i4[0]) | (1 << i4[1]) | (1 << i4[2]) | (1 << i4[3])
                    if (non_S & mask) == mask:
                        # Add a random vertex from this I4 to S
                        idx = random.choice(i4)
                        S |= (1 << idx)
                        improved = True
                        break
                        
            if _valid_adjacency(S, n, k4s, i4s):
                # Found valid S
                # Add new vertex n connected to all in S
                for v in range(n):
                    if S & (1 << v):
                        edges.append((v, n))
                n += 1
                adj = _adj_bitsets(n, edges)
                # Update k4s and i4s
                # This is expensive. O(n^4).
                # For n=37, k4s might be ~100s.
                # For n=42, it's bigger.
                # We only need to check if the *new* graph has K5/I5.
                # But our constraint logic relies on the *current* graph's K4s and I4s.
                # So we must update them.
                
                # Optimization: Only add new K4s/I4s that involve the new vertex?
                # No, existing K4s are still K4s.
                # We just need to append new ones.
                
                new_k4s = []
                # K4s involving new vertex n:
                # n is connected to S.
                # A K4 containing n consists of n and a K3 in S.
                # We need to find all K3s in the induced subgraph on S.
                # This is O(|S|^3).
                # Similarly for I4s: n is non-connected to ~S.
                # I4 containing n consists of n and an I3 in ~S.
                
                # Let's just recompute from scratch for simplicity if n is small enough.
                # n is max 42. 42^4 / 24 is ~150k checks. Very fast.
                k4s = _find_k4s(n, adj)
                i4s = _find_i4s(n, adj)
                
                found = True
                break
                
        if not found:
            # Could not extend further
            break
            
    return (n, edges)
