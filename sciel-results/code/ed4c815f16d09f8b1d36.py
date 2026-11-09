"""Stronger algorithm for dense K3,3-free graphs on 20 vertices."""
import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n = 20

    # Fast K3,3-free check using bitsets
    def is_k33_free(adj_bits):
        for i in range(n):
            for j in range(i+1, n):
                ij = adj_bits[i] & adj_bits[j]
                if ij.bit_count() < 3:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if (ij & adj_bits[k]).bit_count() >= 3:
                        return False
        return True

    def count_edges(adj_bits):
        return sum(b.bit_count() for b in adj_bits) // 2

    def add_edge(adj_bits, u, v):
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)

    def remove_edge(adj_bits, u, v):
        adj_bits[u] &= ~(1 << v)
        adj_bits[v] &= ~(1 << u)

    def get_missing(adj_bits):
        return [(i, j) for i in range(n) for j in range(i+1, n)
                if not (adj_bits[i] >> j) & 1]

    def local_search(adj_bits, deadline):
        """Hill climbing with additions and swaps."""
        current = adj_bits[:]
        current_edges = count_edges(current)
        
        # Precompute missing edges
        missing = get_missing(current)
        random.shuffle(missing)
        existing = [(i, j) for i in range(n) for j in range(i+1, n)
                    if (current[i] >> j) & 1]

        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            random.shuffle(missing)
            
            # Try adding edges
            new_missing = []
            for u, v in missing:
                if time.time() > deadline - 5:
                    new_missing.append((u, v))
                    continue
                add_edge(current, u, v)
                if is_k33_free(current):
                    current_edges += 1
                    improved = True
                    existing.append((u, v))
                else:
                    remove_edge(current, u, v)
                    new_missing.append((u, v))
            
            if improved:
                missing = new_missing
                continue
            
            # Try swaps if stuck
            if existing and missing and time.time() < deadline - 5:
                for _ in range(200):
                    if time.time() > deadline - 5:
                        break
                    e1 = random.choice(existing)
                    e2 = random.choice(missing)
                    u1, v1 = e1
                    u2, v2 = e2
                    
                    remove_edge(current, u1, v1)
                    add_edge(current, u2, v2)
                    if is_k33_free(current):
                        # Accept
                        existing.remove(e1)
                        existing.append(e2)
                        missing.remove(e2)
                        missing.append(e1)
                        improved = True
                    else:
                        remove_edge(current, u2, v2)
                        add_edge(current, u1, v1)
        
        return current, current_edges

    def init_random_greedy(deadline):
        """Build a random maximal K3,3-free graph via greedy edge addition."""
        adj = [0] * n
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() > deadline - 10:
                break
            add_edge(adj, u, v)
            if not is_k33_free(adj):
                remove_edge(adj, u, v)
        return adj

    def init_bipartite_random(deadline):
        """Initialize with a random bipartite graph (naturally K3,3-free if we're careful,
        but a complete bipartite K10,10 has K3,3, so we need sparse bipartite)."""
        adj = [0] * n
        A = list(range(10))
        B = list(range(10, 20))
        # Add random edges between A and B
        for a in A:
            # Each vertex in A connects to a random subset of B
            for b in B:
                if random.random() < 0.4:
                    add_edge(adj, a, b)
        return adj

    def init_complement_of_turan_like(deadline):
        """Try a structure based on Turan-type ideas."""
        adj = [0] * n
        # Start with a random graph and prune
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.6:
                    add_edge(adj, i, j)
        # Prune to make K3,3-free
        # Simple approach: remove edges that create K3,3
        # This is a rough initialization; local search will fix it
        return adj

    best_adj = None
    best_edges = -1

    # Multiple restarts with different initializations
    # Restart 1: Random greedy from empty
    if time.time() < deadline - 20:
        adj1 = init_random_greedy(deadline)
        adj1, e1 = local_search(adj1, deadline - 10)
        if e1 > best_edges:
            best_edges = e1
            best_adj = adj1[:]

    # Restart 2: Random bipartite-like
    if time.time() < deadline - 15:
        adj2 = init_bipartite_random(deadline)
        # First, prune adj2 to be K3,3-free by removing edges greedily
        all_edges2 = [(i, j) for i in range(n) for j in range(i+1, n)
                      if (adj2[i] >> j) & 1]
        random.shuffle(all_edges2)
        adj2_clean = [0] * n
        for u, v in all_edges2:
            add_edge(adj2_clean, u, v)
            if not is_k33_free(adj2_clean):
                remove_edge(adj2_clean, u, v)
        adj2, e2 = local_search(adj2_clean, deadline - 10)
        if e2 > best_edges:
            best_edges = e2
            best_adj = adj2[:]

    # Restart 3: Another random greedy
    if time.time() < deadline - 10:
        adj3 = init_random_greedy(deadline)
        adj3, e3 = local_search(adj3, deadline - 5)
        if e3 > best_edges:
            best_edges = e3
            best_adj = adj3[:]

    # Restart 4: Dense random start
    if time.time() < deadline - 10:
        adj4 = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    add_edge(adj4, i, j)
        # Prune
        all_edges4 = [(i, j) for i in range(n) for j in range(i+1, n)
                      if (adj4[i] >> j) & 1]
        random.shuffle(all_edges4)
        adj4_clean = [0] * n
        for u, v in all_edges4:
            add_edge(adj4_clean, u, v)
            if not is_k33_free(adj4_clean):
                remove_edge(adj4_clean, u, v)
        adj4, e4 = local_search(adj4_clean, deadline - 5)
        if e4 > best_edges:
            best_edges = e4
            best_adj = adj4[:]

    # Final polish on the best
    if best_adj is not None and time.time() < deadline - 2:
        best_adj, best_edges = local_search(best_adj, deadline - 2)

    # Convert to list of edges
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                result.append([i, j])
    
    return result

# EVOLVE-BLOCK-END
