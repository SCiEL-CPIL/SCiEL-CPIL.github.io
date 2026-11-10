"""Improved superpermutation construction for n=7 (and smaller n)."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import sys
from collections import deque

def construct(n):
    deadline = time.time() + 115  # safety margin before 120s limit
    t_start = time.time()
    
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "1234121341243121432142314"
    if n == 5:
        # Use classic construction for n=5, which is optimal at 153
        return _classic_superperm(5)
    if n == 6:
        # Classic is 873, optimal known is 873
        return _classic_superperm(6)
    
    # For n=7, use de Bruijn graph greedy search with multiple restarts
    return _debruijn_greedy(n, deadline)


def _classic_superperm(n):
    """Standard recursive superpermutation construction."""
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
            best_overlap = 0
            max_possible = min(len(new), len(block))
            for o in range(max_possible, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    best_overlap = o
                    break
            new += block[best_overlap:]
        word = new
    return "".join(map(str, word))


def _debruijn_greedy(n, deadline):
    """
    Model as shortest path in de Bruijn graph of order (n-1).
    Nodes: all (n-1)-permutations of 1..n
    Edges: from node u to node v if v = (u[1:] + (c,)) for some c not in u
    A superpermutation corresponds to a walk visiting all nodes.
    We use greedy best-first with multiple random restarts.
    """
    # Generate all (n-1)-permutations
    symbols = list(range(1, n + 1))
    all_perms = list(itertools.permutations(symbols, n - 1))
    num_nodes = len(all_perms)
    
    # Map each permutation to an index
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    
    # Build adjacency: for each node, what nodes can we transition to?
    # From node u = (a1, a2, ..., a_{n-1}), we can go to v = (a2, ..., a_{n-1}, c)
    # where c is not in u
    adj = [[] for _ in range(num_nodes)]
    for i, u in enumerate(all_perms):
        suffix = u[1:]
        for c in symbols:
            if c not in u:
                v = suffix + (c,)
                j = perm_to_idx[v]
                adj[i].append(j)
    
    # Greedy search: maintain visited set, current node, and path
    # Score: number of unvisited neighbors (lower is better for greedy)
    # With random restarts using different initial orderings
    
    best_length = None
    best_word = None
    
    # Precompute classic result as fallback
    classic = _classic_superperm(n)
    best_length = len(classic)
    best_word = classic
    
    # Multiple restarts with different random seeds
    rng = random.Random(42)
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
        
        # Randomize the order in which we consider transitions
        # Use a randomized priority: primarily unvisited count, with random tie-breaking
        visited = [False] * num_nodes
        current = rng.randint(0, num_nodes - 1)
        visited[current] = True
        path = [current]
        
        # The word starts with all_perms[current]
        word_list = list(all_perms[current])
        
        success = True
        steps = 0
        max_steps = num_nodes + 100  # safety
        
        while not all(visited):
            if time.time() > deadline - 5:
                success = False
                break
            
            if steps > max_steps:
                success = False
                break
            steps += 1
            
            # Get neighbors
            neighbors = adj[current]
            
            # Score each neighbor: prefer unvisited, then random
            scored = []
            for nb in neighbors:
                if not visited[nb]:
                    # Higher priority for unvisited
                    score = (0, rng.random())
                else:
                    score = (1, rng.random())
                scored.append((score, nb))
            
            scored.sort()
            
            # Choose the best neighbor
            if not scored:
                success = False
                break
            
            _, next_node = scored[0]
            visited[next_node] = True
            path.append(next_node)
            current = next_node
            
            # Append the last symbol of the new node to the word
            word_list.append(all_perms[current][-1])
        
        if success:
            word_str = "".join(map(str, word_list))
            if len(word_str) < best_length:
                best_length = len(word_str)
                best_word = word_str
                # If we hit the theoretical lower bound or close to it, we can stop
                if best_length <= 5905:
                    break
        
        # Also try with different initial nodes in next iterations
        # The random seed already varies this
    
    return best_word
# EVOLVE-BLOCK-END
