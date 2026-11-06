import math
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s
    
    N = 256
    
    # Build adjacency: adj[v] = set of u (u != v) such that u+v is a perfect square
    # Also track self-loops: v is invalid if 2*v is a perfect square
    squares = set()
    k = 2
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
    
    # Diagonal constraint: 2a must not be a perfect square
    valid = [False] * (N + 1)
    for a in range(1, N + 1):
        if 2 * a not in squares:
            valid[a] = True
    
    # Build adjacency list for off-diagonal
    adj = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        if not valid[a]:
            continue
        for b in range(a + 1, N + 1):
            if not valid[b]:
                continue
            if (a + b) in squares:
                adj[a].add(b)
                adj[b].add(a)
    
    # Candidate set
    candidates = [a for a in range(1, N + 1) if valid[a]]
    
    def is_valid_set(S):
        """Check if S is an independent set."""
        S_set = set(S)
        for a in S:
            for b in adj[a]:
                if b in S_set:
                    return False
        return True
    
    def score(S):
        return len(S)
    
    # Greedy initial solution: process in various orders
    best_set = set()
    best_size = 0
    
    # Multiple greedy attempts with different orderings
    for trial in range(50):
        if time.time() > deadline:
            break
        
        if trial == 0:
            order = list(candidates)
        elif trial == 1:
            order = list(candidates)
            random.shuffle(order)
        elif trial == 2:
            order = list(candidates)
            order.sort(key=lambda x: len(adj[x]))  # least connected first
        elif trial == 3:
            order = list(candidates)
            order.sort(key=lambda x: -len(adj[x]))  # most connected first
        else:
            order = list(candidates)
            random.shuffle(order)
        
        S = set()
        for v in order:
            # Check if v can be added
            conflict = False
            for u in adj[v]:
                if u in S:
                    conflict = True
                    break
            if not conflict:
                S.add(v)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # Local search: hill climbing with swaps
    if time.time() < deadline:
        current_set = set(best_set)
        current_size = len(current_set)
        no_improve = 0
        max_no_improve = 200
        
        while no_improve < max_no_improve and time.time() < deadline:
            improved = False
            
            # Try removing one element and adding another
            for remove_v in list(current_set):
                if time.time() > deadline:
                    break
                new_set = current_set - {remove_v}
                # Find candidates to add
                for add_v in candidates:
                    if add_v in new_set:
                        continue
                    # Check if add_v conflicts with new_set
                    conflict = False
                    for u in adj[add_v]:
                        if u in new_set:
                            conflict = True
                            break
                    if not conflict:
                        new_size = len(new_set) + 1
                        if new_size > current_size:
                            current_set = new_set | {add_v}
                            current_size = new_size
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                no_improve += 1
            else:
                no_improve = 0
        
        if current_size > best_size:
            best_size = current_size
            best_set = current_set
    
    # Try more aggressive local search: remove 2, add 2 or 3
    if time.time() < deadline - 5:
        current_set = set(best_set)
        current_size = len(current_set)
        
        for iteration in range(50):
            if time.time() > deadline - 3:
                break
            
            improved = False
            # Try removing 1 or 2 elements and adding 1 or 2
            for num_remove in [1, 2]:
                if time.time() > deadline - 3:
                    break
                # Get all combinations to remove
                from itertools import combinations
                remove_list = list(current_set)
                for remove_combo in combinations(remove_list, num_remove):
                    if time.time() > deadline - 3:
                        break
                    new_set = current_set - set(remove_combo)
                    # Find best additions
                    add_count = 0
                    for add_v in candidates:
                        if add_v in new_set:
                            continue
                        conflict = False
                        for u in adj[add_v]:
                            if u in new_set:
                                conflict = True
                                break
                        if not conflict:
                            new_set.add(add_v)
                            add_count += 1
                            if add_count >= num_remove + 1:
                                break
                    
                    new_size = len(new_set)
                    if new_size > current_size:
                        current_set = new_set
                        current_size = new_size
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        if current_size > best_size:
            best_size = current_size
            best_set = current_set
    
    # Final verification
    final_list = sorted(best_set)
    S_set = set(final_list)
    for a in final_list:
        for b in S_set:
            if a <= b and (a + b) in squares:
                # Invalid - should not happen
                pass
    
    return final_list if final_list else [1]
