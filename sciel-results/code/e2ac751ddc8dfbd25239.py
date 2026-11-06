import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    n = 40
    target_edges = 401
    
    # All possible edges
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def init_graph():
        """Initialize a random graph with target_edges."""
        A = np.zeros((n, n), dtype=np.int8)
        edges = random.sample(all_pairs, target_edges)
        for u, v in edges:
            A[u, v] = 1
            A[v, u] = 1
        return A, edges

    def calc_books(A):
        """Calculate book sizes for all edges. Returns dict {(u,v): size} and max_book."""
        books = {}
        max_b = 0
        for u, v in all_pairs:
            if A[u, v] == 1:
                b = 0
                for w in range(n):
                    if A[u, w] == 1 and A[v, w] == 1:
                        b += 1
                books[(u, v)] = b
                if b > max_b:
                    max_b = b
            else:
                books[(u, v)] = 0
        return books, max_b

    def get_max_edges(books, max_b):
        return [(u, v) for (u, v), b in books.items() if b == max_b and b > 0]

    def get_non_edges(A):
        return [(u, v) for u, v in all_pairs if A[u, v] == 0]

    best_score = None
    best_A = None
    best_books = None
    best_max_b = None
    
    # Try multiple random restarts
    for trial in range(50):
        if time.time() > deadline - 20:
            break
            
        A, edges = init_graph()
        books, max_b = calc_books(A)
        
        if best_score is None or max_b < best_score[0]:
            best_score = (max_b,)
            best_A = A.copy()
            best_books = books.copy()
            best_max_b = max_b
            
        # If we hit the theoretical lower bound (7), we can stop
        if max_b <= 7:
            break
            
        cur_books = books
        cur_max_b = max_b
        cur_A = A
        
        stuck = 0
        while time.time() < deadline - 5:
            max_edges = get_max_edges(cur_books, cur_max_b)
            if not max_edges:
                break
                
            # Pick a random max edge to remove
            u, v = random.choice(max_edges)
            
            # Get non-edges to add
            non_edges = get_non_edges(cur_A)
            if not non_edges:
                break
                
            # Try a few random swaps
            improved = False
            random.shuffle(non_edges)
            
            for x, y in non_edges[:20]: 
                # Simulate swap
                new_A = cur_A.copy()
                new_A[u, v] = 0
                new_A[v, u] = 0
                new_A[x, y] = 1
                new_A[y, x] = 1
                
                # Update books for affected edges
                affected_edges = set()
                affected_edges.add((u, v))
                affected_edges.add((x, y))
                
                for w in range(n):
                    if new_A[u, w] == 1:
                        e = (min(u, w), max(u, w))
                        affected_edges.add(e)
                    if new_A[v, w] == 1:
                        e = (min(v, w), max(v, w))
                        affected_edges.add(e)
                    if new_A[x, w] == 1:
                        e = (min(x, w), max(x, w))
                        affected_edges.add(e)
                    if new_A[y, w] == 1:
                        e = (min(y, w), max(y, w))
                        affected_edges.add(e)
                
                new_books = cur_books.copy()
                for e in affected_edges:
                    a, b = e
                    if new_A[a, b] == 1:
                        cnt = 0
                        for w in range(n):
                            if new_A[a, w] == 1 and new_A[b, w] == 1:
                                cnt += 1
                        new_books[e] = cnt
                    else:
                        new_books[e] = 0
                        
                new_max_b = max(new_books.values())
                
                # Acceptance criteria
                if new_max_b < cur_max_b:
                    improved = True
                    cur_A = new_A
                    cur_books = new_books
                    cur_max_b = new_max_b
                    break
                elif new_max_b == cur_max_b:
                    # Accept ties with small probability to allow exploration
                    if random.random() < 0.05:
                        improved = True
                        cur_A = new_A
                        cur_books = new_books
                        cur_max_b = new_max_b
                        break

            if improved:
                stuck = 0
                if cur_max_b < best_score[0]:
                    best_score = (cur_max_b,)
                    best_A = cur_A.copy()
                    best_books = cur_books.copy()
                    best_max_b = cur_max_b
            else:
                stuck += 1
                if stuck > 100:
                    break

    # Extract edges from best_A
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_A[i, j] == 1:
                edges.append([i, j])
    
    # Ensure exactly 401 edges
    if len(edges) != target_edges:
        if len(edges) > target_edges:
            edges = edges[:target_edges]
        else:
            existing = set(tuple(e) for e in edges)
            for u, v in all_pairs:
                if len(edges) == target_edges:
                    break
                if (u, v) not in existing and (v, u) not in existing:
                    edges.append([u, v])
                    existing.add((u, v))
                    existing.add((v, u))

    return edges
