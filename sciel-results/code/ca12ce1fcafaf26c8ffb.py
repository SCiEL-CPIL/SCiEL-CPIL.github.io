import time
import sys

def construct():
    """
    Return a polynomial with exactly 12 nonzero terms whose square has
    as few nonzero terms as possible.
    
    Uses a backtracking CSP approach to find a polynomial P with 12 terms
    such that P^2 has minimized nonzeros.
    """
    best_poly = [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1]
    best_score = 19
    
    def count_nonzeros(poly):
        return sum(1 for c in poly if c != 0)
    
    def backtrack(terms, current_sq, depth, start_time):
        nonlocal best_poly, best_score
        
        if time.time() - start_time > 55:
            return

        if depth == 12:
            score = len(current_sq)
            if score < best_score:
                best_score = score
                poly = [0] * 41
                for d, c in terms:
                    if d < 41:
                        poly[d] = c
                while len(poly) > 0 and poly[-1] == 0:
                    poly.pop()
                if len(poly) > 0 and poly[-1] != 0:
                    best_poly = poly
            return

        last_degree = terms[-1][0] if terms else 0
        
        # Candidate degrees
        candidates_d = []
        if len(terms) >= 2:
            step = terms[-1][0] - terms[-2][0]
            candidates_d.append(last_degree + step)
        candidates_d.extend([last_degree + 1, last_degree + 2, last_degree + 3, last_degree + 4, last_degree + 6, last_degree + 10])
        
        seen_d = set(d for d, _ in terms)
        
        for d in candidates_d:
            if d in seen_d or d <= last_degree or d >= 41:
                continue
            seen_d.add(d)
            
            for c in [-1, 1, -2, 2]:
                new_terms = terms + [(d, c)]
                new_sq = dict(current_sq)
                
                key_sq = 2*d
                new_sq[key_sq] = new_sq.get(key_sq, 0) + c*c
                
                for d_prev, c_prev in terms:
                    key_cross = d + d_prev
                    val = 2 * c * c_prev
                    new_sq[key_cross] = new_sq.get(key_cross, 0) + val
                
                new_sq = {k: v for k, v in new_sq.items() if v != 0}
                
                # Pruning
                if len(new_sq) < best_score + 2:
                    backtrack(new_terms, new_sq, depth + 1, start_time)
                    
                if best_score <= 14:
                    return

    initial_terms = [(0, 1)]
    initial_sq = {0: 1}
    start_time = time.time()
    
    backtrack(initial_terms, initial_sq, 1, start_time)
    
    nonzero_count = sum(1 for c in best_poly if c != 0)
    if nonzero_count != 12:
        best_poly = [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1]
        best_score = 19
        
    assert nonzero_count == 12, f"Expected 12 nonzero terms, got {nonzero_count}"
    assert all(-64 <= c <= 64 for c in best_poly), "Coefficient out of range"
    assert len(best_poly) <= 41, "Polynomial too long"
    assert best_poly[-1] != 0, "Final coefficient must be nonzero"
    
    return best_poly
