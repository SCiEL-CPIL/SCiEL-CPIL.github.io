import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    moduli = list(range(20, 201))
    n = len(moduli)
    
    # Precompute gcds
    # g[i][j] = gcd(moduli[i], moduli[j])
    # Indices 0..179 correspond to 20..200
    gcds = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            gcds[i][j] = math.gcd(moduli[i], moduli[j])
            gcds[j][i] = gcds[i][j]
            
    # Graph adjacency: adj[i][j] is True if gcds[i][j] > 1
    adj = [[False]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if gcds[i][j] > 1:
                adj[i][j] = True

    def is_clique(indices):
        """Check if the set of indices forms a clique in the non-coprime graph."""
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                if not adj[indices[i]][indices[j]]:
                    return False
        return True

    def greedy_clique(order):
        """
        Greedy construction of a clique.
        order: list of indices to consider
        """
        selected = []
        for idx in order:
            # Check if idx is connected to all in selected
            compatible = True
            for s in selected:
                if not adj[idx][s]:
                    compatible = False
                    break
            if compatible:
                selected.append(idx)
        return selected

    def assign_residues(selected_indices):
        """
        Assign residues to selected moduli.
        Returns list of [m, r] or None if no solution found.
        """
        if not selected_indices:
            return []
            
        # Sort by modulus size? Or degree?
        # Smaller moduli have smaller domains, so they are harder to satisfy.
        # Let's order by increasing modulus value.
        sorted_indices = sorted(selected_indices, key=lambda i: moduli[i])
        
        residues = {}
        
        def backtrack(pos):
            if pos == len(sorted_indices):
                return True
            
            idx = sorted_indices[pos]
            m = moduli[idx]
            
            # Try all possible residues
            # Optimization: We can try residues in a specific order.
            # Randomizing might help if stuck, but deterministic is safer for reproducibility.
            # Let's try 0 to m-1.
            for r in range(m):
                valid = True
                for prev_idx in sorted_indices[:pos]:
                    g = gcds[idx][prev_idx]
                    if r % g == residues[prev_idx] % g:
                        valid = False
                        break
                if valid:
                    residues[idx] = r
                    if backtrack(pos + 1):
                        return True
                    del residues[idx]
            return False

        if backtrack(0):
            return [[moduli[i], residues[i]] for i in selected_indices]
        return None

    def score_solution(sol):
        return sum(Fraction(1, m) for m, r in sol)

    best_solution = []
    best_score = Fraction(0, 1)
    
    # Initial Greedy
    # Sort by weight 1/m descending (i.e., m ascending)
    initial_order = sorted(range(n), key=lambda i: moduli[i])
    best_clique = greedy_clique(initial_order)
    sol = assign_residues(best_clique)
    if sol:
        best_solution = sol
        best_score = score_solution(sol)
        
    # Local Search on the Clique
    # We maintain a set of indices `current_clique`
    current_clique = set(best_clique)
    
    def try_swap(clique_set, out_idx, in_idx):
        new_clique = list(clique_set)
        new_clique.remove(out_idx)
        new_clique.append(in_idx)
        # Check if new_clique is a clique
        # Optimization: Only check connections of in_idx to others, and out_idx removal doesn't break others
        # So we just need to check if in_idx is connected to all in (clique_set - {out_idx})
        for s in clique_set:
            if s == out_idx:
                continue
            if not adj[in_idx][s]:
                return None
        return new_clique

    iterations = 0
    while time.time() < deadline - 5:
        iterations += 1
        # Pick random out and in
        if not current_clique:
            break
            
        out_idx = random.choice(list(current_clique))
        in_idx = random.randint(0, n - 1)
        
        if in_idx in current_clique:
            continue
            
        new_clique = try_swap(current_clique, out_idx, in_idx)
        if new_clique is None:
            continue
            
        # Check if weight improves
        # Weight of current clique
        # We can maintain current weight
        # But for simplicity, calculate new weight
        # Actually, just swap if weight(in) > weight(out) AND clique is valid
        # Since we only swap, and we want max sum, we should only accept if 1/m_in > 1/m_out
        if Fraction(1, moduli[in_idx]) > Fraction(1, moduli[out_idx]):
            current_clique = set(new_clique)
            
            # Try to assign residues
            sol = assign_residues(list(current_clique))
            if sol:
                s = score_solution(sol)
                if s > best_score:
                    best_score = s
                    best_solution = sol
            else:
                # If residue assignment fails, revert? 
                # Or keep the clique but try different residues?
                # For now, revert to keep state consistent with best_solution
                # Actually, a valid clique might have no residue solution? 
                # Unlikely for m>=20, but possible.
                # If it fails, we should probably not accept the swap to avoid getting stuck in a "bad" clique.
                pass
        else:
            # Accept worse swaps with low probability to escape local optima?
            # Or just accept if it allows adding more nodes later?
            # Let's stick to improving weight swaps for now, plus random restarts.
            pass
            
        # Also try adding nodes if possible (expansion)
        # Find a node not in clique that is connected to all in clique
        # This is O(N * |C|).
        # Do this occasionally.
        if iterations % 10 == 0:
            for in_idx in range(n):
                if in_idx in current_clique:
                    continue
                compatible = True
                for s in current_clique:
                    if not adj[in_idx][s]:
                        compatible = False
                        break
                if compatible:
                    # Try to add
                    test_clique = list(current_clique) + [in_idx]
                    sol = assign_residues(test_clique)
                    if sol:
                        s = score_solution(sol)
                        if s > best_score:
                            best_score = s
                            best_solution = sol
                            current_clique = set(test_clique)
                        break # Only add one at a time
                    
    # Final check: ensure best_solution is valid
    if not best_solution:
        # Fallback: just pick the smallest modulus
        best_solution = [[20, 0]]
        
    return best_solution
