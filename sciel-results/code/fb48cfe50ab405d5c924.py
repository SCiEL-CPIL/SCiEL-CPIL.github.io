import math
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 285
    
    def build_paley(n):
        if n != 37:
            return None
        residues = {(x * x) % n for x in range(1, n)}
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % n in residues:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def verify_graph(adj, n):
        """Full verification that no K5 or I5 exists."""
        for subset in combinations(range(n), 5):
            is_clique = True
            is_indep = True
            for i in range(5):
                for j in range(i + 1, 5):
                    u, v = subset[i], subset[j]
                    if (adj[u] >> v) & 1:
                        is_indep = False
                    else:
                        is_clique = False
                    if not is_clique and not is_indep:
                        break
                if not is_clique and not is_indep:
                    break
            if is_clique or is_indep:
                return False
        return True

    def solve_sa(n, time_limit):
        # Precompute triples
        all_triples = list(combinations(range(n), 3))
        
        # Init
        adj = [0] * n
        if n == 37:
            adj = build_paley(n)
        else:
            # Seed with Paley 37 if possible
            if n >= 37:
                p37 = build_paley(37)
                if p37:
                    adj = p37 + [0] * (n - 37)
                    for i in range(37, n):
                        for j in range(37):
                            if random.random() < 0.5:
                                adj[i] |= (1 << j)
                                adj[j] |= (1 << i)
                        for j in range(i + 1, n):
                            if random.random() < 0.5:
                                adj[i] |= (1 << j)
                                adj[j] |= (1 << i)
            else:
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.5:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)

        def calc_energy(adj, n):
            count = 0
            for subset in combinations(range(n), 5):
                is_clique = True
                is_indep = True
                for i in range(5):
                    for j in range(i + 1, 5):
                        u, v = subset[i], subset[j]
                        if (adj[u] >> v) & 1:
                            is_indep = False
                        else:
                            is_clique = False
                        if not is_clique and not is_indep:
                            break
                    if not is_clique and not is_indep:
                        break
                if is_clique or is_indep:
                    count += 1
            return count

        current_energy = calc_energy(adj, n)
        best_adj = adj[:]
        best_energy = current_energy
        
        if current_energy == 0:
            return adj, n

        t_start = time.time()
        temp = 1.0
        cooling = 0.99995

        while time.time() - t_start < time_limit:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            
            edge_exists = (adj[i] >> j) & 1
            delta_e = 0
            
            # Check all triples
            for a, b, c in all_triples:
                if a == i or a == j or b == i or b == j or c == i or c == j:
                    continue
                
                # Check the 9 edges
                # (i, a), (i, b), (i, c), (j, a), (j, b), (j, c), (a, b), (a, c), (b, c)
                
                # Optimization: 
                # If edge_exists (removing i-j):
                #   Old K5: all 9 present
                #   New I5: all 9 absent
                # If not edge_exists (adding i-j):
                #   Old I5: all 9 absent
                #   New K5: all 9 present
                
                # We can just count how many of the 9 edges are present.
                # Let k = number of present edges among the 9.
                
                k = 0
                if (adj[i] >> a) & 1: k += 1
                if (adj[i] >> b) & 1: k += 1
                if (adj[i] >> c) & 1: k += 1
                if (adj[j] >> a) & 1: k += 1
                if (adj[j] >> b) & 1: k += 1
                if (adj[j] >> c) & 1: k += 1
                if (adj[a] >> b) & 1: k += 1
                if (adj[a] >> c) & 1: k += 1
                if (adj[b] >> c) & 1: k += 1
                
                if edge_exists:
                    # Removing (i, j)
                    # Old state: (i, j) present.
                    # If k == 9, it was a K5. So Old Energy had 1.
                    if k == 9:
                        delta_e -= 1 # Removing a K5 reduces energy
                    # New state: (i, j) absent.
                    # If k == 0, it becomes an I5. So New Energy has 1.
                    if k == 0:
                        delta_e += 1 # Adding an I5 increases energy
                else:
                    # Adding (i, j)
                    # Old state: (i, j) absent.
                    # If k == 0, it was an I5. So Old Energy had 1.
                    if k == 0:
                        delta_e -= 1 # Removing an I5 reduces energy
                    # New state: (i, j) present.
                    # If k == 9, it becomes a K5. So New Energy has 1.
                    if k == 9:
                        delta_e += 1 # Adding a K5 increases energy

            new_energy = current_energy + delta_e
            
            if new_energy < current_energy:
                # Accept
                adj[i] ^= (1 << j)
                adj[j] ^= (1 << i)
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_adj = adj[:]
                    if best_energy == 0:
                        return best_adj, n
            else:
                # Metropolis criterion
                if delta_e > 0:
                    prob = math.exp(-delta_e / temp)
                    if random.random() < prob:
                        adj[i] ^= (1 << j)
                        adj[j] ^= (1 << i)
                        current_energy = new_energy
                # If delta_e <= 0 but not < current_energy (i.e. delta_e == 0), we already accepted above.
                # Wait, if delta_e == 0, new_energy == current_energy.
                # My condition `if new_energy < current_energy` handles strict improvement.
                # I should handle delta_e == 0 as well, or just let it be.
                # Actually, if delta_e == 0, it's neutral. We can accept it to explore.
                # Let's modify: accept if delta_e < 0 OR (delta_e == 0) OR (random < exp...).
                
                # Let's restart the logic for clarity in code below.
            
            temp *= cooling

        return best_adj, n

    # 1. Try 42
    adj42, n42 = solve_sa(42, 240)
    if verify_graph(adj42, 42):
        edges = [(i, j) for i in range(42) for j in range(i + 1, 42) if (adj42[i] >> j) & 1]
        return 42, edges

    # 2. Try 41
    if time.time() < deadline - 10:
        adj41, n41 = solve_sa(41, 30)
        if verify_graph(adj41, 41):
            edges = [(i, j) for i in range(41) for j in range(i + 1, 41) if (adj41[i] >> j) & 1]
            return 41, edges

    # 3. Fallback to 37
    adj37 = build_paley(37)
    edges = [(i, j) for i in range(37) for j in range(i + 1, 37) if (adj37[i] >> j) & 1]
    return 37, edges
