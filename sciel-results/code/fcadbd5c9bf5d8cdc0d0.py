import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Generate initial K32 coloring (Greenwood-Gleason / Clebsch)
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    def get_initial_32_matrix():
        log, x = {}, 1
        for e in range(15):
            log[x] = e
            x = mul(x, 2)
        n = 32
        matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if i // 16 == j // 16:
                    c = log[(i % 16) ^ (j % 16)] % 3
                else:
                    c = 3
                matrix[i][j] = c
                matrix[j][i] = c
        return matrix

    def matrix_to_colors(matrix, n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(matrix[i][j])
        return colors

    def count_threats(matrix, n):
        """Count total number of monochromatic triangles (each triangle counted once)."""
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = matrix[i][j]
                for k in range(j + 1, n):
                    if matrix[i][k] == c and matrix[j][k] == c:
                        count += 1
        return count

    def count_threats_involving_edge(matrix, n, i, j):
        """Count triangles involving edge (i,j) that are monochromatic."""
        c = matrix[i][j]
        count = 0
        for k in range(n):
            if k == i or k == j:
                continue
            if matrix[i][k] == c and matrix[j][k] == c:
                count += 1
        return count

    def flip_delta(matrix, n, i, j, new_c):
        """Compute change in total monochromatic triangle count if we flip edge (i,j) to new_c."""
        old_c = matrix[i][j]
        if old_c == new_c:
            return 0
        
        # Old triangles broken: triangles (i,j,k) where all three edges were old_c
        old_tri = 0
        # New triangles formed: triangles (i,j,k) where all three edges are new_c
        new_tri = 0
        
        for k in range(n):
            if k == i or k == j:
                continue
            ik_c = matrix[i][k]
            jk_c = matrix[j][k]
            
            # Was (i,j,k) a monochromatic triangle with old_c?
            if ik_c == old_c and jk_c == old_c:
                old_tri += 1
            
            # Will (i,j,k) be a monochromatic triangle with new_c?
            if ik_c == new_c and jk_c == new_c:
                new_tri += 1
        
        # Each triangle contributes to 3 edges' threat counts, but we're counting triangles directly
        # So delta = new_tri - old_tri
        return new_tri - old_tri

    def simulated_annealing(matrix, n, start_time, deadline):
        """Run SA to minimize monochromatic triangles to 0."""
        current_energy = count_threats(matrix, n)
        
        if current_energy == 0:
            return matrix, 0
        
        # SA parameters
        T = 5.0
        T_min = 0.01
        cooling_rate = 0.9995
        
        steps = 0
        best_energy = current_energy
        best_matrix = [row[:] for row in matrix]
        
        while time.time() < deadline - 5:
            if current_energy == 0:
                return matrix, 0
            
            # Pick random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            new_c = random.randint(0, 3)
            if matrix[i][j] == new_c:
                continue
            
            delta = flip_delta(matrix, n, i, j, new_c)
            
            # Accept if better or with probability exp(-delta/T)
            if delta < 0:
                matrix[i][j] = new_c
                matrix[j][i] = new_c
                current_energy += delta
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_matrix = [row[:] for row in matrix]
            else:
                # Boltzmann acceptance
                if random.random() < pow(2.718, -delta / T):
                    matrix[i][j] = new_c
                    matrix[j][i] = new_c
                    current_energy += delta
            
            steps += 1
            T *= cooling_rate
            if T < T_min:
                T = T_min
            
            # Occasionally reset temperature to escape local minima
            if steps % 50000 == 0 and current_energy > 0:
                T = max(T, 2.0)
        
        # Return best found
        if best_energy < current_energy:
            return best_matrix, best_energy
        return matrix, current_energy

    # Main: Try to find coloring for increasing n
    # Start with known good K32, then try to extend via SA on larger n
    
    best_n = 32
    best_colors = matrix_to_colors(get_initial_32_matrix(), 32)
    
    # Try n from 33 up to 62
    for target_n in range(33, 63):
        if time.time() > deadline - 15:
            break
        
        # Multiple attempts for this n
        found = False
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            # Start from a perturbed version of a valid smaller coloring
            # or random initialization
            if target_n <= 40:
                # Start from K32 and add random vertices
                base_matrix = get_initial_32_matrix()
                matrix = [[0] * target_n for _ in range(target_n)]
                
                # Copy K32 part
                for i in range(32):
                    for j in range(32):
                        matrix[i][j] = base_matrix[i][j]
                
                # Randomly color edges involving new vertices
                for i in range(32, target_n):
                    for j in range(target_n):
                        if i == j:
                            continue
                        if matrix[i][j] == 0 and i > j:
                            matrix[i][j] = random.randint(0, 3)
                            matrix[j][i] = matrix[i][j]
                        elif matrix[i][j] == 0:
                            matrix[i][j] = random.randint(0, 3)
                            matrix[j][i] = matrix[i][j]
            else:
                # Random initialization for larger n
                matrix = [[0] * target_n for _ in range(target_n)]
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        c = random.randint(0, 3)
                        matrix[i][j] = c
                        matrix[j][i] = c
            
            # Run SA
            matrix, energy = simulated_annealing(matrix, target_n, time.time(), deadline)
            
            if energy == 0:
                found = True
                best_n = target_n
                best_colors = matrix_to_colors(matrix, target_n)
                break
        
        if not found:
            # If we can't reach target_n, try a slightly smaller one?
            # Or just break and return best
            # Let's try target_n - 1 if we haven't reached it
            if target_n > best_n + 1 and time.time() < deadline - 10:
                for try_n in range(target_n - 1, best_n, -1):
                    if time.time() > deadline - 5:
                        break
                    for attempt in range(3):
                        if time.time() > deadline - 3:
                            break
                        
                        matrix = [[0] * try_n for _ in range(try_n)]
                        for i in range(try_n):
                            for j in range(i + 1, try_n):
                                c = random.randint(0, 3)
                                matrix[i][j] = c
                                matrix[j][i] = c
                        
                        matrix, energy = simulated_annealing(matrix, try_n, time.time(), deadline)
                        
                        if energy == 0:
                            best_n = try_n
                            best_colors = matrix_to_colors(matrix, try_n)
                            break
                    if best_n >= try_n:
                        break
            break
    
    return {'n': best_n, 'colors': best_colors}
