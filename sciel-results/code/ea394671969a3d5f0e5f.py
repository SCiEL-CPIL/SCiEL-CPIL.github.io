import numpy as np
import time
import random

def design(problem):
    """
    Designs a 32x32 pixel unit cell to maximize the TM band gap between bands 4 and 5.
    Uses a Genetic Algorithm with a fast Plane Wave Method (PWM) fitness function.
    """
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]  # 1
    eps_high = problem["eps_high"]  # 12
    
    # Time budget management
    start_time = time.time()
    time_limit = 220.0  # Leave 10s margin
    
    # --- Fast Plane Wave Solver ---
    def expand_symmetry(cell_16):
        """
        Expands a 16x16 fundamental domain (top-left quadrant) to a 32x32 cell
        using C4v symmetry (rotations and reflections).
        The 16x16 block represents the region 0 <= x < 16, 0 <= y < 16.
        C4v symmetry about the center (16,16) in 32x32 coordinates.
        Actually, the problem states: "invariant under the square symmetry group about the cell centre".
        This means cell[i][j] == cell[31-i][j] == cell[i][31-j] == cell[31-i][31-j] etc.
        So the top-left 16x16 quadrant determines the whole cell.
        """
        cell = np.zeros((size, size), dtype=int)
        # Top-left quadrant
        cell[0:16, 0:16] = cell_16
        # Top-right quadrant (mirror top-left horizontally)
        cell[0:16, 16:32] = cell_16[:, ::-1]
        # Bottom-left quadrant (mirror top-left vertically)
        cell[16:32, 0:16] = cell_16[::-1, :]
        # Bottom-right quadrant (mirror top-left both)
        cell[16:32, 16:32] = cell_16[::-1, ::-1]
        return cell

    def compute_fourier_coefficients(cell_32):
        """
        Computes Fourier coefficients of 1/epsilon for TM modes.
        epsilon(r) is 1 for air (0) and 12 for dielectric (1).
        We need 1/epsilon.
        """
        # 1/epsilon: air=1, dielectric=1/12
        inv_eps = np.where(cell_32 == 1, 1.0/eps_high, 1.0/eps_low)
        
        # Compute 2D FFT
        # The pixel map is defined on a grid. We assume the unit cell is [0,1)x[0,1).
        # Pixel (i,j) corresponds to x in [i/32, (i+1)/32), y in [j/32, (j+1)/32).
        # The FFT gives coefficients for exp(-i 2pi (nx x + ny y)).
        # We need coefficients for G vectors.
        # G = (nx, ny).
        # The FFT output F[nx, ny] corresponds to sum f(x,y) exp(-i 2pi (nx x + ny y)).
        # Since f is piecewise constant, we can use the FFT of the pixel grid.
        # Note: The standard FFT assumes the function is sampled at points.
        # For a pixel map, the exact Fourier coefficient is:
        # c_G = sum_pixels val * integral_pixel exp(-i 2pi G.r) dr
        # For a 32x32 grid, this is approximately F[nx, ny] / 32^2 * (sinc correction).
        # For small G, the correction is negligible. We will use the raw FFT scaled.
        
        F = np.fft.fft2(inv_eps)
        # Shift so that (0,0) is at center? No, numpy fft2 has (0,0) at [0,0].
        # We need coefficients for specific G vectors.
        return F

    def solve_pw(cell_32, n_waves=7, k_points=None):
        """
        Solves the TM band structure using a reduced set of plane waves.
        n_waves: number of G vectors in x direction (total waves = n_waves^2).
        k_points: list of (kx, ky) in units of 2pi.
        """
        if k_points is None:
            # Key points in irreducible BZ: Gamma(0,0), X(0.5,0), M(0.5,0.5)
            # And some intermediate points for better gap estimation
            k_points = [
                (0.0, 0.0),
                (0.5, 0.0),
                (0.5, 0.5),
                (0.25, 0.0),
                (0.5, 0.25),
                (0.25, 0.25)
            ]
        
        F = compute_fourier_coefficients(cell_32)
        
        # Generate G vectors
        # G vectors are integers (nx, ny).
        # We select G vectors within a radius.
        # For n_waves=7, we take nx, ny in range(-3, 4).
        half_n = n_waves // 2
        G_vectors = []
        for nx in range(-half_n, half_n + 1):
            for ny in range(-half_n, half_n + 1):
                G_vectors.append((nx, ny))
        
        N = len(G_vectors)
        
        # Precompute 1/|k+G|^2 terms? No, k changes.
        
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            # k vector in units of 2pi
            k = np.array([kx, ky])
            
            # Construct the matrix M
            # M_{mn} = delta_{mn} + sum_{G!=0} (epsilon_G / ((k+G_m)^2 * (k-G_n)^2)) 
            # Wait, the standard TM equation is:
            # sum_n [ delta_{mn} - (1/(k+G_m)^2) sum_G' epsilon_{G'} (k+G_m - G')^2 delta_{n, m-G'} ] ... 
            # Let's use the standard form:
            # (k^2) E = sum_G epsilon_G (k+G)^2 E_{k-G}
            # This is a generalized eigenvalue problem or can be transformed.
            # Common formulation:
            # M_{mn} = delta_{mn} + (1/(k+G_m)^2) * epsilon_{G_m - G_n} * (k+G_m - G_n)^2
            # Actually, the most common efficient form for TM is:
            # A_{mn} = delta_{mn} - (1/(k+G_m)^2) * (1/epsilon_{G_m - G_n}) * (k+G_m - G_n)^2
            # No, let's stick to the simplest correct one:
            # The eigenvalue is lambda = (k+G)^2 / epsilon_G ? No.
            
            # Let's use the formulation from Joannopoulos et al.:
            # For TM:
            # (k+G)^2 E_G = sum_{G'} (1/epsilon_{G-G'}) (k+G')^2 E_{G'}
            # This is A E = lambda E, where A_{GG'} = delta_{GG'} (k+G)^2 - (1/epsilon_{G-G'}) (k+G')^2
            # This is not symmetric.
            
            # Alternative symmetric formulation for TM:
            # Define H = sqrt(epsilon). Then it becomes TE-like with effective permittivity?
            # No, for TM, the field H_z satisfies:
            # -1/epsilon * div(epsilon grad H_z) = k^2 H_z
            # In Fourier space:
            # sum_{G'} (1/epsilon_{G-G'}) (k+G)^2 (k+G')^2 E_{G'} = k^2 E_G
            # This is still messy.
            
            # Let's use the standard "Matrix M" approach:
            # M_{mn} = delta_{mn} + (1/(k+G_m)^2) * sum_{G} epsilon_G * (k+G_m - G)^2 * delta_{n, m-G}
            # This is O(N^3).
            
            # Simplified:
            # M_{mn} = delta_{mn} + (1/(k+G_m)^2) * epsilon_{G_m - G_n} * (k+G_m - G_n)^2
            # This assumes the sum collapses to a single term? No, that's only if epsilon is constant.
            # The correct term is:
            # M_{mn} = delta_{mn} + (1/(k+G_m)^2) * sum_{G} epsilon_G (k+G_m - G)^2 delta_{n, m-G}
            # Let G' = G_m - G_n. Then G = G_m - G'.
            # The sum is over G. The delta forces G = G_m - G_n.
            # So the sum collapses!
            # M_{mn} = delta_{mn} + (1/(k+G_m)^2) * epsilon_{G_m - G_n} * (k+G_n)^2
            # Wait, (k+G_m - G) = (k+G_n).
            # So M_{mn} = delta_{mn} + (epsilon_{G_m - G_n} / (k+G_m)^2) * (k+G_n)^2
            # This is correct for the standard PW method.
            
            M = np.zeros((N, N), dtype=complex)
            
            # Precompute k+G vectors
            kG = []
            for G in G_vectors:
                kG.append(k + np.array(G))
            
            for i in range(N):
                for j in range(N):
                    G_diff = np.array(G_vectors[i]) - np.array(G_vectors[j])
                    # Get epsilon coefficient for G_diff
                    # G_diff is in units of 2pi.
                    # Our FFT F[nx, ny] corresponds to G=(nx, ny).
                    # We need to handle negative indices.
                    nx_diff, ny_diff = G_diff
                    # Map to FFT indices
                    # FFT index range: 0..31.
                    # If nx_diff < 0, add 32.
                    idx_x = nx_diff % size
                    idx_y = ny_diff % size
                    eps_coeff = F[idx_x, idx_y]
                    
                    # Note: F is the sum. The coefficient in the expansion is F / size^2?
                    # The pixel map is a sum of rectangles.
                    # The FFT of the pixel map (values 1 or 1/12) gives the sum of values * exp(-i...).
                    # The actual Fourier coefficient c_G is (1/Area) * sum ...
                    # Area = 1. So c_G = F_G.
                    # However, numpy FFT does not normalize by N.
                    # So c_G = F_G / (size*size).
                    
                    eps_val = eps_coeff / (size * size)
                    
                    km = kG[i]
                    kn = kG[j]
                    km_sq = np.dot(km, km)
                    kn_sq = np.dot(kn, kn)
                    
                    if i == j:
                        M[i, j] = 1.0 + (eps_val / km_sq) * kn_sq
                    else:
                        M[i, j] = (eps_val / km_sq) * kn_sq
                        
            # Solve eigenvalue problem
            # M E = lambda E
            # lambda = (k+G)^2 / k^2 ? No.
            # The equation was: sum_n M_{mn} E_n = (k+G_m)^2 / k^2 * E_m ?
            # Let's check the derivation.
            # (k+G_m)^2 E_m = sum_n (1/epsilon_{G_m-G_n}) (k+G_n)^2 E_n
            # Multiply by 1/(k+G_m)^2:
            # E_m = sum_n (1/epsilon_{G_m-G_n}) (k+G_n)^2 / (k+G_m)^2 E_n
            # So M_{mn} = (1/epsilon_{G_m-G_n}) (k+G_n)^2 / (k+G_m)^2
            # And the eigenvalue is 1? No, that's not right.
            
            # Correct standard form:
            # lambda = k^2
            # (k+G_m)^2 E_m = sum_n (1/epsilon_{G_m-G_n}) (k+G_n)^2 E_n
            # This is a generalized eigenvalue problem: A E = lambda B E
            # A_{mn} = - (1/epsilon_{G_m-G_n}) (k+G_n)^2 delta_{mn}? No.
            
            # Let's use the simplest reliable method:
            # Construct A and B.
            # A_{mn} = (k+G_m)^2 delta_{mn}
            # B_{mn} = (1/epsilon_{G_m-G_n}) (k+G_n)^2
            # A E = k^2 B E  =>  B^{-1} A E = k^2 E
            # This requires inverting B, which is expensive and unstable.
            
            # Alternative:
            # M_{mn} = (1/epsilon_{G_m-G_n}) (k+G_n)^2 / (k+G_m)^2
            # Then M E = E ? No.
            
            # Let's use the formulation from "Photonic Crystals: Molding the Flow of Light":
            # For TM:
            # sum_{G'} [ (k+G)^2 delta_{GG'} - (1/epsilon_{G-G'}) (k+G')^2 ] E_{G'} = 0
            # This is a homogeneous system. Non-trivial solution if det = 0.
            # This is not an eigenvalue problem for k^2 directly, but we can treat k^2 as the parameter.
            # However, for a fixed k, we want to find the bands.
            # The equation is linear in E, but the coefficients depend on k.
            # Wait, the term (k+G)^2 is on the diagonal.
            # Let's rearrange:
            # (k+G)^2 E_G = sum_{G'} (1/epsilon_{G-G'}) (k+G')^2 E_{G'}
            # This is A E = B E, where A is diagonal with (k+G)^2, and B is the matrix with 1/eps terms.
            # So B^{-1} A E = E ? No.
            # A E = B E => B^{-1} A E = E.
            # This implies the eigenvalues of B^{-1} A are 1? That can't be right.
            
            # Ah, the equation is:
            # (k+G)^2 E_G = sum_{G'} (1/epsilon_{G-G'}) (k+G')^2 E_{G'}
            # Let lambda = k^2.
            # This is not a standard eigenvalue problem for lambda.
            # It is an eigenvalue problem for the field E for a fixed k.
            # But we want to find k (or lambda) for a given band.
            
            # Correct approach:
            # The matrix M_{GG'} = (1/epsilon_{G-G'}) (k+G')^2
            # The equation is: sum_{G'} M_{GG'} E_{G'} = (k+G)^2 E_G
            # This is M E = D E, where D is diagonal with (k+G)^2.
            # D^{-1} M E = E.
            # This doesn't give k^2.
            
            # Let's look at the TE case, which is simpler:
            # (k+G)^2 E_G = sum (1/epsilon) (k+G')^2 E_{G'}
            # Wait, for TE, it is:
            # (k+G)^2 E_G = sum epsilon_{G-G'} (k+G')^2 E_{G'}
            # For TM, it is:
            # (k+G)^2 E_G = sum (1/epsilon_{G-G'}) (k+G')^2 E_{G'}
            
            # In both cases, the equation is of the form:
            # A(k) E = B(k) E
            # Where A(k) is diagonal with (k+G)^2.
            # This is a generalized eigenvalue problem: A E = B E.
            # We want to find k such that det(A - lambda B) = 0? No, lambda is not k^2.
            # Actually, k is the parameter. We want to find the eigenvalues of the operator.
            # The operator is L = -1/epsilon div(epsilon grad).
            # In Fourier space, L_{GG'} = (1/epsilon_{G-G'}) (k+G')^2 ? No.
            # The eigenvalue is k^2.
            # So: sum_{G'} (1/epsilon_{G-G'}) (k+G')^2 E_{G'} = k^2 E_G
            # This is: M(k) E = k^2 E
            # Where M(k)_{GG'} = (1/epsilon_{G-G'}) (k+G')^2
            # This is a standard eigenvalue problem for k^2, where the matrix M depends on k.
            # This is a nonlinear eigenvalue problem.
            
            # However, we can solve it for a fixed k by finding the eigenvalues of M(k).
            # The bands are the square roots of the eigenvalues of M(k).
            
            # So:
            # 1. Construct M(k) where M_{ij} = (1/epsilon_{G_i-G_j}) (k+G_j)^2
            # 2. Find eigenvalues lambda.
            # 3. omega = sqrt(lambda).
            
            # Let's implement this.
            
            M = np.zeros((N, N), dtype=complex)
            for i in range(N):
                for j in range(N):
                    G_diff = np.array(G_vectors[i]) - np.array(G_vectors[j])
                    nx_diff, ny_diff = G_diff
                    idx_x = nx_diff % size
                    idx_y = ny_diff % size
                    eps_coeff = F[idx_x, idx_y] / (size * size)
                    
                    kn = kG[j]
                    kn_sq = np.dot(kn, kn)
                    
                    # 1/epsilon
                    inv_eps = 1.0 / eps_coeff if eps_coeff != 0 else 1.0
                    
                    M[i, j] = inv_eps * kn_sq
            
            # Solve eigenvalue problem
            # M E = lambda E
            # lambda = k^2
            eigvals = np.linalg.eigvalsh(M)
            # eigvals are real (M is Hermitian if epsilon is real and symmetric)
            # Ensure positive
            eigvals = np.real(eigvals)
            eigvals = np.sort(eigvals)
            
            # Extract bands 4 and 5 (indices 3 and 4)
            if len(eigvals) >= 5:
                omega_4 = np.sqrt(eigvals[3])
                omega_5 = np.sqrt(eigvals[4])
                
                # Update global min/max
                if omega_4 > max_omega_4:
                    max_omega_4 = omega_4
                if omega_5 < min_omega_5:
                    min_omega_5 = omega_5
        
        if min_omega_5 == float('inf'):
            return 0.0
            
        gap = 2 * (min_omega_5 - max_omega_4)
        midgap = min_omega_5 + max_omega_4
        ratio = gap / midgap if midgap != 0 else 0.0
        return ratio

    # --- Genetic Algorithm ---
    
    def random_individual():
        """Generates a random 16x16 binary matrix."""
        return np.random.randint(0, 2, (16, 16))
    
    def fitness(ind):
        """Calculates the fitness of an individual."""
        cell_32 = expand_symmetry(ind)
        return solve_pw(cell_32)
    
    def crossover(parent1, parent2):
        """Uniform crossover."""
        mask = np.random.randint(0, 2, parent1.shape)
        child = np.where(mask == 1, parent1, parent2)
        return child
    
    def mutate(ind, rate=0.05):
        """Bit flip mutation."""
        mask = np.random.rand(ind.shape) < rate
        ind[mask] = 1 - ind[mask]
        return ind
    
    # Initialize population
    pop_size = 100
    generations = 100
    
    population = [random_individual() for _ in range(pop_size)]
    
    # Evaluate initial population
    scores = [fitness(ind) for ind in population]
    
    best_score = max(scores)
    best_ind = population[scores.index(best_score)]
    
    for gen in range(generations):
        if time.time() - start_time > time_limit:
            break
            
        # Sort by fitness
        sorted_inds = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
        new_population = []
        
        # Elitism: keep top 10%
        elite_count = pop_size // 10
        for i in range(elite_count):
            new_population.append(sorted_inds[i][1].copy())
            
        # Fill the rest with offspring
        while len(new_population) < pop_size:
            # Tournament selection
            def select():
                candidates = random.sample(sorted_inds, 3)
                return max(candidates, key=lambda x: x[0])[1]
            
            p1 = select()
            p2 = select()
            child = crossover(p1, p2)
            child = mutate(child)
            new_population.append(child)
            
        population = new_population
        scores = [fitness(ind) for ind in population]
        
        gen_best = max(scores)
        if gen_best > best_score:
            best_score = gen_best
            best_ind = population[scores.index(best_score)].copy()
            
    # Return the best individual found
    best_cell_32 = expand_symmetry(best_ind)
    return best_cell_32.tolist()

