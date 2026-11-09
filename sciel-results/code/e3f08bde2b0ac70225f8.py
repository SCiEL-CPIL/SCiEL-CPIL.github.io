import numpy as np
from scipy.linalg import solve_banded

class ExercisePolicy:
    def __init__(self, boundaries, K, r, dt):
        # boundaries: list of length 11, boundaries[i] is the exercise price for step i+1
        # If S <= boundaries[i], exercise.
        self.boundaries = boundaries
        self.K = K
        self.r = r
        self.dt = dt
        self.df = np.exp(-r * dt)

    def exercise(self, current_state, time):
        current_state = np.asarray(current_state)
        # Map time to step index k (1..12)
        if isinstance(time, (int, np.integer)):
            k = int(time)
        else:
            k = int(round(time * 12))
            
        # Clamp k to valid range 1..12
        k = max(1, min(12, k))
        
        # Immediate payoff
        P = np.maximum(self.K - current_state, 0.0)
        
        if k == 12:
            # Last exercise date: exercise if payoff > 0
            return P > 0.0
        
        # For k < 12, use precomputed boundary
        # boundaries index 0 corresponds to step 1, ..., index 10 corresponds to step 11
        idx = k - 1
        boundary = self.boundaries[idx]
        
        return current_state <= boundary

def fit_exercise_policy(training_paths, contract):
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_steps = contract.get('n_steps', 12)
    dt = T / n_steps
    
    # Grid parameters
    N_S = 1000
    S_max = 2.0 * K
    S = np.linspace(0, S_max, N_S)
    dS = S[1] - S[0]
    
    # Initialize value at T (step 12)
    # V at step 12 is just the payoff
    V_next = np.maximum(K - S, 0.0)
    
    boundaries = []
    
    # Precompute coefficients for Crank-Nicolson
    # The PDE is: V_t + 0.5 sigma^2 S^2 V_SS + r S V_S - r V = 0
    # We solve for V at time t_k given V at t_{k+1}
    # Using Crank-Nicolson: (V^{k+1} - V^k) / dt = 0.5 * (L(V^{k+1}) + L(V^k))
    # Where L is the spatial operator: 0.5 sigma^2 S^2 V_SS + r S V_S - r V
    # Rearranging:
    # V^k - 0.5 dt L(V^k) = V^{k+1} + 0.5 dt L(V^{k+1})
    # Let A be the matrix for (I - 0.5 dt L) and B be the matrix for (I + 0.5 dt L)
    # We need to solve A * V^k = B * V^{k+1}
    
    # Construct tridiagonal matrices
    # We represent the tridiagonal matrix in banded form for scipy.linalg.solve_banded
    # Banded form: (l, u) where l is lower bands, u is upper bands
    # For a tridiagonal matrix, l=1, u=1
    # Shape (3, N)
    # Row 0: upper diagonal
    # Row 1: main diagonal
    # Row 2: lower diagonal
    
    # Note: S=0 is a boundary. We handle it separately or include it in the grid.
    # At S=0, V=K (for put, since we exercise).
    # At S=S_max, V=0.
    
    # We will solve for indices 1 to N_S-2 (interior points)
    # Indices 0 and N_S-1 are fixed by boundary conditions.
    
    # Coefficients for the spatial operator L at point i:
    # L(V)_i = 0.5 sigma^2 S_i^2 (V_{i+1} - 2V_i + V_{i-1})/dS^2 + r S_i (V_{i+1} - V_{i-1})/(2 dS) - r V_i
    
    # Let's compute the coefficients for the tridiagonal system for one time step.
    # The coefficients depend on S_i, so they change for each i.
    
    # We will build the matrices A and B for the interior points.
    # Let M = N_S - 2 be the number of interior points.
    # The system is A * V_int = B * V_next_int
    
    # However, constructing new matrices every step is expensive.
    # But since S is fixed, the coefficients are fixed.
    
    # Let's precompute the diagonal and off-diagonal coefficients for A and B.
    
    # For interior point i (1 <= i <= N_S-2):
    # L(V)_i = a_i V_{i-1} + b_i V_i + c_i V_{i+1}
    # a_i = 0.5 sigma^2 S_i^2 / dS^2 - r S_i / (2 dS)
    # b_i = - sigma^2 S_i^2 / dS^2 - r
    # c_i = 0.5 sigma^2 S_i^2 / dS^2 + r S_i / (2 dS)
    
    S_int = S[1:-1]
    a = 0.5 * sigma**2 * S_int**2 / dS**2 - r * S_int / (2 * dS)
    b = - sigma**2 * S_int**2 / dS**2 - r
    c = 0.5 * sigma**2 * S_int**2 / dS**2 + r * S_int / (2 * dS)
    
    # Matrix A (Left side: I - 0.5 dt L)
    # A_ii = 1 - 0.5 dt b_i
    # A_i,i-1 = - 0.5 dt a_i
    # A_i,i+1 = - 0.5 dt c_i
    
    # Matrix B (Right side: I + 0.5 dt L)
    # B_ii = 1 + 0.5 dt b_i
    # B_i,i-1 = 0.5 dt a_i
    # B_i,i+1 = 0.5 dt c_i
    
    M = len(S_int)
    
    # Construct banded matrices
    # A_banded shape (3, M)
    A_banded = np.zeros((3, M))
    B_banded = np.zeros((3, M))
    
    # Main diagonals
    A_banded[1, :] = 1.0 - 0.5 * dt * b
    B_banded[1, :] = 1.0 + 0.5 * dt * b
    
    # Lower diagonals (A_i, i-1)
    # In banded form, lower diagonal is in row 2, column i-1 corresponds to element (i, i-1)
    # So A_banded[2, 1:] = -0.5 * dt * a[:-1] ? 
    # Let's check scipy docs.
    # ab[i, j] is the coefficient for the (i-j, j) element if i-j >= 0?
    # Actually, for l=1, u=1:
    # ab[0, j] = a[j, j+1] (upper)
    # ab[1, j] = a[j, j] (main)
    # ab[2, j] = a[j, j-1] (lower) -- but this is stored in ab[2, j-1] ?
    # No, standard banded storage:
    # ab[i, j] holds the element a[j, j + i - l]
    # For l=1, u=1:
    # ab[0, j] = a[j, j+1]
    # ab[1, j] = a[j, j]
    # ab[2, j] = a[j, j-1]
    
    # So:
    # A_banded[0, 1:] = -0.5 * dt * c[1:]  # Upper diagonal of A, elements (i, i+1) for i=0..M-2
    # A_banded[2, :-1] = -0.5 * dt * a[:-1] # Lower diagonal of A, elements (i, i-1) for i=1..M-1
    
    # Wait, let's be careful.
    # Element A[i, i+1] is in A_banded[0, i]
    # Element A[i, i-1] is in A_banded[2, i]
    
    # So:
    A_banded[0, :-1] = -0.5 * dt * c[:-1] # A[i, i+1] for i=0..M-2
    A_banded[2, 1:] = -0.5 * dt * a[1:]   # A[i, i-1] for i=1..M-1
    
    B_banded[0, :-1] = 0.5 * dt * c[:-1]  # B[i, i+1]
    B_banded[2, 1:] = 0.5 * dt * a[1:]    # B[i, i-1]
    
    # We will iterate backwards from step 11 to 1
    # V_next is currently V at step 12 (payoff)
    
    for k in range(11, 0, -1):
        # 1. Propagate V_next (which is V_{k+1}) to get Continuation Value C_k
        # Solve A * C_k_int = B * V_next_int
        # Boundary conditions for V_next are already in V_next[0] and V_next[-1]
        # We need to modify the RHS to account for the boundary terms if they are non-zero
        # But for Put, V(0)=K, V(S_max)=0.
        
        RHS = B_banded.copy()
        # Apply B to V_next_int? No, we need to compute the vector b = B * V_next_int
        # But B is a matrix. We can just multiply.
        # However, we stored B in banded form. It's easier to just compute the tridiagonal multiplication.
        
        # Let's construct the vector b = B * V_next_int
        b_vec = np.zeros(M)
        # Main diagonal contribution
        b_vec += B_banded[1, :] * V_next[1:-1]
        # Upper diagonal contribution (from V_next[i+1])
        # B[i, i+1] * V_next[i+1]
        b_vec[:-1] += B_banded[0, :-1] * V_next[2:]
        # Lower diagonal contribution (from V_next[i-1])
        # B[i, i-1] * V_next[i-1]
        b_vec[1:] += B_banded[2, 1:] * V_next[:-2]
        
        # Now solve A * C_k_int = b_vec
        C_k_int = solve_banded((1, 1), A_banded, b_vec)
        
        # Assemble C_k
        C_k = np.zeros(N_S)
        C_k[1:-1] = C_k_int
        # Boundary conditions for Continuation Value
        # At S=0, continuation value of a put is K * exp(-r * dt) ?
        # Actually, if we don't exercise, the value is the discounted expected value.
        # For S=0, the asset stays 0. The put payoff at next step is K.
        # So Continuation Value at S=0 is K * exp(-r * dt).
        C_k[0] = K * np.exp(-r * dt)
        C_k[-1] = 0.0
        
        # 2. Apply Exercise Condition
        # V_k = max(K - S, C_k)
        Payoff = np.maximum(K - S, 0.0)
        V_k = np.maximum(Payoff, C_k)
        
        # 3. Find Exercise Boundary
        # We want the largest S such that Payoff >= C_k
        # Since Payoff is decreasing and C_k is increasing (mostly), there is a crossing.
        # Find where Payoff - C_k changes sign from positive to negative.
        diff = Payoff - C_k
        # Find the last index where diff >= 0
        # Note: diff[0] is K - K*exp(-r dt) > 0
        # diff[-1] is 0 - 0 = 0 (or close to it)
        
        # We want the boundary price S*.
        # Linear interpolation between the last point where diff > 0 and the first point where diff < 0.
        # Or simply find the index.
        
        # Let's find the transition
        # diff is positive at start, negative at end (usually).
        # If diff is always positive, boundary is S_max (but we never exercise at high S).
        # If diff is always negative, boundary is 0 (we always exercise).
        
        # Find indices where diff >= 0
        pos_indices = np.where(diff >= 0)[0]
        if len(pos_indices) == 0:
            boundary = 0.0
        elif len(pos_indices) == N_S:
            boundary = S_max
        else:
            # The last index where diff >= 0
            i_last_pos = pos_indices[-1]
            # The first index where diff < 0
            i_first_neg = i_last_pos + 1
            
            # Linear interpolation
            S1 = S[i_last_pos]
            S2 = S[i_first_neg]
            d1 = diff[i_last_pos]
            d2 = diff[i_first_neg]
            
            if d1 == d2:
                boundary = S1
            else:
                # Root of d1 + (d2-d1)*(S-S1)/(S2-S1) = 0
                boundary = S1 - d1 * (S2 - S1) / (d2 - d1)
        
        boundaries.append(boundary)
        
        # Update V_next for the next iteration (which is step k-1)
        V_next = V_k
    
    # Reverse boundaries because we built them from step 11 down to 1
    # boundaries[0] is for step 11, ..., boundaries[10] is for step 1
    # We need boundaries for step 1 at index 0, ..., step 11 at index 10
    boundaries = boundaries[::-1]
    
    return ExercisePolicy(boundaries, K, r, dt)
