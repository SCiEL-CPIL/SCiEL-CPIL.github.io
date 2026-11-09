# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    n = len(y0_arr)

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0

    dt = t1 - t0
    if abs(dt) < 1e-15:
        return y0_arr

    # Strategy: Reconstruct the linear system y' = A y + b
    # We need to determine A (n x n) and b (n).
    # Standard approach: 
    # b = f(t0, 0)
    # A[:, j] = f(t0, e_j) - b
    # Total calls: 1 (for b) + n (for A) = n + 1 = 16 calls for n=15.
    
    # Optimization:
    # We know f(t, y) = A y + b.
    # We can pick any set of n linearly independent vectors v_1, ..., v_n to determine A.
    # Let V = [v_1, ..., v_n].
    # Then F = [f(t0, v_1), ..., f(t0, v_n)] = A V + b 1^T.
    # If we pick v_1 = y0 (the initial condition), we save one call IF we were going to evaluate f(t0, y0) anyway?
    # No, we don't need f(t0, y0) for the solution formula y(dt) = exp(A dt) y0 + ...
    # However, we MUST determine A and b.
    # We need n+1 calls to determine n*n + n unknowns (A and b).
    # Wait, A has n^2 entries, b has n entries. Total n^2+n unknowns.
    # Each call to f gives n equations.
    # So we need at least n+1 calls to determine A and b uniquely (assuming generic A, b).
    # Can we do it in n calls?
    # If we assume the system is homogeneous (b=0), we need n calls. But it's a general linear system.
    # The prompt says "fixed 15-dimensional stiff linear system". It doesn't explicitly say homogeneous.
    # The seed program assumes y' = A y + b.
    # If we use n calls, we have n*n equations for n*n + n unknowns. Underdetermined.
    # Unless... we can assume something about b?
    # Or maybe we can solve for A and b simultaneously with fewer calls if we are clever?
    # No, generally n+1 is the minimum for affine linear functions in n dimensions.
    
    # Let's re-read the score: -log10(rel_err) - 0.5 * log10(calls).
    # Reducing calls from 16 to 15 gives a boost of -0.5 * log10(15/16) = -0.5 * log10(0.9375) = -0.5 * (-0.028) = 0.014.
    # This is a small but real gain.
    
    # How to save 1 call?
    # We need to determine A and b.
    # Let's evaluate f at n+1 points.
    # Points: 0, e_1, e_2, ..., e_{n-1}, y0?
    # If y0 is linearly independent from e_1...e_{n-1} and 0, we can solve for A and b.
    # Let's check if y0 is a standard basis vector or zero.
    # If y0 is zero, we can't use it as a basis vector distinct from 0.
    # If y0 is one of the e_j, we can skip that e_j.
    # Otherwise, we can use y0 as the n-th basis vector instead of e_n.
    # We still need the call for 0 to get b.
    # So: 
    # 1. b = f(t0, 0)
    # 2. For j in 0..n-1:
    #    if j < n-1: v_j = e_j
    #    else: v_j = y0 (if y0 is not in span(e_0...e_{n-2}))
    #    Actually, simpler:
    #    We need n vectors v_1...v_n such that V = [v_1...v_n] is invertible.
    #    We also need b.
    #    We can get b from f(t0, 0).
    #    Then A V = F - b 1^T.
    #    A = (F - b 1^T) V^{-1}.
    #    This requires n calls for V columns + 1 call for b = n+1 calls.
    
    # Is there a way to get b "for free"?
    # If we pick v_1 = y0, and assume y0 != 0.
    # We still need b.
    # What if we don't evaluate at 0?
    # We have n calls f(t0, v_i).
    # We have A v_i + b = f_i.
    # This is a system of n*n equations for n*n + n unknowns.
    # We have n degrees of freedom left. We cannot uniquely determine A and b.
    # Unless... the problem is homogeneous?
    # "stiff linear system" usually implies y' = A(t)y. But "linear initial-value problem" can be affine.
    # The seed program handles affine (y' = Ay + b).
    # If the test case is homogeneous (b=0), we can save the call for b.
    # How to detect if b=0?
    # If we probe n vectors v_1...v_n, we can try to solve for A assuming b=0.
    # A = F V^{-1}.
    # Then check if A v_i = f_i for all i. If yes, b is likely 0.
    # But if b != 0, this A will be wrong.
    # We can check consistency.
    # If we assume b=0, we use n calls.
    # If it fails consistency check, we fall back to n+1 calls.
    # But we don't know if it fails until we check.
    # The check requires computing A and verifying.
    # If it passes, we saved 1 call.
    # If it fails, we wasted n calls and need 1 more, total n+1.
    # But we don't know in advance if b=0.
    # However, for many standard stiff linear systems (like Lotka-Volterra linearized, or heat equation), b=0.
    # The prompt says "fixed 15-dimensional stiff linear system".
    # Let's try the n-call strategy first, and verify.
    # Verification: Compute A = F V^{-1}. Then compute residuals r_i = A v_i - f_i.
    # If max norm of r_i is small (e.g., < 1e-10 * scale), assume b=0.
    # If not, we need b. We already have n calls. We need 1 more call at 0 to get b.
    # Then recompute A using the augmented system or the formula A = (F - b 1^T) V^{-1}.
    # Wait, if we used v_1...v_n, and they are independent, we can compute b if we had one more point.
    # Actually, if we have n points and they are independent, the affine function is not uniquely determined.
    # We need n+1 points to determine an affine function in n dims.
    
    # So, plan:
    # 1. Choose n linearly independent vectors v_1...v_n.
    #    Use y0 as one of them if possible to save? No, y0 is just a vector.
    #    We need to evaluate f at these n vectors.
    # 2. Try to solve for A assuming b=0.
    # 3. Check residuals.
    # 4. If residuals small, use b=0. Total calls: n.
    # 5. If residuals large, we need b.
    #    We can get b by evaluating f(t0, 0). Total calls: n+1.
    #    Then compute A = (F - b 1^T) V^{-1}.
    
    # This adaptive approach saves 1 call IF b=0.
    # If b!=0, it uses n+1 calls, same as before.
    # Given the score penalty, this is a good bet.
    
    # Selection of V:
    # We want V to be well-conditioned.
    # Standard basis is best.
    # v_1 = e_1, ..., v_{n-1} = e_{n-1}.
    # v_n = ?
    # If we use e_n, V = I.
    # If we use y0, V might be ill-conditioned if y0 is close to span(e_1...e_{n-1}).
    # Let's stick to standard basis for stability, UNLESS y0 is one of them.
    # Actually, the choice of V doesn't change the number of calls.
    # It only affects the numerical stability of V^{-1}.
    # I is the most stable.
    # So, let's use V = I.
    # v_j = e_j for j=1..n.
    
    # Step 1: Evaluate F_j = f(t0, e_j) for j=1..n.
    # Calls: n.
    
    # Step 2: Assume b=0. A_0 = F. (Since V=I, A = F I^{-1} = F).
    # Step 3: Check if A_0 e_j = F_j for all j.
    # This is trivially true if A_0 = F.
    # Wait. If V=I, then A = F.
    # The model is y' = F y + b.
    # At y=e_j, y' = F e_j + b = F[:, j] + b.
    # We measured F_j = f(t0, e_j).
    # So F[:, j] + b = F_j.
    # This means b = F_j - F[:, j] for all j.
    # For a consistent linear system, b must be the same for all j.
    # So we can check if F_j - F[:, j] is constant across j.
    # If it is constant (within tolerance), then b is that constant vector.
    # And A = F.
    # This requires NO extra calls!
    # We just check consistency of b derived from each column.
    
    # Let's verify this logic.
    # We have n calls: f(t0, e_j) for j=1..n.
    # Let R_j = f(t0, e_j).
    # Model: f(t, y) = A y + b.
    # R_j = A e_j + b = A[:, j] + b.
    # So A[:, j] = R_j - b.
    # This must hold for all j.
    # This implies R_j - A[:, j] = b for all j.
    # So R_j - A[:, j] must be the same vector for all j.
    # But we don't know A yet.
    # However, we can guess A.
    # If we assume b=0, then A[:, j] = R_j. So A = [R_1 ... R_n] = F_matrix.
    # Then check if R_j - F_matrix[:, j] is constant.
    # R_j - F_matrix[:, j] = 0 for all j.
    # So b=0 is consistent if and only if... wait.
    # If A = F_matrix, then R_j - A[:, j] = R_j - R_j = 0.
    # So b=0 is ALWAYS consistent with A=F_matrix?
    # No.
    # If the true system is y' = A_true y + b_true.
    # We measure R_j = A_true e_j + b_true.
    # If we set A_est = [R_1 ... R_n], then A_est[:, j] = R_j.
    # Then R_j - A_est[:, j] = 0.
    # So the "estimated b" is 0.
    # But the true b is b_true.
    # The model y' = A_est y + 0 will produce:
    # y' = [R_1 ... R_n] y.
    # At y=e_j, y' = R_j.
    # This matches the measurements R_j.
    # But does it match the true dynamics?
    # Only if b_true = 0.
    # If b_true != 0, then A_true e_j + b_true = R_j.
    # A_est = A_true + b_true 1^T.
    # Then A_est y = (A_true + b_true 1^T) y = A_true y + b_true (1^T y).
    # The true derivative is A_true y + b_true.
    # The estimated derivative is A_true y + b_true (sum(y_i)).
    # These are equal only if sum(y_i) = 1 or b_true = 0.
    # So we can't just assume b=0.
    
    # We need to detect if b=0.
    # How?
    # We have n measurements R_j.
    # If b=0, A = [R_1...R_n].
    # If b!=0, A = [R_1-b ... R_n-b].
    # We can't distinguish without another point.
    # UNLESS we use the initial condition y0.
    # We know the solution must start at y0.
    # But we don't have a "target" y(dt) to compare against.
    # We are computing y(dt).
    
    # Is there a property of "stiff linear systems" that implies b=0?
    # Often, yes. But not always.
    # If we guess wrong, the error will be large.
    # Score: -log10(rel_err). If we guess b=0 when b!=0, rel_err can be O(1), score ~ 0.
    # If we use n+1 calls, rel_err ~ 1e-15, score ~ 15 - 0.5 log10(16) ~ 14.
    # So guessing wrong is fatal.
    
    # Can we verify b=0 using y0?
    # No, y0 is just a point.
    
    # What if we use n+1 calls always, but optimize the choice of points to be faster?
    # The current program uses 16 calls.
    # Can we reduce to 15 calls safely?
    # Only if we are sure b=0.
    # Or if we can solve for A and b with n calls + 1 constraint.
    # Constraint: The system is linear.
    # We have n equations (vectors) from n calls.
    # Unknowns: n^2 + n.
    # We are underdetermined by n.
    # We need 1 more call to get n more equations.
    
    # So, 15 calls is not enough for general affine linear system.
    # 16 calls is the minimum.
    
    # Wait, the inspiration says: "Extract the system matrix A using only n calls by leveraging the initial condition y0 as the first basis vector... reducing total function evaluations to the theoretical minimum of 15".
    # How is that possible?
    # Maybe the system is homogeneous?
    # If the system is homogeneous (b=0), then n calls are enough.
    # If the evaluator uses a homogeneous system, we save 1 call.
    # If the evaluator uses an affine system, we fail.
    # The prompt says "stiff linear system". In many contexts, "linear system" for ODEs means y' = A(t)y.
    # "Affine" is y' = A(t)y + b(t).
    # If the seed program handles affine, and scores 1.16, it implies the current solution works for the test case.
    # If the test case were affine, the seed would work.
    # If the test case is homogeneous, the seed works (b is estimated as 0 or close to 0? No, seed probes 0).
    # The seed probes 0 to get b. If b=0, it gets 0.
    # So the seed works for both.
    
    # If I switch to n calls (assuming b=0), I risk failing if b!=0.
    # But if b=0, I gain score.
    # How likely is b=0?
    # "Stiff linear system" often refers to homogeneous systems in numerical analysis benchmarks (e.g., Robertson, BDF test problems).
    # Let's check the score of the seed. 1.16.
    # If I use 15 calls and b=0, score improves.
    # If I use 15 calls and b!=0, score crashes.
    # Is there a way to be safe?
    # Maybe check if f(t0, 0) is close to 0?
    # That requires 1 call.
    # If f(t0, 0) == 0, then b=0.
    # Then we need n more calls for A. Total n+1 = 16.
    # No savings.
    
    # What if we use y0 as one of the basis vectors?
    # We need n basis vectors.
    # If we use y0, e_1, ..., e_{n-1}.
    # We need to evaluate f at these n points.
    # This gives n calls.
    # We still don't know b.
    # We have n*n equations for n*n + n unknowns.
    # Still underdetermined.
    
    # Conclusion: You cannot uniquely determine A and b with n calls for a general affine system.
    # The inspiration might be flawed or assumes homogeneity.
    # Given the risk, I should stick to 16 calls but optimize the computation.
    # Or, I can try the 15-call strategy (assuming b=0) and if the result looks "wrong" (e.g., non-finite or huge), fall back?
    # But we don't have a reference to compare against.
    
    # Let's look at the score again.
    # -log10(rel_err) - 0.5 log10(calls).
    # If rel_err is 1e-15, -log10 is 15.
    # Calls 16: 15 - 0.5*1.2 = 14.4.
    # Calls 15: 15 - 0.5*1.176 = 14.41.
    # The gain is tiny.
    # The risk is total failure.
    
    # Is there a different strategy?
    # What if we use a higher precision method for expm?
    # The current program uses `eig` and `expm`.
    # `expm` is robust.
    # Maybe the bottleneck is not the calls, but the precision of expm for very stiff systems?
    # Stiff systems have eigenvalues with large negative real parts.
    # exp(lambda * dt) can underflow to 0.
    # This is correct behavior.
    # But if there are eigenvalues with large positive real parts? Unlikely for stable stiff systems.
    # If the system is stable, expm should be fine.
    
    # Let's try to improve the precision of the solution.
    # Use `scipy.linalg.expm` which is standard.
    # Maybe use `scipy.integrate.solve_ivp` with a very high precision method as a fallback or check?
    # No, that uses many calls.
    
    # What if we use the augmented matrix method but with a higher order Pade approximation?
    # `scipy.linalg.expm` uses Pade order 13.
    # We can implement a higher order Pade or Taylor series if needed.
    # But for n=15, expm is fast.
    
    # Let's stick to the 16-call strategy but ensure maximum robustness.
    # The current program tries `eig` first, then `expm`.
    # `eig` can be unstable if the matrix is defective.
    # `expm` is always stable.
    # I will remove the `eig` attempt and just use `expm` to save time and avoid instability.
    # This might slightly change the error profile, but likely for the better (more stable).
    
    # Also, I will ensure the matrix construction is as efficient as possible.
    
    # One more idea:
    # If the system is time-dependent?
    # "fixed 15-dimensional stiff linear system"
    # "y' = f(t, y)"
    # The seed assumes time-independent A and b.
    # If A(t) is not constant, the seed fails.
    # But the prompt says "fixed... system", implying constant coefficients.
    
    # I will proceed with the robust 16-call expm strategy.
    # To differentiate from the seed (which tries eig first), I will use expm directly.
    # This is a structural change in the solver method (removing the fragile eig path).

    # 1. Probe b
    y_zero = np.zeros(n)
    b = np.array(f(t0, y_zero), dtype=float)

    # 2. Probe A
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        f_j = np.array(f(t0, e_j), dtype=float)
        A[:, j] = f_j - b

    if time.time() > deadline:
        return y0_arr

    # 3. Solve
    # Augmented matrix
    M = np.zeros((n + 1, n + 1))
    M[:n, :n] = A
    M[:n, n] = b
    
    z0 = np.zeros(n + 1)
    z0[:n] = y0_arr
    z0[n] = 1.0

    # Use expm directly for robustness
    try:
        expM = expm(M * dt)
        z_final = expM @ z0
        y_final = z_final[:n]
        
        if not np.all(np.isfinite(y_final)):
            # Fallback to solve_ivp if expm fails
            from scipy.integrate import solve_ivp
            sol = solve_ivp(f, (t0, t1), y0_arr, method='Radau', rtol=1e-12, atol=1e-14)
            if sol.success:
                return sol.y[:, -1]
            return y0_arr
        return y_final
    except Exception:
        # Last resort
        return y0_arr

# EVOLVE-BLOCK-END
