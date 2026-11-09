"""
Affine Model Recovery for BBOB f5 (Linear slope).

Strategy:
1. The function f5 is linear (affine). The current strategy finds the minimum
   on the box boundary (corners), which is suboptimal if the unconstrained
   minimum lies inside the box or if the clipping allows reaching it.
2. We recover the exact linear model parameters (offset and gradient vector)
   using d+1 evaluations.
3. We analytically determine the minimizer of this linear model over the box
   [-5, 5]^d. Since f(x) = c + v^T x, the minimum is at x_i = -5 if v_i > 0
   and x_i = 5 if v_i < 0.
4. Crucially, the problem states that func(x) clips x to the box. If the
   "hidden" transformation (shift/rotation) maps the box such that the true
   minimum is outside the visible box, the visible minimum is on the boundary.
   However, if we can estimate the gradient, we can check if the function is
   actually decreasing towards a point inside the box. Wait, for a linear
   function, the minimum over a convex polytope is always at a vertex.
   
   Let's re-read the task carefully: "minimizing func on [-5, 5]^5". 
   "func(x) clips a finite length-5 vector to the box and returns a float."
   This implies the input domain is effectively [-5, 5]^5 because anything
   outside is clipped. Therefore, the domain of optimization is strictly
   [-5, 5]^5.
   
   For a linear function $f(x) = c + v^T x$, the minimum over the hypercube
   $[-5, 5]^d$ is ALWAYS at one of the vertices.
   
   So why is the current score 2.5? And why might a different strategy be better?
   
   The current program evaluates all 32 vertices. This is optimal for finding
   the global minimum of a linear function on a hypercube.
   
   Is it possible the function is NOT purely linear? "BBOB-style function f5
   (Linear slope)". In BBOB, f5 is indeed "Linear Slope". It is defined as
   $f(x) = c + \sum_{i=1}^d w_i x_i$ where $w_i$ are weights. It is strictly
   linear.
   
   If the function is strictly linear, the minimum is at a vertex. The current
   program finds the exact minimum in 32 calls.
   
   Why would the score be less than 1.0 (perfect)?
   The score is AOCC. 
   $n_i = (\log_{10}(e_i) + 8) / 10$.
   $AOCC = 1 - \text{mean}(n_i)$.
   
   If we find the minimum at call 32, then for $i < 32$, $b_i$ is the best
   found so far. For $i \ge 32$, $b_i$ is the global minimum $f_{opt}$.
   $e_i = \text{clip}(b_i - f_{opt}, 1e-8, 100)$.
   
   If $b_i = f_{opt}$, $e_i = 1e-8$.
   $\log_{10}(1e-8) = -8$.
   $n_i = (-8 + 8) / 10 = 0$.
   
   So for $i \ge 32$, $n_i = 0$.
   For $i < 32$, $b_i > f_{opt}$, so $e_i > 1e-8$, so $n_i > 0$.
   
   To maximize AOCC, we want $n_i$ to be as small as possible as soon as
   possible. This means we want to find the global minimum (or a very good
   approximation) in as few calls as possible.
   
   The current program evaluates vertices in a fixed order (likely lexicographic
   or similar). The probability that the best vertex is found early depends on
   the random order or the specific instance.
   
   If we can estimate the gradient with fewer calls, can we find the best
   vertex faster?
   
   To determine the best vertex of a linear function, we need to know the sign
   of each component of the gradient vector $v$.
   $x_i^* = -5 \cdot \text{sign}(v_i)$ (with 0 mapped to arbitrary, but usually
   one side).
   
   If we can estimate the sign of $v_i$ for all $i$ quickly, we can jump
   directly to the best vertex.
   
   How many calls to estimate the sign of all $d$ components?
   We can use a finite difference approach or a simplex.
   
   A simplex of $d+1$ points allows us to solve for the gradient exactly (if
   noise-free).
   Points:
   $p_0 = 0$ (center)
   $p_i = 5 e_i$ (along axes)
   
   $f(p_0) = c$
   $f(p_i) = c + 5 v_i$
   $v_i = (f(p_i) - f(p_0)) / 5$
   
   This takes $d+1 = 6$ calls.
   Then we construct the best vertex $x^*$ based on the signs of $v_i$.
   Evaluate $x^*$ (1 call).
   Total 7 calls to find the global minimum.
   
   This is much faster than 32 calls.
   
   Let's verify if this is robust.
   The function is linear. The evaluation is deterministic (implied by BBOB
   style and lack of noise description).
   
   If we find the global minimum at call 7, then for $i \ge 7$, $n_i = 0$.
   For $i < 7$, we have intermediate values.
   
   Let's trace the first 7 calls for the simplex strategy:
   1. $x = 0$. $b_1 = f(0)$.
   2. $x = 5e_1$. $b_2 = \min(f(0), f(5e_1))$.
   3. $x = 5e_2$. $b_3 = \min(b_2, f(5e_2))$.
   ...
   6. $x = 5e_5$. $b_6 = \min(\dots)$.
   7. $x = x^*$. $b_7 = f(x^*) = f_{opt}$.
   
   This reduces the "warm-up" phase from ~16 average calls (random vertex search)
   or 32 worst-case calls to 7 calls.
   
   Can we do even better?
   Can we find the best vertex in fewer than 7 calls?
   To know the sign of $v_i$, we need information about the derivative in
   direction $e_i$.
   
   If we just pick random points, we might get lucky, but it's not guaranteed.
   The simplex approach is deterministic and guaranteed to find the exact
   gradient (and thus the best vertex) in $d+1$ calls.
   
   Is there a way to do it in $d$ calls?
   If we assume the offset $c$ is known or irrelevant for the sign determination?
   No, $c$ is unknown.
   
   However, note that $v_i = (f(x + h e_i) - f(x)) / h$.
   We need $d$ differences. Each difference requires 2 evaluations if we don't
   reuse points.
   With a star pattern (center + 2d neighbors), it's $2d+1$.
   With a simplex (center + d neighbors), it's $d+1$.
   
   So 6 evaluations to get the model, 1 to verify/optimize. Total 7.
   
   This should significantly improve the AOCC score because the first 6 calls
   will have higher errors, but calls 7-500 will have 0 error.
   Compared to the current program which might take up to 32 calls to find the
   minimum, the average index of finding the minimum drops from ~16 to 7.
   
   Let's calculate the expected AOCC improvement.
   Current: Average min-finding time ~16 (if random order) or 32 (worst case).
   New: Fixed min-finding time 7.
   
   The score is mean over 3 instances.
   
   Let's implement the simplex strategy.
   
   Steps:
   1. Evaluate $f(0)$.
   2. For $i=0..d-1$, evaluate $f(5 e_i)$.
   3. Compute $v_i = (f(5 e_i) - f(0)) / 5$.
   4. Construct $x^*$ where $x^*_i = 5$ if $v_i < 0$ else $-5$.
   5. Evaluate $f(x^*)$.
   6. Verify if $f(x^*)$ is indeed the minimum. Since it's linear, it is.
   7. Stall with $x^*$.
   
   Wait, what if $v_i$ is very close to 0?
   Then the sign might be unstable? No, floating point precision is fine for
   BBOB scales.
   
   What if the function is not perfectly linear due to floating point
   artifacts in the "hidden" transformation?
   The task says "BBOB-style function f5 (Linear slope)". It is generated by
   linear transformations. It should be numerically linear.
   
   One edge case: What if the gradient is zero? Then any point is the minimum.
   The simplex will find $v \approx 0$. $x^*$ will be arbitrary (e.g., all 5s).
   $f(x^*) = c$. This is the minimum.
   
   This strategy is strictly better than corner search in terms of convergence
   speed.
   
   Let's refine the stall phase.
   Once we have $x^*$, we re-evaluate it.
   
   Is there any risk?
   - Exception: No, standard numpy ops.
   - Budget: 7 calls + 493 stalls = 500 calls.
   - Time: Negligible.
   
   Let's write the code.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer that recovers the affine model of the linear function to find
    the global minimum in d+2 calls.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        
        d = self.dim
        calls_made = 0
        
        # Step 1: Evaluate at origin
        x0 = np.zeros(d)
        f0 = func(x0)
        calls_made += 1
        
        # Step 2: Evaluate along each axis at distance 5
        # This allows us to compute the gradient components directly
        # f(x) = f0 + sum(v_i * x_i)
        # f(5 e_i) = f0 + 5 * v_i  =>  v_i = (f(5 e_i) - f0) / 5
        v = np.zeros(d)
        for i in range(d):
            if time.time() > deadline:
                # Fallback if time runs out (should not happen)
                break
            xi = np.zeros(d)
            xi[i] = 5.0
            fi = func(xi)
            calls_made += 1
            v[i] = (fi - f0) / 5.0
            
        # Step 3: Determine the optimal vertex
        # For a linear function f(x) = f0 + v^T x, the minimum over [-5, 5]^d
        # is at x_i = 5 if v_i < 0, and x_i = -5 if v_i > 0.
        # If v_i == 0, either is fine.
        x_best = np.zeros(d)
        for i in range(d):
            if v[i] < 0:
                x_best[i] = 5.0
            else:
                x_best[i] = -5.0
                
        # Step 4: Evaluate the best vertex
        # This should be the global minimum
        f_best = func(x_best)
        calls_made += 1
        
        # Optional: Verify by checking the opposite vertex? 
        # No, linear function guarantees this is the min.
        # But just in case of numerical weirdness or if I misunderstood the domain,
        # let's stick with the analytic solution.
        
        # Step 5: Stall
        # We have used d + 2 calls (1 for center, d for axes, 1 for best).
        # For d=5, that's 7 calls.
        # Remaining budget: 500 - 7 = 493.
        
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            # Re-evaluate the best point
            func(x_best)
            calls_made += 1
            
        return x_best, f_best
# EVOLVE-BLOCK-END
