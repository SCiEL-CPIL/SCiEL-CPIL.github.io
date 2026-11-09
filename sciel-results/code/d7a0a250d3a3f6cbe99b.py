"""Direct Corner Search for BBOB f5 (Linear slope)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Strategy:
    1. The function is linear (affine). The minimum of a linear function over a
       hypercube [-5, 5]^d is always located at one of the 2^d vertices (corners).
    2. Instead of estimating the gradient (which costs 2d+1 calls and risks
       estimation errors or noise), we explicitly evaluate all 32 corners.
    3. This guarantees finding the exact global minimum within the first 32 calls.
    4. The remaining budget is used to "stall" by re-evaluating the best point found
       (or a point extremely close to it) to maintain a low error value b_i for the
       rest of the 500 calls, maximizing the AOCC score.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        # Generate all 2^dim corners of the box [-5, 5]^dim
        # dim=5 -> 32 corners
        corners = np.array([
            np.array([val for val in corner])
            for corner in itertools.product([-5.0, 5.0], repeat=self.dim)
        ])

        x_best = None
        f_best = np.inf
        calls_made = 0

        # Evaluate all corners
        for i in range(len(corners)):
            if time.time() > deadline:
                break
            x = corners[i]
            fx = func(x)
            calls_made += 1

            if fx < f_best:
                f_best = fx
                x_best = x.copy()

        # If we haven't evaluated all corners yet (due to time limit), 
        # just return what we have. But with 32 calls and 300s limit, 
        # this should always complete.
        
        # Now we have found the exact global minimum (for a linear function).
        # The AOCC metric rewards low error early and consistently.
        # We have already made 32 calls. We have budget - 32 calls left.
        # To maximize AOCC, we should keep the best value b_i as low as possible.
        # Since we've found the global minimum, b_i is now f_opt (or very close).
        # e_i = clip(b_i - f_opt, 1e-8, 100) will be very small (approx 0 or 1e-8).
        # n_i = (log10(e_i) + 8) / 10 will be approx 0.
        # AOCC = 1 - mean(n_i) will be approx 1.
        
        # We must make exactly 500 calls if possible, or at least fill the budget.
        # The problem says "repeating the last n_i if fewer calls are made".
        # So we can just stop after 32 calls? No, it says "for each instance... calls it once".
        # And "Every call is recorded". 
        # "More than 500 calls... invalid".
        # "repeating the last n_i if fewer calls are made".
        # This implies we *can* make fewer than 500 calls, and the score is calculated 
        # based on the calls made, with the last value repeated for the remaining "slots".
        # However, making more calls doesn't hurt if we just re-evaluate the best point.
        # But wait, if we re-evaluate the same point, does it count as a new call?
        # "Every call is recorded". Yes.
        # If we re-evaluate x_best, we get f_best again. b_i remains f_best.
        # So n_i remains low.
        # There is no penalty for extra calls as long as we don't exceed 500.
        # And since we already found the global min, any further search is pointless 
        # for improving the value, but necessary to "fill" the timeline?
        # Actually, the score is mean(n_1, ..., n_500).
        # If we stop at call 32, the evaluator repeats n_32 for i=33..500.
        # Since n_32 is already very low (close to 0), repeating it is fine.
        # So we can simply return after evaluating the corners.
        # We don't need to waste time on dummy calls.
        
        # However, we must ensure we don't throw an exception.
        # And we must return something. The return value is ignored, but we should 
        # return the best x and f found to be safe/clean.
        
        if x_best is None:
            # Fallback if no calls were made (should not happen)
            x_best = np.zeros(self.dim)
            f_best = func(x_best)
            calls_made += 1

        return x_best, f_best

# Need to import itertools for product
import itertools
# EVOLVE-BLOCK-END
