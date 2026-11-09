# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Thompson Sampling with a fixed seed for reproducibility and stability.
    # Using a fixed seed ensures that the exploration path is deterministic
    # relative to the sequence of calls, avoiding "bad luck" in the random draws
    # which can be detrimental in small-budget optimization.
    #
    # We use a single z-draw per acquisition call. To improve exploration in
    # the early stages and exploitation later, we could modulate sigma, but
    # the task says to keep the approach. However, we can improve the
    # deterministic seed strategy. The current seed 42 is fixed, so every call
    # uses the same z. This means the first candidate always gets the same
    # z[0], etc. This is actually a form of Latin hypercube sampling of the
    # random draws, which can be good for coverage.
    #
    # Let's try a different approach: use the call count to advance the RNG
    # state, so each call gets fresh random numbers. But we don't have access
    # to call count. We can use f_best as a proxy: as f_best decreases, we
    # can use it to seed the RNG. This way, the exploration pattern changes
    # as we find better solutions, potentially helping to escape local optima.
    #
    # Actually, let's think about this more carefully. The current approach
    # with fixed seed 42 means that z is always the same vector. So the
    # acquisition function is effectively: score = -(mu - sigma * z_fixed).
    # This is a deterministic function of mu and sigma. It's not really
    # "Thompson Sampling" in the traditional sense because the sample is
    # not fresh each time. It's more like a fixed randomized linear combination.
    #
    # For true Thompson Sampling, we want a fresh sample each time. But since
    # we can't track call count, let's use f_best to create a varying seed.
    # As f_best changes, the seed changes, giving us fresh random numbers.
    #
    # However, f_best might not change often (only when a new minimum is
    # found). So let's combine f_best with the current mu/sigma to create
    # a seed that changes more frequently.
    #
    # Actually, the simplest improvement: just use a different fixed seed that
    # might work better. Or, use multiple z draws and take the minimum? No,
    # that changes the approach.
    #
    # Let's try: use f_best to seed the RNG. This ensures that when we find
    # a new best, the exploration pattern changes.
    seed = int(abs(hash(f_best)) % (2**32))
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(size=mu.shape)
    sampled_values = mu - sigma * z
    return -sampled_values
# EVOLVE-BLOCK-END
