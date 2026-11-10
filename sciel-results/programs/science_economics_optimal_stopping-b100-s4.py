"""Bermudan exercise-policy: high-resolution binomial tree backward induction."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = 12  # 12 monthly exercise dates

    # Use a high-resolution binomial tree for accurate continuation values
    # 1200 steps as per the knowledge that improved accuracy significantly
    n_steps = 1200
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # Build the tree states
    # At step i, the stock price can be S0 * u^(i-j) * d^j for j=0..i
    # We only need to track the continuation value at each node
    # But 1200 steps means ~1200*1200/2 = 720,000 nodes, which is manageable

    # Actually, let's think about this more carefully.
    # The Bermudan option can only be exercised at 12 specific dates.
    # We can use a tree that has exactly 12 exercise dates but uses many
    # sub-steps between them for accuracy.

    # Let's use 100 sub-steps between each exercise date = 1200 total steps
    # But we only make exercise decisions at the 12 exercise dates.

    # Build tree: at each sub-step, the price evolves. At exercise dates, we decide.
    # For backward induction, we go from T back to 0.

    # Let's use a simpler approach: compute the continuation value using a
    # high-resolution tree, but only at the exercise dates do we compare
    # with exercise payoff.

    # Actually, the cleanest approach: use the 1200-step tree for the
    # underlying, but the option can only be exercised at 12 dates.
    # We do backward induction on the 1200-step tree, but at non-exercise
    # dates, the option value is just the discounted expected continuation.
    # At exercise dates, it's max(exercise, continuation).

    # Tree size: at step i, there are i+1 nodes. Total nodes = sum(1..1201) = 721,201
    # This is fine memory-wise.

    # But we need to be careful: the "current state" passed to exercise() is
    # the actual spot price at an exercise date, not a tree node index.
    # So we need to map from price to tree node, or use a different approach.

    # Better approach: precompute the optimal exercise boundary at each
    # exercise date. The boundary is a function of the stock price.
    # We can compute the continuation value as a function of price at each
    # exercise date using the tree, then the policy is: exercise if K - S > cont_value(S)

    # Let's compute the continuation value function at each of the 12 exercise
    # dates. We'll use the 1200-step tree.

    # Exercise dates: t_1, t_2, ..., t_12 where t_k = k * T / 12
    # In terms of tree steps: step_k = k * 100 (since 1200/12 = 100)

    # Backward induction on the tree:
    # At the final step (T), the option value is max(K - S, 0) if it's an exercise date
    # Actually, at T, it's the last exercise date, so value = max(K - S, 0)

    # Let's do this properly. We'll store the value at each node.
    # But we only need the value at exercise-date nodes as a function of price.

    # Alternative: use a coarser tree for the exercise decisions but with
    # many sub-steps. Let's use 12 "big" steps, each with 100 sub-steps.

    # Actually, let me just do the full 1200-step backward induction.
    # The state space is S0 * u^i * d^j = S0 * u^(i-2j) for j=0..i
    # So at step i, prices are S0 * u^(i-2j) for j=0..i, i.e., S0 * u^(-i), S0 * u^(-i+2), ..., S0 * u^i

    S0 = 100.0  # reference spot for tree construction

    # We'll work with log-prices for numerical stability
    # log(S) = log(S0) + (i-2j)*log(u)

    # Let's just use the multiplicative form
    # At step i, node j (0<=j<=i): S = S0 * u^(i-j) * d^j = S0 * u^(i-2j)

    # Value array: V[i, j] for step i, node j
    # We'll use a 2D array, but only fill the lower triangle

    # For memory efficiency, we can use two 1D arrays and swap
    # V[j] for current step, V_new[j] for previous step

    # Let's use the two-array approach
    # At step i, there are i+1 nodes
    # V[j] = value at step i, node j

    # Start at step n_steps
    # At the final step, if it's an exercise date, value = max(K - S, 0)
    # Step n_steps = 1200, which is exercise date 12 (since 1200/100 = 12)

    # Initialize at final step
    i = n_steps
    j = np.arange(i + 1)
    S_final = S0 * u ** (i - 2 * j)
    V = np.maximum(K - S_final, 0.0)

    # Now go backward
    # For each step from n_steps-1 down to 0
    # At each step, determine if it's an exercise date
    # Exercise dates are at steps 100, 200, ..., 1200
    # So step i is an exercise date if i % 100 == 0 and i > 0

    for i in range(n_steps - 1, -1, -1):
        # Continuation value: discounted expected value
        # V_cont[j] = disc * (p * V[j+1] + (1-p) * V[j])
        # where V[j+1] is the up-move and V[j] is the down-move
        # Wait, need to be careful with indexing

        # At step i, node j, up goes to step i+1, node j
        # Down goes to step i+1, node j+1
        # So V_cont[j] = disc * (p * V[j] + (1-p) * V[j+1])
        # where V is the array from step i+1 (length i+2)

        V_cont = disc * (p * V[:-1] + (1.0 - p) * V[1:])

        # Check if step i is an exercise date
        if i > 0 and i % 100 == 0:
            # Exercise date: take max of exercise and continuation
            j_curr = np.arange(i + 1)
            S_curr = S0 * u ** (i - 2 * j_curr)
            V_exercise = np.maximum(K - S_curr, 0.0)
            V = np.maximum(V_cont, V_exercise)
        else:
            V = V_cont

        # Check deadline
        if time.time() > deadline:
            break

    # Now V contains the value at step 0, node 0 (just one node)
    # But we need the exercise boundaries at each exercise date.
    # Let's redo this, storing the value functions at exercise dates.

    # Actually, we need to recompute, storing the value at each exercise date.
    # Let's do it again, but this time store the value arrays at exercise dates.

    start_time2 = time.time()

    # Recompute, storing value at exercise dates
    i = n_steps
    j = np.arange(i + 1)
    S_final = S0 * u ** (i - 2 * j)
    V = np.maximum(K - S_final, 0.0)

    # Store the value at the last exercise date
    exercise_values = {}
    exercise_values[12] = (j.copy(), V.copy())  # step 1200, exercise date 12

    for i in range(n_steps - 1, -1, -1):
        V_cont = disc * (p * V[:-1] + (1.0 - p) * V[1:])

        if i > 0 and i % 100 == 0:
            j_curr = np.arange(i + 1)
            S_curr = S0 * u ** (i - 2 * j_curr)
            V_exercise = np.maximum(K - S_curr, 0.0)
            V = np.maximum(V_cont, V_exercise)
            exercise_date = i // 100
            exercise_values[exercise_date] = (j_curr.copy(), V.copy())
        else:
            V = V_cont

        if time.time() > deadline:
            break

    # Now we have, for each exercise date k (1..12), the value V[j] at each node j
    # and the stock price S[j] at each node.
    # The policy is: exercise if K - S > V_cont, i.e., if S < K - V_cont
    # But V[j] = max(V_cont[j], K - S[j]), so exercise if V[j] == K - S[j]
    # i.e., if K - S[j] >= V_cont[j]
    # The boundary is: exercise if S <= boundary_k
    # where boundary_k is the price at which K - S = V_cont

    # For each exercise date, find the threshold price
    # For each node, if it's an exercise node, the threshold is S[j]
    # We need to find, for each exercise date, the maximum S such that exercise is optimal

    # Actually, the policy should be: at exercise date k, given price S,
    # exercise if K - S >= continuation_value(S)
    # The continuation_value is a function of S that we can interpolate from the tree.

    # For the exercise() function, we'll interpolate the continuation value
    # from the tree nodes.

    # Let's precompute, for each exercise date, the arrays of (S, V_cont)
    # so we can interpolate.

    # Recompute to get V_cont at exercise dates
    i = n_steps
    j = np.arange(i + 1)
    S_final = S0 * u ** (i - 2 * j)
    V = np.maximum(K - S_final, 0.0)

    cont_values = {}  # exercise_date -> (S_array, V_cont_array)
    cont_values[12] = (S_final.copy(), np.maximum(K - S_final, 0.0).copy())
    # At the last exercise date, continuation is just the payoff (no future)

    for i in range(n_steps - 1, -1, -1):
        V_cont = disc * (p * V[:-1] + (1.0 - p) * V[1:])

        if i > 0 and i % 100 == 0:
            j_curr = np.arange(i + 1)
            S_curr = S0 * u ** (i - 2 * j_curr)
            V_exercise = np.maximum(K - S_curr, 0.0)
            V = np.maximum(V_cont, V_exercise)
            exercise_date = i // 100
            cont_values[exercise_date] = (S_curr.copy(), V_cont.copy())
        else:
            V = V_cont

        if time.time() > deadline:
            break

    # Now define the policy
    class BermudanPolicy:
        def __init__(self, cont_values, K):
            self.cont_values = cont_values
            self.K = K
            self.num_exercise = 12

        def exercise(self, current_state, time):
            # current_state: array of spot prices
            # time: the current time index (1-based, 1..12 for exercise dates)
            # Return boolean array: True if exercise

            # Determine which exercise date we're at
            # time could be a float or int representing the exercise date index
            if isinstance(time, (int, np.integer)):
                k = int(time)
            else:
                # time might be a continuous time value
                # Map to exercise date index
                k = int(np.round(time * 12))
                k = max(1, min(12, k))

            if k not in self.cont_values:
                return np.zeros(len(current_state), dtype=bool)

            S_nodes, V_cont_nodes = self.cont_values[k]

            # Interpolate continuation value for each current_state
            # S_nodes is sorted? Let's check: S0 * u^(i-2j) for j=0..i
            # As j increases, the exponent decreases, so S decreases
            # So S_nodes is in decreasing order
            # np.interp requires increasing x, so flip

            S_sorted = S_nodes[::-1]
            V_sorted = V_cont_nodes[::-1]

            # Clip current_state to the range of S_nodes
            S_min = S_sorted[0]
            S_max = S_sorted[-1]
            S_clipped = np.clip(current_state, S_min, S_max)

            V_cont_interp = np.interp(S_clipped, S_sorted, V_sorted)

            # Exercise if K - S >= V_cont
            exercise_payoff = np.maximum(self.K - current_state, 0.0)
            should_exercise = exercise_payoff >= V_cont_interp

            return should_exercise

    return BermudanPolicy(cont_values, K)
# EVOLVE-BLOCK-END
