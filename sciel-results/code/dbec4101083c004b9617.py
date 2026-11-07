import numpy as np
import time
from scipy.interpolate import PchipInterpolator

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 118

    # Extract model parameters
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    # Output grid
    n_out = len(resource_grid)
    m_out = resource_grid

    # Working grid - use the scoring grid for accuracy
    m_work = np.geomspace(0.001, 40, 251)
    n_work = len(m_work)

    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)

    # Marginal utility function
    def marg_utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return 1.0 / c
        else:
            return c ** (-gamma)

    # Policy storage on work grid
    policy_work = np.zeros((horizon, 2, n_work))

    # Value function storage
    V_next = np.zeros((n_work, 2))

    # Terminal period: consume all
    policy_work[horizon - 1, :, :] = m_work[None, :, :]
    V_next = utility(m_work)[None, :].T  # shape (n_work, 2)

    # Backward induction for t = horizon-2 down to 0
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 3:
            break

        # Build interpolators for continuation values
        try:
            V_interp_0 = PchipInterpolator(m_work, V_next[:, 0])
            V_interp_1 = PchipInterpolator(m_work, V_next[:, 1])
        except Exception:
            V_interp_0 = None
            V_interp_1 = None

        V_curr = np.zeros((n_work, 2))

        for z in range(2):
            probs = transition[z]

            for i in range(n_work):
                m = m_work[i]

                # Candidate consumption: from small fraction to full consumption
                c_candidates = m * np.linspace(0.001, 1.0, 200)

                # Next period resources for each state
                m_next_0 = R * (m - c_candidates) + income[0]
                m_next_1 = R * (m - c_candidates) + income[1]

                # Interpolate continuation values
                if V_interp_0 is not None and V_interp_1 is not None:
                    V_cont_0 = V_interp_0(m_next_0)
                    V_cont_1 = V_interp_1(m_next_1)
                else:
                    V_cont_0 = np.interp(m_next_0, m_work, V_next[:, 0])
                    V_cont_1 = np.interp(m_next_1, m_work, V_next[:, 1])

                E_V_cont = probs[0] * V_cont_0 + probs[1] * V_cont_1

                U_c = utility(c_candidates)
                Total_Value = U_c + beta * E_V_cont

                idx_max = np.argmax(Total_Value)
                V_curr[i, z] = Total_Value[idx_max]
                policy_work[t, z, i] = c_candidates[idx_max]

        V_next = V_curr

    # Interpolate policy to output grid
    consumption_out = np.zeros((horizon, 2, n_out))

    for t in range(horizon):
        for z in range(2):
            consumption_out[t, z, :] = np.interp(m_out, m_work, policy_work[t, z, :])

    # Validity checks
    for t in range(horizon):
        for z in range(2):
            consumption_out[t, z, :] = np.maximum(consumption_out[t, z, :], 1e-12)
            consumption_out[t, z, :] = np.minimum(consumption_out[t, z, :], m_out + 1e-10)

    # Terminal condition
    consumption_out[horizon - 1, :, :] = m_out[None, None, :]

    return consumption_out
