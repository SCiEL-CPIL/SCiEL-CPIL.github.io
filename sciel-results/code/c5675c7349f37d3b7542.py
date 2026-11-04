    def overlap_penalty(params):
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                ci = np.array([params[i*3], params[i*3+1]])
                cj = np.array([params[j*3], params[j*3+1]])
                dist = np.linalg.norm(ci - cj)
                
                # Calculate dynamic threshold based on relative angle
                ang_i = params[i*3+2]
                ang_j = params[j*3+2]
                d_theta = abs(ang_i - ang_j)
                # Normalize to [0, pi/3]
                d_theta = d_theta % (np.pi/3)
                if d_theta > np.pi/6:
                    d_theta = np.pi/3 - d_theta
                # Linear interpolation between sqrt(3) and 1+sqrt(3)/2
                d_min_0 = np.sqrt(3)
                d_min_30 = 1.0 + np.sqrt(3)/2.0
                threshold = d_min_0 + (d_min_30 - d_min_0) * (d_theta / (np.pi/6))
                
                # Add a small safety margin
                threshold += 0.01
                
                if dist < threshold:
                    penalty += (threshold - dist)**2
        return penalty
