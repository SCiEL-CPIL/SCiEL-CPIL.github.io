    def check_feasibility(route):
        if not route:
            return True, 0.0, 0.0
        
        total_demand = sum(demands[c] for c in route if c not in stations)
        if total_demand > capacity + 1e-9:
            return False, 0.0, 0.0
        
        t = 0.0
        bat = float(battery_max)
        total_d = 0.0
        prev = 0
        
        for c in route:
            d = dist_mat[prev, c]
            t += d / speed
            bat -= d
            if bat < -1e-9:
                return False, 0.0, 0.0
            
            if c in stations:
                # Charging
                charge_time = (battery_max - bat) * recharge_rate
                t += charge_time
                bat = battery_max
            else:
                # Customer
                if t < ready[c]:
                    t = ready[c]
                if t > due[c] + 1e-9:
                    return False, 0.0, 0.0
                t += service[c]
            
            total_d += d
            prev = c
        
        d = dist_mat[prev, 0]
        t += d / speed
        bat -= d
        if bat < -1e-9:
            return False, 0.0, 0.0
        if t > horizon + 1e-9:
            return False, 0.0, 0.0
        
        return True, total_d + d, t
