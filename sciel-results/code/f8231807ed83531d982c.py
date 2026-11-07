    def simulate_route(route):
        if not route:
            return True, 0.0, 0.0, battery_max
            
        curr = 0
        t = 0.0
        b = battery_max
        d_total = 0.0
        
        for node in route:
            d = dist[curr][node]
            b -= d
            d_total += d
            t += d / speed
            
            if b < 0:
                return False, float('inf'), t, b
            
            if node in stations_set:
                # Station
                charge_time = (battery_max - b) * recharge_rate
                t += charge_time
                b = battery_max
            else:
                # Customer
                if t < ready[node]:
                    t = ready[node]
                if t > due[node]:
                    return False, float('inf'), t, b
                t += service[node]
                
            curr = node
            
        # Return to depot
        d = dist[curr][0]
        b -= d
        d_total += d
        t += d / speed
        
        if b < 0 or t > horizon:
            return False, float('inf'), t, b
            
        return True, d_total, t, b
