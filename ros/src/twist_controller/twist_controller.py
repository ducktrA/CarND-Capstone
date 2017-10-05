
GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, veloPID, yawCont):

    	self.veloPID = veloPID
    	self.yawCont = yawCont

        # TODO: Implement
        pass

    def control(self, lin_vel, ang_vel, cur_vel, is_dbw_enabled, delta_t):

        cur_vel = cur_vel * ONE_MPH
        vel_error = lin_vel - cur_vel
 
        if is_dbw_enabled != True:
            self.veloPID.reset()
            vel_error = 0

        print("lin_vel:", lin_vel , "cur_vel: ", cur_vel)

        throttle = self.veloPID.step(vel_error, delta_t)
        steer = self.yawCont.get_steering(lin_vel, ang_vel, cur_vel)
        brake = 0.

        if throttle <= 0.:
        	throttle = 0
        	brake = abs(throttle) * 2500

        # Return throttle, brake, steer
        return throttle, brake, steer
