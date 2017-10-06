
from lowpass import LowPassFilter
from pid import PID

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, veloPID, yawCont, delta_t):

    	self.veloPID = veloPID
    	self.yawCont = yawCont

    	self.lp_yaw = LowPassFilter(0.5, delta_t)
    	self.lp_throttle = LowPassFilter(0.5, delta_t)
    	self.steering_controller = PID(5, 0.05, 1, -0.1, 0.1)

    	self.delta_t = delta_t

        # TODO: Implement
        pass

    def control(self, lin_vel, ang_vel, cur_vel, cur_angvel, is_dbw_enabled):

        cur_vel = cur_vel * ONE_MPH
        vel_error = lin_vel - cur_vel

        throttle = 0.0
        steer = 0.0
        brake = 0.0
 
        if is_dbw_enabled != True:
            self.veloPID.reset()
            self.steering_controller.reset()
            vel_error = 0
        else:
	        print("target ang_vel:", ang_vel , "cur_angvel: ", cur_angvel)

	        throttle = self.veloPID.step(vel_error, self.delta_t)
	        throttle = self.lp_throttle.filt(throttle)

	        #steer_pid_influence = self.steering_controller.step(ang_vel, self.delta_t)
	        steer = self.yawCont.get_steering(lin_vel, ang_vel, cur_vel)
        
        #steer = self.lp_yaw.filt(steer)

        if throttle <= 0.0:
        	throttle = 0.0
        	brake = abs(throttle) * 2500

        # Return throttle, brake, steer
        return throttle, brake, steer
