
from lowpass import LowPassFilter
from pid import PID

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, veloPID, yawCont):

    	self.veloPID = veloPID
    	self.yawCont = yawCont

    	self.lp_throttle = LowPassFilter(0.2, 0.1)

    def control(self, lin_vel, ang_vel, cur_vel, is_dbw_enabled, delta_t):
    	# cur_vel is in m/s
    	
    	print("cur_vel: %f", cur_vel)

        #cur_vel = cur_vel
        vel_error = lin_vel - cur_vel

        throttle = 0.0
        steer = 0.0
        brake = 0.0
 
        if is_dbw_enabled != True:
            self.veloPID.reset()
            vel_error = 0

        else:
	        #print("target ang_vel:", ang_vel)

	        throttle = self.veloPID.step(vel_error, delta_t)
	        throttle = self.lp_throttle.filt(throttle)

	        steer = self.yawCont.get_steering(lin_vel, ang_vel, cur_vel)

        if throttle <= 0.01:
        	throttle = 0.0
        	brake = abs(throttle) * 2500

        # Return throttle, brake, steer
        return throttle, brake, steer
