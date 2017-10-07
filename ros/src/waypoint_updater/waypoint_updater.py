#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 100 # Number of waypoints we will publish. You can change this number
CORRIDOR = math.degrees(60.) # within this angle to the left and right of the ego we accept base_waypoints
INVERSE_WP_DENSITY = 5

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below

        self.pose = None # pose.position, pose.orientation (quaternion)
        self.base_waypoints = None # list of pose's.
        self.closest_before = 0
        self.frame_id = None

        self.frequency = 10
        # TODO: Needs to be adjusted based on target speed
        self.tl_slow_down_dst_wp = 150
        self.tl_stopp_dst_wp = 10
        self.tl_idx = -1 #nearest traffic light with red status
        self.target_speed = (50.0*1000)/(60*60) #[m/s] = [km/h] * 60*60/1000 (e.g.40 km/h = 11.11m/s)
        self.loop()

    def loop(self):
        # publish updates on fixed frequency as in dbw_node
        rospy.loginfo("loop")
        rate = rospy.Rate(self.frequency) # 50Hz
        while not rospy.is_shutdown():
            if self.base_waypoints != None and self.pose != None:
                #(x,y,z) = self.quaternion_to_euler_angle()

                closest = self.closest_wp(self.closest_before)

                # TODO wrap it over when it reaches the end
                points_ahead = min(closest + LOOKAHEAD_WPS, len(self.base_waypoints))
                #rospy.loginfo("Closest Base WP: %d Points Ahead: %d", closest, points_ahead)

                # TODO this is a very basic setup of the Lane
                finalwps = Lane()
                finalwps.header.stamp = rospy.Time.now()
                finalwps.header.frame_id = self.frame_id

                i = 0
                idx = closest

                while i < LOOKAHEAD_WPS:
                    if idx >= len(self.base_waypoints)-1:
                        idx = 0

                    # improve performance by considering only every x waypoints
                    speed_reduction = 0
                    # skip a few waypoints to improve performance but never skip traffic light waypoints (tl_idx)
                    if (i % INVERSE_WP_DENSITY) == 0 or idx == self.tl_idx:
                        if self.tl_idx != -1:
                            dst = self.tl_idx - idx
                            if dst > 0:
                                if dst < self.tl_stopp_dst_wp:
                                    dst = 0
                                speed_reduction = 1-min((dst*dst)/(self.tl_slow_down_dst_wp*self.tl_slow_down_dst_wp), 1) # deaccelerate in the proximity of 10 waypoints around the traffic light
                        velocity = self.target_speed * (1- speed_reduction)
                        self.set_waypoint_velocity(self.base_waypoints, idx, velocity)
                        finalwps.waypoints.append(self.base_waypoints[idx])
                        rospy.loginfo("wp: %d => v = %d", idx, velocity)

                    i = i + 1
                    idx = idx + 1



                self.closest_before = closest
                self.final_waypoints_pub.publish(finalwps)

            rate.sleep()

    def pose_cb(self, msg):
        # TODO: Implement
        self.pose = msg.pose
        pass

    def waypoints_cb(self, waypoints):
        # TODO: Implement
        self.base_waypoints = waypoints.waypoints
        self.frame_id = waypoints.header.frame_id
        # rospy.loginfo("waypoints received: %d", len(self.base_waypoints))
        pass

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        if msg.data != self.tl_idx:
            self.tl_idx = msg.data

        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def closest_wp(self, closest_before):
        dist = 999999999.
        closest = None

        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        #alphal = lambda a, b: math.atan2((a.y - b.y), (a.x - b.x))

        # limit the search space
        upper = min(len(self.base_waypoints), closest_before + 700)
        lower = max(0, closest_before - 50)

        lower = 0
        upper = len(self.base_waypoints)

        for i in range(lower, upper):
            d = dl(self.base_waypoints[i].pose.pose.position, self.pose.position)
            #a = alphal(self.base_waypoints[i].pose.pose.position, self.pose.position)

            # TODO: check if there is an offset
            #if d < dist and ((z - CORRIDOR) < a) and ((z+CORRIDOR) > a):

            if d < dist:
                dist = d
                closest = i

        return closest

    '''
    # it turns out that this is not necessary, waypoints are ascending order, the other lanes coming into the intersections are not modelled
    def get_next_wps(self, closest, z):

        direction = 1 # 1 count upwards, -1 count downwards
        alphal = lambda a, b: math.atan2((a.y - b.y), (a.x - b.x))
        closest_wp = self.base_waypoints[closest].pose.pose.position

        i = 1
        n = 0

        while i < LOOKAHEAD_WPS:
            n = (closest + i * direction) % len(self.base_waypoints)
            a = alphal(self.base_waypoints[n].pose.pose.position, closest_wp)

            if ((z - CORRIDOR) < a) and ((z+CORRIDOR) > a):
                # go ahead
                i = i+1
            else:
                # change direction
                direction = -1

        return n
    '''
    # actually not necessary
    def quat_to_euler(self):
        quaternion = (self.pose.orientation.x, self.pose.orientation.y, self.pose.orientation.z, self.pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        roll = euler[0]
        pitch = euler[1]
        yaw = euler[2]

        return roll, pitch, yaw

    def quaternion_to_euler_angle(self):
        # returns roll, pitch, yaw in radians
        # w, x, y, z):
        w = self.pose.orientation.w
        x = self.pose.orientation.x
        y = self.pose.orientation.y
        z = self.pose.orientation.z

        # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

        ysqr = y * y

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + ysqr)
        X = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        Y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (ysqr + z * z)
        Z = math.atan2(t3, t4)

        return X, Y, Z

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
