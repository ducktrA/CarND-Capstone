#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
#Comment for testing git push
import math

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.tl2wp_idx = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb, queue_size=1)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_rel_dst(self, obj_a, obj_b):
        x, y, z = obj_a.x - obj_b.x, obj_a.y - obj_b.y, obj_a.z - obj_b.z
        return math.sqrt(x*x + y*y + z*z)

    def get_rel_dst_hdg(self, obj_pos):
        """
        Checks if an object at a position "obj_pos" is within the vehicles field of view
        Args:
            obj_pos - object's pose
        Returns:

        """
        # Get position of traffic light and ego
        #light_pos = light.pose.pose.position
        ego_pos = self.pose.pose.position

        # Get ego heading
        q = self.pose.pose.orientation
        q_array = [q.x, q.y, q.z, q.w]
        _, _, hdg_ego = tf.transformations.euler_from_quaternion(q_array)

        # Calcualte the dist and angle to object
        dst_ego2obj = [obj_pos.x - ego_pos.x, obj_pos.y - ego_pos.y]
        hdg_ego2obj = math.atan2(dst_ego2obj[1], dst_ego2obj[0]) - hdg_ego
        # get shortest angle (e.g. -90 deg instead of 270 deg)
        if (hdg_ego2obj < -math.pi):
            hdg_ego2obj += 2.0 * math.pi
        elif (hdg_ego2obj > math.pi):
            hdg_ego2obj -= 2.0 * math.pi

        # Check if it's within FOV
        return self.get_rel_dst(ego_pos, obj_pos), hdg_ego2obj

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement => done OlWi
        close_wp_dst = 999999
        close_wp_idx = -1

        # find the clostes waypoint to the position "pose"
        for i in range(0, len(self.waypoints.waypoints)):
            waypoint_dst = self.get_rel_dst(self.waypoints.waypoints[i].pose.pose.position,pose.position)
            if waypoint_dst < close_wp_dst:
                # current waypoint is closest so far => update result candidate
                close_wp_dst = waypoint_dst
                close_wp_idx = i
        return close_wp_idx


    def project_to_image_plane(self, point_in_world):
        """Project point from 3D world coordinates to 2D camera image location

        Args:
            point_in_world (Point): 3D location of a point in the world

        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image

        """

        fx = self.config['camera_info']['focal_length_x']
        fy = self.config['camera_info']['focal_length_y']
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']

        # get transform between pose of camera and world frame
        trans = None
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link",
                  "/world", now, rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform("/base_link",
                  "/world", now)

        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")

        #TODO Use tranform and rotation to calculate 2D position of light in image

        x = 0
        y = 0

        return (x, y)

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        x, y = self.project_to_image_plane(light.pose.pose.position)

        #TODO use light location to zoom in on traffic light in image

        #Get classification
        #return self.light_classifier.get_classification(cv_image)
        return light.state



    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        #TODO find the closest visible traffic light (if one exists)
        light_state = TrafficLight.UNKNOWN
        stop_line_positions = self.config['stop_line_positions']
        tl_close_dst = 200       # ignore traffic lights farer away than that (range in m)
        tl_close_hdg_deg = 22.5 # ignore traffic lights with an bigger bearing than that (deg)
        tl_close_idx = -1       # will be set to the closest traffic light in the FOV and range
        tl_close_wp_idx = -1    # closest waypoint to the traffic light
        max_hdg_abs_deg = 22.5

        tl_wp = None
        if(self.pose and self.waypoints):
            for idx, tl_xy in enumerate(stop_line_positions):
                tl_pose = Pose()
                tl_pose.position.x = tl_xy[0]
                tl_pose.position.y = tl_xy[1]
                tl_pose.position.z = 2          # assuming the traffic light height is at around 2m (not relevcnt)

                tl_dst, tl_hdg = self.get_rel_dst_hdg(tl_pose.position)
                if tl_hdg < math.radians(tl_close_hdg_deg):
                    if tl_close_dst > tl_dst:
                        tl_close_idx    = idx
                        tl_close_dst    = tl_dst
                        tl_close_wp_idx = self.get_closest_waypoint(tl_pose)
            if tl_close_idx > -1:
                light_state = self.get_light_state(self.lights[tl_close_idx])
                # testing: set the light to red just to see weather the car is braking
            #rospy.loginfo("curr Wp = %d / next tl_wp = %d (%d)", idx, tl_close_idx, light_state)
        return tl_close_wp_idx, light_state



if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
