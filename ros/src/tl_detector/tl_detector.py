#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import tf
import cv2
import yaml
import numpy as np
import os
import math
import datetime
import time

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
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        # Start sync variables:
        # Make sure the data is more or less from one timestamp.
        # It is synced by the image callback. During the image processing the
        # variable sync_active is True in order to avoid that other variables
        # do overwrite pose or light information from this image time.
        self.sync_active = False
        self.rot = None
        self.trans = None
        self.pose_temp = Pose()
        self.waypoints_temp = None
        self.lights_temp = None
        # End sync variables

        # Use this to turn on the image logging. Images will be saved in
        # the path training_img_path
        self.img_logging = 0        #0: off / 1: on
        self.training_img_path = "./training_images"

        # Will create a path to store the images
        if self.img_logging:
            self.last_img_dst = 0
            self.img_idx = 0
            if not (os.path.exists(self.training_img_path)):
                os.mkdir(self.training_img_path)


        rospy.spin()

    def pose_cb(self, msg):
        self.pose_temp = msg
        if not self.sync_active:
            self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints_temp = waypoints
        if self.sync_active:
            self.waypoints = waypoints


    def traffic_cb(self, msg):
        self.lights_temp = msg.lights
        if not self.sync_active:
            self.lights = msg.lights


    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """

        self.has_image = True
        self.sync_active = True # keep the other data being locked to be overwriten during image processing / at the end of this function we sync again
        self.camera_image = msg
        # get transform between pose of camera and world frame
        self.trans = None
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link","/world", self.camera_image.header.stamp, rospy.Duration(1.0))
            (self.trans, self.rot) = self.listener.lookupTransform("/base_link","/world", self.camera_image.header.stamp)
        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")
        #end
        if self.trans != None:
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

        #resync topic information
        self.pose = self.pose_temp
        self.waypoints = self.waypoints_temp
        self.lights = self.lights_temp
        self.sync_active = False


    def get_rel_dst(self, obj_a, obj_b):
        """
        Returns the relative distance as a scalar value in meters.
        Args:
            obj_a - position of first object
            obj_b - position of second object
        Returns:
            Absolute distance between first and second object in meters.
        """
        x, y, z = obj_a.x - obj_b.x, obj_a.y - obj_b.y, obj_a.z - obj_b.z
        return math.sqrt(x*x + y*y + z*z)

    def get_rel_dst_hdg(self, obj_pos):
        """
        Returns the bearing from the ego car to the objects position in radians.
        Args:
            obj_pos - object's position
        Returns:
            Ego vehicle's absolute distance and relative bearing to the object's position.
        """
        # Get position of ego vehicle
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


    def project_to_image_plane(self, point_in_world, width, height):
        """Project point from 3D world coordinates to 2D camera image location
        Args:
            point_in_world (Point): 3D location of a point in the world
            width : Width of the object which will be faced by the camera
            height : Height of the object which will be faced by the camera
        Returns:
            x1 (int): x coordinate of the lower right corner in the image coordinates
            y1 (int): y coordinate of the lower right corner in the image coordinates
            x1 (int): x coordinate of the upper left corner in the image coordinates
            y1 (int): y coordinate of the upper left corner in the image coordinates
        """
        # 1. Step: Coordinate Transformation in 3D world:
        # Get transformation matrix to transfer the object's coordinates (xyz)
        # from world to car's coordinate system
        transWorld2Car = self.listener.fromTranslationRotation(self.trans, self.rot)

        # Get transformation matrix to transfer the object's coordinates (xyz)
        # from car's to camera's coordinate system
        cam_roll     = math.radians(90)          # make make cars y axis (left) direction the direction of of camera y (heigheiheiht)
        cam_pitch    = math.radians(-90.65)      # turn camera left (-) or right (+)
        cam_yaw      = math.radians(-8.85)       # Turn the camera down (-) or up (+)
        cameraRotation = tf.transformations.quaternion_from_euler(cam_roll, cam_pitch, cam_yaw, axes="sxyx")
        transCar2Camera = self.listener.fromTranslationRotation((0,0,0), cameraRotation)

        # Perform the axis rotations from world to car to camera's coordinate system
        xyz_world    = np.array([point_in_world.x, point_in_world.y, point_in_world.z, 1.0])    # Last 1 is for translatation
        xyz_car      = transWorld2Car.dot(xyz_world)
        xyz_camera   = transCar2Camera.dot(xyz_car)     # This is the position for the center of the object in camera's coordinate system

        # Get the pixels for the bounding corners based on object's width and height
        x1 = xyz_camera[0] + 0.5 * width
        y1 = xyz_camera[1] + 0.5 * height
        x2 = xyz_camera[0] - 0.5 * width
        y2 = xyz_camera[1] - 0.5 * height
        z = xyz_camera[2]

        # 2. Step: Project the 3D coordinates in camera system to the
        # camera sensor 2D coorinate system
        camera_info = CameraInfo()
        fx = self.config['camera_info']['focal_length_x']
        fy = self.config['camera_info']['focal_length_y']
        camera_info.width = self.config['camera_info']['image_width']
        camera_info.height = self.config['camera_info']['image_height']

        # The farer the object away the closer its project will be to the image center.
        u1 = int( ((fx * x1) / z) + (camera_info.width * 0.5))
        v1 = int( ((fy * y1) / z) + (camera_info.height * 0.5))
        u2 = int( ((fx * x2) / z) + (camera_info.width * 0.5))
        v2 = int( ((fy * y2) / z) + (camera_info.height * 0.5))
        if u1 > 0 and v1 > 0 and u2 > 0 and v2 > 0 and u1 < camera_info.width and u2 < camera_info.width and v1 < camera_info.height and x2 < camera_info.height:
            #rospy.loginfo("u1 = " + str(u1) + "; v1 = " + str(v1) + "; u2 = " + str(u2) + "; v2 = " + str(v2))
            return u1, v1, u2, v2
        else:
            #rospy.loginfo(image out of camera sensor)
            return -1, -1, -1, -1


    def get_dom_color(self, box_img, color):
	
	img_shape = box_img.shape

	if color == TrafficLight.RED:
		#cut edge of the boxes to focus on the light
		cut_img = box_img[int(img_shape[0]*0.2):int(img_shape[0]*0.9), int(img_shape[1]*0.1):int(img_shape[1]*0.9)]
		#c_min = np.array([200,0,0], np.uint8)
		#c_max = np.array([255,50,50], np.uint8) 
		#HSV red treshold
		c_min = np.array([30,150,50], np.uint8)
		c_max = np.array([255,255,180], np.uint8)
		
	elif color == TrafficLight.YELLOW:
		#cut edge of the boxes to focus on the light
		cut_img = box_img[int(img_shape[0]*0.2):int(img_shape[0]*0.8), int(img_shape[1]*0.1):int(img_shape[1]*0.9)]
		#c_min = np.array([200,150,0], np.uint8)
		#c_max = np.array([255,255,50], np.uint8) 
		#HSV yellow treshold
		c_min = np.array([20,100,100], np.uint8)
		c_max = np.array([30,255,255], np.uint8) 
		
	elif color == TrafficLight.GREEN:
		#cut edge of the boxes to focus on the light
		cut_img = box_img[int(img_shape[0]*0.1):int(img_shape[0]*0.8), int(img_shape[1]*0.1):int(img_shape[1]*0.9)]
		#c_min = np.array([50,200,0], np.uint8)
		#c_max = np.array([120,255,50], np.uint8) 
		#HSV green treshold
		c_min = np.array([57,88,90], np.uint8)
		c_max = np.array([68,242,231], np.uint8)
		
	#get number of pixels with corresponnnnding color
	dst = cv2.inRange(cv2.cvtColor(cut_img, cv2.COLOR_BGR2HSV), c_min, c_max)
	c_cnt = cv2.countNonZero(dst)

	return c_cnt

	
	
    def get_light_state(self, light, dst):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False
        state = TrafficLight.UNKNOWN

        # Get the bounding box of the traffic light in opencv coordinates :
        # 0 1 2 3 4 5 6 .... (=x)
        # 1
        # 2
        # ... =y
        xr, yd, xl, yt = self.project_to_image_plane(light.pose.pose.position, 0.95, 2.2)
        box_height = yd - yt        # yd = vertical down / yt = vertical top
        box_width  = xr - xl        # xr = horizontal right / xl = horizontal left
        if xr > 0 and xl > 0 and yt > 0 and yd > 0:
            # Loada a anddd cd rcorpp trop the igege
            tl_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
            tl_image = tl_image[yt:yd, xl:xr]
            # resize_width =  0.5
            # tl_image = cv2.resize(tl_image, (resize_width, resize_width * crop_height / crop_width ))
            #cv2.imwrite("test.jpg", tl_image)

            # Detect traffic light based on pixel color

	    #get area of red emitter
	    box = tl_image[:box_height/3, :box_width]
	    red_pct = self.get_dom_color(box, TrafficLight.RED)
	    #get area of yellow emitter
	    box = tl_image[box_height/3:2*box_height/3, :box_width]
	    yellow_pct = self.get_dom_color(box, TrafficLight.YELLOW)
	    #get area of green emitter
	    box = tl_image[2*box_height/3:, :box_width]
	    green_pct = self.get_dom_color(box, TrafficLight.GREEN)
	    #determine box with the highest pixel numbers with corresponding color
	    state_pcts = np.array([red_pct, yellow_pct, green_pct])
	    state_idx = np.argmax(state_pcts)
	    #print(state_idx, np.array([red_pct, yellow_pct, green_pct]))
	
            if state_idx == 0:       # 0 = Red
                #rospy.loginfo("Pixel Detection Light State = Red")
                state = TrafficLight.RED
		if state_pcts[1] > 0: #red+yellow -> yellow
			if state_pcts[0]/state_pcts[1] < 0.5:
				state = TrafficLight.YELLOW
		if state_pcts[2] > 0: #red+green -> green
			if state_pcts[0]/state_pcts[2] < 0.5:
				state = TrafficLight.GREEN
            elif state_idx == 1:     # 1 = Yellow
                #rospy.loginfo("Pixel Detection Light State = Yellow")
                state = TrafficLight.YELLOW
            elif state_idx == 2:     # 2 = Green
                #rospy.loginfo("Pixel Detection Light State = Green")
                state = TrafficLight.GREEN
            else:
                #rospy.loginfo("Pixel Detection Light State = Unknown")
                state = TrafficLight.UNKNOWN
                #state = light.state

	    
            if self.img_logging:
                self.img_idx += 1
                if self.last_img_dst != dst:                        # Only store the the image if the distance changed
                    self.last_img_dst = dst
                    # Draw the bounding boxes for red, yellow, green
                    full_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
                    cv2.rectangle(full_image, (xr, yt), (xl, yt + box_height/3), (0,0,255), 2)
                    cv2.rectangle(full_image, (xr, yt + box_height/3), (xl, yt + 2*box_height/3), (0,0,255), 2)
                    cv2.rectangle(full_image, (xr, yt + 2*box_height/3), (xl, yd), (0,0,255), 2)
                    cv2.rectangle(full_image, (xl + box_width/2-5, yt + box_height*5/6-5), (xl + box_width/2+5, yt + box_height*5/6+5), (0,255,0), 5)
		    
                    crop_img_name = self.training_img_path + "/crop_state_" + str(state) + "-dst_" + str(dst) + "-idx" + str(self.img_idx)+".jpg"
                    rect_img_name = self.training_img_path + "/state_" + str(state) + "-dst_" + str(dst) + "-idx" + str(self.img_idx)+".jpg"
                    cv2.imwrite(rect_img_name, full_image)
                    cv2.imwrite(crop_img_name, tl_image)
        else:
            #rospy.loginfo("Traffic light out of image or cutoff => State = Unknown")
            #state = TrafficLight.UNKNOWN
            state = light.state
        return state



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
        tl_close_min_dst = 50   # ignore traffic lights farer than that (range in m)
        tl_min_dst = 10   # ignore traffic lights closer than that (range in m)
        tl_close_fov_deg = 22.5 # ignore traffic lights with an bigger bearing than that (deg)
        tl_close_idx = -1       # will be set to the closest traffic light in the FOV and range
        tl_close_wp_idx = -1    # closest waypoint to the traffic light
        tl_close_hdg_deg = -1
        tl_close_dst = -1
        max_hdg_abs_deg = 22.5

        box_widthp = None
        if(self.pose and self.waypoints) and len(self.lights) > 0 :
	    
	    for idx in range(0, len(self.lights)):
                tl_pose = Pose()
                tl_pose.position.x = self.lights[idx].pose.pose.position.x
                tl_pose.position.y = self.lights[idx].pose.pose.position.y
                tl_pose.position.z = self.lights[idx].pose.pose.position.z          # assuming the traffic light height is at around 2m (not relevcnt)

                tl_dst, tl_bearing = self.get_rel_dst_hdg(tl_pose.position)
		
                #rospy.loginfo("i = " + str(idx) + "; hdg = " + str(tl_bearing) + "; dst = " + str(tl_dst) + "; x/y/z" + str(tl_pose.position.x) + "; " + str(tl_pose.position.y) + "; " + str(tl_pose.position.z))

                if abs(tl_bearing) < math.radians(tl_close_fov_deg):
                    if tl_close_min_dst > tl_dst and tl_min_dst < tl_dst:
			#ts = time.time()
			#st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
			#print(st+', distance, bearing: '+str(tl_dst)+', '+str(tl_bearing))
			#print('distance: '+str(tl_dst))
                        tl_close_idx        = idx
                        tl_close_min_dst    = tl_dst
                        tl_close_hdg_deg    = tl_bearing
                        tl_close_wp_idx     = self.get_closest_waypoint(tl_pose)
            if tl_close_idx > -1:
                #light_state = self.lights[tl_close_idx].state 
                light_state = self.get_light_state(self.lights[tl_close_idx], tl_close_min_dst)
		#print(light_state)
                # testing: set the light to red just to see weather the car is braking
        #rospy.loginfo("next tl = %d (%d)", tl_close_wp_idx, light_state)
        return tl_close_wp_idx, light_state


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
