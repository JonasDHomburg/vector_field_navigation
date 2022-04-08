#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import rospy
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import cv2
import sys
import os


class ui:
    def __init__(self, path, topic='/image', delay=30):
        self.delay = delay
        try:
            self.files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        except OSError as e:
            rospy.logerr('Failed to open folder: ' + str(path))
            raise rospy.ROSInterruptException('Check configuration!')

        self.images = [cv2.imread(filename=filename) for filename in self.files]
        self.bridge = CvBridge()
        self.publisher = rospy.Publisher(topic, Image, queue_size=10)
        self.current_img_idx = 0

    def publish_images(self):
        while not rospy.is_shutdown():
            try:
                image_msg = self.bridge.cv2_to_imgmsg(
                    self.images[self.current_img_idx],
                    "bgr8"
                )
                image_msg.header.frame_id = "asis"
                rospy.loginfo("Publishing: %s", self.files[self.current_img_idx])
                self.publisher.publish(image_msg)
                rospy.sleep(self.delay)
            except CvBridgeError as e:
                rospy.logerr("Failed to convert image: %s", self.files[self.current_img_idx])
                self.files.pop(self.current_img_idx)
                self.images.pop(self.current_img_idx)
                self.current_img_idx -= 1
            self.current_img_idx += 1
            self.current_img_idx = self.current_img_idx % len(self.files)


def main():
    rospy.init_node("interactive_vector_field_drawer")
    path = rospy.get_param('~path', '/media/data1/citrack/Radar/catkin_ws/src/amiro_potential_field_navigation/'
                                    'potential_field_navigation/vectorfields')
    topic = rospy.get_param('~topic', '/image')
    delay = rospy.get_param('~delay', 30)

    drawer = ui(path=path, topic=topic, delay=int(delay))
    drawer.publish_images()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
