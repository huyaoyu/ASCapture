import airsim
from airsim.types import Pose, Vector3r, Quaternionr

import cv2 # debug
import numpy as np

np.set_printoptions(precision=3, suppress=True, threshold=10000)

class ImageClient(object):
    def __init__(self, camlist, typelist, ip=''):
        self.client = airsim.MultirotorClient(ip=ip)
        self.client.confirmConnection()

        self.IMGTYPELIST = typelist
        self.CAMLIST = camlist

        self.imgRequest = []
        for k in self.CAMLIST:
            for imgtype in self.IMGTYPELIST:
                if imgtype == 'Scene':
                    self.imgRequest.append(airsim.ImageRequest(str(k), airsim.ImageType.Scene, False, False))

                elif imgtype == 'DepthPlanner':
                    self.imgRequest.append(airsim.ImageRequest(str(k), airsim.ImageType.DepthPlanner, True))

                elif imgtype == 'Segmentation':
                    self.imgRequest.append(airsim.ImageRequest(str(k), airsim.ImageType.Segmentation, False, False))
                else:
                    print ('Error image type: {}'.format(imgtype))

    def get_cam_pose(self, response):
        cam_pos = response.camera_position # Vector3r
        cam_ori = response.camera_orientation # Quaternionr

        cam_pos_vec = [cam_pos.x_val, cam_pos.y_val, cam_pos.z_val]
        cam_ori_vec = [cam_ori.x_val, cam_ori.y_val, cam_ori.z_val, cam_ori.w_val]

        # print cam_pos_vec, cam_ori_vec
        return cam_pos_vec + cam_ori_vec

    def readimgs(self):
        responses = self.client.simGetImages(self.imgRequest) # discard the first query because of AirSim error
        responses = self.client.simGetImages(self.imgRequest)
        camposelist = []
        rgblist, depthlist, seglist = [], [], []
        idx = 0
        for k in range(len(self.CAMLIST)):
            for imgtype in self.IMGTYPELIST:
                response = responses[idx]
                # response_nsec = response.time_stamp
                # response_time = rospy.rostime.Time(int(response_nsec/1000000000),response_nsec%1000000000)
                if imgtype == 'DepthPlanner': #response.pixels_as_float:  # for depth data
                    img1d = np.array(response.image_data_float, dtype=np.float32)
                    depthimg = img1d.reshape(response.height, response.width)
                    depthlist.append(depthimg)

                elif imgtype == 'Scene':  # raw image data
                    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)  # get numpy array
                    rgbimg = img1d.reshape(response.height, response.width, -1)
                    rgblist.append(rgbimg[:,:,:3])

                elif imgtype == 'Segmentation':
                    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) #get numpy array
                    img_rgba = img1d.reshape(response.height, response.width, -1)
                    img_seg = img_rgba[:,:,0]
                    seglist.append(img_seg)
                idx += 1
            # import ipdb;ipdb.set_trace()
            cam_pose_img = self.get_cam_pose(response) # get the cam pose for each camera
            camposelist.append(cam_pose_img)

        return rgblist, depthlist, seglist, camposelist

    def setpose(self, pose):
        self.client.simSetVehiclePose(pose, ignore_collison=True)

    def getpose(self):
        return self.client.simGetVehiclePose()

    def simPause(self, pause): # this is valid for customized AirSim
        return self.client.simPause(pause)

    def close(self):
        self.client.simPause(False)
        self.client.reset()
