# collect images in cv mode
from airsim.types import Pose, Vector3r, Quaternionr
from airsim.utils import to_eularian_angles, to_quaternion
from ImageClient import ImageClient

import cv2 # debug
import numpy as np
from math import cos, sin, tanh, pi
import time

from os import mkdir, listdir
from os.path import isdir, join
import sys
import random

from settings import get_args

from pyquaternion import Quaternion as Quaternionpy

np.set_printoptions(precision=3, suppress=True, threshold=10000)

# def quaternion_multiply(q,r):
#     q1 = np.zeros(4)
#     q2 = np.zeros(4)
#     res= np.zeros(4)
#     q1[0]   = q[3]
#     q1[1:4] = q[0:3]
#     q2[0]   = r[3]
#     q2[1:4] = r[0:3]
#     #q2q1 means q1 followed by q2
#     t0 = q1[0]*q2[0]-q1[1]*q2[1]-q1[2]*q2[2]-q1[3]*q2[3]
#     t1 = q1[0]*q2[1]+q1[1]*q2[0]-q1[2]*q2[3]+q1[3]*q2[2]
#     t2 = q1[0]*q2[2]+q1[1]*q2[3]+q1[2]*q2[0]-q1[3]*q2[1]
#     t3 = q1[0]*q2[3]-q1[1]*q2[2]+q1[2]*q2[1]+q1[3]*q2[0]
#     res =[t1,t2,t3,t0]
#     return res

# def random_rotation(MaxDegree=15,):
#     '''
#     MaxDegree: max rotation angle
#     '''
#     max_rad = MaxDegree * pi / 180.
#     theta =  np.random.random()*max_rad*2 - max_rad
#     axi = np.random.random(3)
#     axi = axi/np.linalg.norm(axi)
#     rotation = np.zeros(4)
#     rotation[0:3] = axi*np.sin(theta/2)
#     rotation[3] = np.cos(theta/2)

#     return rotation, theta, axi

class QuaternionSampler(object):
    def __init__():
        pass

    def next_quaternion(self,):
        pass

class RandQuaternionSampler(QuaternionSampler):
    def __init__(self, args):

        self.args = args

        self.MaxRandAngle = self.args.rand_degree
        self.SmoothCount =  self.args.smooth_count
        self.MaxYaw = self.args.max_yaw * pi / 180.
        self.MinYaw = self.args.min_yaw * pi / 180.
        self.MaxPitch = self.args.max_pitch * pi / 180.
        self.MinPitch = self.args.min_pitch * pi / 180.
        self.MaxRoll = self.args.max_roll * pi / 180.
        self.MinRoll = self.args.min_roll * pi / 180.

        self.reset()

    def reset(self):
        self.orientation = Quaternionpy(1.,0.,0.,0.)
        self.orilist = []
        self.oriind = self.SmoothCount

    def init_random_yaw(self):
        self.reset()
        randomyaw = np.random.uniform(self.MinYaw, self.MaxYaw)
        print ('Random yaw {}'.format(randomyaw))
        qtn = to_quaternion(0., 0., randomyaw)
        self.orientation = Quaternionpy(qtn.w_val, qtn.x_val, qtn.y_val, qtn.z_val)

    def random_quaternion(self):
        max_rad = self.MaxRandAngle * pi / 180.
        theta =  np.random.random()*max_rad*2 - max_rad
        axi = np.random.random(3)
        axi = axi/np.linalg.norm(axi)
        return Quaternionpy(axis=axi, angle=theta)    

    def next_quaternion(self,):
        if self.oriind >= self.SmoothCount: # sample a new orientation
            rand_ori = self.random_quaternion()
            new_ori = rand_ori * self.orientation

            new_qtn = Quaternionr(new_ori.x, new_ori.y, new_ori.z, new_ori.w)
            (pitch, roll, yaw) = to_eularian_angles(new_qtn)
            # print '  %.2f, %.2f, %.2f' %(pitch, roll, yaw )
            pitch = np.clip(pitch, self.MinPitch, self.MaxPitch)
            roll = np.clip(roll, self.MinRoll, self.MaxRoll)
            yaw = np.clip(yaw, self.MinYaw, self.MaxYaw)
            # print '  %.2f, %.2f, %.2f' %(pitch, roll, yaw )
            new_qtn_clip = to_quaternion(pitch, roll, yaw)

            new_ori_clip = Quaternionpy(new_qtn_clip.w_val, new_qtn_clip.x_val, new_qtn_clip.y_val, new_qtn_clip.z_val)
            qtnlist = Quaternionpy.intermediates(self.orientation, new_ori_clip, self.SmoothCount-2, include_endpoints=True)
            self.orientation = new_ori_clip
            self.orilist = list(qtnlist)
            self.oriind = 1
            # print "sampled new", new_ori, ', after clip', self.orientation #, 'list', self.orilist

        next_qtn = self.orilist[self.oriind]
        self.oriind += 1
        # print "  return next", next_qtn
        return Quaternionr(next_qtn.x, next_qtn.y, next_qtn.z, next_qtn.w)


class DataSampler(object):
    def __init__(self, data_dir):

        self.args = get_args()
        self.datadir = data_dir

        self.imgtypelist = self.args.img_type.split('_')
        self.camlist = self.args.cam_list.split('_')
        self.camlist_name = {'0': 'front', '1': 'right', '2': 'left', '3': 'back', '4': 'bottom'} 

        self.imgclient = ImageClient(self.camlist, self.imgtypelist)
        self.randquaternionsampler = RandQuaternionSampler(self.args)

        self.trajdir = ''
        self.imgdirs = []
        self.depthdirs = []
        self.segdirs = []

        self.posefilelist = [] # save incremental pose files
        self.posenplist = [] # save pose in numpy files

    def create_folders(self):
        mkdir(self.trajdir)
        self.imgdirs = []
        self.depthdirs = []
        self.segdirs = []
        self.posefilelist = [] # save incremental pose files
        self.posenplist = [] # save pose in numpy files
        for camind in self.camlist:
            camname = self.camlist_name[camind]
            if 'Scene' in self.imgtypelist:
                self.imgdirs.append(self.trajdir+'/image_'+camname)
                mkdir(self.imgdirs[-1])
            if 'DepthPlanner' in self.imgtypelist:
                self.depthdirs.append(self.trajdir+'/depth_'+camname)
                mkdir(self.depthdirs[-1])
            if 'Segmentation' in self.imgtypelist:
                self.segdirs.append(self.trajdir+'/seg_'+camname)
                mkdir(self.segdirs[-1])
            # create pose file
            self.posefilelist.append(self.trajdir+'/pose_'+camname+'.txt')
            self.posenplist.append([])

    def init_folders(self, traj_folder):
        '''
        traj_folder: string that denotes the folder name, e.g. T000
        '''
        if not isdir(self.datadir):
            mkdir(self.datadir)
        else: 
            print ('Data folder already exists.. {}'.format(self.datadir))

        self.trajdir = join(self.datadir, traj_folder)
        if not isdir(self.trajdir):
            self.create_folders()
        else:
            print ('Trajectory folder already exists! {}, create folder with time stamp.'.format(self.trajdir))
            self.trajdir = join(self.datadir, traj_folder + '_' + time.strftime('%m%d_%H%M%S',time.localtime()))
            self.create_folders()

    def data_sampling(self, positions, trajname): 

        self.init_folders(trajname)

        self.randquaternionsampler.init_random_yaw()
        start = time.time()
        for k,pose in enumerate(positions):
            position = Vector3r(pose[0], pose[1], pose[2])

            orientation = self.randquaternionsampler.next_quaternion()
            dronepose = Pose(position, orientation)
            self.imgclient.setpose(dronepose)
            time.sleep(0.02)
            self.imgclient.simPause(True)
            rgblist, depthlist, seglist, camposelist = self.imgclient.readimgs()
            self.imgclient.simPause(False)

            # save images and poses
            imgprefix = str(k).zfill(6)+'_'
            for w,camind in enumerate(self.camlist):
                camname = self.camlist_name[camind]
                # save RGB image
                if 'Scene' in self.imgtypelist:
                    img = rgblist[w] # change bgr to rgb
                    cv2.imwrite(join(self.imgdirs[w], imgprefix+camname+'.png'), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                # save depth image
                if 'DepthPlanner' in self.imgtypelist:
                    depthimg = depthlist[w]
                    np.save(join(self.depthdirs[w], imgprefix+camname+'_depth.npy'), depthimg)
                # save segmentation image
                if 'Segmentation' in self.imgtypelist:
                    segimg = seglist[w]
                    np.save(join(self.segdirs[w],imgprefix+camname+'_seg.npy'),segimg)
                # write pose to file
                self.posenplist[w].append(np.array(camposelist[w]))

                # imgshow = np.concatenate((leftimg,rightimg),axis=1)
            print ('  {0}, pose {1}, orientation ({2:.2f},{3:.2f},{4:.2f},{5:.2f})'.format(k, pose, orientation.x_val, orientation.y_val, orientation.z_val, orientation.w_val))
            # cv2.imshow('img',imgshow)
            # cv2.waitKey(1)

        for w in range(len(self.camlist)):
            # save poses into numpy txt
            np.savetxt(self.posefilelist[w], self.posenplist[w])

        end = time.time()
        print('Trajectory sample time {}'.format(end - start))

    def close(self,):
        self.imgclient.close()


if __name__ == '__main__':
    metadir = '/home/wenshan/tmp/maps/carwelding'
    datasampler = DataSampler(join(metadir,'Data'))
    posefolder = 'position_1108_141009'

    posfiles = listdir(join(metadir, posefolder))
    posfiles = [ff for ff in posfiles if ff[-3:]=='txt']
    posfiles.sort()

    for posfilename in posfiles:
        outfoldername = posfilename.split('.txt')[0]
        print ('*** {} ***'.format(outfoldername))
        positions = np.loadtxt(join(metadir, posefolder, posfilename))
        datasampler.data_sampling(positions, outfoldername)

    datasampler.close()

