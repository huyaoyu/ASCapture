from __future__ import print_function

import argparse
import numpy as np
import os
import time

import airsim
from airsim.types import Pose, Vector3r, Quaternionr
from airsim import utils as sim_util
from airsim.utils import to_quaternion

AS_CAM_ID = 1

def handle_arguments(parser):
    parser.add_argument("--pose", type=str, default="0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0", \
        help="The spatial pose of the first camera. 3-element xyz and 4-element quaternion with the last being w.")

    parser.add_argument("--out-dir", type=str, default="./", \
        help="The output directory.")

    return parser.parse_args()

def convert_pose_string_2_array(s):
    """
    s is assumed to be in the form "0.0, 0.0, 0.0, 0.0, 0.0, 0.0"
    """

    return np.fromstring( s, dtype=np.float64, sep="," )

def convert_2_pose(p, e):
    """
    p: A 3-element NumPy array, the position.
    q: A 4-element NumPy array, a quaternion witht he last element being w.
    """

    return Pose( Vector3r( p[0], p[1], p[2] ), to_quaternion( e[0], e[1], e[2] ) )

class CamControl(object):
    def __init__(self, camID, distLimit=10000):
        super(CamControl, self).__init__()

        self.camID = camID

        self.cmdClient = None
        self.imgType   = None

        self.distLimit = distLimit

        self.maxTrial  = 2
    
    def initialize(self):
        # Connect to airsim.
        self.cmdClient = airsim.VehicleClient()
        self.cmdClient.confirmConnection()

        self.imgType = [ airsim.ImageRequest( self.camID, airsim.ImageType.DepthPlanner, True ) ]

    def get_depth_campos(self):
        '''
        cam_pose: 0: [x_val, y_val, z_val] 1: [x_val, y_val, z_val, w_val]
        '''
        imgRes = self.cmdClient.simGetImages(self.imgType)
        
        if imgRes is None or imgRes[0].width==0: # Sometime the request returns no image
            return None, None

        depthFront = sim_util.list_to_2d_float_array( \
            imgRes[0].image_data_float,
            imgRes[0].width, imgRes[0].height )
        
        depthFront[depthFront > self.distLimit] = self.distLimit

        camPose = ( imgRes[0].camera_position, imgRes[0].camera_orientation )

        return depthFront, camPose

    def set_vehicle_pose(self, pose, ignoreCollison=True, vehicleName=""):
        self.cmdClient.simSetVehiclePose( pose, ignoreCollison, vehicleName ) # amigo: this is supposed to be used in CV mode
        time.sleep(0.1)

    def capture_LIDAR_depth(self, p, q):
        """
        p: A 3-element NumPy array, the position.
        q: A 4-element NumPy array, quaternion, w is the last element.
        """

        faces = [ 0, 1, 2, -1 ] # Front, right, back, left.
        depthList = []

        q0 = Quaternionr(q[0], q[1], q[2], q[3])

        for face in faces:
            # Compose a AirSim Pose object.
            yaw = np.pi / 2 * face
            yq  = to_quaternion( 0, 0, yaw )
            # q   = to_quaternion( e[0], e[1], e[2] )
            q1  = yq * q0

            pose = Pose( Vector3r( p[0], p[1], p[2] ), q1 )

            # Change the vehicle pose.
            self.set_vehicle_pose(pose)

            # Get the image.
            for i in range(self.maxTrial):
                depth, _ = self.get_depth_campos()
                if depth is None:
                    print("Fail for trail %d on face %d. " % ( i, face ))
                    continue
        
            if ( depth is None ):
                raise Exception("Could not get depth image for face %d. " % ( face ))

            # Get a valid depth.
            depthList.append(depth)
        
        return depthList
 
def save_depth_list(fn, depthList):
    if ( len(depthList) != 4 ):
        raise Exception("Expecting the depths in the list to be 4. len(depthList) = %d. " % ( len(depthList) ))

    # Test the output directory.
    parts = os.path.split(fn)

    if ( not os.path.isdir( parts[0] ) ):
        os.makedirs( parts[0] )

    np.savez( fn, d0=depthList[0], d1=depthList[1], d2=depthList[2], d3=depthList[3] )

if __name__ == "__main__":
    print("Capture depth images for simulating LIDAR.")

    parser = argparse.ArgumentParser(description="Capture depth images for simulating LIDAR.")
    args   = handle_arguments(parser)

    cc = CamControl(AS_CAM_ID)

    cc.initialize()

    a = convert_pose_string_2_array( args.pose )

    if ( a.size != 7 ):
        raise Exception("Get pose = {}. ".format( a ))

    depthList = cc.capture_LIDAR_depth( a[:3], a[3:] )

    # Save the depth to filesystem.
    save_depth_list("%s/LIDARDepth.npz" % (args.out_dir), depthList)
