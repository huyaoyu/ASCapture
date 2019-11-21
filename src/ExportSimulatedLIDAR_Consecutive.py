from __future__ import print_function

import argparse
import numpy as np
import os
import time

import airsim
from airsim.types import Pose, Vector3r, Quaternionr
from airsim import utils as sim_util
from airsim.utils import to_quaternion

import SimulatedLIDAR as SLD

AS_CAM_ID = 1

def handle_arguments(parser):
    parser.add_argument("input", type=str, \
        help="The input pose file..")

    parser.add_argument("outdir", type=str, \
        help="The output directory.")

    parser.add_argument("--idx-start", type=int, default=0, \
        help="The starting index.")

    parser.add_argument("--idx-num", type=int, default=0, \
        help="The number of poses to execute. 0 for all the pose starting from --idx-start.")

    parser.add_argument("--var-threshold", type=float, default=0, \
        help="Set a positive number as the variance threshold. Should be positive. Set 0 to disable.")

    parser.add_argument("--max-distance", type=float, default=0, \
        help="The maximum allowable distance to be detected. Should be positivel. Set 0 to disable.")

    return parser.parse_args()

def convert_2_pose(a):
    """
    a: A 7-element array.
    """

    return Pose( Vector3r( a[0], a[1], a[2] ), Quaternionr( a[3], a[4], a[5], a[6] ) )

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

        q0 = Quaternionr( q[0], q[1], q[2], q[3] )

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

def capture(outDir, poses, height, width, focal, camID=AS_CAM_ID, idxStart=0, idxNum=0, varThres=0, maxDistance=0, flagSilent=True):
    """
    outDir: The output directory.
    poses: A n by 7 NumPy array.
    height: The image height.
    width: The image width.
    focal: The focal length of the camera.
    camID: The camera ID in AirSim.
    idxStart: The starting index of poses.
    idxNum: The number of indices need to be processed.
    """

    nPoses = len(poses)

    # ========== Check inputs. ==========
    if ( nPoses == 0 ):
        raise Exception("Number of poses is zero. ")

    assert idxStart >= 0
    assert idxStart < nPoses
    assert idxNum >= 0
    assert idxStart + idxNum <= nPoses

    if ( idxNum == 0 ):
        idxNum = nPoses

    if ( idxStart + idxNum > nPoses ):
        idxNum = nPoses - idxStart
    
    # Camera settings.
    assert height > 0
    assert width > 0
    assert focal > 0

    # ========== Done with checking input. ==========

    # Test the output directory.
    if ( not os.path.isdir(outDir) ):
        os.makedirs(outDir)

    # Create the CamControl object.
    cc = CamControl(camID)
    cc.initialize()

    # Create the SimulatedLIDAR object.
    sld = SLD.SimulatedLIDAR( focal, height )
    sld.set_description( SLD.VELODYNE_VLP_32C.desc )
    if ( varThres > 0 ):
        sld.set_variance_threshold(varThres)
    sld.initialize()

    if ( maxDistance <= 0 ):
        maxDistance = None

    # Loop over all the index.
    for idx in range(idxStart, idxStart + idxNum):
        if ( not flagSilent ):
            print("idx: %d/%d " % (idx, idxStart + idxNum - 1))

        # Get the pose.
        p = poses[idx, :]

        # Capture depths.
        depthList = cc.capture_LIDAR_depth( p[:3], p[3:] )

        # Extract LIDAR points.
        lidarPoints = sld.extract( depthList, maxDistance )

        # Convert LIDAR point list to a NumPy array.
        lidarPoints = SLD.convert_lidar_point_list_2_array( lidarPoints )

        # Convert the LIDAR points.
        xyz = SLD.convert_DEA_from_camera_2_Velodyne_XYZ( lidarPoints[:,0], lidarPoints[:,1], lidarPoints[:,2] )

        # Save the LIDAR poitns.
        outFn = "%s/%06d_LIDAR.npy" % ( outDir, idx )
        np.save( outFn, xyz )

    # Export poses.
    outFn = "%s/ExportedPoses.txt" % ( outDir )
    np.savetxt( outFn, poses[idxStart:idxStart + idxNum, :], fmt="%+.12e" )

if __name__ == "__main__":
    print("Capture depth images for simulating LIDAR.")

    parser = argparse.ArgumentParser(description="Capture depth images for simulating LIDAR.")
    args   = handle_arguments(parser)

    # Open the pose file.
    poses = np.loadtxt(args.input, dtype=np.float64)

    # Process.
    capture( args.outdir, poses, 768*2, 1160*2, 580*2, idxStart=args.idx_start, idxNum=args.idx_num, \
        varThres=args.var_threshold, maxDistance=args.max_distance, flagSilent=False )

    print("Done.")