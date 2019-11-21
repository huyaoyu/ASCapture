#!/usr/bin/env python
import rospy

import time
import numpy as np
import matplotlib.pyplot as plt
from os import mkdir, listdir
from os.path import exists
import pickle

from NodePathSampler import NodePathSampler
from PathGraph import PathGraph
from RandomPoseReSampler import RandomPoseReSampler


def node_distance(init, goal):
    diff = np.array([goal.position.x-init.position.x,
                     goal.position.y-init.position.y,
                     goal.position.z-init.position.z])
    return np.linalg.norm(diff, axis=0)


def sample_nodes_edges(pathgraph, pathsampler, outputgraphfilename, Nodenum, Edgenum, TrajlenMinThresh, TrajlenMaxThresh, MaxFailureNum):
    ''' 
    Call sample_nodes_service and get the poselist
    Call roadmap_srv to connect nodes with edges
    Save the result to a graph file
    '''
    # if the graph is not empty, load the graph first
    pre_nodenum = pathgraph.graph.number_of_nodes()
    pre_nodelist = []
    pre_nodeslinknum = np.zeros(pre_nodenum, dtype=np.int32)
    for k, node in enumerate(pathgraph.graph.nodes):
        pre_nodelist.append(node)
        edgenum = len(pathgraph.graph.edges(node))
        pre_nodeslinknum[k] = edgenum

    # node sampling through ros service
    nodeposes = pathsampler.sample_nodes(Nodenum) 
    nodeslinknum = np.zeros(Nodenum, dtype=np.int32)

    if nodeposes is None: # the service call has not returned the nodes
        print ('Node sample service failed')
        return

    # combine the pre-loaded nodes and link 
    if pre_nodenum>0:
        nodeposes = nodeposes + pre_nodelist 
        nodeslinknum = np.concatenate((nodeslinknum, pre_nodeslinknum))
        Nodenum = len(nodeposes)
        rospy.loginfo('Combine pre-loaded nodes and links, total node {}'.format(Nodenum))

    pathsampler.publish_nodes_marker(nodeposes)
    for k,point in enumerate(nodeposes): # add link to the nodes
        initpose = nodeposes[k]
        randlist_after = np.random.permutation(Nodenum - k -1) + k + 1 # query nodes after first
        randlist_befor = np.random.permutation(k)
        randlist = np.concatenate((randlist_after, randlist_befor))
        ind = 0
        failnum = 0
        while nodeslinknum[k] < Edgenum: # randomly sample from all the nodes
            if ind >= Nodenum-1: # Not enough edges after all the nodes have been sampled
                rospy.logwarn('No enough edges '+str(nodeslinknum[k]) + ' sampled')
                break
            randind = randlist[ind]
            ind += 1
            # if nodeslinknum[ind] >= Edgenum: # this node has enough edges already
            #     continue
            goalpose = nodeposes[randind]
            if pathgraph.graph.has_edge(initpose, goalpose):
                rospy.loginfo('edge exists ({}, {})'.format(k,randind))
                continue
            nodedist = node_distance(initpose, goalpose)
            if nodedist > TrajlenMinThresh and nodedist < TrajlenMaxThresh: # do not link to nearby node
                path = pathsampler.plan_edge(initpose, goalpose) # call OMPL
                pathsampler.publish_endpoints_marker(initpose, goalpose)
                if (path is not None) and (len(path)>0): 
                    # path = pathsampler.smooth_path(path)
                    pathsampler.publish_path_marker(path)
                    # pathsampler.publish_path_marker(path)
                    nodeslinknum[k] += 1
                    nodeslinknum[randind] += 1
                    rospy.loginfo('{} - {}, find {} edges'.format(k, randind, nodeslinknum[k]))
                    pathgraph.graph.add_edge(initpose, goalpose, path=path)
                    # import ipdb;ipdb.set_trace()
                else: # planning failed
                    failnum += 1
                    if failnum>MaxFailureNum:
                        rospy.logwarn('Too many failures, got edges of '+str(nodeslinknum[k]))
                        break

    import ipdb;ipdb.set_trace()
    # save the graph to disk
    pickle.dump(pathgraph.graph, open(outputgraphfilename, 'w'))
    rospy.loginfo('Graph file saved '+ outputgraphfilename)

def load_graph_and_resample_positionlist(pathsampler, pathgraph, outdir):


    pathgraph.prune_graph()
    print 'nodes ', pathgraph.graph.number_of_nodes(), 'edges ', pathgraph.graph.number_of_edges()

    # output dir
    timestr = time.strftime('%m%d_%H%M%S',time.localtime())
    dirname = outdir+'/ros_path_'+timestr
    if not exists(dirname):
        mkdir(dirname)

    samplenum = 0
    while True:
        pathsampler.vis_graph(pathgraph.graph, visedge=False)
        cycle = pathgraph.sample_cycle(mode=0)
        if cycle is None:
            break
        samplenum += 1
        cycle_poses = pathgraph.cycle2poselist(cycle)
        print samplenum, ' - cycle nodes:', len(cycle), 'path length:', len(cycle_poses)
        pathsampler.publish_path_marker(cycle_poses,markerid=0) 
        distlist1 = pathgraph.node_dist_statistics(cycle_poses)
        # plt.subplot(121)
        # plt.hist(distlist1, bins=10)

        smooth_cycle_poses = pathsampler.smooth_path(cycle_poses)
        print '     smooth length:', len(smooth_cycle_poses)
        pathsampler.publish_path_marker(smooth_cycle_poses,markerid=1) 
        distlist2 = pathgraph.node_dist_statistics(smooth_cycle_poses)
        # plt.subplot(122)
        # plt.hist(distlist2, bins=10)
        # plt.show()

        positionlist = pathgraph.poselist2positionlist(smooth_cycle_poses)
        positions_np = np.array(positionlist, dtype=np.float32)
        np.savetxt(dirname+'/P'+str(samplenum-1).zfill(3)+'.txt', positions_np)

        pathgraph.delete_path(cycle)
        # import ipdb;ipdb.set_trace()

def rospath2positionlist(posesampler, trajdir):

    trajfiles = listdir(trajdir)
    trajfiles = [ff for ff in trajfiles if ff[-3:]=='txt']
    trajfiles.sort()
    # output dir
    timestr = time.strftime('%m%d_%H%M%S',time.localtime())
    dirname = outdir+'/position_'+timestr
    if not exists(dirname):
        mkdir(dirname)

    for k,trajfile in enumerate(trajfiles):
        filename = dirname+'/P'+str(k).zfill(3)+'.txt'
        figname = dirname+'/P'+str(k).zfill(3)+'.jpg'
        traj_np = np.loadtxt(trajdir + '/' + trajfile)
        positions = posesampler.sample_poses(traj_np.tolist(), figname)
        np.savetxt(filename, positions)


if __name__ == '__main__':

    rospy.init_node('roadmap_path_sample', anonymous=True)
    outdir='/home/wenshan/tmp/maps/carwelding/'

    Nodenum = 40
    Edgenum = 10
    TrajlenMinThresh = 20
    TrajlenMaxThresh = 40
    MaxFailureNum = 20

    # Pathgraph visualization
    graphfilename = '/home/wenshan/tmp/maps/carwelding/OccMap/node40_edge10_len20_40.graph'
    pathsampler = NodePathSampler()
    pathgraph = PathGraph(graphfilename) 
    posesampler = RandomPoseReSampler()

    # outputgraphfilename = outdir + 'OccMap/' + 'node{}_edge{}_len{}_{}.graph'.format(Nodenum, Edgenum, TrajlenMinThresh, TrajlenMaxThresh)
    # sample_nodes_edges(pathgraph, pathsampler, outputgraphfilename, Nodenum, Edgenum, TrajlenMinThresh, TrajlenMaxThresh, MaxFailureNum)


    # pathsampler.vis_graph(pathgraph.graph, visedge=True)
    # load_graph_and_resample_positionlist(pathsampler, pathgraph, outdir)

    rospath2positionlist(posesampler, outdir+'/ros_path_1108_135845')
