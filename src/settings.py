import argparse

def get_args():
    parser = argparse.ArgumentParser(description='sample_pipeline')

    # mapping - expo_control



    # Graph sampling
    parser.add_argument('--node-num', type=int, default=30,
                        help='number of nodes to be sampled (default: 100)')

    parser.add_argument('--edge-num', type=int, default=8,
                        help='number of edges for each node (default: 10)')

    parser.add_argument('--min-dist-thresh', type=int, default=5,
                        help='minimum distance of two linked nodes (default: 10)')

    parser.add_argument('--max-dist-thresh', type=int, default=20,
                        help='maximum distance of two linked nodes (default: 30)')

    parser.add_argument('--max-failure-num', type=int, default=20,
                        help='maximum number of planning failure before giveup (default: 20)')


    # image collection - collect_images
    # parser.add_argument('--outdir', default='./',
    #                     help='output folder (default: ./)')

    parser.add_argument('--cam-list', default='1_2',
                        help='camera list: 0-front, 1-right, 2-left, 3-back, 4-bottom (default: 1_2)')

    parser.add_argument('--img-type', default='Scene_DepthPlanner_Segmentation',
                        help='image type Scene, DepthPlanner, Segmentation (default: Scene_DepthPlanner_Segmentation)')

    parser.add_argument('--rand-degree', type=int, default=30,
                        help='random angle added to the position when sampling (default: 15)')

    parser.add_argument('--smooth-count', type=int, default=10,
                        help='lengh of smoothed trajectory (default: 10)')

    parser.add_argument('--max-yaw', type=int, default=360,
                        help='yaw threshold (default: 360)')

    parser.add_argument('--min-yaw', type=int, default=-360,
                        help='yaw threshold (default: -360)')

    parser.add_argument('--max-pitch', type=int, default=20,
                        help='yaw threshold (default: 45)')

    parser.add_argument('--min-pitch', type=int, default=-45,
                        help='yaw threshold (default: -45)')

    parser.add_argument('--max-roll', type=int, default=20,
                        help='yaw threshold (default: 90)')

    parser.add_argument('--min-roll', type=int, default=-20,
                        help='yaw threshold (default: -90)')


    # parser.add_argument('--load-qnet', action='store_true', default=False,
    #                     help='Read and use an existing policy (default: False)')


    args = parser.parse_args()

    return args
