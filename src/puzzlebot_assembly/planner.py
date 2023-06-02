import numpy as np
from collections import OrderedDict
from itertools import product
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix, distance
from sklearn.metrics import pairwise_distances
from scipy.sparse.csgraph import minimum_spanning_tree
from puzzlebot_assembly.utils import *

class Planner:
    '''
    This planner is the same as in https://github.com/ZoomLabCMU/puzzlebot_v2
    '''
    def __init__(self, N, L=5e-2):
        self.N = N
        self.L = L

        self.mst = None
        self.c = np.ndarray([2, 5])
        self.mesh_graph = None
        self.mesh_ind = None

        self.init_contact_pairs()

    def init_contact_pairs(self):
        L = self.L

        # create offset values for four corners on robot
        corners = [L/2, -L/2]
        self.c[:, 0:4] = np.array(list(product(corners, corners))).T

        l = product([0], [L/2, -L/2])
        self.c[:, 4] = [-0.001, -L/2]

    def generate_mesh_graph(self, mesh_array, x):
        N = self.N
        assert(np.shape(x) == (2, N))
        L = self.L
        
        mesh = np.ndarray([2, N])
        
        prev_num = 0
        for ci in range(len(mesh_array)):
            num = mesh_array[ci]
            col = - np.arange(num) * L
            curr_num = prev_num + num
            mesh[0, prev_num:curr_num] = + ci*L/2 + col
            mesh[1, prev_num:curr_num] = ci*L
            prev_num = curr_num
        # offset all points in mesh to the centroid of x
        #  mesh -= np.mean(mesh, axis=1)[:, None]
        #  mesh += np.mean(x, axis=1)[:, None]

        return mesh

    def generate_pair_pool(self, mesh_array, x, pilot_ids=[]):
        '''
        mesh_array: length M vector, each element is # of rows in mesh
                    currently assuming in descending order
        x: 2-by-N pose
        '''
        N = self.N
        assert(np.sum(mesh_array) == N)
        assert(np.shape(x) == (2, N))
        
        if N < 2:
            return (None, None)

        pair_dict = OrderedDict()

        mesh_graph = self.generate_mesh_graph(mesh_array, x)
        dis_mat = distance_matrix(mesh_graph.T, x.T)
        row_ind, col_ind = linear_sum_assignment(dis_mat)
        mesh_graph = np.append(mesh_graph, np.zeros([1,N]), axis=0)
        print('mesh:', mesh_graph)
        ids, cpair, cids = self.generate_pairs_formed(mesh_graph)
        ids, cpair = self.sort_contact_pairs(mesh_graph[0:2, :], 
                                            ids, cpair, 
                                            pilot_ids=pilot_ids)
        ids = col_ind[ids.flatten()].reshape(ids.shape)
        self.mesh_ind = np.argsort(col_ind)
        print('col:', col_ind)

        for c in range(len(cpair)):
            cid = ids[:, c]
            #  pair_dict[tuple(cid)] = cpair[c]
            pair_dict[tuple(cid)] = np.append(cpair[c], 
                                        np.zeros([1, 2]), axis=0)

        self.mesh_graph = mesh_graph
        print("pair_dict:", pair_dict)
        return pair_dict

    def generate_pairs_formed(self, x, reuse=True):
        '''
        return:
        id_pairs: 2-by-M
        contact_pairs: tuples of length M, element 2-by-2
        contact_ids: 2-by-M, id of contact type
        '''
        N = x.shape[1]
        assert(np.shape(x) == (3, N))
        
        if N < 2:
            return (None, None)
        
        id_pairs = []
        contact_pairs = ()
        contact_pids = []

        # generate the backbone of pairs with 
        # minimum spanning tree from pose graph
        mst = self.mst
        if not reuse or mst is None:
            dis_mat = pairwise_distances(x.T)
            mst = minimum_spanning_tree(dis_mat)
            self.mst = mst
        
        full_id_pairs = np.array(np.nonzero(mst))
        for i in range(full_id_pairs.shape[1]):
            xp = x[:, full_id_pairs[:, i]]
            cpairs, cids = self.get_contact_pair(xp)
            id_pairs.append(full_id_pairs[:, i])
            contact_pairs += (cpairs,)
            contact_pids.append(cids)
        id_pairs = np.array(id_pairs).T
        contact_pids = np.array(contact_pids).T

        return (id_pairs, contact_pairs, contact_pids)

    def sort_contact_pairs(self, x, id_pairs, 
                            contact_pairs, pilot_ids=[]):
        M = id_pairs.shape[1]
        assert(len(contact_pairs) == M)
        dis = np.zeros(M)

        for m in range(M):
            i0, i1 = id_pairs[:, m]
            xc = x[:, [i0, i1]]
            dis[m] = np.linalg.norm(xc[:, 0] - xc[:, 1])
            if i0 in pilot_ids or i1 in pilot_ids:
                dis[m] += 1e-3

        sort_idx = np.argsort(dis)
        new_id_pairs = id_pairs[:, sort_idx]
        new_contact_pairs = ()
        for m in range(M):
            new_contact_pairs += (contact_pairs[sort_idx[m]],)
        
        return new_id_pairs, new_contact_pairs

    def get_contact_pair(self, x):
        '''
        x: pose 3-by-N, N=2
        return: ([contact of x0, contact of x1], contact ids)
        '''
        assert(np.shape(x) == (3, 2))
        
        c = self.c
        x0 = get_R(x[2,0]).dot(c) + x[0:2, 0, None]
        x1 = get_R(x[2,1]).dot(c) + x[0:2, 1, None]
    
        cdis = distance.cdist(x0.T, x1.T)
        np.fill_diagonal(cdis, np.inf)
        #  min_ids = np.unravel_index(np.argmin(cdis), cdis.shape)
        mins = np.array(np.where(cdis == np.min(cdis)))
        num_mins = mins.shape[1]
        min_ids = mins[:, np.random.choice(num_mins)]
        
        return (c[:, min_ids], np.array(min_ids))

    def update_contact_with_ids(self, x, p_dict):
        xp = np.zeros([3, 2])
        for ids in p_dict:
            xp[:, 0] = x[ids[0]*3:(ids[0]+1)*3]
            xp[:, 1] = x[ids[1]*3:(ids[1]+1)*3]
            cpairs, cids = self.get_contact_pair(xp)
            m_ids = self.mesh_ind[list(ids)]
            pd = {tuple(m_ids): cpairs}
            if get_cp_dis(self.mesh_graph.T.flatten(), pd)[0] > 1e-2:
                continue
            p_dict[ids] = np.append(cpairs, np.zeros([1, 2]), axis=0)
        return p_dict
