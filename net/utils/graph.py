import numpy as np
import random
class Graph():


    def __init__(self,
                 layout='openpose_face',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,
                                                                        11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                              (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                              (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                              (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                              (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        elif layout == 'ntu_edge':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                              (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                              (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                              (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2

        elif layout == 'openpose_face_25_points':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
                             (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
                             (12, 13), (13, 14), (14, 8), (13, 14), (14, 8),
                             (0, 17), (1, 17), (9, 17), (11, 17), (1, 18), (11, 18),
                             (17, 18), (1, 16), (7, 16), (11, 15), (12, 15), (16, 15),
                             (18, 15), (19, 15), (12, 15), (16, 15), (16, 18),
                             (16, 21), (16, 22), (16, 23), (24, 21), (24, 22), (24, 23),
                             (19, 20), (12, 20), (7, 20), (8, 20), (14, 20), (7, 19),
                             (16, 19), (21, 22), (22, 23), (1, 21), (2, 21), (3, 21),
                             (5, 23), (6, 23), (7, 23), (3, 24), (4, 24), (5, 24), (12, 19)]
            self.edge = self_link + neighbor_link
            self.center = 16

        elif layout == 'openpose_face_16_points':
            self.num_node = 16
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
                             (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
                             (12, 13), (13, 14), (14, 8), (13, 14), (14, 8),
                             (0, 15), (1, 12), (9, 10), (11, 15), (1, 8),
                             (1, 7), (7, 15), (11, 13), (12, 1), (14, 15),
                             (13, 15), (11, 12), (12, 13), (4, 15),
                             (3, 8), (2, 10), (7, 11), (1, 12), (2, 6), (3, 9),
                             (5, 13), (6, 14), (7, 9), (3, 6), (4, 14)]
            self.edge = self_link + neighbor_link
            self.center = 8




        elif layout == 'openpose_face_68_points':
            self.num_node = 68
            self_link = [(i, i) for i in range(self.num_node)]

            neighbor_link = []
            for i in range(0, 68):
                for j in range(0, 5):
                    neighbor_link.append((i, random.randint(0,67)))


            # neighbor_link = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
            #                  (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
            #                  (12, 13), (13, 14), (14, 8), (13, 14), (14, 8),
            #                  (0, 17), (1, 17), (9, 17), (11, 17), (1, 18), (11, 18),
            #                  (17, 18), (1, 16), (7, 16), (11, 15), (12, 15), (16, 15),
            #                  (18, 15), (19, 15), (12, 15), (16, 15), (16, 18),
            #                  (16, 21), (16, 22), (16, 23), (24, 21), (24, 22), (24, 23),
            #                  (19, 20), (12, 20), (7, 20), (8, 20), (14, 20), (7, 19),
            #                  (16, 19), (21, 22), (22, 23), (1, 21), (2, 21), (3, 21),
            #                  (5, 23), (6, 23), (7, 23), (3, 24), (4, 24), (5, 24), (12, 19)]
            self.edge = self_link + neighbor_link
            self.center = 38

        elif layout == 'openpose_face_19_points':
            self.num_node = 19
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (12, 5), (12, 7), (5, 6),
                             (6, 13), (6, 15), (7, 8), (7, 12), (7, 14), (8, 15), (8, 3), (8, 4),
                             (9, 10), (10, 11), (10, 13), (10, 14), (12, 13), (12, 9), (11, 15),
                             (13, 14), (14, 15), (15, 3), (16 ,1), (16, 2), (16, 13), (17, 9),
                             (17, 10), (17, 11), (18, 2), (18, 3), (18, 14), (18, 11), (1, 9),
                             (3, 11), (9, 16), (5, 1), (12, 1)]
            self.edge = self_link + neighbor_link
            self.center = 10


        elif layout == 'openpose_face_16_points_v2':
            self.num_node = 16
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (12, 5), (12, 7), (5, 6),
                             (6, 13), (6, 15), (7, 8), (7, 12), (7, 14), (8, 15), (8, 3), (8, 4),
                             (9, 10), (10, 11), (10, 13), (10, 14), (12, 13), (12, 9), (11, 15),
                             (13, 14), (14, 15), (12, 3), (13 ,1), (13, 2), (12, 13), (14, 9),
                             (14, 10), (14, 11), (15, 2), (15, 3), (15, 14), (15, 11), (1, 9),
                             (3, 11), (9, 13), (5, 1), (12, 1)]
            self.edge = self_link + neighbor_link
            self.center = 13

        elif layout == 'openpose_skeleton_20_points':
            self.num_node = 20
            self_link = [(i, i) for i in range(self.num_node)]

            neighbor_link = [(0, 1),  (0, 9),  (0, 15), (1, 2) , (1, 8), (1, 9),
                             (2, 3), (2, 7),  (2, 8), (3, 4), (3, 6), (3, 5),
                             (4, 5),  (4, 6),  (4, 19), (5, 6), (5, 10) , (5, 11),
                             (6, 7), (6, 10), (6, 12), (7, 8), (7, 12),  (7, 13),
                             (8, 9), (8, 13),  (8, 14), (9, 14), (9, 13),  (9, 7),
                             (10, 11), (10, 19), (10, 18), (11, 12), (11, 18), (11, 17),
                             (12, 13), (12, 17), (12, 16), (13, 14), (13, 16), (13, 15),
                             (14, 15), (14, 16), (14, 8), (15, 16), (15, 12), (15, 11),
                             (16, 17), (16, 11), (16, 10), (17, 18), (17, 10), (17, 14),
                             (18, 19), (18, 13), (18, 14), (19, 7), (19, 13), (19, 5)
                             ]

            self.edge = self_link + neighbor_link
            self.center = 7


        elif layout == 'openpose_skeleton_14_points':
            self.num_node = 14
            self_link = [(i, i) for i in range(self.num_node)]

            neighbor_link = [(12, 13), (13, 0), (13, 1), (0, 2), (2, 4), (13, 1), (1, 3), (3, 5)
                , (13, 6), (13, 7), (6, 8), (8, 10) , (7, 9), (9, 11), (12, 0), (12, 1), (6, 1), (7, 1) ,
                             (3, 4), (2, 5), (7, 8), (6, 9), (8, 11), (9, 10), (1, 4), (0, 5), (13, 10), (13, 11),
                             (2, 8), (3, 9), (4, 6), (5, 7)
                             ]

            self.edge = self_link + neighbor_link
            self.center = 13

        elif layout == 'openpose_skeleton_face_39_points':
                self.num_node = 39
                self_link = [(i, i) for i in range(self.num_node)]

                neighbor_link_face = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (12, 5), (12, 7), (5, 6),
                                      (6, 13), (6, 15), (7, 8), (7, 12), (7, 14), (8, 15), (8, 3), (8, 4),
                                      (9, 10), (10, 11), (10, 13), (10, 14), (12, 13), (12, 9), (11, 15),
                                      (13, 14), (14, 15), (15, 3), (16, 1), (16, 2), (16, 13), (17, 9),
                                      (17, 10), (17, 11), (18, 2), (18, 3), (18, 14), (18, 11), (1, 9),
                                      (3, 11), (9, 16), (5, 1), (12, 1)]
                neighbor_link_body = [(0, 1), (0, 9), (0, 15), (1, 2), (1, 8), (1, 9),
                                      (2, 3), (2, 7), (2, 8), (3, 4), (3, 6), (3, 5),
                                      (4, 5), (4, 6), (4, 19), (5, 6), (5, 10), (5, 11),
                                      (6, 7), (6, 10), (6, 12), (7, 8), (7, 12), (7, 13),
                                      (8, 9), (8, 13), (8, 14), (9, 14), (9, 13), (9, 7),
                                      (10, 11), (10, 19), (10, 18), (11, 12), (11, 18), (11, 17),
                                      (12, 13), (12, 17), (12, 16), (13, 14), (13, 16), (13, 15),
                                      (14, 15), (14, 16), (14, 8), (15, 16), (15, 12), (15, 11),
                                      (16, 17), (16, 11), (16, 10), (17, 18), (17, 10), (17, 14),
                                      (18, 19), (18, 13), (18, 14), (19, 7), (19, 13), (19, 5)
                                      ]
                # for i in range(0, len(neighbor_link_body)):
                #     for j in range(0, 2):
                #         neighbor_link_body[i][j] = neighbor_link_body[i][j] + 19
                neighbor_link_full = []

                for i in neighbor_link_face:
                    neighbor_link_full.append(i)

                for i in range(0, len(neighbor_link_body)):
                    temp = []
                    for j in range(0, 2):
                        temp.append(neighbor_link_body[i][j] + 19)
                    neighbor_link_full.append((temp[0], temp[1]))



                self.edge = self_link + neighbor_link_full
                self.center = 7
                #self.center = 7
                #selected_land = [1,3,8,13,15,17,21,22,26,31,33,35,36,39,42,45,48,51,54]
            #################[0,1,2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18]


            # neighbor_link = [(0, 1), (6, 5), (1, 2), (6, 4), (0, 7), (0, 17), (0, 8), (1, 2), (0, 5),
            #                  (3, 4), (12, 5), (12, 7), (5, 6),
            #                  (6, 13), (6, 15), (7, 8), (7, 12), (7, 14), (8, 15), (8, 3), (8, 4),
            #                  (9, 10), (10, 11), (10, 13), (10, 14), (12, 13), (12, 9), (11, 15),
            #                  (13, 14), (14, 15), (15, 3), (16 ,1), (16, 2), (16, 13), (17, 9),
            #                  (17, 10), (17, 11),  (18, 3), (18, 14), (18, 11), (1, 9),
            #                  (3, 11), (9, 16), (5, 1), (2, 4), (2, 16), (4, 17)]
            # self.edge = self_link + neighbor_link
            # self.center = 10

        elif layout == 'openpose_face_18_points':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 1), (0, 4), (0, 11), (1, 4), (1, 11), (1, 8), (1, 15),  (2, 3), (2, 7), (2, 10),
                             (2, 14), (2, 17), (3, 7), (3, 14), (4, 11), (4, 5), (5, 14),  (6, 7), (6, 11),
                             (6, 13), (7, 14), (8, 9), (8, 16), (8, 1), (8, 11), (8, 15), (9, 10), (9, 12), (9, 13),
                             (10, 17), (10, 16), (10, 14), (11, 12), (11, 8), (12, 15), (12, 13), (12, 5),
                             (12, 9), (13, 14), (13, 9), (13, 6),  (13, 17), (9, 16)]
            self.edge = self_link + neighbor_link
            self.center = 9

        # elif layout=='customer settings'
        #     pass

        elif layout == 'openpose_skel_11_points':
            self.num_node = 11
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (1, 5)
            , (2, 3), (2, 4), (2, 5), (2, 6), (3, 4), (3, 5), (3, 6), (3, 7), (4, 5), (4, 6)
            , (4, 7), (4, 8), (5, 6), (5, 7), (5, 8), (5, 9), (6, 7), (6, 8), (6, 9), (6, 10)
            , (7, 8), (7, 9), (7, 10), (7, 0), (8, 9), (8, 10), (8, 0), (8, 1), (9, 10), (9, 2)
            , (9, 3), (9, 4), (10, 5), (10, 6), (10, 0), (10, 1)]
            self.edge = self_link + neighbor_link
            self.center = 6
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.
                                    center] > self.hop_dis[i, self.
                                    center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD
