import numpy as np
from queue import Empty

from bubblewrap import Bubblewrap
from improv.actor import Actor, RunManager

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Bubblewrap(Actor):

    def __init__(self, dimension, *args, **kwargs):
        self.d = dimension
        super().__init__(*args, **kwargs)


    def setup(self):
        ## TODO: Input params from file

        ## Parameters
        N = 100             # number of nodes to tile with
        lam = 1e-3          # lambda 
        nu = 1e-3           # nu
        eps = 1e-3          # epsilon sets data forgetting
        step = 8e-2         # for adam gradients
        M = 30              # small set of data seen for initialization
        B_thresh = -10      # threshold for when to teleport (log scale)    
        batch = False       # run in batch mode 
        batch_size = 1      # batch mode size; if not batch is 1
        go_fast = False     # flag to skip computing priors, predictions, and entropy for optimal speed

        ## Load data from datagen/datagen.py 
        s = np.load('vdp_1trajectories_2dim_500to20500_noise0.05.npz')
        # s = np.load('lorenz_1trajectories_3dim_500to20500_noise0.05.npz')
        data = s['y'][0]
        T = data.shape[0]

        self.bw = Bubblewrap(N, self.d, step=step, lam=lam, M=M, eps=eps, nu=nu, B_thresh=B_thresh, batch=batch, batch_size=batch_size, go_fast=go_fast) 

        init = -M
        end = T-M
        step = batch_size

        for i in np.arange(0, M, step): 
            if batch:
                self.bw.observe(data[i:i+step]) 
            else:
                self.bw.observe(data[i])
        self.bw.init_nodes()
        print('Nodes initialized')

    def run(self):
        with RunManager(self.name, self.runBW, self.setup, self.q_sig, self.q_comm) as rm:
            print(rm)

    def runBW(self):

        try:
            ids = self.q_in.get(timeout=0.0005)

            # expect ids of size 2 containing data location and frame number
            new_data = self.client.getID(ids[0])
            self.frame_number = ids[1]

            self.bw.observe(new_data)
            self.bw.e_step()  
            self.bw.grad_Q()

            self.putOutput()

        except Empty:
            pass

        
    def putOutput(self):
        # Function for putting updated results into the store
        ids = []
        ids.append([self.client.put(self.bw.A, 'A'+str(self.frame_number)), 'A'+str(self.frame_number)])
        ids.append([self.client.put(self.bw.L, 'L'+str(self.frame_number)), 'L'+str(self.frame_number)])
        ids.append([self.client.put(self.bw.mu, 'mu'+str(self.frame_number)), 'mu'+str(self.frame_number)])
        ids.append([self.client.put(self.bw.n_obs, 'n_obs'+str(self.frame_number)), 'n_obs'+str(self.frame_number)])
        ids.append([self.client.put(self.bw.pred, 'pred'+str(self.frame_number)), 'pred'+str(self.frame_number)])
        ids.append([self.client.put(self.bw.entropy_list, 'entropy'+str(self.frame_number)), 'entropy'+str(self.frame_number)])
        ids.append([self.frame_number, str(self.frame_number)])

        self.q_out.put(ids)