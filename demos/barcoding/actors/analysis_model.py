from improv.actor import Actor, Signal, RunManager
from improv.store import ObjectNotFoundError
from queue import Empty
import numpy as np
import time
import cv2
import colorsys
import scipy

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ModelAnalysis(Actor):
    def __init__(self, *args):
        super().__init__(*args)
        N = 5  # guess XX neurons initially
        dh = 4
        dk = 8
        w = np.zeros((N, N))
        h = np.zeros((N, dh))
        k = np.zeros((N, dk))
        b = np.zeros(N)
        self.theta = np.concatenate((w, h, b, k), axis=None).flatten()
        self.p = {
            "numNeurons": N,
            "hist_dim": dh,
            "numSamples": 1,
            "dt": 0.5,
            "stim_dim": dk,
        }  # TODO: from config file..

    def setup(self, param_file=None):
        """ """
        np.seterr(divide="ignore")

        # TODO: same as behaviorAcquisition, need number of stimuli here. Make adaptive later
        self.num_stim = 12
        self.frame = 0
        # self.curr_stim = 0 #start with zeroth stim unless signaled otherwise
        self.stim = {}
        self.stimStart = -1
        self.currentStim = None
        self.ests = np.zeros(
            (1, self.num_stim, 2)
        )  # number of neurons, number of stim, on and baseline
        self.counter = np.ones((self.num_stim, 2))
        self.window = 500  # TODO: make user input, choose scrolling window for Visual
        self.C = None
        self.S = None
        self.Call = None
        self.Cx = None
        self.Cpop = None
        self.coords = None
        self.color = None
        self.runMean = None
        self.runMeanOn = None
        self.runMeanOff = None
        self.lastOnOff = None
        self.recentStim = [0] * self.window
        self.currStimID = np.zeros((8, 1000000))  # FIXME
        self.currStim = -10
        self.allStims = {}
        self.estsAvg = None

        self.baseline_coefficient = np.zeros(
            (1, self.num_stim, 3)
        )  # number of neurons, number of stim, on and baseline
        self.stimuli_visited_num = None
        self.stimuli = None
        self.stage = ""

    def run(self):
        self.total_times = []
        self.puttime = []
        self.colortime = []
        self.stimtime = []
        self.timestamp = []
        self.LL = []
        self.fit_times = []

        with RunManager(
            self.name, self.runStep, self.setup, self.q_sig, self.q_comm
        ) as rm:
            logger.info(rm)

        print("Analysis broke, avg time per frame: ", np.mean(self.total_times, axis=0))
        print("Analysis broke, avg time per put analysis: ", np.mean(self.puttime))
        print("Analysis broke, avg time per put analysis: ", np.mean(self.fit_times))
        print("Analysis broke, avg time per color frame: ", np.mean(self.colortime))
        print("Analysis broke, avg time per stim avg: ", np.mean(self.stimtime))
        print("Analysis got through ", self.frame, " frames")

        N = self.p["numNeurons"]

        np.savetxt("output/timing/analysis_frame_time.txt", np.array(self.total_times))
        np.savetxt("output/timing/analysis_timestamp.txt", np.array(self.timestamp))
        np.savetxt("output/analysis_estsAvg.txt", np.array(self.estsAvg))
        np.savetxt("output/analysis_proc_S_full3.txt", self.S)
        np.savetxt("output/analysis_LL.txt", np.array(self.LL))

        np.savetxt("output/used_stims.txt", self.currStimID)

    def runStep(self):
        """Take numpy estimates and frame_number
        Create X and Y for plotting
        """
        t = time.time()
        ids = None

        try:
            ids = self.q_in.get(timeout=0.0001)
            if ids is not None and ids[0] == 1:
                print("analysis: missing frame")
                self.total_times.append(time.time() - t)
                self.q_out.put([1])
                raise Empty
            # t = time.time()
            self.frame = ids[-1]
            (self.coordDict, self.image, self.S) = self.client.getList(ids[:-1])
            self.C = self.S
            self.coords = [o["coordinates"] for o in self.coordDict]

            # Compute tuning curves based on input stimulus
            # Just do overall average activity for now
            try:
                sig = self.links["input_stim_queue"].get(timeout=0.0001)
                self.updateStim_start(sig)
            except Empty as e:
                pass  # no change in input stimulus
            self.stimAvg_start()

            # fit to model once we have enough neurons
            if self.C.shape[0] >= self.p["numNeurons"] and self.frame > (
                self.p["hist_dim"] + 1
            ):
                self.eval_barcode()

            self.globalAvg = np.mean(self.estsAvg[:, :8], axis=0)
            self.tune = [self.estsAvg[:, :8], self.globalAvg]

            # N  = self.p['numNeurons']
            # dh = self.p['hist_dim']
            # ds = self.p['stim_dim']
            # k = np.square(self.theta[N*(N+dh+1):].reshape((N, ds))*1e6)
            # self.globalAvg = np.mean(k, axis=0)
            # self.tune_k = [k, self.globalAvg]

            # Compute coloring of neurons for processed frame
            # Also rotate and stack as needed for plotting
            # TODO: move to viz, but we don't need to compute this 30 times/sec
            self.color = self.plotColorFrame()

            if self.frame >= self.window:
                window = self.window
                self.Cx = np.arange(self.frame - window, self.frame)
            else:
                window = self.frame
                self.Cx = np.arange(0, self.frame)

            if self.C.shape[1] > 0:
                self.Cpop = np.nanmean(self.C, axis=0)
                self.Call = (
                    self.C
                )  # already a windowed version #[:,self.frame-window:self.frame]

            self.putAnalysis()
            self.timestamp.append([time.time(), self.frame])
            self.total_times.append(time.time() - t)

        except ObjectNotFoundError:
            logger.error("Estimates unavailable from store, droppping")
        except Empty as e:
            pass
        except Exception as e:
            logger.exception("Error in analysis: {}".format(e))


    def updateStim_start(self, stim):
        """Rearrange the info about stimulus into
        cardinal directions and frame <--> stim correspondence.

        self.stimStart is the frame where the stimulus started.
        """
        # print('got stim ', stim)
        # get frame number and stimID
        frame = list(stim.keys())[0]
        # print('got frame ', frame)
        whichStim = stim[frame][0]
        # convert stimID into 8 cardinal directions
        stimID = self.IDstim(int(whichStim))

        # assuming we have one of those 8 stimuli
        if stimID != -10:
            if stimID not in self.allStims.keys():
                self.allStims.update({stimID: []})
                self.stumuli[stimID] = np.zeros(2)
                self.stumuli_visited_num[stimID] = 1

                # account for stimuli we haven't yet seen
                # if stimID not in self.stimStart.keys():
                #     self.stimStart.update({stimID:None})

            # determine if this is a new stimulus trial
            if abs(stim[frame][1]) > 1:
                curStim = 1  # on
                self.allStims[stimID].append(frame)
                for i in range(10):
                    self.allStims[stimID].append(frame + i + 1)
            else:
                curStim = 0  # off
            # paradigm for these trials is for each stim: [off, on, off]
            if self.lastOnOff is None:
                self.stage = "pre-stimuli"
                self.lastOnOff = curStim
            elif self.lastOnOff == 0 and curStim == 1:  # was off, now on
                # select this frame as the starting point of the new trial
                # and stimulus has started to be shown
                # All other computations will reference this point
                self.stage = "stimuli"
                self.stimStart = frame
                self.currentStim = stimID
                if stimID < 8:
                    self.currStimID[stimID, frame] = 1
                # NOTE: this overwrites historical info on past trials
                logger.info("Stim {} started at {}".format(stimID, frame))
            else:
                self.stage = "post-stimuli"
                self.currStimID[:, frame] = np.zeros(8)
            self.lastOnOff = curStim

    def putAnalysis(self):
        """Throw things to DS and put IDs in queue for Visual"""
        t = time.time()
        ids = []
        stim = [self.lastOnOff, self.currStim]
        N = self.p["numNeurons"]
        w = self.theta[: N * N].reshape((N, N))

        ids.append(self.client.put(self.Cx, "Cx" + str(self.frame)))
        ids.append(self.client.put(self.Call, "Call" + str(self.frame)))
        ids.append(self.client.put(self.Cpop, "Cpop" + str(self.frame)))
        ids.append(self.client.put(self.tune, "tune" + str(self.frame)))
        ids.append(self.client.put(self.color, "color" + str(self.frame)))
        ids.append(self.client.put(self.coordDict, "analys_coords" + str(self.frame)))
        ids.append(self.client.put(self.allStims, "stim" + str(self.frame)))
        ids.append(self.client.put(w, "w" + str(self.frame)))
        ids.append(self.client.put(np.array(self.LL), "LL" + str(self.frame)))
        ids.append(self.frame)

        self.q_out.put(ids)
        self.puttime.append(time.time() - t)

    def stimAvg_start(self):
        t = time.time()

        ests = self.C

        if self.ests.shape[0] < ests.shape[0]:
            diff = ests.shape[0] - self.ests.shape[0]
            # added more neurons, grow the array
            self.ests = np.pad(self.ests, ((0, diff), (0, 0), (0, 0)), mode="constant")
            # print('------------------Grew:', self.ests.shape)

        if self.baseline_coefficient.shape[0] < ests.shape[0]:
            diff = ests.shape[0] - self.baseline_coefficient.shape[0]
            # added more neurons, grow the array
            self.baseline_coefficient = np.pad(self.baseline_coefficient, ((0, diff), (0, 0), (0, 0, 0)), mode="constant")

        # if self.currentStim is not None:
        #     if self.stimStart == self.frame:
        #         # account for the baseline prior to stimulus onset
        #         self.ests[:, self.currentStim, 1] = (
        #             self.counter[self.currentStim, 1]
        #             * self.ests[:, self.currentStim, 1]
        #             + np.mean(ests[:, self.frame - 10 : self.frame], 1)
        #         ) / (self.counter[self.currentStim, 1] + 1)
        #         self.counter[self.currentStim, 1] += 10

        #     elif self.frame in range(self.stimStart + 1, self.frame + 26):
        #         self.ests[:, self.currentStim, 1] = (
        #             self.counter[self.currentStim, 1]
        #             * self.ests[:, self.currentStim, 1]
        #             + ests[:, self.frame - 1]
        #         ) / (self.counter[self.currentStim, 1] + 1)
        #         self.counter[self.currentStim, 1] += 1

        #         if self.frame in range(self.stimStart + 5, self.stimStart + 19):
        #             self.ests[:, self.currentStim, 0] = (
        #                 self.counter[self.currentStim, 0]
        #                 * self.ests[:, self.currentStim, 0]
        #                 + ests[:, self.frame - 1]
        #             ) / (self.counter[self.currentStim, 0] + 1)
        #             self.counter[self.currentStim, 0] += 1

        if self.stage == 'pre-stimuli':
            self.fit_line(frame_start, duration, ests, stimID)
        elif self.stage == 'stimuli':
            self.cal_signal(duration, neuro_data, stimID)
        else:
            pass

        self.estsAvg = np.squeeze(self.ests[:, :, 0] - self.ests[:, :, 1])
        self.estsAvg = np.where(np.isnan(self.estsAvg), 0, self.estsAvg)
        self.estsAvg[self.estsAvg == np.inf] = 0
        self.estsAvg[self.estsAvg < 0] = 0

        self.stimtime.append(time.time() - t)

    def plotColorFrame(self):
        """Computes colored nicer background+components frame"""
        t = time.time()
        image = self.image
        color = np.stack([image, image, image, image], axis=-1).astype(np.uint8).copy()
        color[..., 3] = 255
        # TODO: don't stack image each time?
        if self.coords is not None:
            for i, c in enumerate(self.coords):
                # c = np.array(c)
                ind = c[~np.isnan(c).any(axis=1)].astype(int)
                # TODO: Compute all colors simultaneously! then index in...
                cv2.fillConvexPoly(
                    color, ind, self._tuningColor(i, color[ind[:, 1], ind[:, 0]])
                )

        # TODO: keep list of neural colors. Compute tuning colors and IF NEW, fill ConvexPoly.

        self.colortime.append(time.time() - t)
        return color

    def _tuningColor(self, ind, inten):
        """ind identifies the neuron by number"""
        ests = self.estsAvg
        # ests = self.tune_k[0]
        if ests[ind] is not None:
            try:
                return self.manual_Color_Sum(ests[ind])
            except ValueError:
                return (255, 255, 255, 0)
            except Exception:
                print("inten is ", inten)
                print("ests[i] is ", ests[ind])
        else:
            return (255, 255, 255, 50)

    def manual_Color_Sum(self, x):
        """x should be length 12 array for coloring
        or, for k coloring, length 8
        Using specific coloring scheme from Naumann lab
        """
        if x.shape[0] == 8:
            mat_weight = np.array(
                [
                    [1, 0.25, 0],
                    [0.75, 1, 0],
                    [0, 1, 0],
                    [0, 0.75, 1],
                    [0, 0.25, 1],
                    [0.25, 0, 1.0],
                    [1, 0, 1],
                    [1, 0, 0.25],
                ]
            )
        elif x.shape[0] == 12:
            mat_weight = np.array(
                [
                    [1, 0.25, 0],
                    [0.75, 1, 0],
                    [0, 2, 0],
                    [0, 0.75, 1],
                    [0, 0.25, 1],
                    [0.25, 0, 1.0],
                    [1, 0, 1],
                    [1, 0, 0.25],
                    [1, 0, 0],
                    [0, 0, 1],
                    [0, 0, 1],
                    [1, 0, 0],
                ]
            )
        else:
            print("Wrong shape for this coloring function")
            return (255, 255, 255, 10)

        color = x @ mat_weight

        blend = 0.8
        thresh = 0.2
        thresh_max = blend * np.max(color)

        color = np.clip(color, thresh, thresh_max)
        color -= thresh
        color /= thresh_max
        color = np.nan_to_num(color)

        if color.any() and np.linalg.norm(color - np.ones(3)) > 0.35:
            color *= 255
            return (color[0], color[1], color[2], 255)
        else:
            return (255, 255, 255, 10)

    def IDstim(self, s):
        """Function to convert stim ID from Naumann lab experiment into
        the 8 cardinal directions they correspond to.
        """
        stim = -10
        if s == 3:
            stim = 0
        elif s == 10:
            stim = 1
        elif s == 9:
            stim = 2
        elif s == 16:
            stim = 3
        elif s == 4:
            stim = 4
        elif s == 14:
            stim = 5
        elif s == 13:
            stim = 6
        elif s == 12:
            stim = 7
        # add'l stim for specific coloring
        elif s == 5:
            stim = 8
        elif s == 6:
            stim = 9
        elif s == 7:
            stim = 10
        elif s == 8:
            stim = 11

        # if s == 3:
        #     stim = 0
        # elif s == 9:
        #     stim = 1
        # elif s == 15:
        #     stim = 2
        # elif s == 21:
        #     stim = 3
        # elif s == 27:
        #     stim = 4
        # elif s == 33:
        #     stim = 5
        # elif s == 39:
        #     stim = 6
        # elif s == 45:
        #     stim = 7

        return stim


    def read_class_info(self, filepath):
        label_df = pd.read_csv(filepath, sep=',', header = None)
        labels = labels_df.to_numpy()
        self.class_name = labels[0]
        self.class_labels = labels[1:]
        self.num_stimuli = np.shape(labels)[1] - 1
        self.num_class = np.shape(labels)[0]


    def cal_integral(self, frame_start, duration, neuro_data):
        x = np.linspace(frame_start, frame_start + duration, np.shape(response)[1])
        integral = [np.trapz(neuro_data[i,:], x) for i in range(np.shape(neuro_data)[0])]
        return integral

    def fit_line(self, frame_start, duration, neuro_data, stimID):
        response = neuro_data.reshape(neuro_data.shape[0], -1)
        regressor = np.linspace(frame_start, frame_start + duration, np.shape(response)[1])
        coefficient = np.polyfit(regressor, response.T, deg=1)
        residuals = response.T - np.polyval(coefficients, regressor)
        std = np.std(residuals, axis = 1)
        self.baseline_coefficient[:, stimID, 0] = coefficient[0]
        self.baseline_coefficient[:, stimID, 1] = coefficient[1]
        self.baseline_coefficient[:, stimID, 2] = std

    def cal_baseline(self, frame_start, duration, neuro_data, stimID):
        x = np.linspace(frame_start, frame_start + duration, np.shape(response)[1])
        baseline_mean = np.polyval([self.baseline_coefficient[:, stimID, 0], self.baseline_coefficient[:, stimID, 1]], x)
        baseline = baseline_mean + 1.8 * self.baseline_coefficient[:, stimID, 2] * duration
        self.ests[:, stimID, 0] = ((self.stumuli_visited_num[stimID] - 1) * self.ests[:, stimID, 0] + baseline) / self.stumuli_visited_num[stimID]

    def cal_signal(self, duration, neuro_data, stimID):
        x = np.linspace(frame_start, frame_start + duration, np.shape(response)[1])
        integral = self.cal_integral(frame_start, duration, neuro_data)
        self.ests[:, stimID, 1] = ((self.stumuli_visited_num[stimID] - 1) * self.ests[:, stimID, 1] + integral) / self.stumuli_visited_num[stimID]

    
    def eval_barcode():
        self.res = self.ests[:, :, 1] - self.ests[:, :, 0]
        self.res[...,2] = np.where(self.res[..., 2] > 0, 1, 0)
