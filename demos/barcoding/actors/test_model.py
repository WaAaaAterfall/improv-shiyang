import numpy as np
# class testAnalysis:
#     def __init__:
#         self.baseline_coefficient = np.zeros((3, 8, 3))

def generate_matrix(num_neuro = 1, num_stimuli = 8, mode = 'responsive'):
    pre_stimuli1 = np.random.normal(2.0, 1.0, (num_neuro, 10))
    pre_stimuli2 = np.random.normal(2.0, 1.0, (num_neuro, 10))
    post_stimuli = np.random.normal(2.0, 1.0, (num_neuro, 10))
    stimuli_responsive = np.random.normal(10, 1.0, (num_neuro, 20))
    print(np.shape(stimuli_responsive))
    stimuli_nonresponsive = np.random.normal(4.0, 2.0, (num_neuro, 20))
    matrix = np.concatenate((pre_stimuli1,stimuli_responsive, post_stimuli), axis = 1)
    if mode == 'pre-stimulus-responsive':
        return pre_stimuli1
    elif mode == 'responsive':
        return stimuli_responsive
    elif mode == 'pre-stimulus-nonresponsive':
        return pre_stimuli2
    elif mode == 'nonresponsive':
        return stimuli_nonresponsive
    else:
        return post_stimuli


def get_barcode(neuro_data):
    return 0


def cal_integral(frame_start, duration, neuro_data):
    x = np.linspace(frame_start, frame_start + duration, np.shape(neuro_data)[1])
    integral = np.trapz(neuro_data, x, axis = 1)
    return integral.reshape(integral.shape[0], -1)

def fit_line(frame_start, duration, neuro_data, stimID):
    print(np.shape(neuro_data))
    response = neuro_data.reshape(neuro_data.shape[0], -1)
    print(np.shape(response))
    regressor = np.linspace(frame_start, frame_start + duration, np.shape(response)[1])
    coefficients = np.polyfit(regressor, response.T, deg=1).T
    print(np.shape(coefficients))
    residuals = response - [np.polyval([coefficients[i, 0],coefficients[i,1]], regressor) for i in range(np.shape(neuro_data)[0])]
    print(residuals)
    std = np.std(residuals, axis = 1)
    print("std: ", std)
    baseline_coefficient = np.zeros((3, 8, 3))
    baseline_coefficient[:, stimID, 0] = coefficients[:,0]
    baseline_coefficient[:, stimID, 1] = coefficients[:,1]
    baseline_coefficient[:, stimID, 2] = std
    return baseline_coefficient

def cal_baseline(frame_start, duration, neuro_data, stimID, coefficients):
    x = np.linspace(frame_start, frame_start + duration, np.shape(neuro_data)[1])
    baseline_mean = [np.mean(np.polyval([coefficients[i, stimID, 0],coefficients[i,stimID, 1]], x)) for i in range(np.shape(neuro_data)[0])]
    print("baseline_mean:", baseline_mean, duration)
    baseline = (baseline_mean + 1.8 * coefficients[:, stimID, 2]) * duration
    #self.ests[:, stimID, 0] = ((self.stumuli_visited_num[stimID] - 1) * self.ests[:, stimID, 0] + baseline) / self.stumuli_visited_num[stimID]
    return baseline.reshape(baseline.shape[0], -1)

def cal_signal(frame_start, duration, neuro_data, stimID):
    integral = cal_integral(frame_start, duration, neuro_data)
    #self.ests[:, stimID, 1] = ((self.stumuli_visited_num[stimID] - 1) * self.ests[:, stimID, 1] + integral) / self.stumuli_visited_num[stimID]
    return integral

def eval_barcode():
    self.res = self.ests[:, :, 1] - self.ests[:, :, 0]
    self.res[...,2] = np.where(self.res[..., 2] > 0, 1, 0)

if __name__ == "__main__":
    # fake_pre_stimulus = generate_matrix(3, 8, "pre-stimulus-responsive")
    # fake_stimulus = generate_matrix(3, 8, "responsive")
    # coefficient = fit_line(0, np.shape(fake_pre_stimulus[0])[0], fake_pre_stimulus, 0)
    # baseline = cal_baseline(0, np.shape(fake_stimulus[0])[0], fake_stimulus, 0, coefficient)
    # signal = cal_signal(0, np.shape(fake_pre_stimulus[0])[0], fake_stimulus ,0)
    # print(baseline, "\n\n", signal, "\n\n", signal - baseline)
    #barcode = get_barcode(fake_neuro_data)
    # print(barcode)
    links = {}
    links['link_in'] = "aaaa"
    links['link_out'] = 'bbbbb'
    print(links["link_in"])
