import numpy as np

def generate_harmonic_noise(max_harmonics, samplesize):
    time = np.linspace(0,1,num=samplesize)
    num_of_harmonics = np.random.randint(low=0, high=max_harmonics )
    harmonics = np.random.randint(low=0, high=round(samplesize/2)+2, size=num_of_harmonics)
    harmonic_noise = np.zeros(samplesize)
    for j in harmonics:
        harmonic_noise = harmonic_noise + np.cos(2*j*np.pi*time)
    return harmonic_noise

def generate_training_data(base_freq, iterations, samplesize, num_phases) :

    time = np.linspace(0,1,num=samplesize)
    X = np.zeros((iterations*num_phases, samplesize))
    Y = np.zeros((iterations*num_phases, num_phases))
    x = np.zeros((num_phases, samplesize))
    y = np.eye(num_phases)
    #The upper bound on the number of harmonics is hardcoded for now.
    max_harmonics = 12

    #These are the signals we are interested in classifying
    #Creating them ahead of time for convenience.
    for i in range(0, num_phases):
        x[i,:] = np.cos(2*base_freq*np.pi*time + i*np.pi/num_phases)                  

    for i in range(0,iterations):
        for j in range(0, num_phases):
            X[num_phases*i+j,:] = x[j,:] + generate_harmonic_noise(max_harmonics, samplesize) + np.random.randn(samplesize)
        Y[num_phases*i:num_phases*(i+1),:] = y

    return X, Y