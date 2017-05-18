import numpy as np
import tensorflow as tf

def generate_training_data(base_freq, iterations, samplesize) :
    time = np.linspace(0,1,num=samplesize)
    g1 = np.cos(2*base_freq*np.pi*time)
    g2 = np.cos(2*base_freq*np.pi*time + np.pi/2)
    g3 = np.cos(2*base_freq*np.pi*time + np.pi)
    g4 = np.cos(2*base_freq*np.pi*time + 3*np.pi/2)
    
    #The upper bound on the number of harmonics is hardcoded for now.
    num_of_harmonics = np.random.randint(low=0, high=12 )
    harmonics = np.random.randint(low=0, high=round(samplesize/2)+2, size=num_of_harmonics)
    harmonic_noise = np.zeros(samplesize)
    for j in harmonics:
        harmonic_noise = harmonic_noise + np.cos(2*j*np.pi*time)
    x1 = g1 + harmonic_noise + np.random.randn(samplesize)
    x2 = g2 + harmonic_noise + np.random.randn(samplesize)
    x3 = g3 + harmonic_noise + np.random.randn(samplesize)
    x4 = g4 + harmonic_noise + np.random.randn(samplesize)

    X = np.column_stack((x1,x2,x3,x4))
    Y = np.eye(4)

    for i in range(1,iterations):
        num_of_harmonics = np.random.randint(low=0, high=12 )
        harmonics = np.random.randint(low=0, high=round(samplesize/2)+2, size=num_of_harmonics)
        harmonic_noise = np.zeros(samplesize)
        for j in harmonics:
            harmonic_noise = harmonic_noise + np.cos(2*j*np.pi*time)
        x1 = g1 + harmonic_noise + np.random.randn(samplesize)
        x2 = g2 + harmonic_noise + np.random.randn(samplesize)
        x3 = g3 + harmonic_noise + np.random.randn(samplesize)
        x4 = g4 + harmonic_noise + np.random.randn(samplesize)
        x = np.column_stack((x1,x2,x3,x4))
        X = np.column_stack((X,x))
        Y = np.column_stack((Y,np.eye(4)))
    
    return X.transpose(), Y.transpose()