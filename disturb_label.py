import numpy as np
import copy
def create_disturb_label(label,noise_rate=0.1):
    def generate_distribution(ground_truth,num_class=20):
        res = [noise_rate*1./float(num_class) for _ in range(num_class)]
        res[ground_truth] = 1.-(num_class-1)/num_class*noise_rate
        return res

    new_label = np.argmax(np.random.multinomial(1,generate_distribution(label)))

    return new_label

