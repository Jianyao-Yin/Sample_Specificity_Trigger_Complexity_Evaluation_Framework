# Sample_Specificity_Trigger_Complexity_Evaluation_Framework
This repository is the code base for the manuscript "SSTC: Towards a Fundamental Understanding of Sample-Specific and Sample-Agnostic Triggers in Backdoor Attacks"
This code base contain one python file of the alghorithm of SSTC evaluation framework described in the manuscript. 
The code is built on Python 3.11.0 and Pytorch 2.5.1+cu118.
The only dataset required by this code file is CIFAR-10, while the code can also go with data free mode. The requirement is that, the resolution of the training samples must be 3*32*32, and the pixel value range is [0, 1].
To perform the evaluation one should first create a function of the poison sample generation process to be tested (the input of this function is a benign sample, the outout is the corresponding poison sample). Then add this fucntion to the inside of generate_poi_sample function, and select this fucntion. After that, run the code and after an amount of thme, the program will output the sample-specificity score, which is the model depth at the termination.
