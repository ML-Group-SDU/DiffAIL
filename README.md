# DiffAIL:Diffusion Adversarial Imitation Learning

PyTorch implementation of Diffusion Adversarial Imitation Learning. If you use our code or data please
cite the paper.

### method
Building on the diffusion model's superior ability to learn data distribution, we present a new method called Diffusion Model Loss as a reward
function for Generative Adversarial Imitation Learning (DiffAIL) in this work. DiffAIL employs the same framework as traditional AIL but incorporates
the diffusion model's learning loss. By combining the diffusion model's loss with AIL, DiffAIL enables the discriminator to accurately capture both 
expert demonstration and policy data distributions, facilitating the successful identification of expert-level behaviours that were previously unseen. 

### Environments

Method is tested on MuJoCo continuous control tasks in OpenAI gym. Our methods are trained using PyTorch 1.11.0+cu113 and Python
3.9.


### Dataset
Since different SOTA methods use different data file formats, to facilitate reproduction of the experimental results,
we provide the datasets for all experiments, including Valuedice, CFIL and OPOLO (the file contents are the same, only 
the suffixes differ). All the dataset files are under ./demos.

### Usage  
The paper results can be reproduced by running:
```
python ./run_experiment.py -e "./exp_specs/ddpm_halfcheetah.yaml" -g 0
python ./run_experiment.py -e "./exp_specs/gail_halfcheetah.yaml" -g 0
```

The algorithms we used BC, Valuedice, CFIL and OPOLO can be found from the following sources:

BC+Valuedice[(code)](https://github.com/google-research/google-research/tree/master/value_dice) : Imitation Learning via Off-Policy Distribution Matching[(paper)](https://arxiv.org/abs/1912.05032)  
CFIL[(code)](https://github.com/gfreund123/cfil) : A Coupled Flow Approach to Imitation Learning[(paper)](https://arxiv.org/abs/2305.00303)  
OPOLO[(code)](https://github.com/illidanlab/opolo-code) : Off-Policy Imitation Learning from Observations[(paper)](https://arxiv.org/abs/2102.13185)  
### Result

Top: Learning curve for different sota imitation learning algorithms with 1 trajectory 5 five seeds in the standard
state-action setting. Bottom: Learning curve for different sota imitation learning algorithms with one trajectory over five seeds
in the state-only setting. The x-axis denotes timesteps, and the y-axis denotes the average return. The shadow areas represent
the standard deviation.
&nbsp;
<p align="center">
<img src='./assets/1-trajectory.png' width=800>
</p>

Top: Learning curve for different sota imitation learning algorithms with 4 trajectories over 5 seeds in the standard
state-action setting. Bottom: Learning curve for different sota imitation learning algorithms with 4 trajectories over 5 seeds in
the state-only setting. The x-axis denotes timesteps, and the y-axis denotes the average return. The shadow areas represent the
standard deviation
&nbsp;
<p align="center">
<img src='./assets/4-trajectories.png' width=800>
</p>
Top: Learning curve for different sota imitation learning algorithms with 16 trajectories over 5 seeds in the standard
state-action setting. Bottom: Learning curve for different sota imitation learning algorithms with 16 trajectories over 5 seeds in
the state-only setting. The x-axis denotes timesteps, and the y-axis denotes the average return. The shadow areas represent the
standard deviation.
&nbsp;
<p align="center">
<img src='./assets/16-trajectories.png' width=800>
</p>

### Acknowledgements
This repo relies on the following existing codebases:
- The diffusion model variant  based on [Diffusion Q](https://github.com/zhendong-wang/diffusion-policies-for-offline-rl)
- The Adversarial Imitation Learning framework is adapted from [here](https://github.com/Ericonaldo/ILSwiss)


