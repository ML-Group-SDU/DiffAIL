import os
import pandas as pd
import numpy as np


def deal_csv_to_entropy_npy(target_dir="./npy_Entropy_result/4-trajectories/",
                            result_path="./ILSwiss/LFD/4-trajectories",
                            env_clips=None):
    task_names = os.listdir(result_path)
    for task_name in task_names:
        env = task_name.split("-")[0]
        file_names = os.listdir(os.path.join(result_path, task_name))
        o = 0
        for file_name in file_names:
            clip = None
            for key in env_clips:
                if key in env:
                    clip = env_clips[key]

            data_file_path = os.path.join(result_path, task_name, file_name, "progress.csv")
            if os.path.exists(data_file_path):
                if clip is not None:
                    try:
                        data = pd.read_csv(data_file_path)["Disc Rew Entropy"].values
                    except:
                        break
                    if len(data) > clip:
                        data = data[0:clip]
                if not os.path.exists(os.path.join(target_dir + task_name)):
                    os.makedirs(os.path.join(target_dir + task_name))

                np.save(target_dir + task_name + '/' + task_name + '-' + str(o) + '-' + str(len(data)) + '.npy',
                        data)
                o = o + 1


def deal_csv_to_npy(target_dir="./npy_result/4-trajectories/", result_path="./ILSwiss/LFD/4-trajectories",
                    env_clips=None):
    task_names = os.listdir(result_path)
    for task_name in task_names:
        env = task_name.split("-")[0]
        file_names = os.listdir(os.path.join(result_path, task_name))
        o = 0
        for file_name in file_names:
            clip = None
            for key in env_clips:
                if key in env:
                    clip = env_clips[key]

            data_file_path = os.path.join(result_path, task_name, file_name, "progress.csv")
            if os.path.exists(data_file_path):
                if clip is not None:
                    data = pd.read_csv(data_file_path)["AverageReturn"].values
                    if len(data) > clip:
                        data = data[0:clip]
                if not os.path.exists(os.path.join(target_dir + task_name)):
                    os.makedirs(os.path.join(target_dir + task_name))

                np.save(target_dir + task_name + '/' + task_name + '-' + str(o) + '-' + str(len(data)) + '.npy',
                        data)
                o = o + 1


def deal_txt_to_npy(target_dir="./npy_result/4-trajectories/", result_path="./dice/LFD/4-trajectories", env_clips=None):
    task_names = os.listdir(result_path)
    for task_name in task_names:
        env = task_name.split("-")[0]
        file_names = os.listdir(os.path.join(result_path, task_name))
        if "CFIL" in task_name:
            for file_name in file_names:
                seeds_files = os.listdir(os.path.join(result_path, task_name, file_name))
                o = 0
                for seed_file in seeds_files:
                    clip = None
                    for key in env_clips:
                        if key in env:
                            clip = env_clips[key]

                    txt_file = os.path.join(result_path, task_name, file_name, seed_file, "progress.txt")

                    with open(txt_file) as f:
                        lines = f.readlines()

                    data = []
                    for n in range(1, len(lines)):
                        data_column = lines[n].split('\t')
                        data.append(float(data_column[1]))

                    if clip is not None and len(data) > clip:
                        data = data[0:clip]

                    if not os.path.exists(os.path.join(target_dir + task_name)):
                        os.makedirs(os.path.join(target_dir + task_name))

                    np.save(target_dir + task_name + '/' + task_name + '-' + str(o) + '-' + str(len(data)) + '.npy',
                            data)
                    o = o + 1


def deal_opolo_to_npy(target_dir="./npy_result/4-trajectories/", result_path="./dice/LFD/4-trajectories",
                      env_clips=None):
    task_names = os.listdir(result_path)
    for task_name in task_names:
        env = task_name.split("-")[0]
        if "OPOLO" in task_name:
            seeds_files = os.listdir(os.path.join(result_path, task_name))
            o = 0
            for seed_file in seeds_files:
                clip = None
                for key in env_clips:
                    if key in env:
                        clip = env_clips[key]

                csv_file = os.path.join(result_path, task_name, seed_file, "agent0.monitor.csv")

                # operation csv
                file = pd.read_csv(csv_file)
                episodes_retruns = file.index[1:]
                episodes_lengths = file.iloc[1:, 0].values

                total_length = 0
                count = 0
                data = []
                for i in range(len(episodes_lengths)):

                    if total_length >= count * 10000:
                        data.append(float(episodes_retruns[i]))
                        count = count + 1
                    total_length = float(episodes_lengths[i]) + total_length

                if clip is not None and len(data) > clip:
                    data = data[0:clip]

                if not os.path.exists(os.path.join(target_dir + task_name)):
                    os.makedirs(os.path.join(target_dir + task_name))

                np.save(target_dir + task_name + '/' + task_name + '-' + str(o) + '-' + str(len(data)) + '.npy',
                        data)
                o = o + 1


def deal_npy_to_npy(target_dir="./npy_result/4-trajectories/", result_path="./dice/LFD/4-trajectories", env_clips=None):
    task_names = os.listdir(result_path)
    for task_name in task_names:
        env = task_name.split("-")[0]
        seeds_files = os.listdir(os.path.join(result_path, task_name))
        if "CFIL" not in task_name:
            if len(seeds_files) > 0:
                o = 0
                for seed_file in seeds_files:
                    clip = None
                    for key in env_clips:
                        if key in env:
                            clip = env_clips[key]
                    data = np.load(os.path.join(result_path, task_name, seed_file))

                    if clip is not None and len(data) > clip:
                        data = data[:clip]

                    if not os.path.exists(os.path.join(target_dir + task_name)):
                        os.makedirs(os.path.join(target_dir + task_name))

                    np.save(target_dir + task_name + '/' + task_name + '-' + str(o) + '-' + str(len(data)) + '.npy',
                            data)
                    o = o + 1


def deal_ablation_to_npy(target_dir="./npy_ablation_result/diffusion_timesteps/",
                         result_path="./Ablation/diffusion_timesteps",
                         env_clips=None):
    task_names = os.listdir(result_path)
    for task_name in task_names:
        env = task_name
        time_stepss = os.listdir(os.path.join(result_path, task_name))

        clip = None
        for key in env_clips:
            if key in env:
                clip = env_clips[key]
                break
        for time_steps in time_stepss:
            if not os.path.exists(os.path.join(target_dir, task_name + '-' + time_steps)):
                os.makedirs(os.path.join(target_dir, task_name + '-' + time_steps))

            o = 0

            data_seed_files = os.listdir(os.path.join(result_path, task_name, time_steps))

            for data_seed_file in data_seed_files:

                # file = os.listdir(os.path.join(result_path, task_name, file_name))[0]
                data_file_path = os.path.join(result_path, task_name, time_steps, data_seed_file, "progress.csv")
                if os.path.exists(data_file_path):
                    if clip is not None:
                        data = pd.read_csv(data_file_path)["AverageReturn"].values
                        if len(data) > clip:
                            data = data[0:clip]

                    np.save(target_dir + task_name + '-' + str(time_steps) + '/' + task_name + '-' + str(o) + '-' + str(
                        len(data)) + '.npy',
                            data)
                    o = o + 1


if __name__ == '__main__':

    env_clips = {"Ant": 50, "Hopper": 75, "Walker": 75, "Halfcheetah": 50, "Swimmer": 50, "Reacher": 50,
                 "InvertedDoublePendulum": 50,
                 "InvertedPendulum": 50}

    target_dirs = ["./npy_result/1-trajectories/", "./npy_result/4-trajectories/",
                   "./npy_result/16-trajectories/"]


    lfo_target_dirs = ["./npy_lfo_result/1-trajectories/", "./npy_lfo_result/4-trajectories/",
                       "./npy_lfo_result/16-trajectories/"]

    entropy_target_dirs = ["./npy_Entropy_result/1-trajectories/", "./npy_Entropy_result/4-trajectories/",
                           "./npy_Entropy_result/16-trajectories/"]

    ablation_target_dirs = ["./npy_ablation_result/diffusion_timesteps/", "./npy_ablation_result/schedule_function/"]
    ablation_result_dirs = ["./Ablation/diffusion_timesteps/", "./Ablation/schedule_function/"]

    dice_result_dirs = ["./dice/LFD/1-trajectories", "./dice/LFD/4-trajectories",
                        "./dice/LFD/16-trajectories"]

    dice_lfo_result_dirs = ["./dice/LFO/1-trajectories", "./dice/LFO/4-trajectories",
                            "./dice/LFO/16-trajectories"]

    ILSwiss_result_dirs = ["./ILSwiss/LFD/1-trajectories", "./ILSwiss/LFD/4-trajectories",
                           "./ILSwiss/LFD/16-trajectories"]
    lfo_ILSwiss_result_dirs = ["./ILSwiss/LFO/1-trajectories", "./ILSwiss/LFO/4-trajectories",
                           "./ILSwiss/LFO/16-trajectories"]
    # ILSwiss  enrtopy
    for i in range(len(ILSwiss_result_dirs)):
        deal_csv_to_entropy_npy(target_dir=entropy_target_dirs[i], result_path=ILSwiss_result_dirs[i],
                                env_clips=env_clips)


    # ILSwiss lfd
    for i in range(len(ILSwiss_result_dirs)):
        deal_csv_to_npy(target_dir=target_dirs[i], result_path=ILSwiss_result_dirs[i], env_clips=env_clips)

    # ILSwiss lfo
    for i in range(len(lfo_ILSwiss_result_dirs)):
        deal_csv_to_npy(target_dir=lfo_target_dirs[i], result_path=lfo_ILSwiss_result_dirs[i], env_clips=env_clips)
    #
    #
    # dice CFIL lfd
    for i in range(len(dice_result_dirs)):
        deal_txt_to_npy(target_dir=target_dirs[i], result_path=dice_result_dirs[i], env_clips=env_clips)

    # dice CFIL lfo
    for i in range(len(dice_lfo_result_dirs)):
        deal_txt_to_npy(target_dir=lfo_target_dirs[i], result_path=dice_lfo_result_dirs[i], env_clips=env_clips)

    # dice OPOLO lfo
    for i in range(len(dice_lfo_result_dirs)):
        deal_opolo_to_npy(target_dir=lfo_target_dirs[i], result_path=dice_lfo_result_dirs[i], env_clips=env_clips)

    for i in range(len(dice_result_dirs)):
        deal_npy_to_npy(target_dir=target_dirs[i], result_path=dice_result_dirs[i], env_clips=env_clips)

    for i in range(len(ablation_result_dirs)):
        deal_ablation_to_npy(target_dir=ablation_target_dirs[i], result_path=ablation_result_dirs[i],
                             env_clips=env_clips)
