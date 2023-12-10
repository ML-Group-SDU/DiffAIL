import seaborn as sns
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

plt.rc('font', size=19)


# add expert score

def insert_expert_return(expert_returns, timestep_list):
    df_list_expert = {}
    timestep_list_expert = {}
    for env in expert_returns.keys():
        returns = expert_returns[env]

        for env_timelist in timestep_list.keys():
            if env in env_timelist:
                timestep = timestep_list[env_timelist]
                break

        data = np.full((int(timestep / 10000), 5), returns, dtype=float)
        for i in range(5):
            if i >= 1:
                time = np.column_stack((time, np.arange(0, timestep, 10000)))
            else:
                time = np.arange(0, timestep, 10000)
        data_flatten = data.flatten()
        time_flatten = time.flatten()
        dat = np.column_stack((time_flatten, data_flatten))
        df = pd.DataFrame(dat, columns=['time', 'reward'])
        df_list_expert[env + "-expert"] = df
        timestep_list_expert[env + "-expert"] = timestep

    return df_list_expert, timestep_list_expert


def get_data(smooth_list, expert_returns=None, file_dir="./npy_result/4-trajectories"):
    task_names = os.listdir(file_dir)
    df_list = {}
    timestep_list = {}
    window_size = 4
    for task_name in task_names:
        file_names = os.listdir(os.path.join(file_dir, task_name))
        i = 0
        data = None
        time = None
        total_timesteps = 0
        smooth = False
        for smooth_ in smooth_list.keys():
            if smooth_ in task_name:
                smooth = smooth_list[smooth_]
                break

        if task_name == "Ant-CFIL":
            print("1")

        for file_name in file_names:
            data_temp = np.load(os.path.join(file_dir, task_name, file_name))
            total_timesteps = data_temp.size * 10000
            if smooth:
                data_temp = np.pad(data_temp, (window_size // 2 - 1, window_size // 2), mode='edge')
                data_temp = np.convolve(data_temp, np.ones(window_size) / window_size, mode='valid')
            if i >= 1:
                data = np.column_stack((data, data_temp))
                time = np.column_stack((time, np.arange(0, data_temp.size * 10000, 10000)))
            else:
                data = data_temp
                time = np.arange(0, data_temp.size * 10000, 10000)
            i = i + 1
        data_flatten = data.flatten()
        time_flatten = time.flatten()
        dat = np.column_stack((time_flatten, data_flatten))
        df = pd.DataFrame(dat, columns=['time', 'reward'])
        df_list[task_name] = df
        timestep_list[task_name] = total_timesteps

    # insert expert returns
    if expert_returns is not None:
        df_list_expert, timestep_list_expert = insert_expert_return(expert_returns, timestep_list)
        df_list.update(df_list_expert)
        timestep_list.update(timestep_list_expert)
    return df_list, timestep_list


def draw_plot(data, lables, titlename, total_timesteps, color, linestyle, suffix="4-trajectories", save_dir=None,
              y_label="Average Return", final=False, setting=None):
    fig = plt.figure(figsize=(5.4, 5.4))
    sns.set_style("darkgrid")
    for i in range(len(data)):
        ax = sns.lineplot(x='time', y='reward', color=color[i], data=data[i], label=lables[i], ci=80)
        ax.lines[i].set_linestyle(linestyle[i])
    total_timesteps = total_timesteps
    plt.xlim(0, total_timesteps)

    plt.legend(fontsize=17)
    plt.ylabel(y_label, fontsize=22)
    plt.xlabel("Timesteps", fontsize=22)
    if final:
        plt.title(titlename, fontsize=28)
    else:
        plt.title(titlename + "-" + suffix, fontsize=25)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax = plt.gca()
    ax.xaxis.major.formatter._useMathText = True
    plt.tight_layout()
    if setting == None:
        plt.savefig("./images/" + save_dir + "_" + titlename + "_" + suffix + ".pdf", format='pdf')
    else:
        plt.savefig("./images/" + save_dir + "_" + titlename + "_" + suffix + "_" + setting + ".pdf", format='pdf')
    plt.show()


def get_most_return(df_list, seeds=5, ):
    print("most high rewards")
    for task in df_list.keys():
        data_temp = df_list[task].iloc[:, 1].values.squeeze()
        data = data_temp.reshape(int(data_temp.size / seeds), seeds)
        mean = np.mean(data, axis=1)
        std = np.std(data, axis=1)
        print(task + ": " + str('%.2f' % np.max(mean)) + " ± " + str('%.2f' % std[np.argmax(mean)]))


def get_plots(envs, df_list, timestep_list, color_algorithms, linestyle_algorithms, suffix="4-trajectories",
              save_dir=None,
              y_label="Average Return", final=False, setting=None):
    for env in envs:
        color = []
        data_list = []
        total_timesteps = 0
        labels = []
        linestyle = []
        for task in df_list.keys():
            if env in task:
                if not final:
                    labels.append(task)
                else:
                    if "Hopper" in task:
                        labels.append(task.split("-")[1])
                    else:
                        labels.append("")
                data_list.append(df_list[task])
                total_timesteps = timestep_list[task]
                for key in color_algorithms.keys():
                    if key == task.split('-')[1]:
                        color.append(color_algorithms[key])
                        linestyle.append(linestyle_algorithms[key])
        draw_plot(data_list, labels, env, total_timesteps, color, linestyle, save_dir=save_dir, suffix=suffix,
                  y_label=y_label, final=final, setting=setting)


def get_plots_ablation(envs, df_list, timestep_list, color_algorithms, linestyle_algorithms, suffix="4-trajectories",
                       save_dir=None,
                       y_label="Average Return"):
    for env in envs:
        color = []
        data_list = []
        total_timesteps = 0
        labels = []
        linestyle = []
        for task in df_list.keys():
            if env in task:
                labels.append(task)
                data_list.append(df_list[task])
                total_timesteps = timestep_list[task]
                for key in color_algorithms.keys():
                    if key in task:
                        color.append(color_algorithms[key])
                        linestyle.append(linestyle_algorithms[key])
        draw_plot(data_list, labels, env, total_timesteps, color, linestyle, save_dir=save_dir, suffix=suffix,
                  y_label=y_label)


if __name__ == '__main__':
    final = True
    target_dirs = ["./npy_result/1-trajectories/", "./npy_result/4-trajectories/",
                   "./npy_result/16-trajectories/"]

    lfo_target_dirs = ["./npy_lfo_result/1-trajectories/", "./npy_lfo_result/4-trajectories/",
                       "./npy_lfo_result/16-trajectories/"]
    # target_dirs = ["./npy_result/1-trajectories/"]

    envs = ["Ant", "Hopper", "Halfcheetah", "Walker"]
    # envs = ["Ant"]
    # envs = ["Walker"]
    smooth_list = {"Ant": True, "Hopper": True, "Halfcheetah": True, "Walker": True}

    # color = ['red', 'green', 'blue', '#44DFD0', '#016300', '#78FF00', 'w', 'peru', 'slategrey', 'm']
    color_algorithms = {"valuedice": 'red', "bc": 'blue', 'CFIL': '#44DFD0',
                        "DiffAIL": 'peru', "gail": 'green', "OPOLO": 'green', "expert": "black"}
    # color_algorithms = {"valuedice": '#E3653C', "bc": '#BDC77E', 'CFIL': '#BFBC49',
    #                     "ddpm": '#D12845', "gail": '#5EAE65', "expert": "black"}
    linestyle_algorithms = {"valuedice": '-', "bc": '-', 'CFIL': '-', "DiffAIL": '-', "gail": '-', "OPOLO": '-',
                            "expert": "--"}

    expert_returns = {"Ant": 4228, "Hopper": 3402, "Walker2d": 6717, "Halfcheetah": 4463}

    entropy_dir = "./npy_Entropy_result/4-trajectories"
    entropy_dirs = ["./npy_Entropy_result/1-trajectories/", "./npy_Entropy_result/4-trajectories",
                    "./npy_Entropy_result/16-trajectories/"]
    # draw entropy
    # entropy_df_list, entropy_timestep_list = get_data(smooth_list, file_dir=entropy_dir)
    # get_plots(envs, entropy_df_list, entropy_timestep_list, color_algorithms, linestyle_algorithms, save_dir='entropy',
    #           y_label="Entropy")
    #
    ablation_envs = ["Ant", "Hopper", "Halfcheetah", "Walker2d"]
    # ablation_envs = ["Ant"]
    ablation_target_dirs = ["./npy_ablation_result/diffusion_timesteps/", "./npy_ablation_result/schedule_function/"]
    color_ablation = [{"2": 'red', "5": 'blue', '10': 'peru',
                       "20": 'green', "expert": "black"},
                      {"cosine": 'red', "vp": 'green', "linear": 'peru', "expert": "black"}]
    linestyle_ablation = [{"2": '-', "5": '-', '10': '-', "20": '-', "expert": "--"},
                          {"cosine": '-', "vp": '-', "linear": '-', "expert": "--"}]

    # for dir_ in target_dirs:
    #     # draw reward
    #     df_list, timestep_list = get_data(smooth_list, expert_returns, file_dir=dir_)
    #     get_plots(envs, df_list, timestep_list, color_algorithms, linestyle_algorithms, save_dir='return',
    #               suffix=dir_.split("/")[2], final=final)
    #     get_most_return(df_list)

    for dir_ in lfo_target_dirs:
        # draw reward
        df_list, timestep_list = get_data(smooth_list, expert_returns, file_dir=dir_)
        get_plots(envs, df_list, timestep_list, color_algorithms, linestyle_algorithms, save_dir='return',
                  suffix=dir_.split("/")[2], final=final, setting="lfo")
        get_most_return(df_list)

    # for i in range(len(ablation_target_dirs)):
    #     # draw ablation
    #     df_list, timestep_list = get_data(smooth_list, expert_returns, file_dir=ablation_target_dirs[i])
    #     get_plots(ablation_envs, df_list, timestep_list, color_ablation[i], linestyle_ablation[i], save_dir='return',
    #               suffix=ablation_target_dirs[i].split("/")[2], final=final)
    #     get_most_return(df_list)

    # for dir_ in entropy_dirs:
    #     # draw entropy
    #     entropy_df_list, entropy_timestep_list = get_data(smooth_list, file_dir=dir_)
    #     get_plots(envs, entropy_df_list, entropy_timestep_list, color_algorithms, linestyle_algorithms,
    #               save_dir='entropy', y_label="Entropy", suffix=dir_.split("/")[2], final=final)
