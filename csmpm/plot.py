import numpy as np
import matplotlib.pyplot as plt
import myutils
from . import utils
import matplotlib as mpl

def plot_masks():
    random_4 = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/masks/random_4.npy')
    # equispaced_4 = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/masks/equispaced_4.npy')
    equispaced_4 = np.zeros((256, 256))
    equispaced_4[:128, :128] = 50 
    uniform_4 = np.ones((256,256)) * 12.5
    halfhalf_4 = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/masks/halfhalf_4_sequency.npy')
    raw_learned_4 = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/masks/fmask_learned_4_raw_sequency.npy')
    sim_learned_4 = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/masks/fmask_learned_4_simulate_sequency.npy')

    random_8 = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/masks/random_8.npy')
    # equispaced_8 = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/masks/equispaced_8.npy')
    equispaced_8 = np.zeros((256, 256))
    equispaced_8[:90, :90] = 50
    uniform_8 = np.ones((256,256)) * 6.25
    halfhalf_8 = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/masks/halfhalf_8_sequency.npy')
    raw_learned_8 = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/masks/fmask_learned_8_raw_sequency.npy')
    sim_learned_8 = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/masks/fmask_learned_8_simulate_sequency.npy')

    fig, axes = plt.subplots(2, 6, figsize=(16, 5.5))
    myutils.plot.plot_img(random_4, ax=axes[0, 0], vlim=[0, 1], title='Random', ylabel=r'$\alpha=0.25$')
    myutils.plot.plot_img(equispaced_4, ax=axes[0, 1], vlim=[0, 50], title='Low-sequency')
    myutils.plot.plot_img(halfhalf_4, ax=axes[0, 2], vlim=[0, 1], title='Half-half')
    myutils.plot.plot_img(uniform_4, ax=axes[0, 3], vlim=[0, 50], title='Uniform')
    myutils.plot.plot_img(sim_learned_4, ax=axes[0, 4], vlim=[0, 50], title='Learned, Simulated')
    myutils.plot.plot_img(raw_learned_4, ax=axes[0, 5], vlim=[0, 50], title='Learned, Raw')

    myutils.plot.plot_img(random_8, ax=axes[1, 0], vlim=[0, 1], ylabel=r'$\alpha=0.125$')
    myutils.plot.plot_img(equispaced_8, ax=axes[1, 1], vlim=[0, 50])
    myutils.plot.plot_img(halfhalf_8, ax=axes[1, 2], vlim=[0, 1])
    myutils.plot.plot_img(uniform_8, ax=axes[1, 3], vlim=[0, 50])
    myutils.plot.plot_img(sim_learned_8, ax=axes[1, 4], vlim=[0, 50])
    ax, im = myutils.plot.plot_img(raw_learned_8, ax=axes[1, 5], vlim=[0, 50])

    cbar_ax = fig.add_axes([0.92, 0.4, 0.01, 0.4])
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=[0,25, 50])
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.set_yticklabels(['0', '25', '50'])  # vertically oriented colorbar
    # fig.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    return fig

def plot_all_boxes():
    fig, axes = plt.subplots(1, 1, figsize=(14, 6))
    plot_boxes(8, ax=axes, global_offset=-0.3, legend=True)
    plot_boxes(4, ax=axes, global_offset=0.3)
    axes.grid()
    plt.show()
    return fig

def plot_boxes(accelrate, ax=None, global_offset=0, legend=False):
    ax = ax or plt.gca()

    vals_simbase = []
    vals_rawbase = []
    vals_sim = []
    vals_raw = []

    if accelrate == 4:
        random_sim_base = np.load('/home/jm2239/MPM/random_4_simulate/save_dir_wv/PSNRs_GD_tv_wv_0.01.npy')
    else:
        random_sim_base = np.load('/home/jm2239/MPM/random_8_simulate/save_dir_wv/PSNRs_GD_tv_wv_0.095.npy')
    # random_4_sim_base = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/metrics/baselines/random_{rate}_simulate_09tv_09wave_psnr.npy'.format(rate=accelrate))
    if accelrate == 4:
        random_raw_base = np.load('/home/jm2239/MPM/random_4_raw/save_dir_wv/PSNRs_GD_tv_wv_0.01.npy')
    else:
        random_raw_base = np.load('/home/jm2239/MPM/random_8_raw/save_dir_wv/PSNRs_GD_tv_wv_0.01.npy')
    # random_4_raw_base = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/metrics/baselines/random_{rate}_raw_09tv_09wave_psnr.npy'.format(rate=accelrate))
    random_sim = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/metrics/random_{rate}_simulate_psnr.npy'.format(rate=accelrate))
    random_raw = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/metrics/random_{rate}_raw_psnr.npy'.format(rate=accelrate))

    if accelrate == 4:
        equispaced_sim_base = np.load('/home/jm2239/MPM/equispaced_4_simulate/save_dir_wv/PSNRs_GD_tv_wv_0.5.npy')
    else:
        equispaced_sim_base = np.load('/home/jm2239/MPM/equispaced_8_simulate/save_dir_wv/PSNRs_GD_tv_wv_0.4.npy')
    # equispaced_4_sim_base = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/metrics/baselines/equispaced_{rate}_simulate_09tv_09wave_psnr.npy'.format(rate=accelrate))
    if accelrate == 4:
        equispaced_raw_base = np.load('/home/jm2239/MPM/equispaced_4_raw/save_dir_wv/PSNRs_GD_tv_wv_0.2.npy')
    else:
        equispaced_raw_base = np.load('/home/jm2239/MPM/equispaced_8_raw/save_dir_wv/PSNRs_GD_tv_wv_0.2.npy')
    # equispaced_4_raw_base = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/metrics/baselines/equispaced_{rate}_raw_06tv_06wave_psnr.npy'.format(rate=accelrate))
    equispaced_sim = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/metrics/equispaced_{rate}_simulate_psnr.npy'.format(rate=accelrate))
    equispaced_raw = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/metrics/equispaced_{rate}_raw_psnr.npy'.format(rate=accelrate))


    if accelrate == 4:
        halfhalf_sim_base = np.load('/home/jm2239/MPM/half_4_sim/save_dir_wv/PSNRs_GD_tv_wv_0.6.npy')
    else:
        halfhalf_sim_base = np.load('/home/jm2239/MPM/half_8_sim/save_dir_wv/PSNRs_GD_tv_wv_0.8.npy')
    # equispaced_4_sim_base = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/metrics/baselines/equispaced_{rate}_simulate_09tv_09wave_psnr.npy'.format(rate=accelrate))
    if accelrate == 4:
        halfhalf_raw_base = np.load('/home/jm2239/MPM/half_4_raw/save_dir_wv/PSNRs_GD_tv_wv_0.30000000000000004.npy')
    else:
        halfhalf_raw_base = np.load('/home/jm2239/MPM/half_8_raw/save_dir_wv/PSNRs_GD_tv_wv_0.47250000000000003.npy')
    # equispaced_4_raw_base = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/metrics/baselines/equispaced_{rate}_raw_06tv_06wave_psnr.npy'.format(rate=accelrate))
    halfhalf_sim = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/metrics/halfhalf_{rate}_simulate_psnr.npy'.format(rate=accelrate))
    halfhalf_raw = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/metrics/halfhalf_{rate}_raw_psnr.npy'.format(rate=accelrate))

    uniform_sim_base = []
    uniform_raw_base = []
    uniform_sim = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/metrics/uniform_{rate}_simulate_psnr.npy'.format(rate=accelrate))
    uniform_raw = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/metrics/uniform_{rate}_raw_psnr.npy'.format(rate=accelrate))

    if accelrate == 4:
        learned_sim_base = np.load('/home/jm2239/MPM/learned_4_simulate/save_dir_wv/PSNRs_GD_tv_wv_0.01.npy')
    else:
        learned_sim_base = np.load('/home/jm2239/MPM/learned_8_simulate/save_dir_wv/PSNRs_GD_tv_wv_0.01.npy')
    # learned_sim_base = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/metrics/baselines/learned_{rate}_simulate_06tv_06wave_psnr.npy'.format(rate=accelrate))+1
    if accelrate == 4:
        learned_raw_base = np.load('/home/jm2239/MPM/learned_4_raw/save_dir_wv/PSNRs_GD_tv_wv_0.01.npy')
    else:
        learned_raw_base = np.load('/home/jm2239/MPM/learned_8_raw/save_dir_wv/PSNRs_GD_tv_wv_0.01.npy')
    # learned_raw_base = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/metrics/baselines/learned_{rate}_raw_07tv_07wave_psnr.npy'.format(rate=accelrate)) +1 
    learned_sim = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/metrics/learned_{rate}_simulate_psnr.npy'.format(rate=accelrate))
    learned_raw = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/metrics/learned_{rate}_raw_psnr.npy'.format(rate=accelrate))

    vals_simbase.append(random_sim_base)
    vals_rawbase.append(random_raw_base)
    vals_sim.append(random_sim)
    vals_raw.append(random_raw)

    vals_simbase.append(equispaced_sim_base)
    vals_rawbase.append(equispaced_raw_base)
    vals_sim.append(equispaced_sim)
    vals_raw.append(equispaced_raw)

    vals_simbase.append(halfhalf_sim_base)
    vals_rawbase.append(halfhalf_raw_base)
    vals_sim.append(halfhalf_sim)
    vals_raw.append(halfhalf_raw)

    vals_simbase.append(uniform_sim_base)
    vals_rawbase.append(uniform_raw_base)
    vals_sim.append(uniform_sim)
    vals_raw.append(uniform_raw)

    vals_simbase.append(learned_sim_base)
    vals_rawbase.append(learned_raw_base)
    vals_sim.append(learned_sim)
    vals_raw.append(learned_raw)

    labels = ['Random', 'Low-Sequency', 'Half-half', 'Uniform', 'Learned']

    if legend:
        myutils.plot.box_plot(vals_simbase, labels, ax=ax, offset=-0.15+global_offset, facecolor='blue', box_alpha=0.4, show_legend=True)
    else:
        myutils.plot.box_plot(vals_simbase, labels, ax=ax, offset=-0.15+global_offset, facecolor='blue', box_alpha=0.4)
    myutils.plot.box_plot(vals_rawbase, labels, ax=ax, offset=-0.05+global_offset, facecolor='red', box_alpha=0.4)
    myutils.plot.box_plot(vals_sim, labels, ax=ax, offset=0.05+global_offset, facecolor='blue')
    myutils.plot.box_plot(vals_raw, labels, ax=ax, offset=0.15+global_offset, facecolor='red', ylabel='PSNR')
    ax.grid()
    return ax

def plot_slices(accelrate):
    inds = [4,22,28,35]
    fig, axes = plt.subplots(len(inds), 10, figsize=(22.5, 9))
    for i, s in enumerate(inds):

        gt_raw = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/test_data/raw_noisy_four_crop.npy').mean(1)[s,0]

        # random_4_raw_base = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/recons/baselines/random_{rate}_simulate_09tv_09wave.npy'.format(rate=accelrate))[s,0]
        random_4_raw_base = np.load('/home/jm2239/MPM/random_4_raw/save_dir_wv/Recons_GD_tv_wv_0.01.npy')[s]
        random_4_raw = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/recons/random_{rate}_raw.npy'.format(rate=accelrate))[s,0]

        random_4_raw_base_psnr = utils.get_metrics(gt_raw, random_4_raw_base, 'psnr', False)[0]
        random_4_raw_psnr = utils.get_metrics(gt_raw, random_4_raw, 'psnr', False)[0]

        # equispaced_4_raw_base = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/recons/baselines/equispaced_{rate}_simulate_09tv_09wave.npy'.format(rate=accelrate))[s,0]
        equispaced_4_raw_base = np.load('/home/jm2239/MPM/equispaced_4_raw/save_dir_wv/Recons_GD_tv_wv_0.2.npy')[s]
        equispaced_4_raw = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/recons/equispaced_{rate}_raw.npy'.format(rate=accelrate))[s,0]

        equispaced_4_raw_base_psnr = utils.get_metrics(gt_raw, equispaced_4_raw_base, 'psnr', False)[0]
        equispaced_4_raw_psnr = utils.get_metrics(gt_raw, equispaced_4_raw, 'psnr', False)[0]

        halfhalf_4_raw_base = np.load('/home/jm2239/MPM/half_4_raw/save_dir_wv/Recons_GD_tv_wv_0.30000000000000004.npy')[s]
        halfhalf_4_raw = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/recons/halfhalf_{rate}_raw.npy'.format(rate=accelrate))[s,0]

        halfhalf_4_raw_base_psnr = utils.get_metrics(gt_raw, halfhalf_4_raw_base, 'psnr', False)[0]
        halfhalf_4_raw_psnr = utils.get_metrics(gt_raw, halfhalf_4_raw, 'psnr', False)[0]

        uniform_4_raw = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/recons/uniform_{rate}_raw.npy'.format(rate=accelrate))[s,0]
        uniform_4_raw_psnr = utils.get_metrics(gt_raw, uniform_4_raw, 'psnr', False)[0]

        # learned_4_raw_base = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/recons/baselines/learned_{rate}_simulate_08tv_08wave.npy'.format(rate=accelrate))[s,0]
        learned_4_raw_base = np.load('/home/jm2239/MPM/learned_4_raw/save_dir_wv/Recons_GD_tv_wv_0.01.npy')[s]
        learned_4_raw = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/recons/learned_{rate}_raw.npy'.format(rate=accelrate))[s,0]

        learned_4_raw_base_psnr = utils.get_metrics(gt_raw, learned_4_raw_base, 'psnr', False)[0]
        learned_4_raw_psnr = utils.get_metrics(gt_raw, learned_4_raw, 'psnr', False)[0]


        white_text2 = None if i != 0 else 'Ground Truth'
        myutils.plot.plot_img(gt_raw[:100,:100], ax=axes[i, 0], title=white_text2)
        white_text2 = None if i != 0 else 'TV-W, R'
        myutils.plot.plot_img(random_4_raw_base[:100,:100], ax=axes[i, 1], white_text='PSNR='+str(np.round(random_4_raw_base_psnr, 2)), border='blue', title=white_text2)
        white_text2 = None if i != 0 else 'TV-W, LS'
        myutils.plot.plot_img(equispaced_4_raw_base[:100,:100], ax=axes[i, 2], white_text='PSNR='+str(np.round(equispaced_4_raw_base_psnr, 2)), border='blue', title=white_text2)
        white_text2 = None if i != 0 else 'TV-W, HH'
        myutils.plot.plot_img(halfhalf_4_raw_base[:100,:100], ax=axes[i, 3], white_text='PSNR='+str(np.round(halfhalf_4_raw_base_psnr, 2)), border='blue', title=white_text2)
        white_text2 = None if i != 0 else 'TV-W, Learned'
        myutils.plot.plot_img(learned_4_raw_base[:100,:100], ax=axes[i, 4], white_text='PSNR='+str(np.round(learned_4_raw_base_psnr, 2)), border='blue', title=white_text2)

        white_text2 = None if i != 0 else 'U-Net, R'
        myutils.plot.plot_img(random_4_raw[:100,:100], ax=axes[i, 5], white_text='PSNR='+str(np.round(random_4_raw_psnr, 2)), border='red', title=white_text2)
        white_text2 = None if i != 0 else 'U-Net, LS'
        myutils.plot.plot_img(equispaced_4_raw[:100,:100], ax=axes[i, 6], white_text='PSNR='+str(np.round(equispaced_4_raw_psnr, 2)), border='red', title=white_text2)
        white_text2 = None if i != 0 else 'U-Net, HH'
        myutils.plot.plot_img(halfhalf_4_raw[:100,:100], ax=axes[i, 7], white_text='PSNR='+str(np.round(halfhalf_4_raw_psnr, 2)), border='red', title=white_text2)
        white_text2 = None if i != 0 else 'U-Net, U'
        myutils.plot.plot_img(learned_4_raw[:100,:100], ax=axes[i, 8], white_text='PSNR='+str(np.round(uniform_4_raw_psnr, 2)), border='red', title=white_text2)
        white_text2 = None if i != 0 else 'U-Net, Learned'
        myutils.plot.plot_img(learned_4_raw[:100,:100], ax=axes[i, 9], white_text='PSNR='+str(np.round(learned_4_raw_psnr, 2)), border='red', title=white_text2)

    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    # fig.tight_layout()
    return fig

# def plot_slices(accelrate):
#     inds = [10, 169, 130]#, 19]
#     fig, axes = plt.subplots(len(inds), 7, figsize=(21, 9))
#     for i, s in enumerate(inds):

#         gt_sim = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/test_data/test_mix_four_crop.npy'.format(rate=accelrate))[s,0]

#         random_4_sim_base = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/recons/baselines/random_{rate}_simulate_09tv_09wave.npy'.format(rate=accelrate))[s,0]
#         random_4_sim = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/recons/random_{rate}_simulate.npy'.format(rate=accelrate))[s,0]

#         random_4_sim_base_psnr = utils.get_metrics(gt_sim, random_4_sim_base, 'psnr', False)[0]
#         random_4_sim_psnr = utils.get_metrics(gt_sim, random_4_sim, 'psnr', False)[0]

#         equispaced_4_sim_base = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/recons/baselines/equispaced_{rate}_simulate_09tv_09wave.npy'.format(rate=accelrate))[s,0]
#         equispaced_4_sim = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/recons/equispaced_{rate}_simulate.npy'.format(rate=accelrate))[s,0]

#         equispaced_4_sim_base_psnr = utils.get_metrics(gt_sim, equispaced_4_sim_base, 'psnr', False)[0]
#         equispaced_4_sim_psnr = utils.get_metrics(gt_sim, equispaced_4_sim, 'psnr', False)[0]

#         learned_4_sim_base = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/recons/baselines/learned_{rate}_simulate_08tv_08wave.npy'.format(rate=accelrate))[s,0]
#         learned_4_sim = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/recons/learned_{rate}_simulate.npy'.format(rate=accelrate))[s,0]

#         learned_4_sim_base_psnr = utils.get_metrics(gt_sim, learned_4_sim_base, 'psnr', False)[0]
#         learned_4_sim_psnr = utils.get_metrics(gt_sim, learned_4_sim, 'psnr', False)[0]


#         white_text2 = None if i != 0 else 'Ground Truth'
#         myutils.plot.plot_img(gt_sim[:100,:100], ax=axes[i, 0], white_text2=white_text2)
#         white_text2 = None if i != 0 else 'TV, Random'
#         myutils.plot.plot_img(random_4_sim_base[:100,:100], ax=axes[i, 1], white_text='PSNR='+str(np.round(random_4_sim_base_psnr, 2)), border='blue', white_text2=white_text2)
#         white_text2 = None if i != 0 else 'TV, LS'
#         myutils.plot.plot_img(equispaced_4_sim_base[:100,:100], ax=axes[i, 3], white_text='PSNR='+str(np.round(equispaced_4_sim_base_psnr, 2)), border='blue', white_text2=white_text2)
#         white_text2 = None if i != 0 else 'TV, Learned'
#         myutils.plot.plot_img(learned_4_sim_base[:100,:100], ax=axes[i, 2], white_text='PSNR='+str(np.round(learned_4_sim_base_psnr, 2)), border='blue', white_text2=white_text2)

#         white_text2 = None if i != 0 else 'U-Net, Random'
#         myutils.plot.plot_img(random_4_sim[:100,:100], ax=axes[i, 4], white_text='PSNR='+str(np.round(random_4_sim_psnr, 2)), border='red', white_text2=white_text2)
#         white_text2 = None if i != 0 else 'U-Net, LS'
#         myutils.plot.plot_img(equispaced_4_sim[:100,:100], ax=axes[i, 5], white_text='PSNR='+str(np.round(equispaced_4_sim_psnr)), border='red', white_text2=white_text2)
#         white_text2 = None if i != 0 else 'U-Net, Learned'
#         myutils.plot.plot_img(learned_4_sim[:100,:100], ax=axes[i, 6], white_text='PSNR='+str(np.round(learned_4_sim_psnr, 2)), border='red', white_text2=white_text2)

#     plt.subplots_adjust(wspace=0, hspace=0)
#     fig.tight_layout()
#     return fig
