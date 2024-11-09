"""
***** Object-based attention (OBA) project
***** Experiment 01: FBA

    Mo Shams <m.shams.ahmar@gmail.com>
    Nov, 2024

Two superimposed random dot color patches flicker around the center (7.5Hz and
12Hz) while the subject fixates at the center.
One of the two colors is cued in the beginning of each trial and subject is
prompted to detect a tilt (zero to two times in each trial) by pressing a key.

There are two conditions:
    CND1: attend blue patch
    CND2: attend red patch

"""
import os
import random
import numpy as np
import pandas as pd
from lib import stim_flow_control as sfc, gen_events, \
    gen_random_path as gen_path
from psychopy import event, visual, core
from lib.evaluate_responses import eval_resp
from egi_pynetstation.NetStation import NetStation

# from stimulus.archive.exp01_v1 import full_screen

# disable Panda's false warning message
pd.options.mode.chained_assignment = None  # default='warn'

# ----------------------------------------------------------------------------
# /// INSERT SESSION'S META DATA ///

subID = "test"
connect2ECI = False  # True/False
screen_num = 1  # 0: ctrl room    1: test room
monitor_name = 'dell'  # dell/asus/mac
keyboard = "numpad"  # numpad/mac
freq1 = 12
freq2 = 7.5

# ----------------------------------------------------------------------------
# /// CONFIGURATION ///

# create file name
date = sfc.get_date()
time = sfc.get_time()

image_root = os.path.join("image", "cyc03", "FBA")
output_name = f"cyc03_exp01_FBA_{date}_{time}_{subID}.json"

# set data directory
save_path = os.path.join("..", "data", "cyc03", output_name)

# find out the last recorded block number
# todo: make sure if we need the temp.json file and if we need to pause
#  recording after each block
# temp_data = 'temp.json'
# try:
# read from file
# df = pd.read_json(temp_data)
# read file name
# file_name = df.file_name[0]
# update the block number
# iblock = df['last_block_num'][0] + 1
# df['last_block_num'][0] = iblock
# write to file
# df.to_json(temp_data)
# except:
#     iblock = 1
#     create file name
# date = sfc.get_date()
# time = sfc.get_time()
# file_name = f"{subID}_{date}_{time}_exp01_fba_cnt.json"
# create a dictionary of variables to be saved
# trial_dict = {'last_block_num': [iblock],
#               'file_name': [file_name]}
# convert to data frame
# df = pd.DataFrame(trial_dict)
# Note: saving is postponed to the end of the first trial

# ----------------------------------------------------------------------------
# /// CONFIGURE ECI CONNECTION ///
if connect2ECI:
    # Set an IP address for the computer running NetStation as an IPv4 string
    IP_ns = '10.10.10.42'
    # Set a port that NetStation will be listening to as an integer
    port_ns = 55513
    ns = NetStation(IP_ns, port_ns)
    # Set an NTP clock server (the amplifier) address as an IPv4 string
    IP_amp = '10.10.10.51'
    ns.connect(ntp_ip=IP_amp)
    # Begin recording
    ns.begin_rec()
else:
    ns = None

# ----------------------------------------------------------------------------
# /// SET STIMULUS PARAMETERS ///

# todo: add Asus computer
refresh_rate = 60
trial_duration = 7 * refresh_rate  # duration of a trial [frames]

if subID == 'test':
    full_screen = False
win = []
if monitor_name == 'dell':
    mon = sfc.config_mon_dell()
    win_testSize = (1920, 700)
    win = sfc.config_win(mon=mon, fullscr=full_screen,
                         screen=screen_num, win_size=win_testSize)

sfc.test_refresh_rate(win, refresh_rate)

fixmark_radius = .25
fixmark_color = 'white'
fixmark_x = 0
fixmark_y = 0

cue_radius = .5
cue_array_base = [1, 2]
cue_color_base = np.array([[255, 50, 50], [0, 153, 255]])

tilt_array_base = [0, 0, 1]
tilt_duration_frames = int(refresh_rate / 2)

size_factor = 10
image1_size = np.array([size_factor, size_factor])
image2_size = np.array([size_factor, size_factor])
image3_size = np.array([size_factor, size_factor])

# opacity (1: opac | 0: transparent)
# image1_trans = .5  # image1 (blue) is always on top
# image2_trans = .6  # image2 (red) is always behind

# jittering properties
# jitter_repetition = int(refresh_rate / 5)  # number of frames where the relevant
# images keep their positions (equal to 50 ms)

# rel_imgpath_n = trial_duration // jitter_repetition + 1
# rel_imgpath_sigma = .0002
# rel_imgpath_step = .0003

# rel_image_pos0_x = fixmark_x
# rel_image_pos0_y = fixmark_y

gap_durations_base = range(int(.75 * refresh_rate),
                           int(1.25 * refresh_rate) + 1, 1)

if keyboard == "numpad":
    command_keys = {"quit_key": "backspace", "response_key": "num_0"}
elif keyboard == "mac":
    command_keys = {"quit_key": "escape", "response_key": "space"}
else:
    raise NameError(f"Keyboard name '{keyboard}' not recognized.")

timer = core.Clock()

# todo: check the block message
# sfc.block_msg(win, iblock, N_BLOCKS, command_keys)

mouse = event.Mouse(win=win, visible=False)
# todo: calculate the first trial number of the current block
# acc_trial = (iblock - 1) * N_TRIALS
acc_trial = 0

# ----------------------------------------------------------------------------
# /// CONDITIONS ///
ncnds = 2 * 3
# ncnds = 2 cue x 3 tilt (2x w/o tilt and 1x w/ tilt)

cue_array = np.repeat(cue_array_base, 3)
tilt_array = np.tile(tilt_array_base, 2)

rep_per_cnd = 15
cue_array = np.repeat(cue_array, rep_per_cnd)
tilt_array = np.repeat(tilt_array, rep_per_cnd)

ntrials = ncnds * rep_per_cnd
ind_shuffle = np.arange(ntrials)
np.random.shuffle(ind_shuffle)
cue_array = cue_array[ind_shuffle]
tilt_array = tilt_array[ind_shuffle]

assert (cue_array.size == ntrials)
assert (tilt_array.size == ntrials)

# ----------------------------------------------------------------------------
# /// TRIAL BEGINS ///

for itrial in range(ntrials):
    iblue = 1
    ired = 1

    # /// set up the stimulus behavior in current trial
    acc_trial += 1
    print(f"[Trial {acc_trial:03d}]   ", end="")
    if acc_trial > 1:
        # read current running performance
        df_temp = pd.read_json(save_path)
        prev_run_perf = df_temp.loc[acc_trial - 2, 'running_performance']
        prev_tilt_mag = df_temp.loc[acc_trial - 2, 'tilt_magnitude']
    else:
        prev_run_perf = None
        prev_tilt_mag = None

    # randomly select frames, in which change happens
    # todo: make sure the number of events is 2/3 of the times
    change_start_frames = gen_events.gen_events2(refresh_rate)
    n_total_evnts = len(change_start_frames)
    change_frames = np.array(change_start_frames)
    change_times = np.empty((n_total_evnts,))
    change_times[:] = np.nan
    response_times = [np.nan]

    for i in change_start_frames:
        for j in range(tilt_duration_frames - 1):
            change_frames = \
                np.hstack((change_frames, [i + j + 1]))

    iti_dur = random.choice(gap_durations_base)
    postFixGap_dur = random.choice(gap_durations_base)

    irr_image1_nframes = refresh_rate / freq1
    irr_image2_nframes = refresh_rate / freq2

    cue_image = cue_array[itrial]
    tilt_images = np.random.choice([1, 2], n_total_evnts)
    tilt_dirs = np.random.choice(['CW', 'CCW'], n_total_evnts)

    # --------------------------------
    image1_directory = os.path.join(image_root, f"image1.png")
    image2_directory = os.path.join(image_root, f"image2.png")
    image1 = visual.ImageStim(win,
                              image=image1_directory,
                              size=image1_size)
    image2 = visual.ImageStim(win,
                              image=image2_directory,
                              size=image2_size)
    fixmark = visual.Circle(win,
                            radius=fixmark_radius,
                            pos=(fixmark_x, fixmark_y),
                            fillColor=cue_color_base[cue_image - 1] / 255)
    print('===\n')
    print(cue_color_base[cue_image - 1] / 255)
    print('===')
    # --------------------------------
    # # generate the brownian path
    # path1_x = gen_path.brownian_2d(
    #     n_samples=rel_imgpath_n,
    #     distribution_sigma=rel_imgpath_sigma,
    #     max_step=rel_imgpath_step) + rel_image_pos0_x
    # path1_y = gen_path.brownian_2d(
    #     n_samples=rel_imgpath_n,
    #     distribution_sigma=rel_imgpath_sigma,
    #     max_step=rel_imgpath_step) + rel_image_pos0_y
    #
    # path2_x = gen_path.brownian_2d(
    #     n_samples=rel_imgpath_n,
    #     distribution_sigma=rel_imgpath_sigma,
    #     max_step=rel_imgpath_step) + rel_image_pos0_x
    # path2_y = gen_path.brownian_2d(
    #     n_samples=rel_imgpath_n,
    #     distribution_sigma=rel_imgpath_sigma,
    #     max_step=rel_imgpath_step) + rel_image_pos0_y
    #
    # # slow down the jittering speed by reducing the position change rate
    # path1_x = np.repeat(path1_x, jitter_repetition)
    # path1_y = np.repeat(path1_y, jitter_repetition)
    # path2_x = np.repeat(path2_x, jitter_repetition)
    # path2_y = np.repeat(path2_y, jitter_repetition)

    if acc_trial == 1:
        tilt_mag = 50
        tilt_change = 0
    else:
        # calculate what titl angle (magnitude) to use
        tilt_change = sfc.cal_next_tilt(goal_perf=80, run_perf=prev_run_perf)
        tilt_mag = int(prev_tilt_mag + tilt_change)
        # take care of saturated scenarios
        if tilt_mag > 99:
            tilt_mag = 99
        elif tilt_mag < 1:
            tilt_mag = 1
    print(f"TiltAng: {(tilt_mag / 10):3.1f}deg   ", end="")

    # load the changed image
    # image3_directory1cw = os.path.join(image_root,
    #                                    f"blue{iblue}_tilt{tilt_mag}_CW.png")
    # image3_directory1ccw = os.path.join(image_root,
    #                                     f"blue{iblue}_tilt{tilt_mag}_CCW.png")
    # image3_directory2cw = os.path.join(image_root,
    #                                    f"red{ired}_tilt{tilt_mag}_CW.png")
    # image3_directory2ccw = os.path.join(image_root,
    #                                     f"red{ired}_tilt{tilt_mag}_CCW.png")
    #
    # rel_image3_1cw = visual.ImageStim(win,
    #                                   image=image3_directory1cw,
    #                                   size=image3_size,
    #                                   opacity=image1_trans)
    # rel_image3_1ccw = visual.ImageStim(win,
    #                                    image=image3_directory1ccw,
    #                                    size=image3_size,
    #                                    opacity=image1_trans)
    # rel_image3_2cw = visual.ImageStim(win,
    #                                   image=image3_directory2cw,
    #                                   size=image3_size,
    #                                   opacity=image2_trans)
    # rel_image3_2ccw = visual.ImageStim(win,
    #                                    image=image3_directory2ccw,
    #                                    size=image3_size,
    #                                    opacity=image2_trans)

    # --------------------------------
    # /// run the stimulus

    cur_evnt_n = 0

    # inter-trial period
    for iframe in range(iti_dur):
        win.flip()

    # cue period
    for iframe in range(2 * refresh_rate):
        fixmark.draw()
        win.flip()

    # # run gap period
    # for iframe in range(postFixGap_dur):
    #     win.flip()

    if connect2ECI:
        # send a trigger to indicate beginning of each trial
        ns.send_event(event_type=f"CUE{cue_image}",
                      label=f"CUE{cue_image}")

    # todo: create a photo stimuli for trial begin

    for iframe in range(trial_duration):
        image1.ori = 0
        image2.ori = 0
        pressed_key = event.getKeys(keyList=list(command_keys.values()))
        # set the position of each task-relevant image
        # image1.pos = (path1_x[iframe], path1_y[iframe])
        # image2.pos = (path2_x[iframe], path2_y[iframe])

        # get the time of change
        if iframe in change_start_frames:
            ch_t = timer.getTime()
            change_times[cur_evnt_n] = round(ch_t * 1000)
            cur_evnt_n += 1

        # if conditions satisfied tilt the image
        if iframe in change_frames:
            if tilt_dirs[cur_evnt_n - 1] == 'CW':
                if tilt_images[cur_evnt_n - 1] == 1:
                    image1.ori = -tilt_mag / 10
                    # rel_image3_1cw.pos = (
                    #     path1_x[iframe], path1_y[iframe])
                    if sfc.decide_on_show(iframe, irr_image2_nframes):
                        image2.draw()
                    if sfc.decide_on_show(iframe, irr_image1_nframes):
                        image1.draw()
                        # rel_image3_1cw.draw()
                elif tilt_images[cur_evnt_n - 1] == 2:
                    image2.ori = -tilt_mag / 10
                    # rel_image3_2cw.pos = (
                    #     path2_x[iframe], path2_y[iframe])
                    if sfc.decide_on_show(iframe, irr_image2_nframes):
                        image2.draw()
                        # rel_image3_2cw.draw()
                    if sfc.decide_on_show(iframe, irr_image1_nframes):
                        image1.draw()
            else:
                if tilt_images[cur_evnt_n - 1] == 1:
                    image1.ori = tilt_mag / 10
                    # rel_image3_1ccw.pos = (
                    #     path1_x[iframe], path1_y[iframe])
                    if sfc.decide_on_show(iframe, irr_image2_nframes):
                        image2.draw()
                    if sfc.decide_on_show(iframe, irr_image1_nframes):
                        image1.draw()
                        # rel_image3_1ccw.draw()
                elif tilt_images[cur_evnt_n - 1] == 2:
                    image2.ori = tilt_mag / 10
                    # rel_image3_2ccw.pos = (
                    #     path2_x[iframe], path2_y[iframe])
                    if sfc.decide_on_show(iframe, irr_image2_nframes):
                        image2.draw()
                        # rel_image3_2ccw.draw()
                    if sfc.decide_on_show(iframe, irr_image1_nframes):
                        image1.draw()
        # if not, show the unchanged versions
        else:
            if sfc.decide_on_show(iframe, irr_image2_nframes):
                image2.draw()
            if sfc.decide_on_show(iframe, irr_image1_nframes):
                image1.draw()

        fixmark.draw()
        win.flip()

        # response period
        if command_keys['quit_key'] in pressed_key:
            core.quit()
        # check if space bar is pressed within 1 sec from tilt
        if command_keys['response_key'] in pressed_key:
            res_t = timer.getTime()
            response_times.append(round(res_t * 1000))
    response_times.pop(0)

    # evaluate the response
    [instant_perf, avg_rt] = eval_resp(cue_image,
                                       tilt_images,
                                       change_times,
                                       response_times)
    if np.isnan(avg_rt):
        print(f"Perf:{int(instant_perf):3d}%   avgRT:  nan    ", end="")
    else:
        print(f"Perf:{int(instant_perf):3d}%   avgRT:{int(avg_rt):4d}ms   ",
              end="")

    # --------------------------------
    # /// save trial parameters

    if subID != 'test':
        trial_dict = {
            'trial_num': [acc_trial],
            'Frequency_tags': [freq1, freq2],
            'cued_image': [cue_image],
            'n_events': n_total_evnts,
            'tilted_images': [tilt_images],
            'tilt_directions': [tilt_dirs],
            'tilt_magnitude': [tilt_mag],
            'avg_rt': [avg_rt],
            'instant_performance': [instant_perf],
            'cummulative_performance': [np.nan],
            'running_performance': [np.nan]
        }

        dfnew = pd.DataFrame(trial_dict)
        if itrial > 0:
            df = pd.read_json(save_path)
            dfnew = pd.concat([df, dfnew], ignore_index=True)
        dfnew.to_json(save_path)

        if itrial == ntrials - 1:
            sfc.end_screen(win, color='white')

    # --------------------------------
    # /// calculate cummulative and running performances

    if itrial > 0:
        # calculate the cumulative performance (all recorded trials)
        eval_series = dfnew.instant_performance
        eval_array = eval_series.values
        cum_perf = round(sum(eval_array) / len(eval_array), 2)
        print(f"CumPerf:{cum_perf:6.2f}%   ", end="")
        # calculate the running performance over last 10 trials
        run_perf = round(sum(eval_array[-10:]) / len(eval_array[-10:]), 2)
        print(f"RunPerf:{run_perf:6.2f}%")
        # fill the remaining values in the data frame
        dfnew.loc[acc_trial - 1,
        ['cummulative_performance', 'running_performance']] = \
            [cum_perf, run_perf]
        dfnew.to_json(save_path)

    for iframe in np.arange(refresh_rate / 2):
        win.flip()

# --------------------------------
if connect2ECI:
    ns.disconnect()
    print(f"\n    *** Recording finished ***")

win.close()
