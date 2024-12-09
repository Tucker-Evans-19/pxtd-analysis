import matplotlib.pyplot as plt
import numpy as np

#pulling in one of the data files
directory = '/Users/tuckerevans/Documents/MIT/HEDP/ptd-analysis/results/csv_output/'
file = 'deconvolution_data_103741_PTD_channel_0_DTn.csv'
files = ['deconvolution_data_103741_PTD_channel_0_DTn.csv', 'deconvolution_data_103741_PTD_channel_1_D3Hep.csv']

general_reactions = ['DTn', 'DDn', 'D3Hep', 'xray']
general_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', 'black']
color_dict = dict(zip(general_reactions, general_colors))

shot_num_min = 103743
shot_num_max = 103746

shot_nums = [103740, 103741, 103743,103744, 103745, 103746, 103747,103748, 103749, 103750, 103751, 109483, 109485, 109486, 109488, 109491, 109492, 109493, 109495, 109496, 109498]
shot_delays = [11, 120, 283, 339, 315, 301, 297, 260, 545, 672, 600, 0, 0, 0, 0,0,0,0,0,0, 0,0,0]
delay_dict = dict(zip(shot_nums, shot_delays))


# MultiIon-22A
shot_nums = [103743, 103744, 103745]
shot_nums = [103739, 103740, 103741]
shot_nums = [103746, 103747, 103748]
shot_nums = [103749, 103750, 103751]

# MultiIon-23A
#shot_nums = [109483, 109485, 109486]
#shot_nums = [109488, 109491, 109492]
#shot_nums = [109493, 109495, 109496]
shot_nums = [109498]


reactions = ['DTn', 'D3Hep']
fig, ax = plt.subplots(len(shot_nums), sharex=True, sharey = True)
for count, shot_num in enumerate(shot_nums):
    for channel in range(3):
        for reaction in reactions:
            try:
                file  = f'deconvolution_data_{shot_num}_PTD_channel_{channel}_{reaction}.csv'
                A = np.genfromtxt(directory + file)
                ax[count].plot(A[:,0] - delay_dict[shot_num], A[:,1], color = color_dict[reaction], label = reaction)
                ax[count].fill_between(A[:,0]- delay_dict[shot_num], A[:,2], A[:,3], alpha = .4, color = color_dict[reaction])
                ax[count].set_title(f'{shot_num}')
            except(FileNotFoundError):
                pass
        ax[channel].legend()
ax[0].set_xlim([1400, 2800])
ax[0].set_ylim([0,.0006])
plt.tight_layout()


plt.show()

