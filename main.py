# Strategy 1:
# Show ML-Synchorny-Curve vs. ANOVA-maxSynchrony 
# 0) plot spike trains 
# 1) Split recordings into windows
# 2) Calculate features (minFR=6 1/min): 
# Synchrony-Curve 
# 3) Combine features into one table, also add div as a feature
# 4) ML

import os
import warnings

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    #exec(open(os.path.join(os.getcwd(), "script_0_1_plot_spiketrains.py")).read())
    #exec(open(os.path.join(os.getcwd(), "script_1_0_split_data.py")).read())
    exec(open(os.path.join(os.getcwd(), "script_2_0_calc_synchrony_curve.py")).read())
    exec(open(os.path.join(os.getcwd(), "script_2_1_calc_synchrony_stats.py")).read())
    exec(open(os.path.join(os.getcwd(), "script_2_2_calc_bursts.py")).read())
    exec(open(os.path.join(os.getcwd(), "script_3_0_make_feature_set.py")).read())
    exec(open(os.path.join(os.getcwd(), "script_3_1_preprocess_feature_set.py")).read())
    exec(open(os.path.join(os.getcwd(), "script_4_0_machine_learning.py")).read())
    exec(open(os.path.join(os.getcwd(), "script_4_1_find_best_result.py")).read())
    exec(open(os.path.join(os.getcwd(), "script_5_0_shap.py")).read())
    exec(open(os.path.join(os.getcwd(), "script_6_0_statistics.py")).read())
    

    print("Finished all scripts.")