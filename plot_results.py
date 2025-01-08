import pandas as pd

from utils import plot_results

test_rep = pd.read_pickle('results_uncertainty_estimation_ablation.pkl')
test_unc_ps_sw = pd.read_pickle('results_uncertainty_rank_pseudo_labels_sample_weight.pkl')
test_rep_pseudo_sw = pd.read_pickle('BHI/results_uncertainty_criterion_ablation.pkl')
test_unc_rep_pseudo_sw = pd.read_pickle('BHI/results_uncertainty_rank_pseudo_labels_sample_weight_representativeness.pkl')
test_rnd = pd.read_pickle('results_random_selection.pkl')

test_rep = test_rep.values
test_unc_ps_sw = test_unc_ps_sw.values
test_rep_pseudo_sw = test_rep_pseudo_sw.values
test_unc_rep_pseudo_sw = test_unc_rep_pseudo_sw.values
test_rnd = test_rnd.values

plot_results(test_rep_pseudo_sw, test_rnd, 'Uncertainty criterion ablation')

plot_results(test_rep, test_rnd, 'Uncertainty estimation ablation')

plot_results(test_unc_ps_sw, test_rnd, 'Representativeness ablation')

plot_results(test_unc_rep_pseudo_sw, test_rnd, 'Proposed method')

