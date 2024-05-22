# section 4.2
python MOT_models.py --start_epsilon=1 --target_epsilon=1e-3 --iter_gap=1 --epsilon_scale_num=0.99 --epsilon_scale_gap=5 --data_file=multifactorEpitaxialgrowth --max_iter=5000

python MOT_models.py --start_epsilon=1 --target_epsilon=1e-3 --iter_gap=1 --epsilon_scale_num=0.99 --epsilon_scale_gap=5 --data_file=multidimEducationCovariance --max_iter=5000

python MOT_models.py --start_epsilon=1 --target_epsilon=1e-3 --iter_gap=1 --epsilon_scale_num=0.99 --epsilon_scale_gap=5 --data_file=multidimHelpfulness --max_iter=5000

# section 4.3
python MOT_models.py --start_epsilon=1 --target_epsilon=1e-3 --iter_gap=1 --epsilon_scale_num=0.99 --epsilon_scale_gap=5 --data_file=multidimEducationCovariance --max_iter=5000 --cost_type=cov

python MOT_models.py --start_epsilon=1 --target_epsilon=1e-3 --iter_gap=1 --epsilon_scale_num=0.99 --epsilon_scale_gap=5 --data_file=multidimInvEducationCovariance --max_iter=5000 --cost_type=cov

# section 4.4
python MOT_models.py --start_epsilon=1 --target_epsilon=1e-3 --iter_gap=1 --epsilon_scale_num=0.99 --epsilon_scale_gap=5 --data_file=finitesampleEpitaxialgrowth --max_iter=5000

