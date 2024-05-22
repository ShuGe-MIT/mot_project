# section 4.2
python MOT_models.py --start_epsilon=1 --target_epsilon=1e-3 --iter_gap=1 --epsilon_scale_num=0.99 --epsilon_scale_gap=5 --data_file=multifactorEpitaxialgrowth --max_iter=5000

python MOT_models.py --start_epsilon=1 --target_epsilon=1e-3 --iter_gap=1 --epsilon_scale_num=0.99 --epsilon_scale_gap=5 --data_file=multidimEducationCovariance --max_iter=5000

python MOT_models.py --start_epsilon=1 --target_epsilon=1e-3 --iter_gap=1 --epsilon_scale_num=0.99 --epsilon_scale_gap=5 --data_file=multidimHelpfulness --max_iter=5000

# section 4.3
python MOT_models.py --start_epsilon=1 --target_epsilon=1e-3 --iter_gap=1 --epsilon_scale_num=0.99 --epsilon_scale_gap=5 --data_file=multidimEducationCovariance --max_iter=5000 --cost_type=cov

python MOT_models.py --start_epsilon=1 --target_epsilon=1e-3 --iter_gap=1 --epsilon_scale_num=0.99 --epsilon_scale_gap=5 --data_file=multidimInvEducationCovariance --max_iter=5000 --cost_type=cov

## iterate over margins
python MOT_models.py --start_epsilon=1 --target_epsilon=1e-3 --iter_gap=1 --epsilon_scale_num=0.99 --epsilon_scale_gap=5 --data_file=multidimInvEducationCovariance-GPA_year1-GPA_year2 --max_iter=5000 --cost_type=cov

python MOT_models.py --start_epsilon=1 --target_epsilon=1e-3 --iter_gap=1 --epsilon_scale_num=0.99 --epsilon_scale_gap=5 --data_file=multidimInvEducationCovariance-GPA_year1-grade_20059_fall --max_iter=5000 --cost_type=cov

python MOT_models.py --start_epsilon=1 --target_epsilon=1e-3 --iter_gap=1 --epsilon_scale_num=0.99 --epsilon_scale_gap=5 --data_file=multidimInvEducationCovariance-GPA_year1-goodstanding_year1 --max_iter=5000 --cost_type=cov

python MOT_models.py --start_epsilon=1 --target_epsilon=1e-3 --iter_gap=1 --epsilon_scale_num=0.99 --epsilon_scale_gap=5 --data_file=multidimInvEducationCovariance-GPA_year1-goodstanding_year2 --max_iter=5000 --cost_type=cov

python MOT_models.py --start_epsilon=1 --target_epsilon=1e-3 --iter_gap=1 --epsilon_scale_num=0.99 --epsilon_scale_gap=5 --data_file=multidimInvEducationCovariance-GPA_year2-grade_20059_fall --max_iter=5000 --cost_type=cov

python MOT_models.py --start_epsilon=1 --target_epsilon=1e-3 --iter_gap=1 --epsilon_scale_num=0.99 --epsilon_scale_gap=5 --data_file=multidimInvEducationCovariance-GPA_year2-goodstanding_year1 --max_iter=5000 --cost_type=cov

python MOT_models.py --start_epsilon=1 --target_epsilon=1e-3 --iter_gap=1 --epsilon_scale_num=0.99 --epsilon_scale_gap=5 --data_file=multidimInvEducationCovariance-GPA_year2-goodstanding_year2 --max_iter=5000 --cost_type=cov

python MOT_models.py --start_epsilon=1 --target_epsilon=1e-3 --iter_gap=1 --epsilon_scale_num=0.99 --epsilon_scale_gap=5 --data_file=multidimInvEducationCovariance-grade_20059_fall-goodstanding_year1 --max_iter=5000 --cost_type=cov

python MOT_models.py --start_epsilon=1 --target_epsilon=1e-3 --iter_gap=1 --epsilon_scale_num=0.99 --epsilon_scale_gap=5 --data_file=multidimInvEducationCovariance-grade_20059_fall-goodstanding_year2 --max_iter=5000 --cost_type=cov

python MOT_models.py --start_epsilon=1 --target_epsilon=1e-3 --iter_gap=1 --epsilon_scale_num=0.99 --epsilon_scale_gap=5 --data_file=multidimInvEducationCovariance-goodstanding_year1-goodstanding_year2 --max_iter=5000 --cost_type=cov

python MOT_models.py --start_epsilon=1 --target_epsilon=1e-3 --iter_gap=1 --epsilon_scale_num=0.99 --epsilon_scale_gap=5 --data_file=multidimEducationCovariance-GPA_year1-GPA_year2 --max_iter=5000 --cost_type=cov

python MOT_models.py --start_epsilon=1 --target_epsilon=1e-3 --iter_gap=1 --epsilon_scale_num=0.99 --epsilon_scale_gap=5 --data_file=multidimEducationCovariance-GPA_year1-grade_20059_fall --max_iter=5000 --cost_type=cov

python MOT_models.py --start_epsilon=1 --target_epsilon=1e-3 --iter_gap=1 --epsilon_scale_num=0.99 --epsilon_scale_gap=5 --data_file=multidimEducationCovariance-GPA_year1-goodstanding_year1 --max_iter=5000 --cost_type=cov

python MOT_models.py --start_epsilon=1 --target_epsilon=1e-3 --iter_gap=1 --epsilon_scale_num=0.99 --epsilon_scale_gap=5 --data_file=multidimEducationCovariance-GPA_year1-goodstanding_year2 --max_iter=5000 --cost_type=cov

python MOT_models.py --start_epsilon=1 --target_epsilon=1e-3 --iter_gap=1 --epsilon_scale_num=0.99 --epsilon_scale_gap=5 --data_file=multidimEducationCovariance-GPA_year2-grade_20059_fall --max_iter=5000 --cost_type=cov

python MOT_models.py --start_epsilon=1 --target_epsilon=1e-3 --iter_gap=1 --epsilon_scale_num=0.99 --epsilon_scale_gap=5 --data_file=multidimEducationCovariance-GPA_year2-goodstanding_year1 --max_iter=5000 --cost_type=cov

python MOT_models.py --start_epsilon=1 --target_epsilon=1e-3 --iter_gap=1 --epsilon_scale_num=0.99 --epsilon_scale_gap=5 --data_file=multidimEducationCovariance-GPA_year2-goodstanding_year2 --max_iter=5000 --cost_type=cov

python MOT_models.py --start_epsilon=1 --target_epsilon=1e-3 --iter_gap=1 --epsilon_scale_num=0.99 --epsilon_scale_gap=5 --data_file=multidimEducationCovariance-grade_20059_fall-goodstanding_year1 --max_iter=5000 --cost_type=cov

python MOT_models.py --start_epsilon=1 --target_epsilon=1e-3 --iter_gap=1 --epsilon_scale_num=0.99 --epsilon_scale_gap=5 --data_file=multidimEducationCovariance-grade_20059_fall-goodstanding_year2 --max_iter=5000 --cost_type=cov

python MOT_models.py --start_epsilon=1 --target_epsilon=1e-3 --iter_gap=1 --epsilon_scale_num=0.99 --epsilon_scale_gap=5 --data_file=multidimEducationCovariance-goodstanding_year1-goodstanding_year2 --max_iter=5000 --cost_type=cov

# section 4.4
python MOT_models.py --start_epsilon=1 --target_epsilon=1e-3 --iter_gap=1 --epsilon_scale_num=0.99 --epsilon_scale_gap=5 --data_file=finitesampleEpitaxialgrowth --max_iter=5000

