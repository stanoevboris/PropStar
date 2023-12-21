python3 benchmark.py --learner random_forest_learner --n_estimators 16 --representation_type sklearn_tfidf &&
python3 benchmark.py --learner random_forest_learner --n_estimators 32 --representation_type sklearn_tfidf &&
python3 benchmark.py --learner extra_tree_learner --n_estimators 16 --representation_type sklearn_tfidf &&
python3 benchmark.py --learner extra_tree_learner --n_estimators 32 --representation_type sklearn_tfidf &&
python3 benchmark.py --learner ada_boost_learner --n_estimators 16 --representation_type sklearn_tfidf &&
python3 benchmark.py --learner ada_boost_learner --n_estimators 32 --representation_type sklearn_tfidf &&
python3 benchmark.py --learner gradient_boost_learner --n_estimators 16 --learning_rate 0.8 --representation_type sklearn_tfidf &&
python3 benchmark.py --learner gradient_boost_learner --n_estimators 16 --learning_rate 1.0 --representation_type sklearn_tfidf &&
python3 benchmark.py --learner gradient_boost_learner --n_estimators 32 --learning_rate 0.8 --representation_type sklearn_tfidf &&
python3 benchmark.py --learner gradient_boost_learner --n_estimators 32 --learning_rate 1.0 --representation_type sklearn_tfidf &&
python3 benchmark.py --learner DRM  --n_estimators 32 --learning_rate 0.001  --representation_type sklearn_tfidf --hidden_size 16 &&
python3 benchmark.py --learner DRM  --n_estimators 32 --learning_rate 0.001  --representation_type sklearn_tfidf --hidden_size 32 &&
python3 benchmark.py --learner DRM  --n_estimators 32 --learning_rate 0.01  --representation_type sklearn_tfidf --hidden_size 16 &&
python3 benchmark.py --learner DRM  --n_estimators 32 --learning_rate 0.01  --representation_type sklearn_tfidf --hidden_size 32 &&
python3 benchmark.py --learner starspace  --n_estimators 32 --learning_rate 0.01  --representation_type sklearn_tfidf --hidden_size 32 --negative_samples_limit 5 &&
python3 benchmark.py --learner starspace  --n_estimators 32 --learning_rate 0.01  --representation_type sklearn_tfidf --hidden_size 64 --negative_samples_limit 5