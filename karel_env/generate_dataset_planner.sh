# Generate karel datasets
python karel_env/generator.py --min_demo_length=2 --use_planner
# Append unseen demonstrations to each programs
python karel_env/append_demonstration.py --min_demo_length=2 --use_planner
# Add perception primitives to each demonstrations
python karel_env/add_per.py
