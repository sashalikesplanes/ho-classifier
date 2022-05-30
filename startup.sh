git config --global user.email "a.kiselev@student.tudelft.nl"
git config --global user.name "Alexander Kiselev"

pip install llvmlite==0.38.0 --ignore-installed
pip install tsai -U >> /dev/null
pip install --upgrade sktime==0.9.0
pip3 install --upgrade numpy==1.20
pip install wandb -U
pip install optuna -U