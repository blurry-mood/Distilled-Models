pip install --quiet jupyterlab jupyterlab-git
echo "==> Jupyterlab installed successfully."

pip install Pillow --quiet
echo "==> PIL installed successfully."

pip install albumentations --quiet
echo "==> Albumentations installed successfully."

pip install timm --quiet
echo "==> Timm installed successfully."

pip install pytorch-lightning --quiet
echo "==> Pytorch-Lightning installed successfully."

pip install tqdm --quiet
conda install -c conda-forge ipywidgets
conda install -c conda-forge nodejs
jupyter nbextension enable --py widgetsnbextension
echo "==> Tqdm installed successfully."

pip install wandb --quiet
echo "==> WandB installed successfully."
