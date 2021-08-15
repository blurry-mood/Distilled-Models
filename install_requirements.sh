pip install --upgrade --quiet jupyterlab jupyterlab-git
echo "==> Jupyterlab installed successfully."

pip install torchvision torchtext torch --upgrade --quiet
echo "==> Pytorch installed successfully."

pip install Pillow --upgrade --quiet
echo "==> PIL installed successfully."

pip install albumentations --upgrade --quiet
echo "==> Albumentations installed successfully."

pip install timm --upgrade --quiet
echo "==> Timm installed successfully."

pip install pytorch-lightning --upgrade --quiet
echo "==> Pytorch-Lightning installed successfully."

pip install torchmetrics --upgrade --quiet
echo "==> Torchmetrics installed successfully."

pip install tqdm --upgrade --quiet
conda install -c conda-forge ipywidgets
jupyter nbextension enable --py widgetsnbextension
echo "==> Tqdm installed successfully."
