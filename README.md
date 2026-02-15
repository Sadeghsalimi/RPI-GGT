
# Heterogeneous GNN for lncRNA-Protein Interaction Prediction

This repository contains a deep learning framework designed to predict interactions between long non-coding RNAs (lncRNAs) and Proteins. It utilizes a Heterogeneous Graph Neural Network (GNN) combined with Convolutional Neural Networks (CNNs) to process multi-modal features, including sequence k-mers, binding motifs, protein 3D structures, and distance maps.


## 🛠️ Installation & Dependencies

This project relies on **PyTorch** and **PyTorch Geometric**.

Dependencies:

"
  - _openmp_mutex=4.5=2_gnu
  - _py-xgboost-mutex=2.0=cpu_0
  - aiohappyeyeballs=2.4.4=pyhd8ed1ab_1
  - aiohttp=3.11.11=py39hf73967f_0
  - aiosignal=1.3.2=pyhd8ed1ab_0
  - asttokens=3.0.0=pyhd8ed1ab_1
  - async-timeout=5.0.1=pyhd8ed1ab_1
  - attrs=24.3.0=pyh71513ae_0
  - aws-c-auth=0.8.0=h2219d47_15
  - aws-c-cal=0.8.1=h099ea23_3
  - aws-c-common=0.10.6=h2466b09_0
  - aws-c-compression=0.3.0=h099ea23_5
  - aws-c-event-stream=0.5.0=h85d8506_11
  - aws-c-http=0.9.2=h3888f84_4
  - aws-c-io=0.15.3=hc5a9e45_5
  - aws-c-mqtt=0.11.0=h2c94728_12
  - aws-c-s3=0.7.7=h6a38c86_0
  - aws-c-sdkutils=0.2.1=h099ea23_4
  - aws-checksums=0.2.2=h099ea23_4
  - aws-crt-cpp=0.29.7=h0642867_7
  - aws-sdk-cpp=1.11.458=h5f5f9c4_4
  - backcall=0.2.0=pyh9f0ad1d_0
  - biopython=1.79=py39hb82d6ee_2
  - blas=1.0=mkl
  - brotli=1.1.0=h2466b09_2
  - brotli-bin=1.1.0=h2466b09_2
  - brotli-python=1.1.0=py39ha51f57c_2
  - bzip2=1.0.8=h2466b09_7
  - c-ares=1.34.4=h2466b09_0
  - ca-certificates=2024.12.14=h56e8100_0
  - cccl=2.5.0=h49adc43_0
  - certifi=2024.12.14=pyhd8ed1ab_0
  - cffi=1.17.1=py39ha55e580_0
  - charset-normalizer=3.4.1=pyhd8ed1ab_0
  - colorama=0.4.6=pyhd8ed1ab_1
  - comm=0.2.2=pyhd8ed1ab_1
  - contourpy=1.2.0=py39h59b6b97_0
  - cuda=12.1.0=0
  - cuda-cccl=12.6.77=h57928b3_0
  - cuda-cccl_win-64=12.6.77=h57928b3_0
  - cuda-command-line-tools=12.1.0=0
  - cuda-compiler=12.1.0=0
  - cuda-cudart=12.1.55=0
  - cuda-cudart-dev=12.1.55=0
  - cuda-cuobjdump=12.6.77=he0c23c2_1
  - cuda-cupti=12.1.62=0
  - cuda-cuxxfilt=12.6.77=he0c23c2_1
  - cuda-demo-suite=12.1.55=0
  - cuda-documentation=12.1.55=0
  - cuda-libraries=12.1.0=0
  - cuda-libraries-dev=12.1.0=0
  - cuda-nsight-compute=12.1.0=0
  - cuda-nvcc=12.1.66=0
  - cuda-nvdisasm=12.6.77=he0c23c2_1
  - cuda-nvml-dev=12.6.77=he0c23c2_1
  - cuda-nvprof=12.6.80=he0c23c2_0
  - cuda-nvprune=12.6.77=he0c23c2_1
  - cuda-nvrtc=12.1.55=0
  - cuda-nvrtc-dev=12.1.55=0
  - cuda-nvtx=12.1.66=0
  - cuda-nvvp=12.6.80=he0c23c2_1
  - cuda-opencl=12.6.77=he0c23c2_0
  - cuda-opencl-dev=12.6.77=he0c23c2_0
  - cuda-profiler-api=12.6.77=h57928b3_0
  - cuda-runtime=12.1.0=0
  - cuda-sanitizer-api=12.6.77=he0c23c2_1
  - cuda-toolkit=12.1.0=0
  - cuda-tools=12.1.0=0
  - cuda-version=12.6=h7480c83_3
  - cuda-visual-tools=12.1.0=0
  - cycler=0.12.1=pyhd8ed1ab_1
  - datasets=3.2.0=pyhd8ed1ab_0
  - debugpy=1.8.11=py39ha51f57c_0
  - decorator=5.1.1=pyhd8ed1ab_1
  - dill=0.3.8=pyhd8ed1ab_0
  - dssp=2.0.4=0
  - et_xmlfile=2.0.0=pyhd8ed1ab_1
  - exceptiongroup=1.2.2=pyhd8ed1ab_1
  - executing=2.1.0=pyhd8ed1ab_1
  - filelock=3.16.1=pyhd8ed1ab_1
  - font-ttf-dejavu-sans-mono=2.37=hab24e00_0
  - font-ttf-inconsolata=3.000=h77eed37_0
  - font-ttf-source-code-pro=2.038=h77eed37_0
  - font-ttf-ubuntu=0.83=h77eed37_3
  - fontconfig=2.15.0=h765892d_1
  - fonts-conda-ecosystem=1=0
  - fonts-conda-forge=1=0
  - fonttools=4.55.3=py39hf73967f_1
  - freetype=2.12.1=hdaf720e_2
  - frozenlist=1.5.0=py39ha55e580_0
  - fsspec=2024.3.1=pyhca7485f_0
  - gensim=4.3.3=py39h0a2e257_0
  - glib=2.82.2=h7025463_0
  - glib-tools=2.82.2=h4394cf3_0
  - gst-plugins-base=1.24.7=hb0a98b8_0
  - gstreamer=1.24.7=h5006eae_0
  - h2=4.1.0=pyhd8ed1ab_1
  - hpack=4.0.0=pyhd8ed1ab_1
  - huggingface_hub=0.26.5=pyhd8ed1ab_1
  - hyperframe=6.0.1=pyhd8ed1ab_1
  - icu=75.1=he0c23c2_0
  - idna=3.10=pyhd8ed1ab_1
  - importlib-metadata=8.5.0=pyha770c72_1
  - importlib-resources=6.4.5=pyhd8ed1ab_1
  - importlib_resources=6.4.5=pyhd8ed1ab_1
  - intel-openmp=2025.0.0=h57928b3_1164
  - ipykernel=6.25.0=py39h9909e9c_0
  - ipython=8.15.0=py39haa95532_0
  - jedi=0.19.2=pyhd8ed1ab_1
  - jinja2=3.1.5=pyhd8ed1ab_0
  - joblib=1.4.2=pyhd8ed1ab_1
  - jupyter_client=8.6.3=pyhd8ed1ab_1
  - jupyter_core=5.7.2=py39hcbf5309_0
  - khronos-opencl-icd-loader=2024.10.24=h2466b09_1
  - kiwisolver=1.4.7=py39h2b77a98_0
  - krb5=1.21.3=hdf4eb48_0
  - lcms2=2.16=h67d730c_0
  - lerc=4.0.0=h63175ca_0
  - libabseil=20240722.0=cxx17_h4eb7d71_4
  - libarrow=18.1.0=ha929de4_7_cuda
  - libarrow-acero=18.1.0=h7d8d6a5_7_cuda
  - libarrow-dataset=18.1.0=h7d8d6a5_7_cuda
  - libarrow-substrait=18.1.0=h3dbecdf_7_cuda
  - libblas=3.9.0=12_win64_mkl
  - libbrotlicommon=1.1.0=h2466b09_2
  - libbrotlidec=1.1.0=h2466b09_2
  - libbrotlienc=1.1.0=h2466b09_2
  - libcblas=3.9.0=12_win64_mkl
  - libclang13=19.1.6=default_ha5278ca_0
  - libcrc32c=1.1.2=h0e60522_0
  - libcublas=12.1.0.26=0
  - libcublas-dev=12.1.0.26=0
  - libcufft=11.0.2.4=0
  - libcufft-dev=11.0.2.4=0
  - libcurand=10.3.7.77=he0c23c2_0
  - libcurand-dev=10.3.7.77=he0c23c2_0
  - libcurl=8.11.1=h88aaa65_0
  - libcusolver=11.4.4.55=0
  - libcusolver-dev=11.4.4.55=0
  - libcusparse=12.0.2.55=0
  - libcusparse-dev=12.0.2.55=0
  - libdeflate=1.23=h9062f6e_0
  - libevent=2.1.12=h3671451_1
  - libexpat=2.6.4=he0c23c2_0
  - libffi=3.4.2=h8ffe710_5
  - libgcc=14.2.0=h1383e82_1
  - libglib=2.82.2=h7025463_0
  - libgomp=14.2.0=h1383e82_1
  - libgoogle-cloud=2.33.0=h95c5cb2_1
  - libgoogle-cloud-storage=2.33.0=he5eb982_1
  - libgrpc=1.67.1=h0ac93cb_1
  - libiconv=1.17=hcfcfb64_2
  - libintl=0.22.5=h5728263_3
  - libintl-devel=0.22.5=h5728263_3
  - libjpeg-turbo=3.0.0=hcfcfb64_1
  - liblapack=3.9.0=12_win64_mkl
  - liblzma=5.6.3=h2466b09_1
  - libnpp=12.0.2.50=0
  - libnpp-dev=12.0.2.50=0
  - libnvjitlink=12.1.105=0
  - libnvjitlink-dev=12.1.55=0
  - libnvjpeg=12.1.0.39=0
  - libnvjpeg-dev=12.1.0.39=0
  - libnvvm-samples=12.1.55=0
  - libogg=1.3.5=h2466b09_0
  - libparquet=18.1.0=ha850022_7_cuda
  - libpng=1.6.44=h3ca93ac_0
  - libprotobuf=5.28.3=h8309712_1
  - libre2-11=2024.07.02=h4eb7d71_2
  - libsodium=1.0.20=hc70643c_0
  - libsqlite=3.47.2=h67fdade_0
  - libssh2=1.11.1=he619c9f_0
  - libthrift=0.21.0=hbe90ef8_0
  - libtiff=4.7.0=h797046b_3
  - libutf8proc=2.9.0=h2466b09_1
  - libuv=1.49.2=h2466b09_0
  - libvorbis=1.3.7=h0e60522_0
  - libwebp-base=1.5.0=h3b0e114_0
  - libwinpthread=12.0.0.r4.gg4f2fc60ca=h57928b3_8
  - libxcb=1.17.0=h0e4246c_0
  - libxgboost=2.0.3=cuda120_he04f013_4
  - libzlib=1.3.1=h2466b09_2
  - lz4-c=1.10.0=h2466b09_1
  - markupsafe=3.0.2=py39hf73967f_1
  - matplotlib=3.8.4=py39hcbf5309_0
  - matplotlib-base=3.8.4=py39he1095e7_2
  - matplotlib-inline=0.1.7=pyhd8ed1ab_1
  - mkl=2021.4.0=h0e2418a_729
  - mpmath=1.3.0=pyhd8ed1ab_1
  - multidict=6.1.0=py39hf73967f_1
  - multiprocess=0.70.14=py39ha55989b_3
  - munkres=1.1.4=pyh9f0ad1d_0
  - nest-asyncio=1.6.0=pyhd8ed1ab_1
  - networkx=3.2.1=pyhd8ed1ab_0
  - nsight-compute=2024.3.2.3=h5173278_0
  - numpy=1.26.4=py39hddb5d58_0
  - obonet=0.2.3=py_0
  - opencl-headers=2024.10.24=he0c23c2_0
  - openjpeg=2.5.3=h4d64b90_0
  - openpyxl=3.0.10=py39h2bbff1b_0
  - openssl=3.4.0=h2466b09_0
  - orc=2.0.3=haf104fe_2
  - packaging=24.2=pyhd8ed1ab_2
  - pandas=2.2.3=py39h2366fc2_1
  - parso=0.8.4=pyhd8ed1ab_1
  - pcre2=10.44=h3d7b363_2
  - pickleshare=0.7.5=pyhd8ed1ab_1004
  - pillow=11.1.0=py39h73ef694_0
  - pip=24.3.1=pyh8b19718_2
  - platformdirs=4.3.6=pyhd8ed1ab_1
  - ply=3.11=pyhd8ed1ab_3
  - prompt-toolkit=3.0.48=pyha770c72_1
  - propcache=0.2.1=py39ha55e580_0
  - psutil=6.1.1=py39ha55e580_0
  - pthread-stubs=0.4=h0e40799_1002
  - pure_eval=0.2.3=pyhd8ed1ab_1
  - py-xgboost=2.0.3=cpu_pyh995e691_4
  - pyarrow=18.1.0=py39hcbf5309_0
  - pyarrow-core=18.1.0=py39h0b3d880_0_cuda
  - pycparser=2.22=pyh29332c3_1
  - pygments=2.18.0=pyhd8ed1ab_1
  - pyparsing=3.2.1=pyhd8ed1ab_0
  - pyqt=5.15.9=py39hb77abff_5
  - pyqt5-sip=12.12.2=py39h99910a6_5
  - pysocks=1.7.1=pyh09c184e_7
  - python=3.9.21=h8205438_1
  - python-dateutil=2.9.0.post0=pyhff2d567_1
  - python-tzdata=2024.2=pyhd8ed1ab_1
  - python-xxhash=3.5.0=py39ha55e580_1
  - python_abi=3.9=2_cp39
  - pytorch=2.3.0=py3.9_cuda12.1_cudnn8_0
  - pytorch-cuda=12.1=hde6ce7c_5
  - pytorch-mutex=1.0=cuda
  - pytz=2024.1=pyhd8ed1ab_0
  - pywin32=307=py39ha51f57c_3
  - pyyaml=6.0.2=py39ha55e580_1
  - pyzmq=26.2.0=py39h03e5c00_3
  - qt-main=5.15.15=h9151539_2
  - re2=2024.07.02=haf4117d_2
  - regex=2024.11.6=py39ha55e580_0
  - requests=2.32.3=pyhd8ed1ab_1
  - safetensors=0.5.0=py39h92a245a_0
  - scikit-learn=1.6.0=py39hdd013cc_0
  - scipy=1.13.1=py39h1a10956_0
  - seaborn=0.12.2=py39haa95532_0
  - sentence-transformers=3.3.1=pyhd8ed1ab_1
  - setuptools=75.6.0=pyhff2d567_1
  - sip=6.7.12=py39h99910a6_0
  - six=1.17.0=pyhd8ed1ab_0
  - smart_open=7.1.0=pyhd8ed1ab_0
  - snappy=1.2.1=h500f7fa_1
  - sqlite=3.47.2=h2466b09_0
  - stack_data=0.6.3=pyhd8ed1ab_1
  - sympy=1.13.3=pyh04b8f61_4
  - texttable=1.6.4=pyhd3eb1b0_0
  - threadpoolctl=3.5.0=pyhc1e730c_0
  - tk=8.6.13=h5226925_1
  - tokenizers=0.21.0=py39hb5dfeee_0
  - toml=0.10.2=pyhd8ed1ab_1
  - tomli=2.2.1=pyhd8ed1ab_1
  - tornado=6.4.2=py39ha55e580_0
  - tqdm=4.67.1=pyhd8ed1ab_1
  - traitlets=5.14.3=pyhd8ed1ab_1
  - transformers=4.47.1=pyhd8ed1ab_0
  - typing-extensions=4.12.2=hd8ed1ab_1
  - typing_extensions=4.12.2=pyha770c72_1
  - tzdata=2024b=hc8b5060_0
  - ucrt=10.0.22621.0=h57928b3_1
  - unicodedata2=15.1.0=py39ha55e580_1
  - urllib3=2.3.0=pyhd8ed1ab_0
  - vc=14.3=ha32ba9b_23
  - vc14_runtime=14.42.34433=he29a5d6_23
  - vs2015_runtime=14.42.34433=hdffcdeb_23
  - wcwidth=0.2.13=pyhd8ed1ab_1
  - wheel=0.45.1=pyhd8ed1ab_1
  - win_inet_pton=1.1.0=pyh7428d3b_8
  - wrapt=1.17.0=py39ha55e580_0
  - xgboost=2.0.3=cpu_pyhb8f9a19_4
  - xorg-libxau=1.0.12=h0e40799_0
  - xorg-libxdmcp=1.1.5=h0e40799_0
  - xxhash=0.8.2=hcfcfb64_0
  - yaml=0.2.5=h8ffe710_2
  - yarl=1.18.3=py39ha55e580_0
  - zeromq=4.3.5=ha9f60a1_7
  - zipp=3.21.0=pyhd8ed1ab_1
  - zstandard=0.23.0=py39h9bf74da_1
  - zstd=1.5.6=h0ea2cb4_0
"


## ⚙️ Data Setup

The model relies on specific feature files stored in JSON format. 

## 🚀 Usage

You can run the model using `train.py`. The script handles training, cross-validation, evaluation, and plotting.



### Custom Training Configuration

You can modify hyperparameters and feature toggles via command-line arguments:

```bash
python train.py \
  --dataset NPInter5 \
  --Neg_method BalancedNegGen \
  --epochs 50 \
  --batch_size 512 \
  --lr 0.0005 \
  --hidden 128

```

### Feature Selection

Enable or disable specific biological features:

```bash
python train.py \
  --proteinkmer True \
  --lncRNAkmer True \
  --motif True \
  --pro3D True \
  --distance False

```

### Prediction Mode

To train the model and then use it to predict new interactions for all proteins in the dataset against all lncRNAs:

```bash
python train.py --Predicting True

```

*Results will be saved to `results/{timestamp}/predicted_interactions.json*`

## 📊 Arguments Reference

| Argument | Default | Description |
| --- | --- | --- |
| `--dataset` | `NPInter5` | Dataset name (NPInter2, RPI369, RPI7317, etc.) |
| `--Neg_method` | `BalancedNegGen` | Negative sampling method |
| `--epochs` | `30` | Number of training epochs |
| `--lr` | `0.001` | Learning rate |
| `--hidden` | `64` | Hidden layer dimension size |
| `--batch_size` | `1024` | GNN batch size |
| `--seeds` | `[50, 100...]` | List of seeds for cross-validation folds |
| `--pro3D` | `True` | Use Protein 3D structure (CNN processing) |
| `--motif` | `True` | Use lncRNA binding motifs |

## 🧠 Model Architecture

The model (`model.py`) consists of three main components:

1. **Feature Encoders:**
* **Sequence:** Linear layers process k-mer frequencies.
* **Structure:** A 1D Convolutional Neural Network (CNN) processes flattened Protein 3D structure data.
* **Embeddings:** Learnable embeddings for node IDs.


2. **Heterogeneous GNN:**
* Uses `GENConv`, `SAGEConv`, and `TransformerConv`.
* Operates on the heterogeneous graph (`ncRNA` and `Protein` nodes).


3. **Classifier:**
* An MLP that concatenates the learned node features of an edge pair to predict the interaction probability.



## 📄 Outputs

Upon completion, the script creates a timestamped folder in `results/` containing:

* **`model_info_result.txt`**: Logs of metrics (AUC, Accuracy, Recall, Precision, MCC).
* **Confusion Matrices**: PNG images.
* **Performance Plots**: Accuracy vs. Interaction Ratios/Counts (Scatter plots and Heatmaps).
* **`saved model/`**: Contains the model weights (`.pth`) and configuration.
