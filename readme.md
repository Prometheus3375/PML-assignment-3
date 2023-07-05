# Installation

1. Install [Python 3.9](https://www.python.org/downloads/).
1. Open terminal in the root directory of this assignment.
1. (Optional) Initialize virtual environment and activate it according to the
   [tutorial](https://docs.python.org/3/library/venv.html).
1. Run `python -m pip install -U pip setuptools wheel` to update pip, setuptools and wheel packages.
1. Install necessary NVIDIA drivers to allow PyTorch use GPU.
   *Note*: you do not need to install CUDA binaries.
    - [GPUs with CUDA](https://developer.nvidia.com/cuda-gpus). *This is not a complete list.*
    - [CUDA executables and documentation](https://developer.nvidia.com/cuda-downloads).
    - Read [docs](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
      to find out which drivers you need to support necessary CUDA version.
1. Install [PyTorch 1.8.x](https://pytorch.org/) with CUDA supported by your graphic card and drivers.

# Train data

[comment]: <> (1. [EN-RU plain-text bitexts of UN Parallel Corpus]&#40;https://conferences.unite.un.org/UNCORPUS/en/DownloadOverview&#41;)

1. [Yandex English-Russian corpus](https://translate.yandex.ru/corpus?lang=en)
1. [ParaCrawl Russian Corpus](https://www.paracrawl.eu/)
