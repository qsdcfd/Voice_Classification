{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "lazypredict",
      "provenance": [],
      "authorship_tag": "ABX9TyNdPiwnRfjV0iZKPPlAQM2v"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "### 필요한 라이브러리 호출\n"
      ],
      "metadata": {
        "id": "Xzl_bneIelj3"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "RT8P55dWIRO0"
      },
      "outputs": [],
      "source": [
        "#오디오 처리 라이브러리\n",
        "import librosa\n",
        "import librosa.display as dsp\n",
        "from IPython.display import Audio"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/gdrive/')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "5z4JKANdISbW",
        "outputId": "37b9a3ff-bf3e-4d13-b351-c2e7f3ece157"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Drive already mounted at /content/gdrive/; to attempt to forcibly remount, call drive.mount(\"/content/gdrive/\", force_remount=True).\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#데이터 전처리\n",
        "\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import os\n",
        "from tqdm import tqdm"
      ],
      "metadata": {
        "id": "RbKTZT_LchTC"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#모델 재현성을 위하여 random seed고정\n",
        "import random\n",
        "\n",
        "def seed_everything(seed):\n",
        "    os.environ['PYTHONHASHSEED'] = str(seed)\n",
        "    np.random.seed(seed)\n",
        "\n",
        "seed_everything(929)"
      ],
      "metadata": {
        "id": "t1CE4r8EeNt0"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        ""
      ],
      "metadata": {
        "id": "zMKET-gJeROY"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "### 데이터 불러오기"
      ],
      "metadata": {
        "id": "38LqQVh2eo0t"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "train = pd.read_csv('/content/gdrive/MyDrive/drive-download-20220614T024615Z-001/train.csv')\n",
        "train.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "G0TWca73etek",
        "outputId": "f17b839d-9186-4bfa-d948-b63efa8a72db"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "  file_name  label\n",
              "0   001.wav      9\n",
              "1   002.wav      0\n",
              "2   004.wav      1\n",
              "3   005.wav      8\n",
              "4   006.wav      0"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-36e993de-78ef-40e1-a2a7-d160da579f76\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>file_name</th>\n",
              "      <th>label</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>001.wav</td>\n",
              "      <td>9</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>002.wav</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>004.wav</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>005.wav</td>\n",
              "      <td>8</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>006.wav</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-36e993de-78ef-40e1-a2a7-d160da579f76')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-36e993de-78ef-40e1-a2a7-d160da579f76 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-36e993de-78ef-40e1-a2a7-d160da579f76');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 5
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "14CFwUqvetjR",
        "outputId": "4aa1d737-b243-4ff7-e0c1-b5d0eb423da4"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 400 entries, 0 to 399\n",
            "Data columns (total 2 columns):\n",
            " #   Column     Non-Null Count  Dtype \n",
            "---  ------     --------------  ----- \n",
            " 0   file_name  400 non-null    object\n",
            " 1   label      400 non-null    int64 \n",
            "dtypes: int64(1), object(1)\n",
            "memory usage: 6.4+ KB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        ""
      ],
      "metadata": {
        "id": "6XEjJXSXe7V0"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "### 데이터 전처리\n",
        "\n",
        "- 소리는 보통 특정 주파수를 갖느 sin함수\n",
        "\n",
        "- 주파수 분석 기법 활용\n",
        "\n",
        "- 아날로그 데이터를 디지털 신호로 변환"
      ],
      "metadata": {
        "id": "jj0sxAdRe-__"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "data,sample_rate = librosa.load('/content/gdrive/MyDrive/Dacon/train/001.wav',sr=16000)\n",
        "print('sample_rate:', sample_rate, ',audio shape:', data.shape)\n",
        "print('length:', data.shape[0]/float(sample_rate), 'secs')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7Za6eVRHfJYv",
        "outputId": "d23350ec-27cf-446c-a80d-782b7d0c53f8"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "sample_rate: 16000 ,audio shape: (10192,)\n",
            "length: 0.637 secs\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "sampling rate의 의미는 초당 16000개(16000Hz 주파수)의 샘플을 가지고 있는 데이터라는 의미입니다. (1초에 음성 신호를 16000번 sampling)\n",
        "\n",
        "sampling rate의 defult값은 22050Hz인데, 16000Hz으로 설정한 이유는 사람의 목소리는 대부분 16000Hz 안에 포함된다고 합니다.\n",
        "\n",
        "또한 audio shape와 sampling rate를 이용해서 오디오 길이 계산을 할 수 있습니다."
      ],
      "metadata": {
        "id": "gTRGdwXRfsLr"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#train_wav데이터 프레임 생성\n",
        "def train_dataset():\n",
        "    folder = \"/content/gdrive/MyDrive/Dacon/train\"\n",
        "    dataset = []\n",
        "    for file in tqdm(os.listdir(folder),colour='green'):\n",
        "        if 'wav' in file:\n",
        "            abs_file_path = os.path.join(folder,file)\n",
        "            data, sr = librosa.load(abs_file_path, sr = 16000)\n",
        "            class_label = int(train[train.file_name == file].label)\n",
        "            dataset.append([data,class_label])\n",
        "    \n",
        "    print(\"Dataset 생성 완료\")\n",
        "    return pd.DataFrame(dataset,columns=['data','label'])"
      ],
      "metadata": {
        "id": "3Ol0Q_KwfkmT"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_wav = train_dataset()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Ua1glMjFggmc",
        "outputId": "8679bda0-7287-4aef-db3f-94409da7f00b"
      },
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|\u001b[32m██████████\u001b[0m| 400/400 [00:31<00:00, 12.90it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Dataset 생성 완료\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_wav.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "5lTd3kXSglSk",
        "outputId": "5afd767b-de04-4c77-a620-99c1fe4200f0"
      },
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                                                data  label\n",
              "0  [3.6655838e-05, -3.7366447e-06, 3.4776433e-05,...      5\n",
              "1  [0.00011985076, 0.00016174652, 0.00017246709, ...      9\n",
              "2  [-0.00043892296, -0.00073743664, -0.0006132907...      8\n",
              "3  [-4.289015e-05, 9.891299e-05, 2.6636611e-05, 0...      4\n",
              "4  [1.2653453e-05, 2.3892262e-05, -7.51332e-06, 4...      0"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-7732ccbe-ab0f-430c-91b7-7c39061b95b0\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>data</th>\n",
              "      <th>label</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>[3.6655838e-05, -3.7366447e-06, 3.4776433e-05,...</td>\n",
              "      <td>5</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>[0.00011985076, 0.00016174652, 0.00017246709, ...</td>\n",
              "      <td>9</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>[-0.00043892296, -0.00073743664, -0.0006132907...</td>\n",
              "      <td>8</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>[-4.289015e-05, 9.891299e-05, 2.6636611e-05, 0...</td>\n",
              "      <td>4</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>[1.2653453e-05, 2.3892262e-05, -7.51332e-06, 4...</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-7732ccbe-ab0f-430c-91b7-7c39061b95b0')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-7732ccbe-ab0f-430c-91b7-7c39061b95b0 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-7732ccbe-ab0f-430c-91b7-7c39061b95b0');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 10
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#train_wav.to_csv(\"train_wav.csv\")"
      ],
      "metadata": {
        "id": "bJOaP4Dbg8z7"
      },
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        ""
      ],
      "metadata": {
        "id": "k4mVFU_HhHPj"
      },
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "### 음성 데이터 특징 추출\n",
        "\n",
        "- 퓨리에 변환\n",
        "\n",
        "    - 입력 신호를 다양한 주파수를 가지는 주기함수들로 분해\n",
        "\n",
        "    - 노이즈 및 배경 소리로부터 실제로 유용한 소리의 특징 추출\n",
        "\n",
        "- MFCC\n",
        "\n",
        "   - 음성데이터를 특징벡터화\n",
        "\n",
        "   - 사람이 인지하기 좋은 20~40ms로 나눠준다\n",
        "\n",
        "- Mel-scale\n",
        "\n",
        "   - 주파수 간격이 넓어지면 감지 못함\n",
        "   - filter, scaling\n",
        "\n",
        "- Argument\n",
        "\n",
        "   - y : audio data\n",
        " \n",
        "   - sr : sampling rate\n",
        "\n",
        "   - n_mfcc : return 될 mfcc의 개수를 정해주는 파라미터. 더 다양한 데이터 특징을 추출하기 위해서 증가 시킵니다.\n",
        "\n",
        "   - n_fft : frame의 length를 결정하는 파라미터. n_fft를 설정하면 window size가 디폴트 값으로 n_fft가 됩니다.\n",
        "\n",
        "   - 사람의 목소리는 대부분 16000Hz 안에 포함이 되는데, 일반적으로 자연어 처리에서는 음성을 25m의 크기를 기본으로 하고 있습니다.\n",
        "\n",
        "(ex. 16000Hz인 음성에서는 25m의 음성의 크기를 가지고 있으면 n_fft는 16000 * 0.025 = 400 (sampling rate * frame_length = n_fft)가 됩니다.)\n",
        "\n",
        "\n",
        "   - hop_length : 윈도우 길이를 나타냅니다. 길이만큼 옆으로 가면서 데이터를 읽습니다.\n",
        "\n",
        "   - hop_length도 마찬가지로 window 간의 거리이므로 sampling rate * frame_stride 가 됩니다."
      ],
      "metadata": {
        "id": "AsxxxM8ZhMTO"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def extract_features(file):\n",
        "    audio, sample_rate = librosa.load(file, sr = 16000)\n",
        "    extracted_features = librosa.feature.mfcc(y=audio,\n",
        "                                              sr=sample_rate,\n",
        "                                              n_mfcc=40)\n",
        "\n",
        "    extracted_features = np.mean(extracted_features.T,axis=0)\n",
        "    return extracted_features\n"
      ],
      "metadata": {
        "id": "iNs_VubmhziH"
      },
      "execution_count": 12,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "extract_features('/content/gdrive/MyDrive/Dacon/train/001.wav')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_qZ7d3kliUZV",
        "outputId": "60830843-ef0b-4d01-f4a0-45db131236a5"
      },
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([-5.4157184e+02,  1.0199717e+02, -1.0018576e+01,  4.5054615e+01,\n",
              "        7.3112822e+00,  1.0971639e+01, -1.2032940e+01, -5.8687968e+00,\n",
              "       -1.8881404e+00,  4.2930880e+00, -8.1847525e+00, -2.3072269e+00,\n",
              "       -9.1721897e+00,  1.4182716e+01, -1.2839543e+01, -3.1000304e+00,\n",
              "       -3.0502682e+00, -2.1911802e+00, -6.2639456e+00, -5.1691580e+00,\n",
              "       -1.3974123e+01,  3.3810470e+00, -6.9977813e+00,  3.7736294e-01,\n",
              "       -4.4287410e+00,  1.0799456e+00, -1.3639281e+00,  4.2418461e+00,\n",
              "        2.3687012e+00,  2.8972092e+00,  2.6670651e+00,  1.8590590e+00,\n",
              "       -3.8219371e+00, -1.6171485e-01, -1.4186367e+00, -4.1422081e+00,\n",
              "       -5.3374414e+00, -9.9907315e-01, -3.3392251e+00, -4.2987284e-01],\n",
              "      dtype=float32)"
            ]
          },
          "metadata": {},
          "execution_count": 13
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def preprocess_train_dataset(data):\n",
        "    mfccs = []\n",
        "    for i in data:\n",
        "        extracted_features = librosa.feature.mfcc(y=i,\n",
        "                                              sr=16000,\n",
        "                                              n_mfcc=40)\n",
        "        extracted_features = np.mean(extracted_features.T,axis=0)\n",
        "        mfccs.append(extracted_features)\n",
        "            \n",
        "    return mfccs\n",
        "\n",
        "mfccs = preprocess_train_dataset(train_wav.data)\n",
        "mfccs = np.array(mfccs)"
      ],
      "metadata": {
        "id": "-GvucILyijgW"
      },
      "execution_count": 14,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "mfccs"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "l0pbv-iTkmu4",
        "outputId": "f7c32d2d-36db-44f4-8334-1a333f021d93"
      },
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[-5.0092383e+02,  1.0522231e+02, -3.0754614e+01, ...,\n",
              "         2.7968862e+00,  2.0259068e+00,  9.0582504e+00],\n",
              "       [-5.4157184e+02,  1.0199717e+02, -1.0018576e+01, ...,\n",
              "        -9.9907315e-01, -3.3392251e+00, -4.2987284e-01],\n",
              "       [-5.4121655e+02,  6.4172516e+01,  1.1352415e+01, ...,\n",
              "         9.8170424e-01, -1.2399026e+00,  1.7476788e+00],\n",
              "       ...,\n",
              "       [-6.4978589e+02,  8.1067482e+01,  2.5364252e+01, ...,\n",
              "         3.1881747e+00, -3.2863741e+00,  9.8691696e-01],\n",
              "       [-5.7801672e+02,  8.9649864e+01,  6.7791195e+00, ...,\n",
              "         3.5183573e+00, -6.8778282e-01,  2.9434049e+00],\n",
              "       [-5.1473218e+02,  6.9641624e+01, -2.2522993e+01, ...,\n",
              "        -8.1666952e-01, -7.3845682e+00, -2.4838428e+00]], dtype=float32)"
            ]
          },
          "metadata": {},
          "execution_count": 15
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        ""
      ],
      "metadata": {
        "id": "HMXQj0-Dkoy8"
      },
      "execution_count": 15,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "### 변수 및 모델 정의"
      ],
      "metadata": {
        "id": "UzcM1y_wkuT3"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "train_X, test_X, train_y, test_y = train_test_split(mfccs, train_wav.label, test_size=0.4)"
      ],
      "metadata": {
        "id": "91vN-XIYkwyD"
      },
      "execution_count": 16,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print('학습시킬 train셋:', train_X.shape, train_y.shape)\n",
        "print('검증할 val 셋:', test_X.shape, test_y.shape)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "FU5oDW5VlZsP",
        "outputId": "188a2c15-c227-4d53-9500-7b06b8f6c360"
      },
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "학습시킬 train셋: (240, 40) (240,)\n",
            "검증할 val 셋: (160, 40) (160,)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.ensemble import RandomForestClassifier\n",
        "\n",
        "model = RandomForestClassifier()\n",
        "\n",
        "model.fit(train_X, train_y)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "LeKUwVPgllFG",
        "outputId": "5ebb63e5-5c88-4942-b8f1-63aa5fd2340e"
      },
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "RandomForestClassifier()"
            ]
          },
          "metadata": {},
          "execution_count": 18
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        ""
      ],
      "metadata": {
        "id": "4LhP7v78l0fB"
      },
      "execution_count": 18,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "### 모델 성눙"
      ],
      "metadata": {
        "id": "k_cr_X07l2ek"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "\n",
        "def ACCURACY(true, pred):\n",
        "    score= np.mean(true==pred)\n",
        "    return score"
      ],
      "metadata": {
        "id": "azgebA4Al4ed"
      },
      "execution_count": 19,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "prediction = model.predict(test_X)\n",
        "\n",
        "score = ACCURACY(test_y,  prediction)\n",
        "\n",
        "print(f'모델의 정확도는 {score*100:.2f}% 입니다')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_Qs0MxMVmBLb",
        "outputId": "09a293c0-ccd0-4db2-d035-99dfdc18cbf2"
      },
      "execution_count": 20,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "모델의 정확도는 66.25% 입니다\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        ""
      ],
      "metadata": {
        "id": "1owDp93bmOcj"
      },
      "execution_count": 20,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "### test.csv분류하기"
      ],
      "metadata": {
        "id": "R3T4VAyWmXvM"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "test = pd.read_csv(\"/content/gdrive/MyDrive/drive-download-20220614T024615Z-001/test.csv\")\n",
        "test.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "Ovj9NQQsmaLs",
        "outputId": "6619670c-22c1-4d51-b275-6b1f290d3f92"
      },
      "execution_count": 21,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "  file_name\n",
              "0   003.wav\n",
              "1   008.wav\n",
              "2   010.wav\n",
              "3   015.wav\n",
              "4   024.wav"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-3c1fcf28-1944-4fd9-baf7-d29a46f0687e\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>file_name</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>003.wav</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>008.wav</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>010.wav</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>015.wav</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>024.wav</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-3c1fcf28-1944-4fd9-baf7-d29a46f0687e')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-3c1fcf28-1944-4fd9-baf7-d29a46f0687e button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-3c1fcf28-1944-4fd9-baf7-d29a46f0687e');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 21
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def test_dataset():\n",
        "    folder = '/content/gdrive/MyDrive/test'\n",
        "    dataset=[]\n",
        "    for file in tqdm(os.listdir(folder),colour='green'):\n",
        "        if 'wav' in file:\n",
        "            abs_file_path = os.path.join(folder,file)\n",
        "            data, sr = librosa.load(abs_file_path, sr = 16000)\n",
        "            dataset.append([data, file])\n",
        "    \n",
        "    print(\"Dataset 생성 완료\")\n",
        "    return pd.DataFrame(dataset,columns=['data', 'file_name'])"
      ],
      "metadata": {
        "id": "Cls0DSugm85H"
      },
      "execution_count": 22,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "test_wav = test_dataset()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "hEnoeA-9nMmw",
        "outputId": "b2c75a67-cc07-4286-e445-388b16f9516f"
      },
      "execution_count": 23,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|\u001b[32m██████████\u001b[0m| 201/201 [00:07<00:00, 26.00it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Dataset 생성 완료\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#test_wav.to_csv('test_wav.csv')"
      ],
      "metadata": {
        "id": "pxKtLn_dnOmS"
      },
      "execution_count": 24,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "mfccs_2 = preprocess_train_dataset(test_wav.data)\n",
        "mfccs_2 = np.array(mfccs_2)"
      ],
      "metadata": {
        "id": "XSB6vmsHnSah"
      },
      "execution_count": 25,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# 모델 선언\n",
        "model = RandomForestClassifier()\n",
        "\n",
        "# 모델 학습\n",
        "model.fit(mfccs, train_wav.label)\n",
        "\n",
        "# 모델 예측\n",
        "prediction = model.predict(mfccs_2)\n",
        "\n",
        "test_wav['label'] = prediction"
      ],
      "metadata": {
        "id": "oOGlll7hnhcx"
      },
      "execution_count": 26,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "test_wav.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "PgkCFizdnmbK",
        "outputId": "d1d83033-5f0a-42b9-8f8a-586564a611a2"
      },
      "execution_count": 27,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                                                data file_name  label\n",
              "0  [0.00027645202, 0.00047594117, 0.0004510493, 0...   010.wav      3\n",
              "1  [0.00018057936, 0.00026723626, 0.00017172625, ...   024.wav      2\n",
              "2  [-9.729892e-05, -0.00013003226, -8.376751e-05,...   025.wav      0\n",
              "3  [-0.00012691804, -0.00019610435, -0.0001295761...   015.wav      3\n",
              "4  [0.00021518696, 0.00039842026, 0.00028533765, ...   078.wav      0"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-31ff2e33-d83f-4111-a621-5f726bdb8fb8\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>data</th>\n",
              "      <th>file_name</th>\n",
              "      <th>label</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>[0.00027645202, 0.00047594117, 0.0004510493, 0...</td>\n",
              "      <td>010.wav</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>[0.00018057936, 0.00026723626, 0.00017172625, ...</td>\n",
              "      <td>024.wav</td>\n",
              "      <td>2</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>[-9.729892e-05, -0.00013003226, -8.376751e-05,...</td>\n",
              "      <td>025.wav</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>[-0.00012691804, -0.00019610435, -0.0001295761...</td>\n",
              "      <td>015.wav</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>[0.00021518696, 0.00039842026, 0.00028533765, ...</td>\n",
              "      <td>078.wav</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-31ff2e33-d83f-4111-a621-5f726bdb8fb8')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-31ff2e33-d83f-4111-a621-5f726bdb8fb8 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-31ff2e33-d83f-4111-a621-5f726bdb8fb8');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 27
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "testset = test_wav[['file_name', 'label']]\n",
        "\n",
        "pred_df = testset.copy()\n",
        "pred_df = pred_df.sort_values(by=[pred_df.columns[0]], ascending=[True]).reset_index(drop=True)\n",
        "pred_df.head()\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "SUOVgqXJnopm",
        "outputId": "955c2fcc-6538-4530-fdbc-b51297f400e5"
      },
      "execution_count": 28,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "  file_name  label\n",
              "0   003.wav      0\n",
              "1   008.wav      1\n",
              "2   010.wav      3\n",
              "3   015.wav      3\n",
              "4   024.wav      2"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-fdf159c5-9db8-45c4-9974-91a8325f2954\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>file_name</th>\n",
              "      <th>label</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>003.wav</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>008.wav</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>010.wav</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>015.wav</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>024.wav</td>\n",
              "      <td>2</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-fdf159c5-9db8-45c4-9974-91a8325f2954')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-fdf159c5-9db8-45c4-9974-91a8325f2954 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-fdf159c5-9db8-45c4-9974-91a8325f2954');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 28
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "submission = pd.read_csv('/content/gdrive/MyDrive/drive-download-20220614T024615Z-001/sample_submission.csv')\n",
        "submission['label'] = pred_df['label']\n",
        "submission.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "0bMs_uj5n_dl",
        "outputId": "eeb7200e-7b47-4206-ed59-367f1fc7ec29"
      },
      "execution_count": 29,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "  file_name  label\n",
              "0   003.wav      0\n",
              "1   008.wav      1\n",
              "2   010.wav      3\n",
              "3   015.wav      3\n",
              "4   024.wav      2"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-ca16ec6a-79a7-4463-bd2d-9c4c3fa8ea10\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>file_name</th>\n",
              "      <th>label</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>003.wav</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>008.wav</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>010.wav</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>015.wav</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>024.wav</td>\n",
              "      <td>2</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-ca16ec6a-79a7-4463-bd2d-9c4c3fa8ea10')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-ca16ec6a-79a7-4463-bd2d-9c4c3fa8ea10 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-ca16ec6a-79a7-4463-bd2d-9c4c3fa8ea10');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 29
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#submission.to_csv('submit.csv', index=False)\n"
      ],
      "metadata": {
        "id": "gZ7FEdCwoMiZ"
      },
      "execution_count": 30,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Lazypredict"
      ],
      "metadata": {
        "id": "h-5XM3UJs5fE"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#!pip install lazypredict\n"
      ],
      "metadata": {
        "id": "_TUnlLKPoQQ7"
      },
      "execution_count": 31,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import lazypredict\n"
      ],
      "metadata": {
        "id": "gQcFotc8ptTs"
      },
      "execution_count": 32,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from lazypredict.Supervised import LazyClassifier\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "tmf32AGKqPGc",
        "outputId": "65e4df04-db5b-430f-a338-285c4722a70b"
      },
      "execution_count": 33,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/sklearn/utils/deprecation.py:143: FutureWarning: The sklearn.utils.testing module is  deprecated in version 0.22 and will be removed in version 0.24. The corresponding classes / functions should instead be imported from sklearn.utils. Anything that cannot be imported from sklearn.utils is now part of the private API.\n",
            "  warnings.warn(message, FutureWarning)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n"
      ],
      "metadata": {
        "id": "BhqGm2KIqRLb"
      },
      "execution_count": 34,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "test_wav.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "v0db266oqqC5",
        "outputId": "96fcf9c7-db77-4000-c490-1cf36fa13359"
      },
      "execution_count": 35,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                                                data file_name  label\n",
              "0  [0.00027645202, 0.00047594117, 0.0004510493, 0...   010.wav      3\n",
              "1  [0.00018057936, 0.00026723626, 0.00017172625, ...   024.wav      2\n",
              "2  [-9.729892e-05, -0.00013003226, -8.376751e-05,...   025.wav      0\n",
              "3  [-0.00012691804, -0.00019610435, -0.0001295761...   015.wav      3\n",
              "4  [0.00021518696, 0.00039842026, 0.00028533765, ...   078.wav      0"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-a2e82356-236b-41a7-a2b0-76fec8cb276e\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>data</th>\n",
              "      <th>file_name</th>\n",
              "      <th>label</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>[0.00027645202, 0.00047594117, 0.0004510493, 0...</td>\n",
              "      <td>010.wav</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>[0.00018057936, 0.00026723626, 0.00017172625, ...</td>\n",
              "      <td>024.wav</td>\n",
              "      <td>2</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>[-9.729892e-05, -0.00013003226, -8.376751e-05,...</td>\n",
              "      <td>025.wav</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>[-0.00012691804, -0.00019610435, -0.0001295761...</td>\n",
              "      <td>015.wav</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>[0.00021518696, 0.00039842026, 0.00028533765, ...</td>\n",
              "      <td>078.wav</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-a2e82356-236b-41a7-a2b0-76fec8cb276e')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-a2e82356-236b-41a7-a2b0-76fec8cb276e button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-a2e82356-236b-41a7-a2b0-76fec8cb276e');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 35
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_wav.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "Tj1V-b1-qsMW",
        "outputId": "738f4dc7-2362-41b7-fe08-d75a90b5b82b"
      },
      "execution_count": 36,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                                                data  label\n",
              "0  [3.6655838e-05, -3.7366447e-06, 3.4776433e-05,...      5\n",
              "1  [0.00011985076, 0.00016174652, 0.00017246709, ...      9\n",
              "2  [-0.00043892296, -0.00073743664, -0.0006132907...      8\n",
              "3  [-4.289015e-05, 9.891299e-05, 2.6636611e-05, 0...      4\n",
              "4  [1.2653453e-05, 2.3892262e-05, -7.51332e-06, 4...      0"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-c5494dd8-32b2-4a61-81ce-13d727599a14\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>data</th>\n",
              "      <th>label</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>[3.6655838e-05, -3.7366447e-06, 3.4776433e-05,...</td>\n",
              "      <td>5</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>[0.00011985076, 0.00016174652, 0.00017246709, ...</td>\n",
              "      <td>9</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>[-0.00043892296, -0.00073743664, -0.0006132907...</td>\n",
              "      <td>8</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>[-4.289015e-05, 9.891299e-05, 2.6636611e-05, 0...</td>\n",
              "      <td>4</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>[1.2653453e-05, 2.3892262e-05, -7.51332e-06, 4...</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-c5494dd8-32b2-4a61-81ce-13d727599a14')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-c5494dd8-32b2-4a61-81ce-13d727599a14 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-c5494dd8-32b2-4a61-81ce-13d727599a14');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 36
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.utils import shuffle\n",
        "from lazypredict.Supervised  import LazyClassifier\n",
        "from matplotlib import pyplot as plt"
      ],
      "metadata": {
        "id": "i1RGDVzFspXB"
      },
      "execution_count": 37,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X_train, X_test, y_train, y_test = train_test_split(mfccs, train_wav.label,test_size=.2,random_state =123)\n",
        "\n",
        "clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)\n",
        "models,predictions = clf.fit(X_train, X_test, y_train, y_test)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7rN95Ay-uEKq",
        "outputId": "d89a4853-9ec7-48fd-83a7-e08eaf927cf7"
      },
      "execution_count": 38,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|██████████| 29/29 [00:03<00:00,  8.05it/s]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(models)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7drItSyIuM2j",
        "outputId": "48ab4f67-74fe-4f15-b4fe-d34cc2a80079"
      },
      "execution_count": 39,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                               Accuracy  Balanced Accuracy ROC AUC  F1 Score  \\\n",
            "Model                                                                          \n",
            "LinearDiscriminantAnalysis         0.72               0.74    None      0.73   \n",
            "LGBMClassifier                     0.70               0.73    None      0.70   \n",
            "XGBClassifier                      0.70               0.73    None      0.70   \n",
            "ExtraTreesClassifier               0.65               0.67    None      0.65   \n",
            "LogisticRegression                 0.64               0.67    None      0.64   \n",
            "BaggingClassifier                  0.66               0.66    None      0.65   \n",
            "RidgeClassifierCV                  0.61               0.64    None      0.60   \n",
            "RandomForestClassifier             0.62               0.63    None      0.63   \n",
            "NuSVC                              0.57               0.62    None      0.59   \n",
            "RidgeClassifier                    0.59               0.62    None      0.58   \n",
            "GaussianNB                         0.59               0.61    None      0.58   \n",
            "LinearSVC                          0.59               0.60    None      0.60   \n",
            "CalibratedClassifierCV             0.57               0.59    None      0.57   \n",
            "SVC                                0.54               0.59    None      0.54   \n",
            "PassiveAggressiveClassifier        0.56               0.58    None      0.57   \n",
            "Perceptron                         0.56               0.58    None      0.57   \n",
            "SGDClassifier                      0.56               0.58    None      0.58   \n",
            "NearestCentroid                    0.56               0.56    None      0.56   \n",
            "DecisionTreeClassifier             0.49               0.47    None      0.49   \n",
            "KNeighborsClassifier               0.39               0.42    None      0.39   \n",
            "BernoulliNB                        0.41               0.42    None      0.41   \n",
            "AdaBoostClassifier                 0.21               0.23    None      0.17   \n",
            "QuadraticDiscriminantAnalysis      0.21               0.20    None      0.22   \n",
            "ExtraTreeClassifier                0.16               0.18    None      0.16   \n",
            "DummyClassifier                    0.10               0.11    None      0.11   \n",
            "LabelSpreading                     0.09               0.10    None      0.01   \n",
            "LabelPropagation                   0.09               0.10    None      0.01   \n",
            "\n",
            "                               Time Taken  \n",
            "Model                                      \n",
            "LinearDiscriminantAnalysis           0.03  \n",
            "LGBMClassifier                       1.05  \n",
            "XGBClassifier                        0.52  \n",
            "ExtraTreesClassifier                 0.22  \n",
            "LogisticRegression                   0.06  \n",
            "BaggingClassifier                    0.10  \n",
            "RidgeClassifierCV                    0.03  \n",
            "RandomForestClassifier               0.30  \n",
            "NuSVC                                0.04  \n",
            "RidgeClassifier                      0.03  \n",
            "GaussianNB                           0.02  \n",
            "LinearSVC                            0.13  \n",
            "CalibratedClassifierCV               0.44  \n",
            "SVC                                  0.05  \n",
            "PassiveAggressiveClassifier          0.03  \n",
            "Perceptron                           0.03  \n",
            "SGDClassifier                        0.04  \n",
            "NearestCentroid                      0.01  \n",
            "DecisionTreeClassifier               0.02  \n",
            "KNeighborsClassifier                 0.02  \n",
            "BernoulliNB                          0.02  \n",
            "AdaBoostClassifier                   0.21  \n",
            "QuadraticDiscriminantAnalysis        0.02  \n",
            "ExtraTreeClassifier                  0.02  \n",
            "DummyClassifier                      0.01  \n",
            "LabelSpreading                       0.04  \n",
            "LabelPropagation                     0.03  \n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        ""
      ],
      "metadata": {
        "id": "XDLK1o3BuQ2m"
      },
      "execution_count": 39,
      "outputs": []
    }
  ]
}